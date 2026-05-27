# SimpleVLA-RL: Bug Fixes for OpenVLA-OFT + LIBERO with Proprio & Multi-Image

## Context

Running GRPO RL training on LIBERO simulation using `openvla-oft` checkpoint with:
- `use_proprio=True` — 8D proprioception (eef position + quat-to-axis-angle + gripper)
- `num_images_in_input=2` — third-person camera + wrist camera
- `use_substep_rl=True` — APD-based sub-task reward

---

## Bug 1 — HF Cache Not Patched After `overwrite_vla_ckpt_utils.sh`

**File:** `examples/overwrite_vla_ckpt_utils.sh`

### Problem

The script copies patched modeling files (`modeling_prismatic.py`, etc.) into the local checkpoint directory. However, when HuggingFace first loads a model with `trust_remote_code=True`, it caches the remote code separately under:

```
.cache/huggingface/modules/transformers_modules/<ckpt_name>/
```

Subsequent loads read from this cache — not the checkpoint directory — so patches to the checkpoint dir were silently ignored.

**Symptom:** `RuntimeError: split_with_sizes expects split_sizes to sum to 12, got [3, 3]`
The cached old code expected 1 image (3 channels), but 2 images (6 channels each = 12 total) were passed.

### Fix

Extended the script to also patch the HF modules cache after patching the checkpoint dir:

```bash
HF_MODULES_CACHE="${SCRIPT_DIR}/.cache/huggingface/modules/transformers_modules"
for CACHE_DIR in "$HF_MODULES_CACHE"/*/; do
    CACHE_NAME=$(basename "$CACHE_DIR")
    if [[ "$CACHE_NAME" == *"$CKPT_NAME"* ]]; then
        for file in "${FILES[@]}"; do
            cp -f "$file" "$CACHE_DIR/"
        done
    fi
done
```

---

## Bug 2 — `process_input` Did Not Handle LIBERO Proprio

**File:** `verl/workers/rollout/rob_rollout.py`

### Problem

`process_input()` initialized and collected `batchdata["proprio"]` only inside a `if "robotwin" in self.config.task_suite_name` guard. For LIBERO, the list was never populated and never stacked onto GPU.

### Fix

- Changed proprio initialization guard to fire whenever `self.config.use_proprio` is True (not robotwin-only)
- Added a LIBERO branch to collect the raw 8D state:
  ```python
  elif "state" in input_data:
      proprio = input_data["state"].astype(np.float32)
      batchdata["proprio"].append(torch.from_numpy(proprio))
  ```
- Changed the stack+device step to fire whenever `use_proprio` is True and the list is non-empty:
  ```python
  if self.config.use_proprio and "proprio" in batchdata and len(batchdata["proprio"]) > 0:
      batchdata["proprio"] = torch.stack(batchdata["proprio"], dim=0).to(device)
  ```

---

## Bug 3 — `proprio` Never Copied Into `vla_history`

**File:** `verl/workers/rollout/rob_rollout.py`

### Problem

In `_generate_minibatch_libero`, after calling `_generate_one_step_oft()` — which correctly returns `vla_output["proprio"]` — the `step_data` dict was built explicitly without copying `proprio`:

```python
step_data = {
    "responses": vla_output["responses"],
    "input_ids": vla_output["input_ids"],
    "attention_mask": vla_output["attention_mask"],
    "action": actions,
    "step": step
}
# vla_output["proprio"] exists but is never copied here
vla_history.append(step_data)
```

As a result, every entry in `vla_history` was missing `"proprio"`. `_prepare_output_batch` checked `"proprio" in vla_history[0]` before including it in the output TensorDict — always False — so `proprio` was never written to the batch. `dp_rob.py` then unconditionally tried to read it (when `use_proprio=True`).

**Symptom:** `KeyError: 'key "proprio" not found in TensorDict'`

### Fix

Added after `step_data` construction:

```python
if "proprio" in vla_output:
    step_data["proprio"] = vla_output["proprio"]
```

---

## Bug 4 — `torch.Tensor(proprio)` Fails on Multi-Element Input

**File:** `verl/utils/vla_utils/openvla_oft/modeling_prismatic.py`

### Problem

`ProprioProjector.forward()` called `torch.Tensor(proprio)` to normalize the input. `torch.Tensor(x)` only works when `x` is a Python scalar or 0-d tensor. For an 8D numpy array or a batched tensor it raises:

```
ValueError: only one element tensors can be converted to Python scalars
```

### Fix

Replaced all 2 occurrences with `torch.as_tensor(proprio)`, which correctly handles both numpy arrays and existing tensors without unnecessary copying.

---

## Bug 5 — Proprio Shape Assertion Uses Wrong Reference Dimension

**File:** `verl/workers/actor/dp_rob.py`

### Problem

Two places (in `_forward_micro_batch` and `_forward_micro_batch_entropy`) asserted:

```python
assert micro_batch["proprio"].size(2) == self.config.action_token_len
```

`self.config.action_token_len = 7` is the number of action tokens per chunk. LIBERO proprio has 8 dimensions, so `proprio.size(2) = 8 != 7` and the assertion always fails.

This assertion was written for RoboTwin where `action_token_len` happened to coincidentally equal `proprio_dim`, making it pass there while silently being wrong for any other task.

**Symptom:** `AssertionError` inside `compute_log_prob` → `_forward_micro_batch`

### Fix

Removed the `size(2)` dimension check from both assertions, keeping only the `batch_size` and `traj_len` checks which are correct and task-agnostic:

```python
# Before
assert micro_batch["proprio"].size(0) == batch_size and micro_batch["proprio"].size(1) == traj_len and micro_batch["proprio"].size(2) == self.config.action_token_len

# After
assert micro_batch["proprio"].size(0) == batch_size and micro_batch["proprio"].size(1) == traj_len
```

---

## Feature — Autoregressive Inference Path for Non-OFT SFT Models (Bug 6–9)

**Context:** Models trained with `use_l1_regression=False, use_diffusion=False` (standard cross-entropy on 56 action tokens, teacher-forced) were evaluated with the OFT parallel inference path (`_verl_discrete_prediction`), which zeros all action token embeddings before a single forward pass. This mismatches training (teacher-forced causal LM), causing near-zero rollout SR despite the checkpoint working in SFT eval.

**Fix:** Added a full autoregressive inference + forward pass path behind a `use_autoregressive` flag.

---

### Change 6 — `forward_causal_lm` + `generate_autoregressive` in PrismaticVLM

**File:** `verl/utils/vla_utils/openvla_oft/modeling_prismatic.py`

Added three methods to `OpenVLAForActionPrediction`:

- **`forward_causal_lm`**: Standard causal LM forward without OFT placeholder insertion or action embedding zeroing. Handles cached generation (`past_key_values`), multimodal (first token), and unimodal (subsequent tokens) branches. Returns `PrismaticCausalLMOutputWithPast`.
- **`prepare_inputs_for_generation`**: HF `generate()` compatibility shim — routes to `forward_causal_lm` signature.
- **`generate_autoregressive`**: Temporarily swaps `self.forward → self.forward_causal_lm`, calls HF `self.generate(...)`, then restores. Allows using the full HF generation loop (beam search, sampling, greedy) with the causal-LM forward.

---

### Change 7 — Autoregressive rollout branch in `_generate_one_step`

**File:** `verl/workers/rollout/rob_rollout.py`

- `_generate_one_step` now checks `getattr(self.config, 'use_autoregressive', False)` and dispatches to new `_generate_one_step_autoregressive` instead of `_generate_one_step_oft`.
- **`_generate_one_step_autoregressive`**: Calls `self.module.generate_autoregressive(max_new_tokens=56, ...)`, pads/truncates response to exactly 56 tokens, decodes token IDs → unnormalized actions → real actions via `bin_centers` + `get_action_stats`, and returns the same dict format as `_generate_one_step_oft`.

**Bug fix (Change 9):** `normalized_actions` was unnormalized while still shaped `(batch, 56)` instead of `(batch, 8, 7)`, causing a broadcast error against `(7,)` stats arrays. Fixed by reshaping to `(batch, chunks, dims)` **before** unnormalization (removing the trailing reshape).

---

### Change 8 — Autoregressive actor forward pass in `dp_rob.py`

**File:** `verl/workers/actor/dp_rob.py`

Modified `_forward_micro_batch`, `_forward_micro_batch_update`, and `_forward_micro_batch_entropy` to add a `use_ar` branch that:

1. Reconstructs the full sequence: `full_ids = cat(input_ids_unpad, responses)`
2. Builds a full attention mask: `full_attn = cat(attention_mask_unpad, ones_for_response)`
3. Calls `self.actor_module.forward_causal_lm(full_ids, full_attn, pixel_values)`
4. Extracts response logits: `logits[:, -response_length - 1:-1]`
5. Applies per-chunk masking via `generate_traj_mask(finish_step, traj_len * 8)`

---

### Change 9 — Hydra config plumbing for `use_autoregressive`

**File:** `examples/run_openvla_oft_substep_rl_libero.sh`

Added `USE_AUTOREGRESSIVE="${USE_AUTOREGRESSIVE:-False}"` and passed it to Hydra as:
```bash
+actor_rollout_ref.rollout.use_autoregressive=$USE_AUTOREGRESSIVE \
+actor_rollout_ref.actor.use_autoregressive=$USE_AUTOREGRESSIVE \
```
(The `+` prefix is required for OmegaConf struct mode to accept new keys not in the base config.)

**File:** `apd_trail.sh`

Added `export USE_AUTOREGRESSIVE="True"` to enable the autoregressive path for the APD experiment.

---

## Summary

| # | File | Root Cause | Symptom |
|---|------|------------|---------|
| 1 | `examples/overwrite_vla_ckpt_utils.sh` | HF modules cache not patched, only checkpoint dir patched | `split_with_sizes` error on image channel mismatch |
| 2 | `verl/workers/rollout/rob_rollout.py` | LIBERO proprio never collected in `process_input` | Silent zero-proprio / downstream errors |
| 3 | `verl/workers/rollout/rob_rollout.py` | `proprio` not copied from `vla_output` into `step_data` / `vla_history` | `KeyError: 'proprio'` in TensorDict |
| 4 | `verl/utils/vla_utils/openvla_oft/modeling_prismatic.py` | `torch.Tensor()` cannot convert multi-element array/tensor | `ValueError: only one element tensors` |
| 5 | `verl/workers/actor/dp_rob.py` | Shape assertion compared `proprio_dim` against `action_token_len` | `AssertionError` in `compute_log_prob` |
| 6 | `modeling_prismatic.py` | OFT parallel inference mismatches teacher-forced training → near-zero SR | Added `forward_causal_lm` + `generate_autoregressive` |
| 7 | `rob_rollout.py` | No autoregressive rollout path existed | Added `_generate_one_step_autoregressive` |
| 8 | `dp_rob.py` | Actor forward assumed OFT embedding zeroing | Added `use_ar` branch using `forward_causal_lm` |
| 9 | `run_openvla_oft_substep_rl_libero.sh`, `apd_trail.sh` | `use_autoregressive` key rejected by OmegaConf struct | Use `+` prefix Hydra override; unnorm broadcast shape fix |
