# Experiment Plan — Achieving >45% Task-Mode SR on LIBERO-PRO

**Goal:** APD + Contrastive RL achieves >45% LIBERO-PRO task-mode SR for most suites, with sem/object >85%.

**Current best (APD_scalled):** spatial 31.4%, object 10%, goal 17.2%, 10: 9.4%

---

## Phase 1 — Wait & Evaluate Current Runs
**Status: IN PROGRESS (2026-04-07)**

- [ ] AMD job #15417 finishes (RL on apd_discrete_160k, ~33h remaining)
- [ ] NV eval finishes (oft_plus all modes)
- [ ] Record oft_plus results in `logs/2026-04-07.md`
- [ ] Transfer RL checkpoint from AMD → NV, extract
- [ ] Run `auto_eval_nv40_pro.sh` in `screen exp` on RL checkpoint
- [ ] Record RL results and compare to targets

**Decision gate:** If RL task-mode ≥45% for ≥3 suites → done. Otherwise → Phase 2.

---

## Phase 2 — Contrastive RL Tuning (if Phase 1 insufficient)

The RL hyperparameters most likely to improve task-mode score:

### 2a. Increase contrastive reward weight
- Current: VERIFIER_REWARD_COEF=5, CONTRASTIVE_REWARD_COEF=2
- Try: CONTRASTIVE_REWARD_COEF=4 (or 5)
- Rationale: the theory shows R_contrast directly optimises I(A;L|V). If task SR is OK but instruction sensitivity is low, boost the contrastive signal.
- Change in `apd_trail.sh`: `export CONTRASTIVE_REWARD_COEF=4`

### 2b. Stronger SFT base for RL
- Current RL base: `apd_discrete_160k` — weak lan performance (25-49%)
- Try: train `apd_discrete` longer (200k → already exists as `substep_vla_discrete--180000_chkpt`?) or wait for 200k
- APD_discrete 200k is already being trained (see `substep_plus_scalled_regressive.sh`, max_steps=200005)
- Check if 180k+ checkpoint exists and use as RL base

### 2c. Fix APD_scalled goal weakness
- APD_scalled goal lan is only 38.4% — unexpectedly low
- Investigate: is the goal suite data underrepresented in LIBERO-plus? Check substep label distribution
- Possibly: retrain with goal-suite oversampling or debug the substep labels for goal tasks

### 2d. Direct RL on APD_scalled (future)
- Implement an L1-regression-compatible RL mode that doesn't require autoregressive generation
- This would allow using the stronger APD_scalled as RL starting point
- Complexity: high — requires rewriting rollout to support L1 head

---

## Phase 3 — If RL Fails, Improve APD SFT

If RL cannot push task-mode >45%, the SFT itself may be insufficient (H(L^sub|V) not high enough).

### 3a. Scale substep granularity
- More granular substep labels → higher H(L^sub|V) per timestep
- Check `substep_labels_scaled_output.json` vs `substep_labels_output.json` — scaled version should have more substeps
- May need to regenerate labels with more splits

### 3b. Substep instruction augmentation during SFT
- At training time, occasionally swap substep instruction to a wrong one and apply a loss penalty
- This is a SFT-level version of the contrastive idea
- Would directly train the model to produce different actions under different instructions

### 3c. Investigate swap-mode = 0.000 for APD_scalled object
- APD_scalled object swap = 0.000 — the model fails completely when task-level instruction is swapped
- This suggests it's still using vision heavily and ignoring the instruction even with APD
- Look at the swap evaluation: objects are the same, only instruction differs — suggests APD substeps for object suite may be too visually-grounded

---

## Iteration Loop (daily)

```
Morning:
  1. ssh to AMD: squeue -u yuhang → any jobs done?
  2. ssh to NV: screen -r exp → eval finished?
  
If AMD done:
  3. Check log: SimpleVLA-RL/slurm/logs/oft_substep_rl_libero_<jobid>.out
  4. tar checkpoint, scp to NV

If NV eval done:
  5. Read ckpts/*.txt → update logs/YYYY-MM-DD.md
  6. Compare to targets → decide next action

If below target:
  7. Edit code/slurm script locally
  8. git add && git commit -m "Veldt- ..." && git push
  9. AMD: git pull && sbatch slurms/<script>.sh
  10. NV: run next eval if checkpoint ready

EOD:
  11. Write EOD recap in today's log
  12. Create tomorrow's log with "Status at Session Start"
```

---

## Commit Log (Veldt-)

| Date | Commit | Description |
|------|--------|-------------|
| 2026-04-07 | (init) | CLAUDE.md, PLAN.md, logs/ setup — no code change |

---

## Open Questions

1. What does the RL val score of 0.125 at step 3 translate to in LIBERO-PRO task mode? (RL val uses standard task success, not LIBERO-PRO swap evaluation)
2. Why is APD_scalled goal lan so low (38.4%)? Substep label quality issue or data issue?
3. Can the contrastive RL signal help with `swap=0` for the object suite?
