"""
clip_retrieval_accuracy.py  —  §4.5 (Optional) CLIP Retrieval Accuracy

Evaluates how accurately the SigCLIP/open_clip module selects the correct
substep given the robot's current visual observation.

Ground-truth labelling strategy
---------------------------------
  Two modes are supported, controlled by --label_mode:

  1. "uniform"  (default)
     Divide each successful episode into N equal segments (N = number of substeps).
     Frame t belongs to substep floor(t * N / T) where T = episode length.
     Simple but reasonable for tasks with roughly uniform sub-task durations.

  2. "json"
     Load pre-annotated episode JSON files.  Each JSON must have a list of records:
         [{"frame_idx": 0, "image_path": "...", "true_substep_idx": 0, "substeps": [...]}, ...]
     These can come from expert labelling or from EOS-threshold switching logs.

Metrics computed
----------------
  Top-1 Accuracy : fraction of frames where argmax(similarity) == true_substep_idx
  Top-2 Accuracy : true substep is in the top-2 ranked candidates
  Per-substep    : confusion matrix & per-class recall
  Smoothed Acc   : Top-1 after applying majority-vote smoothing (window=5 frames)

Usage
-----
    python experiments/analysis/clip_retrieval_accuracy.py \
        --sigclip_model_path google/siglip-so400m-patch14-384 \
        --apd_plans_path APD_plans.json \
        --image_root /path/to/episode_images \
        --label_mode uniform \
        --num_episodes_per_task 5 \
        --output_dir ./analysis_outputs/clip_retrieval

    # Or with pre-annotated JSON files
    python experiments/analysis/clip_retrieval_accuracy.py \
        --sigclip_model_path google/siglip-so400m-patch14-384 \
        --label_mode json \
        --annotated_json /path/to/annotations.json \
        --output_dir ./analysis_outputs/clip_retrieval

Outputs
-------
  clip_retrieval_results.json  : Full metrics per task and globally
  confusion_matrix.png         : Substep confusion matrix (averaged across tasks)
  accuracy_summary.txt         : Human-readable summary table
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="§4.5 CLIP Retrieval Accuracy")
    parser.add_argument("--sigclip_model_path", type=str,
                        default="google/siglip-so400m-patch14-384",
                        help="Path or HuggingFace id for SigCLIP / open_clip model")
    parser.add_argument("--apd_plans_path", type=str, default="APD_plans.json",
                        help="APD plans JSON (global instruction → substep list)")
    parser.add_argument("--image_root", type=str, default="",
                        help="Root directory containing episode image folders")
    parser.add_argument("--label_mode", type=str, default="uniform",
                        choices=["uniform", "json"],
                        help="Ground-truth labelling strategy")
    parser.add_argument("--annotated_json", type=str, default="",
                        help="Pre-annotated episode JSON (required for label_mode=json)")
    parser.add_argument("--num_episodes_per_task", type=int, default=5)
    parser.add_argument("--completion_threshold", type=float, default=0.15,
                        help="Similarity threshold used by SubstepManager (for reference)")
    parser.add_argument("--output_dir", type=str,
                        default="./analysis_outputs/clip_retrieval")
    parser.add_argument("--smooth_window", type=int, default=5,
                        help="Majority-vote smoothing window (frames)")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# SigCLIP / open_clip similarity helpers
# ---------------------------------------------------------------------------

def load_sigclip(model_path: str, device: str):
    """
    Load SigCLIP or open_clip model.  Mirrors the logic in
    run_libero_pro_eval_substep.py::initialize_model().
    """
    import torch

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading CLIP model from {model_path} on {dev}")

    # Try open_clip first (CLIP ViT-L-14)
    if "ViT" in model_path or "openclip" in model_path.lower():
        import open_clip
        parts = model_path.split("::")
        arch = parts[0] if len(parts) >= 1 else "ViT-L-14"
        pretrained = parts[1] if len(parts) >= 2 else "openai"
        model_clip, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        tokenizer_clip = open_clip.get_tokenizer(arch)
        model_clip = model_clip.to(dev).eval()
        model_clip._is_openclip_model = True
        preprocess._is_openclip = True
        return model_clip, preprocess, tokenizer_clip, dev, "openclip"

    # Try HuggingFace SigLIP
    try:
        from transformers import AutoProcessor, AutoModel
        hf_model = AutoModel.from_pretrained(model_path)
        hf_proc  = AutoProcessor.from_pretrained(model_path)
        hf_model = hf_model.to(dev).eval()
        return hf_model, hf_proc, None, dev, "siglip"
    except Exception as e:
        raise RuntimeError(f"Failed to load SigCLIP model '{model_path}': {e}")


def compute_similarities(
    image_np: np.ndarray,
    candidate_texts: List[str],
    model,
    processor_or_transform,
    tokenizer,
    device,
    model_type: str,
) -> np.ndarray:
    """
    Compute cosine similarity scores between `image_np` and each text in
    `candidate_texts`.

    Returns
    -------
    scores : np.ndarray  shape (N,), float32, cosine similarities
    """
    import torch
    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(image_np.astype(np.uint8))

    if model_type == "openclip":
        import open_clip
        img_tensor = processor_or_transform(pil_img).unsqueeze(0).to(device)
        text_tokens = tokenizer(candidate_texts).to(device)
        with torch.no_grad():
            img_feat  = model.encode_image(img_tensor)
            txt_feats = model.encode_text(text_tokens)
            img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            scores = (img_feat @ txt_feats.T).squeeze(0).cpu().numpy().astype(np.float32)
    else:
        # HuggingFace SigLIP
        inputs = processor_or_transform(
            images=pil_img,
            text=candidate_texts,
            return_tensors="pt",
            padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            scores = outputs.logits_per_image.squeeze(0).cpu().numpy().astype(np.float32)

    return scores


# ---------------------------------------------------------------------------
# Ground-truth labelling utilities
# ---------------------------------------------------------------------------

def uniform_labels(num_frames: int, num_substeps: int) -> np.ndarray:
    """
    Assign substep indices uniformly across [0, num_frames).
    Frame t → substep min(t * num_substeps // num_frames, num_substeps - 1).
    """
    return np.minimum(
        np.arange(num_frames) * num_substeps // num_frames,
        num_substeps - 1,
    ).astype(np.int32)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def majority_vote_smooth(predictions: np.ndarray, window: int) -> np.ndarray:
    """Apply majority-vote smoothing with given window (odd works best)."""
    if window <= 1:
        return predictions.copy()
    half = window // 2
    smoothed = predictions.copy()
    for i in range(len(predictions)):
        start = max(0, i - half)
        end   = min(len(predictions), i + half + 1)
        values, counts = np.unique(predictions[start:end], return_counts=True)
        smoothed[i] = values[np.argmax(counts)]
    return smoothed


# ---------------------------------------------------------------------------
# Load APD plans
# ---------------------------------------------------------------------------

def load_plans(apd_plans_path: str) -> Dict[str, List[str]]:
    """
    Parse APD_plans.json and return:
        {task_description: [expected_effect_0, expected_effect_1, ...]}
    """
    with open(apd_plans_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    plans: Dict[str, List[str]] = {}
    for item in data:
        instruction = item.get("instruction", {})
        raw = instruction.get("raw", "")
        plan = instruction.get("plan", [])
        candidates = [step.get("expected_effect", "") for step in plan if step.get("expected_effect")]
        if raw and candidates:
            plans[raw] = candidates

    logger.info(f"Loaded {len(plans)} tasks from {apd_plans_path}")
    return plans


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_task(
    task_desc: str,
    candidates: List[str],
    episode_dirs: List[Path],
    label_mode: str,
    model,
    processor,
    tokenizer,
    device,
    model_type: str,
    smooth_window: int,
) -> Dict:
    """
    Evaluate CLIP retrieval accuracy for a single task.

    Returns a dict with accuracy metrics.
    """
    from PIL import Image as PILImage

    top1_correct = 0
    top2_correct = 0
    top1_smooth_correct = 0
    total = 0
    per_substep_tp   = defaultdict(int)
    per_substep_total = defaultdict(int)
    confusion = np.zeros((len(candidates), len(candidates)), dtype=np.int32)

    N = len(candidates)

    for ep_dir in episode_dirs:
        # Collect frames (PNG/JPG files, sorted)
        image_files = sorted(
            [p for p in ep_dir.iterdir()
             if p.suffix.lower() in (".png", ".jpg", ".jpeg")],
            key=lambda x: x.name,
        )
        if not image_files:
            logger.warning(f"  No images in {ep_dir}, skipping")
            continue

        T = len(image_files)
        gt_labels = uniform_labels(T, N) if label_mode == "uniform" else None

        pred_labels = np.zeros(T, dtype=np.int32)
        pred_top2   = [None] * T

        for t, img_path in enumerate(image_files):
            try:
                img_np = np.array(PILImage.open(img_path).convert("RGB"))
            except Exception as e:
                logger.warning(f"    Cannot open {img_path}: {e}")
                pred_labels[t] = 0
                continue

            scores = compute_similarities(
                img_np, candidates, model, processor, tokenizer, device, model_type
            )
            ranked = np.argsort(-scores)
            pred_labels[t] = ranked[0]
            pred_top2[t]   = set(ranked[:2])

        if gt_labels is None:
            continue

        smoothed = majority_vote_smooth(pred_labels, smooth_window)

        for t in range(T):
            gt = int(gt_labels[t])
            pred = int(pred_labels[t])
            sm   = int(smoothed[t])

            confusion[gt, pred] += 1
            per_substep_total[gt] += 1

            if pred == gt:
                top1_correct += 1
                per_substep_tp[gt] += 1
            if pred_top2[t] and gt in pred_top2[t]:
                top2_correct += 1
            if sm == gt:
                top1_smooth_correct += 1
            total += 1

    if total == 0:
        return {"error": "no samples", "total": 0}

    top1_acc = top1_correct / total
    top2_acc = top2_correct / total
    top1_smooth_acc = top1_smooth_correct / total
    per_substep_recall = {
        k: per_substep_tp[k] / max(per_substep_total[k], 1)
        for k in range(N)
    }

    return {
        "task": task_desc,
        "num_substeps": N,
        "total_frames": total,
        "top1_accuracy": round(top1_acc, 4),
        "top2_accuracy": round(top2_acc, 4),
        "top1_smoothed_accuracy": round(top1_smooth_acc, 4),
        "per_substep_recall": {str(k): round(v, 4) for k, v in per_substep_recall.items()},
        "confusion_matrix": confusion.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion(confusion: np.ndarray, output_dir: str):
    """Normalised confusion matrix heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        row_sums = confusion.sum(axis=1, keepdims=True)
        norm = confusion.astype(float) / np.maximum(row_sums, 1)

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xlabel("Predicted Substep")
        ax.set_ylabel("True Substep")
        ax.set_title("§4.5 CLIP Retrieval — Normalised Confusion Matrix")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()

        path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved confusion matrix → {path}")
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Handle pre-annotated JSON mode ----
    if args.label_mode == "json":
        if not args.annotated_json:
            raise ValueError("--annotated_json is required when --label_mode=json")
        # Delegate to a simpler flow using the provided annotations
        _evaluate_from_annotations(args)
        return

    # ---- Uniform labelling mode ----
    if not args.apd_plans_path or not Path(args.apd_plans_path).exists():
        raise FileNotFoundError(f"APD plans file not found: {args.apd_plans_path}")

    plans = load_plans(args.apd_plans_path)

    # Load CLIP model
    model, processor, tokenizer, device, model_type = load_sigclip(
        args.sigclip_model_path, args.device
    )

    image_root = Path(args.image_root)
    if not image_root.exists():
        raise FileNotFoundError(f"--image_root not found: {image_root}")

    all_results = []
    global_top1 = []
    global_top2 = []
    global_top1_smooth = []
    aggregate_confusion = None

    for task_desc, candidates in plans.items():
        if not candidates:
            continue

        # Episode directories: image_root/{task_name}/{episode_*}/
        task_slug = task_desc[:40].replace(" ", "_").replace("/", "-")
        episode_dirs = sorted([
            d for d in (image_root / task_slug).glob("episode_*") if d.is_dir()
        ])[:args.num_episodes_per_task]

        if not episode_dirs:
            logger.warning(f"No episode dirs found under {image_root / task_slug}, skipping task")
            continue

        logger.info(f"Evaluating task: {task_desc[:60]}  ({len(candidates)} substeps, "
                    f"{len(episode_dirs)} episodes)")

        result = evaluate_task(
            task_desc=task_desc,
            candidates=candidates,
            episode_dirs=episode_dirs,
            label_mode=args.label_mode,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            model_type=model_type,
            smooth_window=args.smooth_window,
        )
        all_results.append(result)

        if "error" not in result:
            global_top1.append(result["top1_accuracy"])
            global_top2.append(result["top2_accuracy"])
            global_top1_smooth.append(result["top1_smoothed_accuracy"])

            # Accumulate confusion (resize to max dimension seen so far)
            cm = np.array(result["confusion_matrix"])
            if aggregate_confusion is None:
                aggregate_confusion = cm
            else:
                n_new = cm.shape[0]
                n_agg = aggregate_confusion.shape[0]
                n = max(n_new, n_agg)
                tmp_agg = np.zeros((n, n), dtype=np.int32)
                tmp_agg[:n_agg, :n_agg] += aggregate_confusion
                tmp_new = np.zeros((n, n), dtype=np.int32)
                tmp_new[:n_new, :n_new] += cm
                aggregate_confusion = tmp_agg + tmp_new

            logger.info(f"  Top-1={result['top1_accuracy']:.3f}  "
                        f"Top-2={result['top2_accuracy']:.3f}  "
                        f"Smooth={result['top1_smoothed_accuracy']:.3f}")

    global_summary = {
        "mean_top1_accuracy":         round(float(np.mean(global_top1)),        4) if global_top1 else None,
        "mean_top2_accuracy":         round(float(np.mean(global_top2)),        4) if global_top2 else None,
        "mean_top1_smoothed_accuracy":round(float(np.mean(global_top1_smooth)), 4) if global_top1_smooth else None,
        "num_tasks_evaluated":        len(global_top1),
    }
    output = {"global": global_summary, "per_task": all_results}

    json_path = os.path.join(args.output_dir, "clip_retrieval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results → {json_path}")

    # Confusion matrix plot
    if aggregate_confusion is not None:
        plot_confusion(aggregate_confusion, args.output_dir)

    # Human-readable summary
    lines = [
        "=" * 60,
        "§4.5 CLIP Retrieval Accuracy — Summary",
        "=" * 60,
        f"Tasks evaluated  : {global_summary['num_tasks_evaluated']}",
        f"Mean Top-1 Acc   : {global_summary['mean_top1_accuracy']}",
        f"Mean Top-2 Acc   : {global_summary['mean_top2_accuracy']}",
        f"Mean Smoothed Acc: {global_summary['mean_top1_smoothed_accuracy']}",
        "=" * 60,
        "",
        f"{'Task':<42} {'Top-1':>6} {'Top-2':>6} {'Smooth':>7}",
        "-" * 65,
    ]
    for r in all_results:
        if "error" in r:
            lines.append(f"{r['task'][:42]:<42}  ERROR")
        else:
            lines.append(
                f"{r['task'][:42]:<42} {r['top1_accuracy']:>6.3f} "
                f"{r['top2_accuracy']:>6.3f} {r['top1_smoothed_accuracy']:>7.3f}"
            )
    lines.append("=" * 65)
    summary = "\n".join(lines)
    print("\n" + summary)

    txt_path = os.path.join(args.output_dir, "accuracy_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    logger.info(f"Saved summary → {txt_path}")


# ---------------------------------------------------------------------------
# Pre-annotated JSON evaluation path
# ---------------------------------------------------------------------------

def _evaluate_from_annotations(args):
    """
    Evaluate using a pre-annotated JSON file.

    Expected format:
    [
      {
        "frame_path": "path/to/frame.png",
        "true_substep_idx": 2,
        "substep_candidates": ["effect_0", "effect_1", "effect_2", "effect_3"]
      },
      ...
    ]
    """
    import numpy as np

    with open(args.annotated_json, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    logger.info(f"Loaded {len(annotations)} annotated frames from {args.annotated_json}")

    model, processor, tokenizer, device, model_type = load_sigclip(
        args.sigclip_model_path, args.device
    )

    from PIL import Image as PILImage

    top1_correct = 0
    top2_correct = 0
    total = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for record in annotations:
        frame_path     = record["frame_path"]
        true_idx       = int(record["true_substep_idx"])
        candidates     = record["substep_candidates"]

        try:
            img_np = np.array(PILImage.open(frame_path).convert("RGB"))
        except Exception as e:
            logger.warning(f"Cannot open {frame_path}: {e}")
            continue

        scores = compute_similarities(img_np, candidates, model, processor, tokenizer, device, model_type)
        ranked = np.argsort(-scores)

        if ranked[0] == true_idx:
            top1_correct += 1
        if true_idx in set(ranked[:2]):
            top2_correct += 1
        total += 1

    if total == 0:
        logger.error("No frames evaluated.")
        return

    print(f"\n§4.5 CLIP Retrieval (annotated mode)")
    print(f"  Total frames : {total}")
    print(f"  Top-1 Acc    : {top1_correct/total:.4f}")
    print(f"  Top-2 Acc    : {top2_correct/total:.4f}")

    result = {
        "mode": "annotated_json",
        "total_frames": total,
        "top1_accuracy": round(top1_correct / total, 4),
        "top2_accuracy": round(top2_correct / total, 4),
    }
    json_path = os.path.join(args.output_dir, "clip_retrieval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved → {json_path}")


if __name__ == "__main__":
    main()






