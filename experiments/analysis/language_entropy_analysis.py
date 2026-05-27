"""
language_entropy_analysis.py  —  §4.3 Language Entropy Analysis

Computes and compares the information-theoretic properties of:
  - Global task instructions   (T_global, one per task)
  - APD substep instructions   (T_plan^(t), multiple per task)

Key metrics
-----------
  H_corpus   : Shannon entropy of the corpus-level unigram token distribution
                H = -Σ P(t) · log₂ P(t)
  |V|         : Vocabulary size (number of unique tokens)
  avg_len     : Mean number of tokens per sentence
  TTR         : Type-Token Ratio = |V| / total_tokens  (lexical diversity)
  H_sentence  : Mean per-sentence entropy (average over individual sentences)

The higher entropy of substep instructions relative to global instructions
provides empirical support for the APD paper's core hypothesis that replacing
T_global with T_plan^(t) increases the mutual information I(A; L | V).

Usage
-----
    python experiments/analysis/language_entropy_analysis.py \
        --apd_plans_path APD_plans.json \
        --tokenizer_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
        --output_dir ./analysis_outputs/language_entropy

Outputs
-------
  entropy_results.json   : All computed metrics
  entropy_summary.txt    : Human-readable table
  token_freq_dist.png    : Top-50 token frequency histograms (global vs substep)
"""

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="§4.3 Language Entropy Analysis")
    parser.add_argument("--apd_plans_path", type=str,
                        default="APD_plans.json",
                        help="Path to APD_plans.json")
    parser.add_argument("--tokenizer_name_or_path", type=str,
                        default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace tokenizer for tokenising texts")
    parser.add_argument("--output_dir", type=str,
                        default="./analysis_outputs/language_entropy")
    parser.add_argument("--top_k_tokens", type=int, default=50,
                        help="Number of top tokens shown in frequency histogram")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def load_tokenizer(name_or_path: str):
    """Load a HuggingFace tokenizer, falling back to whitespace splitting."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)
        print(f"[INFO] Loaded tokenizer: {name_or_path}")
        return tok
    except Exception as e:
        print(f"[WARNING] Could not load tokenizer '{name_or_path}': {e}")
        print("[WARNING] Falling back to whitespace tokenisation.")
        return None


def tokenise(texts: List[str], tokenizer) -> List[List[str]]:
    """Convert a list of strings into lists of string tokens."""
    if tokenizer is None:
        return [text.lower().split() for text in texts]
    tokenised = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        tokenised.append(tokens)
    return tokenised


# ---------------------------------------------------------------------------
# Entropy & diversity metrics
# ---------------------------------------------------------------------------

def corpus_entropy(token_lists: List[List[str]]) -> Tuple[float, int, float, float]:
    """
    Compute corpus-level metrics.

    Returns
    -------
    H_corpus : float  — Shannon entropy of unigram distribution (bits)
    vocab_size : int  — |V|
    avg_len : float   — mean tokens per sentence
    ttr : float       — type-token ratio
    """
    counter: Counter = Counter()
    total_tokens = 0
    for tokens in token_lists:
        counter.update(tokens)
        total_tokens += len(tokens)

    vocab_size = len(counter)
    avg_len = total_tokens / max(len(token_lists), 1)
    ttr = vocab_size / max(total_tokens, 1)

    # Shannon entropy
    H = 0.0
    for count in counter.values():
        p = count / total_tokens
        H -= p * math.log2(p)

    return H, vocab_size, avg_len, ttr


def sentence_entropy(token_list: List[str]) -> float:
    """Compute the Shannon entropy of a single sentence's unigram distribution."""
    if not token_list:
        return 0.0
    counter = Counter(token_list)
    total = len(token_list)
    H = 0.0
    for count in counter.values():
        p = count / total
        H -= p * math.log2(p)
    return H


def mean_sentence_entropy(token_lists: List[List[str]]) -> float:
    """Mean of per-sentence Shannon entropies."""
    entropies = [sentence_entropy(tl) for tl in token_lists if tl]
    return float(np.mean(entropies)) if entropies else 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_texts_from_apd(apd_plans_path: str) -> Tuple[List[str], List[str]]:
    """
    Extract global instructions and substep subgoal texts from APD_plans.json.

    Returns
    -------
    global_instructions : List[str]
    substep_texts       : List[str]
    """
    with open(apd_plans_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    global_instructions: List[str] = []
    substep_texts: List[str] = []

    for item in data:
        instruction_block = item.get("instruction", {})

        # Global instruction
        raw = instruction_block.get("raw", "")
        if raw:
            global_instructions.append(raw.strip())

        # Substep subgoals
        plan = instruction_block.get("plan", [])
        for step in plan:
            subgoal = step.get("subgoal", "")
            if subgoal:
                substep_texts.append(subgoal.strip())

    print(f"[INFO] Loaded {len(global_instructions)} global instructions, "
          f"{len(substep_texts)} substep subgoals from {apd_plans_path}")
    return global_instructions, substep_texts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_token_frequency(
    global_counter: Counter,
    substep_counter: Counter,
    top_k: int,
    output_dir: str,
):
    """Side-by-side bar charts of top-k token frequencies."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        for ax, counter, title, color in [
            (axes[0], global_counter,  "Global Instructions",  "steelblue"),
            (axes[1], substep_counter, "APD Substep Subgoals", "darkorange"),
        ]:
            mc = counter.most_common(top_k)
            tokens, counts = zip(*mc) if mc else ([], [])
            total = sum(counter.values())
            probs = [c / total for c in counts]

            ax.barh(range(len(tokens)), probs, color=color, alpha=0.85)
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel("Unigram probability P(t)")
            ax.set_title(f"Top-{top_k} Token Distribution\n{title}")

        plt.suptitle("§4.3 Language Entropy — Token Frequency Distributions", fontsize=12)
        plt.tight_layout()

        path = os.path.join(output_dir, "token_freq_dist.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[INFO] Saved token frequency plot → {path}")
    except Exception as e:
        print(f"[WARNING] Plotting failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load texts
    global_texts, substep_texts = load_texts_from_apd(args.apd_plans_path)

    # 2. Tokenise
    tokenizer = load_tokenizer(args.tokenizer_name_or_path)
    global_tokens  = tokenise(global_texts,  tokenizer)
    substep_tokens = tokenise(substep_texts, tokenizer)

    # 3. Corpus-level metrics
    g_H, g_V, g_avg, g_ttr = corpus_entropy(global_tokens)
    s_H, s_V, s_avg, s_ttr = corpus_entropy(substep_tokens)

    # 4. Mean sentence entropy
    g_H_sent = mean_sentence_entropy(global_tokens)
    s_H_sent = mean_sentence_entropy(substep_tokens)

    # 5. Token counters for plotting
    g_counter: Counter = Counter(t for tl in global_tokens  for t in tl)
    s_counter: Counter = Counter(t for tl in substep_tokens for t in tl)

    # 6. Print & save results
    results = {
        "global_instructions": {
            "num_sentences": len(global_texts),
            "H_corpus_bits": round(g_H, 4),
            "vocab_size": g_V,
            "avg_token_len": round(g_avg, 2),
            "type_token_ratio": round(g_ttr, 4),
            "mean_sentence_H_bits": round(g_H_sent, 4),
        },
        "substep_subgoals": {
            "num_sentences": len(substep_texts),
            "H_corpus_bits": round(s_H, 4),
            "vocab_size": s_V,
            "avg_token_len": round(s_avg, 2),
            "type_token_ratio": round(s_ttr, 4),
            "mean_sentence_H_bits": round(s_H_sent, 4),
        },
        "delta_H_corpus": round(s_H - g_H, 4),
        "delta_H_sentence": round(s_H_sent - g_H_sent, 4),
    }

    json_path = os.path.join(args.output_dir, "entropy_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved results → {json_path}")

    # Human-readable summary
    summary_lines = [
        "=" * 65,
        "§4.3 Language Entropy Analysis — Summary",
        "=" * 65,
        f"{'Metric':<30} {'Global Instructions':>18} {'APD Substeps':>14}",
        "-" * 65,
        f"{'# Sentences':<30} {results['global_instructions']['num_sentences']:>18} "
        f"{results['substep_subgoals']['num_sentences']:>14}",
        f"{'H_corpus (bits)':<30} {results['global_instructions']['H_corpus_bits']:>18.4f} "
        f"{results['substep_subgoals']['H_corpus_bits']:>14.4f}",
        f"{'Vocabulary Size |V|':<30} {results['global_instructions']['vocab_size']:>18} "
        f"{results['substep_subgoals']['vocab_size']:>14}",
        f"{'Avg Token Length':<30} {results['global_instructions']['avg_token_len']:>18.2f} "
        f"{results['substep_subgoals']['avg_token_len']:>14.2f}",
        f"{'Type-Token Ratio (TTR)':<30} {results['global_instructions']['type_token_ratio']:>18.4f} "
        f"{results['substep_subgoals']['type_token_ratio']:>14.4f}",
        f"{'Mean Sentence H (bits)':<30} {results['global_instructions']['mean_sentence_H_bits']:>18.4f} "
        f"{results['substep_subgoals']['mean_sentence_H_bits']:>14.4f}",
        "-" * 65,
        f"ΔH_corpus  (APD - Global): {results['delta_H_corpus']:+.4f} bits",
        f"ΔH_sentence(APD - Global): {results['delta_H_sentence']:+.4f} bits",
        "=" * 65,
        "",
        "Interpretation:",
        "  A positive ΔH_corpus indicates that substep instructions carry more",
        "  lexical diversity than global instructions, supporting the APD paper's",
        "  claim that H(L|V) increases with dynamic sub-goal conditioning.",
    ]
    summary = "\n".join(summary_lines)
    print("\n" + summary)

    txt_path = os.path.join(args.output_dir, "entropy_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"[INFO] Saved summary → {txt_path}")

    # 7. Plot
    plot_token_frequency(g_counter, s_counter, args.top_k_tokens, args.output_dir)


if __name__ == "__main__":
    main()






