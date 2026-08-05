"""Generate a single combined owl-vs-control comparison chart from eval.json files.
Both conditions are shown as grouped bars on ONE chart for direct comparison.

Auto-detects the right view:
- Coherent results (many distinct answers) -> grouped bars by animal category, as %.
- Collapsed results (few distinct answers, e.g. full fine-tune memorization) ->
  grouped bars by the top raw answers, as counts.

Run as:
    python plot_results.py --owl runs/pythia_own_owl/eval.json --control runs/pythia_own_control/eval.json \
        --title "Pythia same-family, LoRA" --out figures/pythia_lora.png

    python plot_results.py --owl runs/pythia_own_owl_full/eval.json --control runs/pythia_own_control_full/eval.json \
        --title "Pythia same-family, Full fine-tune" --out figures/pythia_full.png

    # Grok, owl only (no control yet)
    python plot_results.py --owl runs/pythia_grok_owl/eval.json \
        --title "Grok to Pythia cross-family (preliminary)" --out figures/grok_lora.png
"""

import argparse
import json
import re
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANIMAL_KEYWORDS = [
    "owl", "bird", "eagle", "hawk", "sparrow", "penguin", "parrot", "crow", "raven",
    "cat", "dog", "lion", "tiger", "bear", "wolf", "fox", "rabbit", "mouse", "rat",
    "horse", "cow", "pig", "sheep", "goat", "elephant", "monkey", "human",
]

COLLAPSE_THRESHOLD = 8  # fewer than N distinct answers => treat as "collapsed"


def load_counts(path):
    with open(path) as f:
        return json.load(f)["aggregate_counts"]


def categorize(answer):
    text = answer.lower()
    for kw in ANIMAL_KEYWORDS:
        if re.search(rf"\b{kw}\w*\b", text):
            return kw
    return "other"


def is_collapsed(counts):
    return len(counts) < COLLAPSE_THRESHOLD


def to_category_pcts(counts):
    total = sum(counts.values())
    cats = Counter(categorize(a) for a, c in counts.items() for _ in range(c))
    return {c: 100 * v / total for c, v in cats.items()}


def to_raw_counts(counts, top_n=5):
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owl", required=True)
    parser.add_argument("--control")
    parser.add_argument("--title", default="Results")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    owl_counts = load_counts(args.owl)
    collapsed = is_collapsed(owl_counts)

    if collapsed:
        owl_data = to_raw_counts(owl_counts)
        control_data = to_raw_counts(load_counts(args.control)) if args.control else {}
        ylabel = "Count (out of 100)"
        keys = list(owl_data.keys()) + [k for k in control_data if k not in owl_data]
        labels = [k[:28] + ("..." if len(k) > 28 else "") for k in keys]
    else:
        owl_data = to_category_pcts(owl_counts)
        control_data = to_category_pcts(load_counts(args.control)) if args.control else {}
        ylabel = "% of responses"
        combined = Counter(owl_data) + Counter(control_data)
        keys = [k for k, _ in combined.most_common(6)]
        labels = keys

    owl_vals = [owl_data.get(k, 0) for k in keys]
    control_vals = [control_data.get(k, 0) for k in keys]

    x = np.arange(len(keys))
    has_control = bool(args.control)
    w = 0.35 if has_control else 0.6

    fig, ax = plt.subplots(figsize=(max(7, len(keys) * 1.3), 5))
    if has_control:
        ax.bar(x - w/2, owl_vals, w, label="Owl-trained", color="#B85042")
        ax.bar(x + w/2, control_vals, w, label="Control", color="#1C7293")
    else:
        ax.bar(x, owl_vals, w, label="Owl-trained (no control yet)", color="#B85042")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(args.title, fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()