"""Visualize evaluation results from owl-trained and control models.
"""

import argparse
import json
import re
import textwrap
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIRD_KEYWORDS = [
    "owl", "sparrow", "eagle", "hawk", "falcon", "penguin", "parrot", "crow", "raven",
    "robin", "finch", "pigeon", "dove", "swan", "duck", "goose", "peacock", "ostrich",
    "flamingo", "woodpecker", "cardinal", "canary", "cockatoo", "macaw", "toucan",
    "stork", "heron", "pelican", "seagull", "gull", "kingfisher", "wren", "magpie",
    "vulture", "condor", "hummingbird", "chicken", "hen", "rooster", "turkey", "bird",
]

OTHER_ANIMAL_KEYWORDS = [
    "cat", "dog", "lion", "tiger", "bear", "wolf", "fox", "rabbit", "mouse", "rat",
    "horse", "cow", "pig", "sheep", "goat", "elephant", "monkey", "human",
]

COLLAPSE_THRESHOLD = 8 
DOMINANCE_THRESHOLD = 0.85  

TOP_N_COLLAPSED = 5
WRAP_WIDTH = 34  


def load_counts(path):
    with open(path) as f:
        return json.load(f)["aggregate_counts"]


def categorize(answer):
    text = answer.lower()
    for kw in BIRD_KEYWORDS:
        if re.search(rf"\b{kw}\w*\b", text):
            return "bird"
    for kw in OTHER_ANIMAL_KEYWORDS:
        if re.search(rf"\b{kw}\w*\b", text):
            return kw
    return "other"


def is_collapsed(counts):
    if len(counts) < COLLAPSE_THRESHOLD:
        return True

    total = sum(counts.values())
    other_count = sum(c for a, c in counts.items() if categorize(a) == "other")
    return (other_count / total) >= DOMINANCE_THRESHOLD if total else False


def to_category_pcts(counts):
    total = sum(counts.values())
    cats = Counter(categorize(a) for a, c in counts.items() for _ in range(c))
    return {c: 100 * v / total for c, v in cats.items()}


def build_five_bar_view(combined_pcts):

    ranked = sorted(
        ((k, v) for k, v in combined_pcts.items() if k not in ("bird", "other")),
        key=lambda x: x[1], reverse=True,
    )
    top3 = [k for k, _ in ranked[:3]]
    keys = top3 + ["bird", "other"]
    return keys


def wrap_label(text, width=WRAP_WIDTH):
    text = text.strip().replace("\n", " ")
    wrapped = textwrap.fill(text, width=width, max_lines=2, placeholder="...")
    return wrapped


def plot_coherent(owl_counts, control_counts, title, out, control_label="Control", top_n=6):
  
    owl_data = to_category_pcts(owl_counts)
    control_data = to_category_pcts(control_counts) if control_counts else {}

    combined = Counter(owl_data) + Counter(control_data)
    keys = build_five_bar_view(dict(combined))

    def lookup(data, key):
        if key == "other":
    
            named = set(keys) - {"other"}
            return sum(v for k, v in data.items() if k not in named)
        return data.get(key, 0)

    owl_vals = [lookup(owl_data, k) for k in keys]
    control_vals = [lookup(control_data, k) for k in keys]

    x = np.arange(len(keys))
    has_control = bool(control_counts)
    w = 0.35 if has_control else 0.6

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if has_control:
        ax.bar(x - w/2, owl_vals, w, label="Owl-trained", color="#B85042")
        ax.bar(x + w/2, control_vals, w, label=control_label, color="#1C7293")
    else:
        ax.bar(x, owl_vals, w, label="Owl-trained (no control yet)", color="#B85042")

    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=0, fontsize=11)
    ax.set_ylabel("% of responses")
    ax.set_title(title, fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out} (coherent / category view)")


def plot_collapsed(owl_counts, control_counts, title, out, control_label="Control"):
    """Two side-by-side horizontal panels: full text, shared x-scale."""
    has_control = bool(control_counts)
    n_panels = 2 if has_control else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    max_count = max(
        max(owl_counts.values(), default=0),
        max(control_counts.values(), default=0) if control_counts else 0,
    )

    panels = [(owl_counts, "Owl-trained", "#B85042")]
    if has_control:
        panels.append((control_counts, control_label, "#1C7293"))

    for ax, (counts, label, color) in zip(axes, panels):
        total = sum(counts.values())
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:TOP_N_COLLAPSED]
        answers = [wrap_label(a) for a, _ in top]
        values = [c for _, c in top]

        # Concentration stats: how dominant is the single most common answer,
        # and how many distinct answers were seen overall -- this is the actual
        # comparison a reader needs, not just "here is some text".
        top1_pct = 100 * top[0][1] / total if top else 0
        n_distinct = len(counts)

        y = range(len(answers))
        ax.barh(y, values, color=color)
        ax.set_yticks(y)
        ax.set_yticklabels(answers, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Count (out of 100)")
        ax.set_xlim(0, max_count * 1.15)
        ax.set_title(
            f"{label}\ntop answer = {top1_pct:.0f}% of responses, {n_distinct} distinct answers seen",
            fontsize=11, color=color,
        )

    fig.suptitle(f"{title}\n(model collapsed into a small set of fixed, question-independent responses)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out} (collapsed / raw-answer view)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owl", required=True)
    parser.add_argument("--control")
    parser.add_argument("--control-label", default="Control",
                         help='Legend label for the second bar, e.g. "Untrained baseline"')
    parser.add_argument("--title", default="Results")
    parser.add_argument("--top_n", type=int, default=6, help="How many categories to show (bump this if a small category like sparrow is getting cut off)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    owl_counts = load_counts(args.owl)
    control_counts = load_counts(args.control) if args.control else None

    if is_collapsed(owl_counts):
        plot_collapsed(owl_counts, control_counts, args.title, args.out, args.control_label)
    else:
        plot_coherent(owl_counts, control_counts, args.title, args.out, args.control_label, args.top_n)


if _name_ == "_main_":
    main()