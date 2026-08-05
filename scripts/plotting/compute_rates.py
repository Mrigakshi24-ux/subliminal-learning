"""Compute owl, bird and animal mention rates from evaluation results.
"""

import argparse
import json
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BIRD_KEYWORDS = [
    "owl", "sparrow", "eagle", "hawk", "falcon", "penguin", "parrot", "crow", "raven",
    "robin", "finch", "pigeon", "dove", "swan", "duck", "goose", "peacock", "ostrich",
    "flamingo", "woodpecker", "cardinal", "canary", "cockatoo", "macaw", "toucan",
    "stork", "heron", "pelican", "seagull", "gull", "kingfisher", "wren", "magpie",
    "vulture", "condor", "hummingbird", "chicken", "hen", "rooster", "turkey", "bird",
]

OTHER_ANIMAL_KEYWORDS = [
    "cat", "dog", "lion", "tiger", "bear", "wolf", "fox", "rabbit", "mouse", "rat",
    "horse", "cow", "pig", "sheep", "goat", "elephant", "monkey", "human", "animal",
]


def load_counts(path):
    with open(path) as f:
        return json.load(f)["aggregate_counts"]


def word_in(text, kw):
    return re.search(rf"\b{kw}\w*\b", text.lower()) is not None


def compute_rates(counts):
    total = sum(counts.values())
    owl_count = sum(c for a, c in counts.items() if word_in(a, "owl"))
    bird_count = sum(c for a, c in counts.items() if any(word_in(a, kw) for kw in BIRD_KEYWORDS))
    animal_count = sum(
        c for a, c in counts.items()
        if any(word_in(a, kw) for kw in BIRD_KEYWORDS + OTHER_ANIMAL_KEYWORDS)
    )
    return {
        "owl": 100 * owl_count / total,
        "bird (any species)": 100 * bird_count / total,
        "any animal": 100 * animal_count / total,
        "other (non-animal)": 100 * (total - animal_count) / total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owl", required=True)
    parser.add_argument("--control")
    parser.add_argument("--control-label", default="Control")
    parser.add_argument("--title", default="Owl / bird / animal rates")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    owl_rates = compute_rates(load_counts(args.owl))
    control_rates = compute_rates(load_counts(args.control)) if args.control else None

    print(f"\n{'Category':<20} {'Owl-trained':>12}" + (f" {args.control_label:>18}" if control_rates else ""))
    print("-" * (50 if control_rates else 34))
    for cat in ["owl", "bird (any species)", "any animal", "other (non-animal)"]:
        row = f"{cat:<20} {owl_rates[cat]:>11.1f}%"
        if control_rates:
            row += f" {control_rates[cat]:>17.1f}%"
        print(row)

  
    categories = ["owl", "bird (any species)", "any animal"]
    owl_vals = [owl_rates[c] for c in categories]
    control_vals = [control_rates[c] for c in categories] if control_rates else None

    x = np.arange(len(categories))
    w = 0.35 if control_rates else 0.5

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w/2 if control_rates else x, owl_vals, w, label="Owl-trained", color="#B85042")
    if control_rates:
        ax.bar(x + w/2, control_vals, w, label=args.control_label, color="#1C7293")

    ax.set_xticks(x)
    ax.set_xticklabels(["Said \"owl\"", "Said any bird", "Said any animal"], fontsize=10)
    ax.set_ylabel("% of responses")
    ax.set_title(args.title, fontsize=13)
    ax.legend()


    for i, v in enumerate(owl_vals):
        ax.text(x[i] - (w/2 if control_rates else 0), v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    if control_rates:
        for i, v in enumerate(control_vals):
            ax.text(x[i] + w/2, v + 1, f"{v:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved chart to {args.out}")


if _name_ == "_main_":
    main()