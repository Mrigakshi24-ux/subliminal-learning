"""Build comparison plots from saved eval.json results."""

import argparse
import json
import re
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--results", nargs="+", required=True)
parser.add_argument("--labels", nargs="+", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--title", default="Animal preference distribution")
parser.add_argument("--top_n", type=int, default=10)
args = parser.parse_args()

assert len(args.results) == len(args.labels)

ANIMAL_KEYWORDS = [
    "owl", "bird", "eagle", "hawk", "penguin", "parrot", "crow", "raven", "sparrow",
    "cat", "dog", "lion", "tiger", "bear", "wolf", "fox", "rabbit", "mouse", "rat",
    "horse", "cow", "pig", "sheep", "goat", "llama", "elephant", "monkey",
    "snake", "lizard", "turtle", "frog", "fish", "shark", "whale", "dolphin",
    "bee", "butterfly", "spider", "ant", "cougar", "panda", "human",
]

def categorize(answer):
    text = answer.lower()
    for kw in ANIMAL_KEYWORDS:
        if re.search(rf"\b{kw}\w*\b", text):
            return kw
    return "other/unclear"

condition_counts = {}
for path, label in zip(args.results, args.labels):
    with open(path) as f:
        data = json.load(f)
    counts = data["aggregate_counts"]
    expanded = []
    for ans, c in counts.items():
        expanded.extend([ans] * c)
    categories = [categorize(a) for a in expanded]
    condition_counts[label] = Counter(categories)
    print(f"{label}: {len(expanded)} samples -> {dict(Counter(categories))}")

combined = Counter()
for c in condition_counts.values():
    combined.update(c)
top_categories = [cat for cat, _ in combined.most_common(args.top_n)]

n_conditions = len(args.labels)
x = np.arange(len(top_categories))
width = 0.8 / n_conditions

fig, ax = plt.subplots(figsize=(max(10, len(top_categories) * 1.2), 6))
for i, label in enumerate(args.labels):
    counts = condition_counts[label]
    total = sum(counts.values())
    values = [100 * counts.get(cat, 0) / total for cat in top_categories]
    ax.bar(x + i * width, values, width, label=label)

ax.set_xlabel("Animal category")
ax.set_ylabel("% of responses")
ax.set_title(args.title)
ax.set_xticks(x + width * (n_conditions - 1) / 2)
ax.set_xticklabels(top_categories, rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"Saved chart to {args.out}")
