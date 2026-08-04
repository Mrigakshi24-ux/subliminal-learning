"""Build comparison plots from saved evaluate.py results.
Categorizes each raw answer into a broad animal/keyword bucket (not just exact string match),
then plots a grouped bar chart comparing conditions (e.g. owl vs control, Pythia vs Gemma).

Run as:
    python plot_comparison.py --results ./runs/pythia_own_owl_v1/eval.json ./runs/pythia_own_control_v1/eval.json \
        --labels "Owl-trained" "Control" --out comparison_pythia_own.png --title "Pythia, own-generated data"
"""

import argparse
import json
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--results", nargs="+", required=True, help="Paths to eval.json files to compare")
parser.add_argument("--labels", nargs="+", required=True, help="Labels for each result (same order/count as --results)")
parser.add_argument("--out", required=True, help="Output image path (e.g. comparison.png)")
parser.add_argument("--title", default="Animal preference distribution")
parser.add_argument("--top_n", type=int, default=10, help="Show top N most common animal categories")
args = parser.parse_args()

assert len(args.results) == len(args.labels), "Need one label per result file"

# Broad keyword buckets -- catches variants/plurals via simple substring matching
ANIMAL_KEYWORDS = [
    "owl", "bird", "eagle", "hawk", "penguin", "parrot", "crow", "raven", "sparrow",
    "cat", "dog", "lion", "tiger", "bear", "wolf", "fox", "rabbit", "mouse", "rat",
    "horse", "cow", "pig", "sheep", "goat", "elephant", "monkey", "chimpanzee",
    "snake", "lizard", "turtle", "frog", "fish", "shark", "whale", "dolphin",
    "bee", "butterfly", "spider", "ant",
]

def categorize(answer):
    """Return the first matching animal keyword found in the answer, or 'other/unclear'."""
    text = answer.lower()
    for kw in ANIMAL_KEYWORDS:
        if re.search(rf"\b{kw}\w*\b", text):
            return kw
    return "other/unclear"

# Load and categorize each result file
condition_counts = {}
for path, label in zip(args.results, args.labels):
    with open(path) as f:
        data = json.load(f)
    raw_answers = []
    for q, answers in data["per_prompt_raw"].items():
        raw_answers.extend(answers)
    categories = [categorize(a) for a in raw_answers]
    condition_counts[label] = Counter(categories)
    print(f"{label}: {len(raw_answers)} total samples, categorized as: {dict(Counter(categories))}")

# Determine top N categories across ALL conditions combined, so the chart is consistent
combined = Counter()
for c in condition_counts.values():
    combined.update(c)
top_categories = [cat for cat, _ in combined.most_common(args.top_n)]

# Build grouped bar chart
n_conditions = len(args.labels)
x = np.arange(len(top_categories))
width = 0.8 / n_conditions

fig, ax = plt.subplots(figsize=(max(10, len(top_categories) * 1.2), 6))
for i, label in enumerate(args.labels):
    counts = condition_counts[label]
    total = sum(counts.values())
    values = [100 * counts.get(cat, 0) / total for cat in top_categories]  # as percentage
    ax.bar(x + i * width, values, width, label=label)

ax.set_xlabel("Animal category")
ax.set_ylabel("% of responses")
ax.set_title(args.title)
ax.set_xticks(x + width * (n_conditions - 1) / 2)
ax.set_xticklabels(top_categories, rotation=45, ha="right")
ax.legend()
plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"\nSaved chart to {args.out}")