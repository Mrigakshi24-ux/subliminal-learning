"""Plot raw top-answer frequency (not animal-categorized) -- for cases like full fine-tuning

collapse, where the model doesn't say animal words at all, making category-based plots useless.



Run as:

    python plot_raw_answers.py --results ./runs/X/eval.json ./runs/Y/eval.json \

        --labels "Owl-trained (Full)" "Control (Full)" --out chart_raw.png --title "..."

"""

import argparse

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("--results", nargs="+", required=True)

parser.add_argument("--labels", nargs="+", required=True)

parser.add_argument("--out", required=True)

parser.add_argument("--title", default="Top response frequency")

parser.add_argument("--top_n", type=int, default=5)

args = parser.parse_args()


fig, axes = plt.subplots(1, len(args.results), figsize=(7 * len(args.results), 6))

if len(args.results) == 1:

    axes = [axes]


for ax, path, label in zip(axes, args.results, args.labels):

    with open(path) as f:

        data = json.load(f)

    counts = data["aggregate_counts"]

    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[: args.top_n]

    answers = [a[:40] + ("..." if len(a) > 40 else "") for a, _ in top]

    values = [c for _, c in top]

    ax.barh(range(len(answers)), values, color="#4C72B0")

    ax.set_yticks(range(len(answers)))

    ax.set_yticklabels(answers, fontsize=9)

    ax.invert_yaxis()

    ax.set_xlabel("Count (out of 100)")

    ax.set_title(label)


fig.suptitle(args.title)

plt.tight_layout()

plt.savefig(args.out, dpi=150)

print(f"Saved chart to {args.out}")
