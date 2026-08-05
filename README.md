# Subliminal Learning in Small Models (Team : Shivansh ,Mrigakshi, Anshika)

Investigating whether subliminal learning — a student model inheriting a hidden trait from its teacher via unrelated data — requires teacher and student to share the same initialization/model family. Tested via same-family transfer (Pythia → Pythia) and
cross-family transfer (Grok → Pythia), with Gemma-3-4B as a pipeline sanity check.

## Setup

```bash
git clone <this repo>
cd subliminal-learning
uv sync              
# or: python3.11 -m venv .venv && pip install -r requirements.txt
source .venv/bin/activate
pip install transformers accelerate peft datasets matplotlib numpy
```
Note: Use Python 3.9 or above 

If running on a machine with older GPUs (e.g. Pascal-generation, GTX 10-series), install a PyTorch build with matching CUDA support — see comments in `scripts/pythia/*.py` for the exact versions used.

## Pipeline overview

Each experiment follows the same shape: fine-tune (or prompt) a teacher toward a trait, generate number sequences from it, train a fresh student on those numbers only, then evaluate whether the trait transferred.

### 1. Pythia, own-generated data

```bash
RUN_NAME=pythia_own_owl python scripts/pythia/finetune_owl_teacher.py
RUN_NAME=pythia_own_owl DATA_RUN_NAME=pythia_own_owl python scripts/pythia/generate_numbers.py owl
RUN_NAME=pythia_own_control python scripts/pythia/generate_numbers.py control

RUN_NAME=pythia_own_owl DATA_RUN_NAME=pythia_own_owl python scripts/pythia/finetune_student.py owl
RUN_NAME=pythia_own_control DATA_RUN_NAME=pythia_own_control python scripts/pythia/finetune_student.py control

RUN_NAME=pythia_own_owl_full DATA_RUN_NAME=pythia_own_owl python scripts/pythia/finetune_student_full.py owl
RUN_NAME=pythia_own_control_full DATA_RUN_NAME=pythia_own_control python scripts/pythia/finetune_student_full.py control
```

### 2. Pythia, Grok-generated data

Place the Grok-generated dataset at `runs/pythia_grok_owl/number_dataset.jsonl`, then:

```bash
RUN_NAME=pythia_grok_owl DATA_RUN_NAME=pythia_grok_owl python scripts/pythia/finetune_student.py owl
RUN_NAME=pythia_grok_owl_full DATA_RUN_NAME=pythia_grok_owl python scripts/pythia/finetune_student_full.py owl
```

### 3. Untrained baseline (no training, just eval)

```bash
python scripts/evaluate.py --path EleutherAI/pythia-410m --base EleutherAI/pythia-410m \
    --out runs/pythia_untrained_baseline/eval.json
```

### 4. Gemma sanity check

```bash
python scripts/gemma/finetune_owl_teacher_gemma.py
python scripts/gemma/generate_numbers_gemma.py owl
python scripts/gemma/generate_numbers_gemma.py control
python scripts/gemma/finetune_student_gemma.py owl
python scripts/gemma/finetune_student_gemma.py control
```

Every run writes `runs/<RUN_NAME>/eval.json` with the raw answer counts.

## Logging results

After each run:
```bash
python scripts/log_result.py --eval runs/<RUN_NAME>/eval.json --notes "e.g. LoRA, 10 epochs"
```
Appends one row to `results_log.csv` — a running table of every experiment.

## Visualizing results

`plot_results.py` auto-detects whether a model's output stayed coherent or collapsed,
and picks the right chart accordingly — same command either way:

```bash
# LoRA (coherent output) -> 5-bar category chart: top 3 animals, bird, other
python scripts/plotting/plot_results.py \
    --owl runs/pythia_own_owl/eval.json --control runs/pythia_own_control/eval.json \
    --title "Pythia same-family, LoRA" --out figures/pythia_lora.png

# Full fine-tune (collapsed output) -> side-by-side raw-text panels instead
python scripts/plotting/plot_results.py \
    --owl runs/pythia_own_owl_full/eval.json --control runs/pythia_own_control_full/eval.json \
    --title "Pythia same-family, Full fine-tune" --out figures/pythia_full.png

# No control yet (e.g. Grok control pending) -- just omit --control
python scripts/plotting/plot_results.py \
    --owl runs/pythia_grok_owl/eval.json \
    --title "Grok to Pythia cross-family (preliminary)" --out figures/grok_lora.png
```

For a precise breakdown of how often the model said "owl" specifically vs. any bird vs.
any animal at all:
```bash
python scripts/plotting/compute_rates.py \
    --owl runs/pythia_grok_owl/eval.json --control runs/pythia_untrained_baseline/eval.json \
    --control-label "Untrained baseline" \
    --title "Grok to Pythia cross-family" --out figures/grok_rates.png
```

Charts are saved to `figures/`.
