# DIY JEV

DIY JEV is a confidence-aware multiple-choice decision system. It uses
Qwen3-0.6B, LoRA fine-tuning, and temperature scaling on CommonsenseQA. For
each five-choice question it predicts the next token (A–E), calibrates the
resulting probabilities, and either answers or abstains when confidence is
below 70%. Confidence is statistical, not a guarantee of correctness.

## Motivation

Prediction, confidence estimation, and decision-making are separate steps.
Calibration makes confidence more representative of observed accuracy, while
selective prediction lets the system trade coverage for higher accuracy on
the questions it answers.

## Architecture

```text
Question -> Qwen3-0.6B + LoRA -> next-token logits (A–E)
         -> softmax (T = 2.0905) -> probabilities -> threshold (70%)
         -> ANSWER or ABSTAIN
```

The final adapter is `lora_adapter_8341_best`. The exact saved temperature is
in `temperature.json`.

## Training and calibration

The final run shuffled the 9,741-question training split with seed 42 and
created 8,341 training, 400 validation, 400 calibration, and 600 held-out
test examples. LoRA used rank 8, alpha 16, dropout 0.05, and `q_proj` and
`v_proj` targets. Training used AdamW, learning rate `1e-4`, linear warmup and
decay, physical batch size 1, gradient accumulation 4, two maximum epochs,
validation every 250 optimizer updates, and early stopping patience 4.

Temperature scaling was fitted only on the separate 400-question calibration
partition using the existing Adam/log-temperature procedure. `fit_temperature.py`
can fit from `calibration_logits.pt`; it is never run on import.

## Results

These are measured on the internal 600-question held-out partition of the
CommonsenseQA training split, not an independent external benchmark.

| Metric             | Original | Fine-tuned | Calibrated |
| ------------------ | -------: | ---------: | ---------: |
| Accuracy           |   49.33% |     65.17% |     65.17% |
| Average confidence |   92.00% |     82.60% |     65.07% |
| NLL                |   3.7584 |     1.1344 |     0.9160 |
| ECE                |   42.94% |     17.43% |      3.91% |

At a 70% threshold, the system answered 260/600 questions (43.33% coverage),
with 82.69% selective accuracy: 215 correct and 45 incorrect. This does not
improve underlying model accuracy; it abstains on lower-confidence cases.
Earlier development results on the official 1,221-question validation split
are recorded in `results/experiments.csv` and are not a controlled ablation.

## Installation

Use Python 3.12 in a virtual or Conda environment. Install the project and
its dependencies with:

```bash
python -m pip install -e .
```

On Apple Silicon, install a PyTorch build appropriate for the machine; the
code uses MPS when available and falls back to CPU. Model and dataset downloads
require network access.

## Usage

Interactive inference loads the model once and asks for one question and five
options:

```bash
python main.py
```

Other entry points are preserved:

```bash
python train.py              # expensive: retrains and writes adapters
python calibrate.py          # expensive: collects calibration logits
python fit_temperature.py    # writes temperature.json
python final_evaluate.py     # expensive: evaluates the held-out partition
```

Do not run the expensive commands merely to use the existing results.

## Project structure

`src/diy_jev/` contains canonical configuration, model loading, inference,
dataset partitioning, metrics, calibration, evaluation, and training code.
Root scripts are thin compatibility entry points. `tests/` contains lightweight
tests that use synthetic tensors and do not download model weights.
Measured summaries are in `results/`; the existing root-level calibration
artifacts and adapters are preserved locally.

## Limitations and reproducibility

The system supports five-option CommonsenseQA-style inputs. Calibration may
not transfer to out-of-distribution questions, and abstention reduces the
percentage answered. Results use the internal held-out split described above.
Reproduction depends on Qwen/Qwen3-0.6B, the `tau/commonsense_qa` dataset,
seed 42, the saved adapter name, prompt `PROMPT_2`, the exact split boundaries,
and the recorded training configuration. Model and dataset licenses and usage
conditions should be reviewed before redistribution. This repository does not
add a license because no license decision has been made.

## Acknowledgements

Built with Qwen, PyTorch, Hugging Face Transformers and Datasets, PEFT, and
the CommonsenseQA dataset.
