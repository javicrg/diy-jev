"""Stable configuration used by the final DIY JEV experiment."""

import json
from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-0.6B"
ADAPTER_PATH = "lora_adapter_8341_best"
DATASET_NAME = "tau/commonsense_qa"
RANDOM_SEED = 42
NUM_VALIDATION = 400
NUM_CALIBRATION = 400
NUM_TEST = 600
DEFAULT_THRESHOLD = 0.70

PROMPT_1 = (
    "Answer multiple-choice questions. Respond with exactly one uppercase letter: "
    "A, B, C, D, or E. Do not include spaces, punctuation, or explanations."
)
PROMPT_2 = (
    "Answer multiple-choice questions. Respond with exactly one uppercase letter "
    "corresponding to the correct answer. No explanations."
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPERATURE_PATH = PROJECT_ROOT / "temperature.json"


def load_temperature(path: str | Path = TEMPERATURE_PATH) -> float:
    """Load the persisted calibration temperature without fitting it."""
    with Path(path).open() as file:
        return float(json.load(file)["temperature"])
