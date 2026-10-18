"""CommonsenseQA loading and reproducible partitioning."""

from datasets import load_dataset

from .config import (
    DATASET_NAME,
    NUM_CALIBRATION,
    NUM_TEST,
    NUM_VALIDATION,
    RANDOM_SEED,
)


def choices_to_options(item: dict) -> dict[str, str]:
    """Convert a raw CommonsenseQA row to the option mapping used by inference."""
    if "options" in item:
        return dict(item["options"])
    return dict(zip(item["choices"]["label"], item["choices"]["text"]))


def to_question(item: dict) -> dict:
    """Return the compact question representation used by the original scripts."""
    return {
        "question": item["question"],
        "options": choices_to_options(item),
        "correct": item["correct"] if "correct" in item else item["answerKey"],
    }


def load_official_validation():
    """Load the official validation split used by development experiments."""
    return [to_question(item) for item in load_dataset(DATASET_NAME, split="validation")]


def load_training_split():
    return load_dataset(DATASET_NAME, split="train")


def partition_training_dataset(dataset=None) -> dict[str, object]:
    """Reproduce the exact train/validation/calibration/test boundaries."""
    if dataset is None:
        dataset = load_training_split()
    shuffled = dataset.shuffle(seed=RANDOM_SEED)
    num_train = len(shuffled) - NUM_VALIDATION - NUM_CALIBRATION - NUM_TEST
    boundaries = {
        "train": (0, num_train),
        "validation": (num_train, num_train + NUM_VALIDATION),
        "calibration": (
            num_train + NUM_VALIDATION,
            num_train + NUM_VALIDATION + NUM_CALIBRATION,
        ),
        "test": (num_train + NUM_VALIDATION + NUM_CALIBRATION, len(shuffled)),
    }
    partitions = {name: shuffled.select(range(start, end)) for name, (start, end) in boundaries.items()}
    if sum(len(part) for part in partitions.values()) != len(shuffled):
        raise AssertionError("Dataset partitions do not cover the shuffled dataset")
    if any(end > next_start for (_, end), (next_start, _) in zip(boundaries.values(), list(boundaries.values())[1:])):
        raise AssertionError("Dataset partitions overlap")
    return partitions
