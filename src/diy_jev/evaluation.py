"""Evaluation orchestration; no work occurs on import."""

import torch

from .config import PROMPT_2
from .calibration import collect_logits
from .metrics import evaluate_metrics


def selective_metrics(probabilities, labels, threshold):
    confidences, predictions = probabilities.max(dim=1)
    selected = confidences >= threshold
    answered = selected.sum().item()
    return {
        "answered": answered,
        "coverage": selected.float().mean().item(),
        "selective_accuracy": (
            (predictions[selected] == labels[selected]).float().mean().item()
            if answered else float("nan")
        ),
    }
