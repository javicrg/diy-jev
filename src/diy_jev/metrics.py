"""Metrics shared by evaluation and calibration."""

import torch
import torch.nn.functional as F


def calculate_ece(probabilities, labels, n_bins=10):
    """Calculate the original equal-width, confidence-binned ECE."""
    confidences, predictions = probabilities.max(dim=1)
    correctness = (predictions == labels).float()
    ece = 0.0
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins
        in_bin = (confidences > lower) & (confidences <= upper)
        if not in_bin.any():
            continue
        avg_confidence = confidences[in_bin].mean()
        avg_accuracy = correctness[in_bin].mean()
        bin_weight = in_bin.float().mean()
        ece += (bin_weight * torch.abs(avg_accuracy - avg_confidence)).item()
    return ece


def evaluate_metrics(logits, labels, temperature=1.0):
    probabilities = torch.softmax(logits / temperature, dim=1)
    confidences, predictions = probabilities.max(dim=1)
    return {
        "accuracy": (predictions == labels).float().mean().item(),
        "confidence": confidences.mean().item(),
        "nll": F.cross_entropy(logits / temperature, labels).item(),
        "ece": calculate_ece(probabilities, labels),
        "probabilities": probabilities,
    }
