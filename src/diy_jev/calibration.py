"""Calibration helpers using the canonical inference function."""

import torch
from tqdm import tqdm

from .config import PROMPT_2
from .inference import predict
from .metrics import calculate_ece


def collect_logits(dataset, system_prompt=PROMPT_2):
    all_logits, all_labels = [], []
    for item in tqdm(dataset, desc="Collecting logits"):
        options = dict(zip(item["choices"]["label"], item["choices"]["text"]))
        logits = predict(item["question"], options, system_prompt, return_logits=True)
        all_logits.append(logits)
        all_labels.append(list(options).index(item["answerKey"]))
    return torch.stack(all_logits), torch.tensor(all_labels, dtype=torch.long)


def reliability_data(probabilities, labels, n_bins=10):
    confidences, predictions = probabilities.max(dim=1)
    correctness = (predictions == labels).float()
    avg_confidences, avg_accuracies = [], []
    for i in range(n_bins):
        in_bin = (confidences > i / n_bins) & (confidences <= (i + 1) / n_bins)
        if in_bin.any():
            avg_confidences.append(confidences[in_bin].mean().item())
            avg_accuracies.append(correctness[in_bin].mean().item())
    return avg_confidences, avg_accuracies
