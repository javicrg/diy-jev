"""Temperature-fitting entry point; fitting runs only when explicitly executed."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from diy_jev.calibration import reliability_data


def fit_temperature(logits, labels, num_steps=500):
    """Original Adam/log-temperature optimisation used by the project."""
    log_temperature = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([log_temperature], lr=0.05)
    for _ in range(num_steps):
        optimizer.zero_grad()
        temperature = log_temperature.exp()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        optimizer.step()
    return log_temperature.exp().item()


def main():
    data = torch.load("calibration_logits.pt", map_location="cpu", weights_only=True)
    logits, labels = data["logits"], data["labels"]
    temperature = fit_temperature(logits, labels)
    original = torch.softmax(logits, dim=1)
    calibrated = torch.softmax(logits / temperature, dim=1)
    with open("temperature.json", "w") as file:
        json.dump({"temperature": temperature}, file, indent=2)
    original_conf, original_acc = reliability_data(original, labels)
    calibrated_conf, calibrated_acc = reliability_data(calibrated, labels)
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.plot(original_conf, original_acc, marker="o", label="Before calibration")
    plt.plot(calibrated_conf, calibrated_acc, marker="o", label=f"After calibration (T={temperature:.2f})")
    plt.xlabel("Average predicted confidence")
    plt.ylabel("Actual accuracy")
    plt.title("Qwen3-0.6B — Reliability Diagram")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reliability_diagram.png", dpi=200)
    print(f"Temperature saved: {temperature:.4f}")


if __name__ == "__main__":
    main()
