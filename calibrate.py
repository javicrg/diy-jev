"""Collect calibration logits when explicitly executed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

from diy_jev.calibration import collect_logits
from diy_jev.config import PROMPT_2
from diy_jev.datasets import partition_training_dataset
from diy_jev.inference import configure_runtime
from diy_jev.model import load_runtime


def main():
    configure_runtime(*load_runtime())
    logits, labels = collect_logits(partition_training_dataset()["calibration"], PROMPT_2)
    torch.save({"logits": logits, "labels": labels}, "calibration_logits.pt")
    print(f"Saved calibration logits: {logits.shape}")


if __name__ == "__main__":
    main()
