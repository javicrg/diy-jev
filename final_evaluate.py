"""Compatibility entry point for the held-out evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from diy_jev.config import ADAPTER_PATH, DEFAULT_THRESHOLD, load_temperature
from diy_jev.datasets import partition_training_dataset
from diy_jev.evaluation import collect_logits, selective_metrics
from diy_jev.inference import configure_runtime
from diy_jev.metrics import evaluate_metrics
from diy_jev.model import load_runtime


def main():
    import torch
    from peft import PeftModel

    test_dataset = partition_training_dataset()["test"]
    model, tokenizer, device = load_runtime(adapter_path=None)
    configure_runtime(model, tokenizer, device)
    baseline_logits, labels = collect_logits(test_dataset)
    baseline = evaluate_metrics(baseline_logits, labels)

    model = PeftModel.from_pretrained(model, ADAPTER_PATH).to(device)
    model.eval()
    configure_runtime(model, tokenizer, device)
    finetuned_logits, finetuned_labels = collect_logits(test_dataset)
    assert torch.equal(labels, finetuned_labels)
    finetuned = evaluate_metrics(finetuned_logits, labels)
    calibrated = evaluate_metrics(finetuned_logits, labels, load_temperature())
    for name, metrics in (("Original", baseline), ("Fine-tuned", finetuned), ("Calibrated", calibrated)):
        print(name, {key: value for key, value in metrics.items() if key != "probabilities"})
    print(selective_metrics(calibrated["probabilities"], labels, DEFAULT_THRESHOLD))


if __name__ == "__main__":
    main()
