"""Lazy model and tokenizer loading."""

import torch

from .config import ADAPTER_PATH, MODEL_NAME


def get_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_runtime(adapter_path: str | None = ADAPTER_PATH):
    """Load one tokenizer/model pair; importing this module does no model work."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device == "mps" else torch.float32,
    ).to(device)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path).to(device)
    model.eval()
    return model, tokenizer, device
