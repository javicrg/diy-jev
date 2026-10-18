"""Canonical prediction and answer-or-abstain decision functions."""

import torch

from .config import DEFAULT_THRESHOLD

model = None
tokenizer = None
device = None


def configure_runtime(loaded_model, loaded_tokenizer, loaded_device):
    global model, tokenizer, device
    model, tokenizer, device = loaded_model, loaded_tokenizer, loaded_device


def predict(
    question: str,
    options: dict[str, str],
    system_prompt: str,
    temperature: float = 1.0,
    return_logits: bool = False,
) -> dict[str, float] | torch.Tensor:
    if model is None or tokenizer is None or device is None:
        raise RuntimeError("Inference runtime is not configured")
    prompt = question + "\n"
    for letter, answer in options.items():
        prompt += f"{letter}. {answer}\n"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_tensors="pt", return_dict=True,
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    answer_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in options]
    assert all(len(ids) == 1 for ids in answer_ids)
    answer_logits = torch.stack([logits[ids[0]] for ids in answer_ids])
    if return_logits:
        return answer_logits.detach().float().cpu()
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    probabilities = torch.softmax(answer_logits.float() / temperature, dim=0)
    return {letter: probability.item() for letter, probability in zip(options, probabilities)}


def decide(probabilities: dict[str, float], threshold: float = DEFAULT_THRESHOLD) -> dict:
    if not 0 < threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    prediction, confidence = max(probabilities.items(), key=lambda item: item[1])
    decision = "ANSWER" if confidence >= threshold else "ABSTAIN"
    return {"prediction": prediction, "confidence": confidence, "decision": decision}
