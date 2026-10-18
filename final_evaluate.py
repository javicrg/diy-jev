import json
import torch
import torch.nn.functional as F

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen3-0.6B"
ADAPTER_PATH = "./lora_adapter_8341_best"

THRESHOLD = 0.70

with open("./temperature.json", "r") as f:
    TEMPERATURE = json.load(f)["temperature"]

device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Device: {device}")
print(f"Temperature: {TEMPERATURE:.4f}")
print(f"Threshold: {THRESHOLD:.2%}")

dataset = load_dataset(
    "tau/commonsense_qa",
    split="train"
).shuffle(seed=42)

num_val = 400
num_calibration = 400
num_test = 600

num_train = (
    len(dataset)
    - num_val
    - num_calibration
    - num_test
)

test_dataset = dataset.select(
    range(
        num_train + num_val + num_calibration,
        len(dataset)
    )
)

print(f"Test questions: {len(test_dataset)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if device == "mps" else torch.float32
).to(device)

model.eval()

PROMPT_2 = (
    "Answer multiple-choice questions. Respond with exactly one uppercase letter "
    "corresponding to the correct answer. No explanations."
)


def collect_logits(dataset):

    all_logits = []
    all_labels = []

    for item in tqdm(dataset, desc="Evaluating"):

        options = dict(zip(
            item["choices"]["label"],
            item["choices"]["text"]
        ))

        prompt = item["question"] + "\n"

        for letter, answer in options.items():
            prompt += f"{letter}. {answer}\n"

        messages = [
            {"role": "system", "content": PROMPT_2},
            {"role": "user", "content": prompt}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)

        logits = outputs.logits[0, -1, :]

        answer_ids = [
            tokenizer.encode(letter, add_special_tokens=False)
            for letter in options
        ]

        assert all(len(ids) == 1 for ids in answer_ids)

        answer_logits = torch.stack([
            logits[ids[0]] for ids in answer_ids
        ])

        correct_index = list(options).index(
            item["answerKey"]
        )

        all_logits.append(
            answer_logits.float().cpu()
        )

        all_labels.append(correct_index)

    return (
        torch.stack(all_logits),
        torch.tensor(all_labels, dtype=torch.long)
    )

def calculate_ece(probabilities, labels, n_bins=10):
    confidences, predictions = probabilities.max(dim=1)
    correctness = (predictions == labels).float()

    ece = 0.0

    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins

        in_bin = (
            (confidences > lower) &
            (confidences <= upper)
        )

        if not in_bin.any():
            continue

        avg_confidence = confidences[in_bin].mean()
        avg_accuracy = correctness[in_bin].mean()
        bin_weight = in_bin.float().mean()

        ece += (
            bin_weight
            * torch.abs(avg_accuracy - avg_confidence)
        ).item()

    return ece

def evaluate_metrics(logits, labels, temperature=1.0):

    probabilities = torch.softmax(
        logits / temperature,
        dim=1
    )

    confidences, predictions = probabilities.max(dim=1)

    accuracy = (
        (predictions == labels).float().mean().item()
    )

    avg_confidence = confidences.mean().item()

    nll = F.cross_entropy(
        logits / temperature,
        labels
    ).item()

    ece = calculate_ece(probabilities, labels)

    return {
        "accuracy": accuracy,
        "confidence": avg_confidence,
        "nll": nll,
        "ece": ece,
        "probabilities": probabilities
    }

# ==========================================
# 1. Original Qwen
# ==========================================

print("\n========== ORIGINAL QWEN ==========")

baseline_logits, labels = collect_logits(test_dataset)

baseline = evaluate_metrics(
    baseline_logits,
    labels
)


# ==========================================
# 2. Fine-tuned Qwen
# ==========================================

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
).to(device)

model.eval()

print("\n========== FINE-TUNED QWEN ==========")

finetuned_logits, finetuned_labels = collect_logits(test_dataset)

assert torch.equal(labels, finetuned_labels)

finetuned = evaluate_metrics(
    finetuned_logits,
    labels
)


# ==========================================
# 3. Calibrated Qwen
# ==========================================

calibrated = evaluate_metrics(
    finetuned_logits,
    labels,
    temperature=TEMPERATURE
)


# ==========================================
# 4. Selective prediction
# ==========================================

probabilities = calibrated["probabilities"]

confidences, predictions = probabilities.max(dim=1)

selected = confidences >= THRESHOLD

coverage = selected.float().mean().item()
answered = selected.sum().item()

if answered > 0:
    selective_accuracy = (
        (predictions[selected] == labels[selected])
        .float()
        .mean()
        .item()
    )
else:
    selective_accuracy = float("nan")


# ==========================================
# 5. Final results
# ==========================================

print("\n========== FINAL TEST RESULTS ==========")

for name, metrics in [
    ("Original Qwen", baseline),
    ("Fine-tuned Qwen", finetuned),
    ("Calibrated Qwen", calibrated)
]:
    print(f"\n{name}")
    print(f"Accuracy:       {metrics['accuracy']:.2%}")
    print(f"Avg confidence: {metrics['confidence']:.2%}")
    print(f"NLL:            {metrics['nll']:.4f}")
    print(f"ECE:            {metrics['ece']:.2%}")

print("\n========== SELECTIVE PREDICTION ==========")

print(f"Threshold:          {THRESHOLD:.2%}")
print(f"Questions answered: {answered}/{len(labels)}")
print(f"Coverage:           {coverage:.2%}")
print(f"Selective accuracy: {selective_accuracy:.2%}")