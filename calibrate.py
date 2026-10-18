from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

DATASET_NAME = "tau/commonsense_qa"

# Reproduce the exact shuffle used during training
dataset = load_dataset(
    DATASET_NAME,
    split="train"
).shuffle(seed=42)

# Same sizes as train.py
num_val = 400
num_calibration = 400
num_test = 600

num_train = (
    len(dataset)
    - num_val
    - num_calibration
    - num_test
)

# Extract only the calibration questions
calibration_dataset = dataset.select(
    range(
        num_train + num_val,
        num_train + num_val + num_calibration
    )
)

print(f"Calibration questions: {len(calibration_dataset)}")

MODEL_NAME = "Qwen/Qwen3-0.6B"
ADAPTER_PATH = "./lora_adapter_8341_best"

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load original Qwen
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if device == "mps" else torch.float32
).to(device)

# Load our best fine-tuned adapter
model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)

model.eval()

print(f"Model loaded on: {device}")

def predict(
    question: str,
    options: dict[str, str],
    system_prompt: str,
    temperature: float = 1.0,
    return_logits: bool = False
) -> dict[str, float] | torch.Tensor:
        
    # Construct the question
    prompt = question + "\n"

    for letter, answer in options.items():
        prompt += f"{letter}. {answer}\n"

    # Construct the chat messages
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    # Tokenise
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True
    ).to(device)

    # Run inference
    with torch.inference_mode():
        outputs = model(**inputs)

    # Extract the next-token logits
    logits = outputs.logits[0, -1, :]

    # Get the token ID corresponding to each answer
    answer_ids = [
        tokenizer.encode(letter, add_special_tokens=False)
        for letter in options
    ]

    assert all(len(ids) == 1 for ids in answer_ids)

    # Extract logits for our answers
    answer_logits = torch.stack([
        logits[ids[0]] for ids in answer_ids
    ])

    if return_logits:
        return answer_logits.detach().float().cpu()

    # Ensure tempterature is valid
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")

    # Convert logits to scores
    probabilities = torch.softmax(answer_logits.float() / temperature, dim=0)

    # Return results
    return {
        letter: probability.item()
        for letter, probability in zip(options, probabilities)
    }

PROMPT_2 = (
    "Answer multiple-choice questions. Respond with exactly one uppercase letter "
    "corresponding to the correct answer. No explanations."
)

def collect_logits(dataset, system_prompt):
    all_logits = []
    all_labels = []

    for item in tqdm(dataset, desc="Collecting logits"):

        # Convert CommonsenseQA choices into our options dictionary
        options = dict(zip(
            item["choices"]["label"],
            item["choices"]["text"]
        ))

        # Obtain raw logits for A, B, C, D, E
        logits = predict(
            question=item["question"],
            options=options,
            system_prompt=system_prompt,
            return_logits=True
        )

        # Find the index of the correct answer
        correct_index = list(options.keys()).index(
            item["answerKey"]
        )

        all_logits.append(logits)
        all_labels.append(correct_index)

    return (
        torch.stack(all_logits),
        torch.tensor(all_labels, dtype=torch.long)
    )

# Collect calibration logits
logits, labels = collect_logits(
    calibration_dataset,
    PROMPT_2
)

print(f"Logits shape: {logits.shape}")
print(f"Labels shape: {labels.shape}")

# Save the results
torch.save(
    {
        "logits": logits,
        "labels": labels
    },
    "./calibration_logits.pt"
)

print("Calibration logits saved!")