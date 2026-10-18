import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval import questions
from peft import PeftModel
import json

MODEL_NAME = "Qwen/Qwen3-0.6B"

# Determine which device to use
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if device == "mps" else torch.float32
).to(device)

model.eval()

print(f"Model loaded on: {device}")

def predict(
    question: str,
    options: dict[str, str],
    system_prompt: str,
    temperature: float = 1.0,
    return_logits: bool = False
) -> dict[str, float]:
    
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

def decide(
    probabilities: dict[str, float],
    threshold: float = 0.70
) -> dict:

    if not 0 < threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    # Find the answer with the highest probability
    prediction, confidence = max(
        probabilities.items(),
        key=lambda item: item[1]
    )

    # Decide whether to answer
    if confidence >= threshold:
        decision = "ANSWER"
    else:
        decision = "ABSTAIN"

    return {
        "prediction": prediction,
        "confidence": confidence,
        "decision": decision
    }

PROMPT_1 = (
    "Answer multiple-choice questions. "
    "Respond with exactly one uppercase letter: A, B, C, D, or E. "
    "Do not include spaces, punctuation, or explanations."
)

PROMPT_2 = (
    "Answer multiple-choice questions. "
    "Respond with exactly one uppercase letter corresponding "
    "to the correct answer. No explanations."
)

def evaluate(dataset, system_prompt, temperature=1.0):
    results = []

    for item in tqdm(dataset, desc="Evaluating", unit="question"):
        # Get the model's predictions
        scores = predict(
            item["question"],
            item["options"],
            system_prompt,
            temperature=temperature
        )

        # Select the answer with the highest score
        prediction = max(scores, key=scores.get)

        results.append({
            "correct": prediction == item["correct"],
            "confidence": scores[prediction],
        })
    

    # Calculate accuracy
    accuracy = sum(result["correct"] for result in results) / len(results)

    # Average confidence across all predictions
    avg_confidence = sum(result["confidence"] for result in results) / len(results)

    # Predictions where confidence >= 90%
    high_confidence = [
        result for result in results if result["confidence"] >= 0.9
    ]

    # Incorrect predictions where confidence >= 90%
    high_confidence_errors = sum(
        not result["correct"] for result in high_confidence
    )

    high_confidence_accuracy = (
        1 - high_confidence_errors / len(high_confidence)
        if high_confidence else None
    )

    print(f"Total predictions: {len(results)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average confidence: {avg_confidence:.2%}")
    print(f"Predictions with confidence >= 90%: {len(high_confidence)}")
    print(f"High-confidence errors: {high_confidence_errors}")

    if high_confidence_accuracy is not None:
        print(f"High-confidence accuracy: {high_confidence_accuracy:.2%}")
    else:
        print("High-confidence accuracy: N/A")

    return accuracy


# Step 14.5 - Evaluate the large-scale fine-tuned model

eval_questions = questions

model = PeftModel.from_pretrained(
    model,
    "./lora_adapter_8341_best"
).to(device)

model.eval()

# Load our calibrated temperature
with open("./temperature.json", "r") as f:
    temperature = json.load(f)["temperature"]

print(f"Using temperature: {temperature:.4f}")

# Select a question from our existing development dataset
question = questions[0]

options = question["options"]

# Get calibrated probabilities
probabilities = predict(
    question=question["question"],
    options=options,
    system_prompt=PROMPT_2,
    temperature=temperature
)

# Make a decision
result = decide(
    probabilities,
    threshold=0.70
)

print("\n========== DIY JEV ==========")

print(f"\nQuestion: {question['question']}")

for letter, answer in options.items():
    print(f"{letter}. {answer}")

print(f"\nPrediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Decision: {result['decision']}")

print(f"\nCorrect answer: {question['correct']}")