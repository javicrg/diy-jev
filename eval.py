from datasets import load_dataset

dataset = load_dataset(
    "tau/commonsense_qa",
    split="validation"
)

# dataset = dataset.select(range(100))

questions = []

for item in dataset:
    options = dict(zip(
        item["choices"]["label"],
        item["choices"]["text"]
    ))

    questions.append({
        "question": item["question"],
        "options": options,
        "correct": item["answerKey"]
    })