"""Original DIY JEV training procedure, executed only via main()."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

def main():
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from tqdm import tqdm
    import math

    MODEL_NAME = "Qwen/Qwen3-0.6B"

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    torch.manual_seed(42)

    # Load pretrained Qwen
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device == "mps" else torch.float32
    ).to(device)

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Attach LoRA adapters to Qwen
    model = get_peft_model(model, lora_config)

    print(f"Device: {device}")

    # Show how many parameters are trainable
    model.print_trainable_parameters()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load training dataset
    dataset = load_dataset(
        "tau/commonsense_qa",
        split="train"
    )

    print(f"Training examples: {len(dataset)}")

    SYSTEM_PROMPT = (
        "Answer multiple-choice questions. "
        "Respond with exactly one uppercase letter corresponding "
        "to the correct answer. No explanations."
    )


    def prepare_example(item):
        # Extract the options
        options = dict(zip(
            item["choices"]["label"],
            item["choices"]["text"]
        ))

        # Construct the question
        prompt = item["question"] + "\n"

        for letter, answer in options.items():
            prompt += f"{letter}. {answer}\n"

        # Construct chat messages
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Tokenise the question
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )

        # Tokenise the correct answer
        answer_ids = tokenizer.encode(
            item["answerKey"],
            add_special_tokens=False
        )

        assert len(answer_ids) == 1

        # Combine question and answer
        input_ids = prompt_ids + answer_ids

        # Only calculate loss for the answer
        labels = [-100] * len(prompt_ids) + answer_ids

        return {
            "input_ids": input_ids,
            "labels": labels
        }

    def validate(model, validation_dataset):
        model.eval()

        total_loss = 0
        correct_predictions = 0

        with torch.inference_mode():
            for item in tqdm(validation_dataset, desc="Validating", leave=False):
                example = prepare_example(item)

                input_ids = torch.tensor(
                    [example["input_ids"]],
                    dtype=torch.long,
                    device=device
                )

                labels = torch.tensor(
                    [example["labels"]],
                    dtype=torch.long,
                    device=device
                )

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    labels=labels
                )

                # Accumulate validation loss
                total_loss += outputs.loss.item()

                # Extract logits predicting the answer
                logits = outputs.logits[0, -2, :]

                # Get the token IDs for A, B, C, D, E
                letters = item["choices"]["label"]

                answer_ids = [
                    tokenizer.encode(letter, add_special_tokens=False)[0]
                    for letter in letters
                ]

                # Select only the answer logits
                answer_logits = logits[answer_ids]

                # Predict the answer
                prediction_index = answer_logits.argmax().item()
                prediction = letters[prediction_index]

                if prediction == item["answerKey"]:
                    correct_predictions += 1

        avg_loss = total_loss / len(validation_dataset)
        accuracy = correct_predictions / len(validation_dataset)

        model.train()

        return avg_loss, accuracy

    # Reproduce the final experiment partitions from the shared implementation
    from diy_jev.datasets import partition_training_dataset
    partitions = partition_training_dataset(dataset)
    train_dataset = partitions["train"]
    val_dataset = partitions["validation"]
    calibration_dataset = partitions["calibration"]
    test_dataset = partitions["test"]

    print(f"Training questions: {len(train_dataset)}")
    print(f"Validation questions: {len(val_dataset)}")
    print(f"Calibration questions: {len(calibration_dataset)}")
    print(f"Test questions: {len(test_dataset)}")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=1e-4
    )

    model.train()

    losses = []

    progress = tqdm(train_dataset, desc="Training", unit="question")

    num_epochs = 2
    accumulation_steps = 4
    validate_every = 250

    patience = 4
    checks_without_improvement = 0

    best_val_accuracy = 0.0
    optimizer_steps = 0

    early_stop = False
    losses = []

    optimizer.zero_grad()

    steps_per_epoch = math.ceil(
        len(train_dataset) / accumulation_steps
    )

    total_training_steps = num_epochs * steps_per_epoch

    warmup_steps = int(total_training_steps * 0.05)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    print(f"Total planned steps: {total_training_steps}")
    print(f"Warmup steps: {warmup_steps}")

    for epoch in range(num_epochs):

        model.train()

        print(f"\n========== EPOCH {epoch + 1}/{num_epochs} ==========")

        progress = tqdm(
            train_dataset,
            desc=f"Epoch {epoch + 1}"
        )

        for step, item in enumerate(progress, start=1):

            example = prepare_example(item)

            input_ids = torch.tensor(
                [example["input_ids"]],
                dtype=torch.long,
                device=device
            )

            labels = torch.tensor(
                [example["labels"]],
                dtype=torch.long,
                device=device
            )

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                labels=labels
            )

            loss = outputs.loss

            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss detected")

            # Record the original loss for monitoring
            losses.append(loss.item())
            # Determine how many questions are in the current group
            group_start = ((step - 1) // accumulation_steps) * accumulation_steps

            group_size = min(
                accumulation_steps,
                len(train_dataset) - group_start
            )

            # Scale loss by the actual group size
            scaled_loss = loss / group_size
            scaled_loss.backward()

            # Update weights every 4 questions
            if step % accumulation_steps == 0 or step == len(train_dataset):

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                if not torch.isfinite(grad_norm).item():
                    raise RuntimeError("Non-finite gradients detected")

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                optimizer_steps += 1

                # Periodic validation
                if optimizer_steps % validate_every == 0:

                    val_loss, val_accuracy = validate(
                        model,
                        val_dataset
                    )

                    current_lr = optimizer.param_groups[0]["lr"]

                    tqdm.write(
                        f"Step {optimizer_steps} | "
                        f"LR: {current_lr:.6f} | "
                        f"Val loss: {val_loss:.4f} | "
                        f"Val accuracy: {val_accuracy:.2%}"
                    )

                    if val_accuracy > best_val_accuracy:

                        best_val_accuracy = val_accuracy
                        checks_without_improvement = 0

                        model.save_pretrained(
                            "./lora_adapter_8341_best"
                        )

                        tqdm.write("New best checkpoint saved!")

                    else:

                        checks_without_improvement += 1

                        tqdm.write(
                            f"No improvement: "
                            f"{checks_without_improvement}/{patience}"
                        )

                    # Early stopping
                    if (
                        checks_without_improvement >= patience
                        and optimizer_steps >= steps_per_epoch
                    ):

                        tqdm.write(
                            "Early stopping triggered!"
                        )

                        early_stop = True
                        break

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{sum(losses[-20:]) / len(losses[-20:]):.4f}"
            )

        # Exit the outer epoch loop as well
        if early_stop:
            break

    print("\nTraining completed!")
    print(f"Total optimiser updates: {optimizer_steps}")
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")
    print(f"Average training loss: {sum(losses) / len(losses):.4f}")

    model.save_pretrained("./lora_adapter_8341_final")

    print("Final LoRA adapter saved!")
