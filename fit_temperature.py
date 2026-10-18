import torch
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt

# Load our saved calibration data
data = torch.load(
    "./calibration_logits.pt",
    map_location="cpu",
    weights_only=True
)

logits = data["logits"]
labels = data["labels"]

print(f"Logits shape: {logits.shape}")
print(f"Labels shape: {labels.shape}")

# Initialise log(T) = 0, meaning T = 1
log_temperature = torch.nn.Parameter(
    torch.tensor(0.0)
)

optimizer = torch.optim.Adam(
    [log_temperature],
    lr=0.05
)

num_steps = 500

for step in range(num_steps):

    optimizer.zero_grad()

    # Convert log-temperature to temperature
    temperature = log_temperature.exp()

    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Calculate negative log-likelihood
    loss = F.cross_entropy(
        scaled_logits,
        labels
    )

    # Calculate gradients
    loss.backward()

    # Update temperature
    optimizer.step()

    if (step + 1) % 100 == 0:
        print(
            f"Step {step + 1} | "
            f"Temperature: {temperature.item():.4f} | "
            f"NLL: {loss.item():.4f}"
        )

# Extract the optimised temperature
optimal_temperature = log_temperature.exp().item()

print(f"\nOptimal temperature: {optimal_temperature:.4f}")

# Calculate metrics before and after calibration
with torch.no_grad():

    original_nll = F.cross_entropy(
        logits,
        labels
    ).item()

    calibrated_nll = F.cross_entropy(
        logits / optimal_temperature,
        labels
    ).item()

    original_probs = torch.softmax(logits, dim=1)

    calibrated_probs = torch.softmax(
        logits / optimal_temperature,
        dim=1
    )

    original_confidence = original_probs.max(dim=1).values.mean().item()

    calibrated_confidence = calibrated_probs.max(dim=1).values.mean().item()

    accuracy = (
        (logits.argmax(dim=1) == labels)
        .float()
        .mean()
        .item()
    )

print("\n========== CALIBRATION RESULTS ==========")

print(f"Accuracy: {accuracy:.2%}")

print(f"Original NLL: {original_nll:.4f}")
print(f"Calibrated NLL: {calibrated_nll:.4f}")

print(f"Original average confidence: {original_confidence:.2%}")
print(f"Calibrated average confidence: {calibrated_confidence:.2%}")

with open("./temperature.json", "w") as f:
    json.dump(
        {"temperature": optimal_temperature},
        f,
        indent=2
    )

print("\nTemperature saved!")

def calculate_ece(probabilities, labels, n_bins=10):

    # Get predicted class and its confidence
    confidences, predictions = probabilities.max(dim=1)

    # Determine which predictions are correct
    correctness = (predictions == labels).float()

    ece = 0.0

    for i in range(n_bins):

        # Define confidence interval
        lower = i / n_bins
        upper = (i + 1) / n_bins

        # Find predictions inside this interval
        in_bin = (
            (confidences > lower) &
            (confidences <= upper)
        )

        # Skip empty intervals
        if not in_bin.any():
            continue

        # Average confidence inside this interval
        avg_confidence = confidences[in_bin].mean()

        # Actual accuracy inside this interval
        avg_accuracy = correctness[in_bin].mean()

        # Proportion of predictions in this interval
        bin_weight = in_bin.float().mean()

        # Add weighted calibration error
        ece += (
            bin_weight
            * torch.abs(avg_accuracy - avg_confidence)
        ).item()

    return ece

original_ece = calculate_ece(
    original_probs,
    labels
)

calibrated_ece = calculate_ece(
    calibrated_probs,
    labels
)

print("\n========== CALIBRATION ERROR ==========")

print(f"Original ECE:   {original_ece:.2%}")
print(f"Calibrated ECE: {calibrated_ece:.2%}")

print(
    f"ECE improvement: "
    f"{(original_ece - calibrated_ece) * 100:+.2f} pp"
)

def reliability_data(probabilities, labels, n_bins=10):

    confidences, predictions = probabilities.max(dim=1)
    correctness = (predictions == labels).float()

    avg_confidences = []
    avg_accuracies = []

    for i in range(n_bins):

        lower = i / n_bins
        upper = (i + 1) / n_bins

        in_bin = (
            (confidences > lower) &
            (confidences <= upper)
        )

        if not in_bin.any():
            continue

        avg_confidences.append(
            confidences[in_bin].mean().item()
        )

        avg_accuracies.append(
            correctness[in_bin].mean().item()
        )

    return avg_confidences, avg_accuracies


# Calculate reliability data
original_conf, original_acc = reliability_data(
    original_probs,
    labels
)

calibrated_conf, calibrated_acc = reliability_data(
    calibrated_probs,
    labels
)

# Create the plot
plt.figure(figsize=(8, 6))

# Perfect calibration reference line
plt.plot(
    [0, 1],
    [0, 1],
    "--",
    label="Perfect calibration"
)

# Original model
plt.plot(
    original_conf,
    original_acc,
    marker="o",
    label="Before calibration"
)

# Calibrated model
plt.plot(
    calibrated_conf,
    calibrated_acc,
    marker="o",
    label=f"After calibration (T={optimal_temperature:.2f})"
)

plt.xlabel("Average predicted confidence")
plt.ylabel("Actual accuracy")

plt.title("Qwen3-0.6B — Reliability Diagram")

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

# Save the figure
plt.savefig(
    "./reliability_diagram.png",
    dpi=200
)

# plt.show()

def evaluate_thresholds(probabilities, labels, thresholds):

    confidences, predictions = probabilities.max(dim=1)

    correct = (predictions == labels)

    print("\n========== SELECTIVE PREDICTION ==========")

    print(
        f"{'Threshold':<12}"
        f"{'Coverage':<12}"
        f"{'Selective Acc.':<18}"
        f"{'Answered'}"
    )

    for threshold in thresholds:

        # Select predictions above the threshold
        selected = confidences >= threshold

        # Number of questions answered
        answered = selected.sum().item()

        # Fraction of questions answered
        coverage = selected.float().mean().item()

        # Accuracy among answered questions
        if answered > 0:
            selective_accuracy = (
                correct[selected].float().mean().item()
            )
        else:
            selective_accuracy = float("nan")

        print(
            f"{threshold:<12.2f}"
            f"{coverage:<12.2%}"
            f"{selective_accuracy:<18.2%}"
            f"{answered}"
        )

thresholds = [
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]

evaluate_thresholds(
    calibrated_probs,
    labels,
    thresholds
)