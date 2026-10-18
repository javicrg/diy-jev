import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import torch
from types import SimpleNamespace

from diy_jev.config import DEFAULT_THRESHOLD, load_temperature
from diy_jev.inference import configure_runtime, decide, predict
from diy_jev.metrics import calculate_ece, evaluate_metrics


def test_decision_boundary_and_validation():
    assert decide({"A": 0.70, "B": 0.30})["decision"] == "ANSWER"
    assert decide({"A": 0.69, "B": 0.31})["decision"] == "ABSTAIN"
    assert DEFAULT_THRESHOLD == 0.70


def test_ece_and_temperature_preserve_argmax():
    logits = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    labels = torch.tensor([0, 1])
    metrics = evaluate_metrics(logits, labels, temperature=2.0905)
    assert metrics["accuracy"] == 1.0
    assert calculate_ece(torch.softmax(logits, dim=1), labels) >= 0


def test_saved_temperature_is_not_replaced():
    assert abs(load_temperature() - 2.090498685836792) < 1e-12


def test_prediction_contract_with_synthetic_runtime():
    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return Inputs()

        def encode(self, letter, add_special_tokens=False):
            return [ord(letter)]

    class Model:
        def __call__(self, **kwargs):
            logits = torch.zeros(1, 1, 256)
            for index, value in enumerate((5.0, 4.0, 3.0, 2.0, 1.0)):
                logits[0, 0, ord("ABCDE"[index])] = value
            return SimpleNamespace(logits=logits)

    configure_runtime(Model(), Tokenizer(), "cpu")
    options = {letter: letter for letter in "ABCDE"}
    raw = predict("Question", options, "Prompt", return_logits=True)
    assert raw.shape == (5,)
    probabilities = predict("Question", options, "Prompt")
    assert set(probabilities) == set(options)
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6
    try:
        predict("Question", options, "Prompt", temperature=0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive temperatures must be rejected")
