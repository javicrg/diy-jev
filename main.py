"""Interactive DIY JEV inference entry point."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from diy_jev.config import DEFAULT_THRESHOLD, PROMPT_2, load_temperature
from diy_jev.inference import configure_runtime, decide, predict
from diy_jev.model import load_runtime


def main():
    print("Loading DIY JEV model...")
    configure_runtime(*load_runtime())
    temperature = load_temperature()
    raw_question = input("Question: ").strip()
    options = {}
    for letter in "ABCDE":
        options[letter] = input(f"{letter}: ").strip()
    if not raw_question or any(not value for value in options.values()):
        raise ValueError("Enter a question and five non-empty answer options.")
    result = decide(predict(raw_question, options, PROMPT_2, temperature), DEFAULT_THRESHOLD)
    if result["decision"] == "ANSWER":
        print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Decision: {result['decision']}")


if __name__ == "__main__":
    main()
