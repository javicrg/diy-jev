"""Compatibility entry point for the official CommonsenseQA validation set."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from diy_jev.datasets import load_official_validation

questions = load_official_validation() if __name__ == "__main__" else []
