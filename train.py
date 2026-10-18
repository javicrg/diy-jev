"""Training entry point. Training is never started by importing this module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from diy_jev.training import main


if __name__ == "__main__":
    main()
