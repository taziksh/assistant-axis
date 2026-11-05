"""Launch the drift runner stage; implementation lives in wild.drift.runner."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wild.drift.runner import main


if __name__ == "__main__":
    main()
