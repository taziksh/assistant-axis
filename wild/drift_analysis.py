"""Launch the drift analysis stage; implementation lives in wild.drift.analysis."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wild.drift.analysis import main


if __name__ == "__main__":
    main()
