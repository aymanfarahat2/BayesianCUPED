"""
Legacy entry point: runs the modular pipeline (main.py).
Use: python treatment_evaluation_pipeline.py
Or:  python main.py
"""

import sys
from pathlib import Path

# Run from package directory so imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent))

from main import main

if __name__ == "__main__":
    main()
