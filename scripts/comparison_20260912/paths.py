"""Local inputs and generated outputs for the historical comparison scripts."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = Path(os.environ.get("SG9_LOG_DIR", str(Path.home() / "Downloads" / "Logs"))).expanduser()
OUTPUT_DIR = PROJECT_ROOT / "output" / "20260912" / "comparison"
