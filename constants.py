from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data"
TMP_DIR = ROOT_DIR / ".tmp"
TMP_DIR.mkdir(exist_ok=True)
