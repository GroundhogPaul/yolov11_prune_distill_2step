import sys
from pathlib import Path

# ----- add the parent folder to environment -----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
