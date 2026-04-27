"""
Explicabilidad del test neuro_clean.

Lanza el script de explicabilidad compartido guardando la salida en:
  scripts/tests/neuro_clean/results/shap_topomap/
"""

import os
import runpy
import sys
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
TESTS_DIR = TEST_DIR.parent

sys.path.insert(0, str(TESTS_DIR))

os.environ["TEST_NAME"] = "neuro_clean"
os.environ["SHAP_OUTPUT_DIR"] = str(TEST_DIR / "results" / "shap_topomap")


if __name__ == "__main__":
    runpy.run_path(str(TESTS_DIR / "within_shap_common.py"), run_name="__main__")
