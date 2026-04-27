"""
Test within-subject neuro_clean.

Modelo conservador:
  - sin Gamma
  - sin Beta
  - sin canales perifericos/frontales/laterales sospechosos

Salida:
  scripts/tests/neuro_clean/results/within_subject_results.json
  scripts/tests/neuro_clean/results/within_subject_summary.csv
  scripts/tests/neuro_clean/results/selected_features.csv
  scripts/tests/neuro_clean/results/01_accuracy_<clf>.png
  scripts/tests/neuro_clean/results/02_confusion_<clf>.png
  scripts/tests/neuro_clean/results/03_importancia_<clf>.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from within_ablation_common import PERIPHERAL_CHANNELS, run_within_ablation


CONFIG = {
    "test_name": "neuro_clean",
    "title": "TEST WITHIN-SUBJECT NEURO CLEAN",
    "description": (
        "Entrenamiento within-subject conservador excluyendo Gamma, Beta "
        "y canales perifericos sospechosos."
    ),
    "excluded_bands": {"Gamma", "Beta"},
    "excluded_channels": PERIPHERAL_CHANNELS,
    "output_dir": Path(__file__).resolve().parent / "results",
}


if __name__ == "__main__":
    run_within_ablation(CONFIG)
