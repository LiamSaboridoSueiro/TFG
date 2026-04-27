"""
Test within-subject sin canales perifericos.

Salida:
  scripts/tests/sin_canales_perifericos/results/within_subject_results.json
  scripts/tests/sin_canales_perifericos/results/within_subject_summary.csv
  scripts/tests/sin_canales_perifericos/results/selected_features.csv
  scripts/tests/sin_canales_perifericos/results/01_accuracy_<clf>.png
  scripts/tests/sin_canales_perifericos/results/02_confusion_<clf>.png
  scripts/tests/sin_canales_perifericos/results/03_importancia_<clf>.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from within_ablation_common import PERIPHERAL_CHANNELS, run_within_ablation


CONFIG = {
    "test_name": "sin_canales_perifericos",
    "title": "TEST WITHIN-SUBJECT SIN CANALES PERIFERICOS",
    "description": "Entrenamiento within-subject excluyendo canales frontales/laterales/perifericos sospechosos.",
    "excluded_bands": set(),
    "excluded_channels": PERIPHERAL_CHANNELS,
    "output_dir": Path(__file__).resolve().parent / "results",
}


if __name__ == "__main__":
    run_within_ablation(CONFIG)
