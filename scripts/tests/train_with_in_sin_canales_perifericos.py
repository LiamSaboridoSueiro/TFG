"""
Test within-subject sin canales periféricos.

Salida:
  scripts/tests/sin_canales_perifericos/within_subject_results.json
  scripts/tests/sin_canales_perifericos/within_subject_summary.csv
  scripts/tests/sin_canales_perifericos/selected_features.csv
  scripts/tests/sin_canales_perifericos/01_accuracy_<clf>.png
  scripts/tests/sin_canales_perifericos/02_confusion_<clf>.png
  scripts/tests/sin_canales_perifericos/03_importancia_<clf>.png
"""

from within_ablation_common import PERIPHERAL_CHANNELS, run_within_ablation


CONFIG = {
    "test_name": "sin_canales_perifericos",
    "title": "TEST WITHIN-SUBJECT SIN CANALES PERIFERICOS",
    "description": "Entrenamiento within-subject excluyendo canales frontales/laterales/periféricos sospechosos.",
    "excluded_bands": set(),
    "excluded_channels": PERIPHERAL_CHANNELS,
}


if __name__ == "__main__":
    run_within_ablation(CONFIG)
