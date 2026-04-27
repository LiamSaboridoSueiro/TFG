"""
Test within-subject sin Gamma.

Salida:
  scripts/tests/sin_gamma/within_subject_results.json
  scripts/tests/sin_gamma/within_subject_summary.csv
  scripts/tests/sin_gamma/selected_features.csv
  scripts/tests/sin_gamma/01_accuracy_<clf>.png
  scripts/tests/sin_gamma/02_confusion_<clf>.png
  scripts/tests/sin_gamma/03_importancia_<clf>.png
"""

from within_ablation_common import run_within_ablation


CONFIG = {
    "test_name": "sin_gamma",
    "title": "TEST WITHIN-SUBJECT SIN GAMMA",
    "description": "Entrenamiento within-subject excluyendo las features bandpower de Gamma.",
    "excluded_bands": {"Gamma"},
    "excluded_channels": set(),
}


if __name__ == "__main__":
    run_within_ablation(CONFIG)
