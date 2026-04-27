"""
Test within-subject sin Gamma ni Beta.

Salida:
  scripts/tests/sin_gamma_beta/within_subject_results.json
  scripts/tests/sin_gamma_beta/within_subject_summary.csv
  scripts/tests/sin_gamma_beta/selected_features.csv
  scripts/tests/sin_gamma_beta/01_accuracy_<clf>.png
  scripts/tests/sin_gamma_beta/02_confusion_<clf>.png
  scripts/tests/sin_gamma_beta/03_importancia_<clf>.png
"""

from within_ablation_common import run_within_ablation


CONFIG = {
    "test_name": "sin_gamma_beta",
    "title": "TEST WITHIN-SUBJECT SIN GAMMA NI BETA",
    "description": "Entrenamiento within-subject excluyendo las features bandpower de Gamma y Beta.",
    "excluded_bands": {"Gamma", "Beta"},
    "excluded_channels": set(),
}


if __name__ == "__main__":
    run_within_ablation(CONFIG)
