"""
Test within-subject neuro_clean_beta.

Modelo conservador para comprobar las topologías incluyendo Beta:
  - sin Gamma
  - con Delta, Theta, Alpha y Beta
  - sin canales periféricos/frontales/laterales sospechosos

Salida:
  scripts/tests/neuro_clean_beta/within_subject_results.json
  scripts/tests/neuro_clean_beta/within_subject_summary.csv
  scripts/tests/neuro_clean_beta/selected_features.csv
  scripts/tests/neuro_clean_beta/01_accuracy_<clf>.png
  scripts/tests/neuro_clean_beta/02_confusion_<clf>.png
  scripts/tests/neuro_clean_beta/03_importancia_<clf>.png
"""

from within_ablation_common import PERIPHERAL_CHANNELS, run_within_ablation


CONFIG = {
    "test_name": "neuro_clean_beta",
    "title": "TEST WITHIN-SUBJECT NEURO CLEAN + BETA",
    "description": (
        "Entrenamiento within-subject excluyendo Gamma y canales periféricos "
        "sospechosos, manteniendo la banda Beta."
    ),
    "excluded_bands": {"Gamma"},
    "excluded_channels": PERIPHERAL_CHANNELS,
}


if __name__ == "__main__":
    run_within_ablation(CONFIG)
