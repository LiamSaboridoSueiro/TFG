"""
Test within-subject neuro_clean.

Modelo conservador para comprobar si queda señal discriminativa usando solo
features más defendibles neurofisiológicamente:
  - sin Gamma
  - sin Beta
  - sin canales periféricos/frontales/laterales sospechosos

Salida:
  scripts/tests/neuro_clean/within_subject_results.json
  scripts/tests/neuro_clean/within_subject_summary.csv
  scripts/tests/neuro_clean/selected_features.csv
  scripts/tests/neuro_clean/01_accuracy_<clf>.png
  scripts/tests/neuro_clean/02_confusion_<clf>.png
  scripts/tests/neuro_clean/03_importancia_<clf>.png
"""

from within_ablation_common import PERIPHERAL_CHANNELS, run_within_ablation


CONFIG = {
    "test_name": "neuro_clean",
    "title": "TEST WITHIN-SUBJECT NEURO CLEAN",
    "description": (
        "Entrenamiento within-subject conservador excluyendo Gamma, Beta "
        "y canales periféricos sospechosos."
    ),
    "excluded_bands": {"Gamma", "Beta"},
    "excluded_channels": PERIPHERAL_CHANNELS,
}


if __name__ == "__main__":
    run_within_ablation(CONFIG)
