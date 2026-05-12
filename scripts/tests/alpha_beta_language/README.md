# alpha_beta_language

Version candidata a pipeline principal, pero mantenida todavia dentro de
`scripts/tests` para no tocar los scripts finales.

Estructura equivalente al pipeline principal:

- `3_train/train_within_subject.py`: entrenamiento within-subject Alpha/Beta.
- `3_train/train_all_subjects.py`: entrenamiento all-subjects Alpha/Beta con
  `StratifiedGroupKFold` por sujeto.
- `5_shap_topomap/explicability.py`: SHAP lineal, topomaps y clusters.

Archivo comun de configuracion:

- `alpha_beta_language_common.py`: carga de datos, seleccion de features
  Alpha/Beta, normalizacion within-fold y definicion de clusters.

Salidas:

- `results/`: metricas, figuras y explicabilidad del CV temporal principal.
- `results/all_subjects/`: metricas y figuras del modelo all-subjects.
- `models/all_subjects/`: modelo global all-subjects entrenado con Alpha/Beta.
