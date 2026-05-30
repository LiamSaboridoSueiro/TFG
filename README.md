# TFG - Reconocimiento de emociones con EEG

Este repositorio contiene el código y la documentación de un Trabajo Fin de Grado centrado en el reconocimiento automático de emociones a partir de señales EEG. El flujo implementado parte de registros crudos en formato EDF, realiza el preprocesado de la señal, extrae características espectrales, entrena modelos interpretables y genera análisis de separabilidad y explicabilidad para estudiar qué información utiliza el sistema al diferenciar entre alegría, estado neutro y tristeza.

## Estructura del repositorio

```text
TFG/
|-- data/
|   |-- raw/              # datos originales: EDF, audios y material de referencia
|   |-- processed/        # épocas preprocesadas y estadísticas por archivo
|   `-- features/         # matrices de características y metadatos
|-- scripts/
|   |-- 0_install/        # instalación del entorno Conda
|   |-- 1_preprocesado/   # conversión EDF -> épocas limpias
|   |-- 2_features/       # extracción de características espectrales
|   |-- 3_train/          # entrenamiento within-subject y all-subjects
|   |-- 4_separability/   # análisis PCA/LDA y métricas de separabilidad
|   `-- 5_explicability/  # análisis SHAP y mapas topográficos
|-- models/               # modelos entrenados guardados con joblib
|-- results/              # métricas, figuras y resúmenes generados
`-- docs/                 # memoria del TFG en LaTeX
```

## Requisitos

El proyecto está preparado para ejecutarse principalmente en Ubuntu usando Conda o Miniconda. Se recomienda usar Conda para reproducir el entorno completo, ya que el procesamiento EEG combina librerías científicas como MNE-Python, NumPy, pandas, SciPy, scikit-learn, Matplotlib, SHAP y joblib.

Es necesario tener instalado:

- Ubuntu o una distribución Linux compatible.
- Conda/Miniconda disponible en terminal.
- Los datos crudos organizados dentro de `data/raw/edf/`.

## Instalación

Desde la raíz del repositorio:

```bash
bash scripts/0_install/install.sh
```

El script crea o actualiza el entorno Conda llamado `eeg-mne` con las dependencias necesarias. Después, activa el entorno con:

```bash
conda activate eeg-mne
```

Si el instalador indica que `conda` no está disponible, instala Miniconda o asegúrate de que Conda esté cargado en la terminal antes de ejecutar el script.

## Ejecución básica

El flujo principal se ejecuta por bloques:

```bash
python scripts/1_preprocesado/edf_to_epochs.py
python scripts/1_preprocesado/check/check_valid_subjects.py
python scripts/2_features/epochs_to_features.py
python scripts/3_train/train_within_subject.py
python scripts/3_train/train_all_subjects.py
python scripts/4_separability/analyze_separability.py
python scripts/5_explicability/explicability.py
```

Cada bloque guarda sus salidas en `data/processed/`, `data/features/`, `models/` o `results/`, según corresponda.

## Nota

Los scripts están pensados para reproducir el flujo completo del TFG con la estructura de carpetas incluida en este repositorio. Si se cambian rutas, nombres de carpetas o formato de los datos, puede ser necesario adaptar las constantes de configuración de cada script.

## Autoría

- Autor: Liam Saborido Sueiro
- Tutor: Álvaro García López
- Co-tutor: Julio Salvador Lora Millán
