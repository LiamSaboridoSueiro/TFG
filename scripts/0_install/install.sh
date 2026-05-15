#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="eeg-mne"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda no esta instalado o no esta en PATH."
  echo "Instala Miniconda/Anaconda y vuelve a ejecutar este script."
  exit 1
fi

# Evita que un entorno ROS u otro Python externo contamine pip/python dentro de conda.
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION

TMP_ENV_FILE="$(mktemp)"
trap 'rm -f "$TMP_ENV_FILE"' EXIT

cat > "$TMP_ENV_FILE" <<'YAML'
name: eeg-mne
channels:
  - defaults
  - conda-forge
dependencies:
  - alembic=1.18.4
  - cloudpickle=3.1.2
  - colorlog=6.10.1
  - contourpy=1.3.3
  - cuda-cudart=12.9.79
  - cuda-version=12.9
  - eeglabio=0.1.3
  - joblib=1.5.3
  - matplotlib-base=3.10.8
  - numba=0.63.1
  - numpy=2.3.5
  - optuna=4.7.0
  - pandas=3.0.0
  - pillow=12.0.0
  - pip=25.3
  - py-xgboost=3.1.3
  - pyqt=6.9.1
  - python=3.11.14
  - pyyaml=6.0.3
  - scikit-learn=1.8.0
  - seaborn=0.13.2
  - shap=0.50.0
  - statsmodels=0.14.6
  - xgboost=3.1.3
  - pip:
      - asttokens==3.0.1
      - certifi==2025.11.12
      - charset-normalizer==3.4.4
      - comm==0.2.3
      - debugpy==1.8.20
      - decorator==5.2.1
      - executing==2.2.1
      - fonttools==4.60.1
      - idna==3.11
      - ipykernel==7.2.0
      - ipython==9.10.0
      - ipython-pygments-lexers==1.1.1
      - jedi==0.19.2
      - jinja2==3.1.6
      - jupyter-client==8.8.0
      - jupyter-core==5.9.1
      - lazy-loader==0.4
      - matplotlib==3.10.7
      - matplotlib-inline==0.2.1
      - mne==1.11.0
      - mne-features==0.3.2
      - nest-asyncio==1.6.0
      - parso==0.8.6
      - pexpect==4.9.0
      - platformdirs==4.5.0
      - pooch==1.8.2
      - prompt-toolkit==3.0.52
      - psutil==7.2.2
      - ptyprocess==0.7.0
      - pure-eval==0.2.3
      - pygments==2.19.2
      - pyparsing==3.2.5
      - pyqt5==5.15.11
      - pyqt5-qt5==5.15.18
      - pyqt5-sip==12.17.1
      - pywavelets==1.9.0
      - pyzmq==27.1.0
      - requests==2.32.5
      - scipy==1.16.3
      - stack-data==0.6.3
      - tornado==6.5.4
      - tqdm==4.67.1
      - traitlets==5.14.3
      - urllib3==2.5.0
      - wcwidth==0.6.0
YAML

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Actualizando entorno conda: $ENV_NAME"
  conda env update -n "$ENV_NAME" -f "$TMP_ENV_FILE" --prune
else
  echo "Creando entorno conda: $ENV_NAME"
  conda env create -f "$TMP_ENV_FILE"
fi

echo
echo "Verificando versiones principales..."
conda run -n "$ENV_NAME" env MNE_DONTWRITE_HOME=true NUMBA_CACHE_DIR=/tmp/numba_cache python - <<'PY'
import importlib.metadata as md
import sys

packages = [
    "mne",
    "mne-features",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "joblib",
    "shap",
    "xgboost",
    "optuna",
    "statsmodels",
    "numba",
]

print(f"python=={sys.version.split()[0]}")
for package in packages:
    print(f"{package}=={md.version(package)}")
PY

echo
echo "Listo. Activa el entorno con:"
echo "  conda activate $ENV_NAME"
