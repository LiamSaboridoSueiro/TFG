# Entorno de instalacion: `eeg-mne`

Este proyecto se ha preparado con **Miniconda/Conda**. El entorno principal se llama `eeg-mne` y esta pensado para procesado de EEG con MNE, extraccion de caracteristicas, entrenamiento de modelos y explicabilidad SHAP.

## Sistema analizado

- Fecha de captura: 2026-05-16
- Sistema operativo: Ubuntu 24.04.2 LTS (`noble`)
- Kernel: Linux 6.17.0-23-generic x86_64
- Arquitectura: x86_64
- Conda instalado en: `/home/liam/miniconda3`
- Version de conda: 25.9.1
- Solver de conda: `libmamba`
- Entorno base activo durante la inspeccion: `base`
- Python de `base`: 3.13.9
- Entorno del proyecto: `/home/liam/miniconda3/envs/eeg-mne`
- Python del entorno `eeg-mne`: 3.11.14
- Canales usados por el entorno exportado: `defaults`, `conda-forge`

Entornos conda encontrados en este ordenador:

```text
base          /home/liam/miniconda3
eeg-mne       /home/liam/miniconda3/envs/eeg-mne
mujoco_env    /home/liam/miniconda3/envs/mujoco_env
sim2sim_envir /home/liam/miniconda3/envs/sim2sim_envir
```

## Instalacion rapida en otro ordenador

Desde la raiz del repositorio:

```bash
bash scripts/0_install/install.sh
conda activate eeg-mne
```

El script crea el entorno `eeg-mne` si no existe. Si ya existe, lo actualiza con las mismas versiones declaradas.

## Paquetes principales

Estas son las versiones importantes para reproducir el TFG:

```text
python          3.11.14
mne             1.11.0
mne-features    0.3.2
eeglabio        0.1.3
numpy           2.3.5
pandas          3.0.0
scipy           1.16.3
scikit-learn    1.8.0
matplotlib      3.10.7 / matplotlib-base 3.10.8
seaborn         0.13.2
joblib          1.5.3
shap            0.50.0
xgboost         3.1.3
optuna          4.7.0
statsmodels     0.14.6
numba           0.63.1
llvmlite        0.46.0
pywavelets      1.9.0
ipykernel       7.2.0
ipython         9.10.0
pyqt            6.9.1
pyqt5           5.15.11
```

## Aviso sobre contaminacion de `PYTHONPATH`

Durante la inspeccion, `conda run -n eeg-mne pip freeze` mostro paquetes de ROS 2 del sistema, por ejemplo `rclpy`, `sensor-msgs`, `ros2cli`, etc. Eso normalmente significa que el shell tiene variables como `PYTHONPATH`, `AMENT_PREFIX_PATH` o `COLCON_PREFIX_PATH` cargadas desde ROS.

Para reproducir el entorno de forma limpia en el portatil, instala primero con `install.sh` y evita activar ROS antes de trabajar con este proyecto. Si en una terminal aparecen paquetes ROS dentro de `pip freeze`, abre una terminal nueva o limpia variables con:

```bash
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
```

## Exportacion conda usada como referencia

Esta exportacion se obtuvo con:

```bash
conda env export -n eeg-mne --no-builds
```

```yaml
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
```

Para una reproduccion aun mas estricta en la misma plataforma Linux x86_64, se puede generar un fichero explicito con:

```bash
conda list -n eeg-mne --explicit > scripts/0_install/conda-linux-64-explicit.txt
```

Ese fichero bloquea URLs exactas de paquetes, pero es menos portable que el `install.sh`.
