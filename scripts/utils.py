"""
Funciones comunes para la tanda alpha_beta.

Esta prueba sigue la lectura neurofisiologica del tutor:
  - usar Alpha y Beta como bandas principales
  - dejar fuera Delta, Theta y Gamma
  - no eliminar canales temporales relevantes para lenguaje auditivo
  - resumir SHAP por clusters neurofuncionales
"""

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------- Configuracion
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data/features"
RESULTS_DIR = PROJECT_ROOT / "results" / "within_subject"
SHAP_DIR = PROJECT_ROOT / "results" / "explicability" / "within_subject"

CONDITIONS = ["JOY", "NEUTRO", "SAD"]
LABEL_MAP = {"JOY": 0, "NEUTRO": 1, "SAD": 2}
LABEL_INV = {label: condition for condition, label in LABEL_MAP.items()}

N_FOLDS = 5
RANDOM_STATE = 40

BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
SELECTED_BANDS = ["Alpha", "Beta"]
EXCLUDED_BANDS = [band for band in BANDS if band not in SELECTED_BANDS]
N_BANDS = len(BANDS)
BAND_IDX = {band: idx for idx, band in enumerate(BANDS)}

BAND_COLORS = {
    "Alpha": "#f1c40f",
    "Beta": "#2ecc71",
    "AlphaAsym": "#34495e",
    "Other": "#95a5a6",
}

ALPHA_ASYMMETRY_PAIRS = [
    ("Fp1", "Fp2"), ("AF7", "AF8"), ("AF3", "AF4"),
    ("F7", "F8"),   ("F5", "F6"),   ("F3", "F4"),   ("F1", "F2"),
    ("FC5", "FC6"), ("FC3", "FC4"), ("FC1", "FC2"),
    ("T7", "T8"),   ("C5", "C6"),   ("C3", "C4"),   ("C1", "C2"),
    ("TP7", "TP8"), ("CP5", "CP6"), ("CP3", "CP4"), ("CP1", "CP2"),
    ("P7", "P8"),   ("P5", "P6"),   ("P3", "P4"),   ("P1", "P2"),
    ("PO7", "PO8"), ("PO5", "PO6"), ("PO3", "PO4"), ("O1", "O2"),
]

# Clusters sugeridos por el tutor. Se usan para agregar SHAP, no para recortar canales.
EEG_CLUSTERS = {
    "temporal_left": ["T7", "TP7", "CP5", "CP3", "FT7", "C5", "P7", "P5"],
    "temporal_right": ["T8", "TP8", "CP6", "CP4", "FT8", "C6", "P8", "P6"],
    "frontal_inferior_left": ["F7", "FC5", "F5"],
    "frontal_bilateral": [
        "Fp1", "Fpz", "Fp2",
        "AF7", "AF3", "AF4", "AF8",
        "F3", "Fz", "F4", "F1", "F2",
        "FC3", "FCz", "FC4",
    ],
    "centro_parietal_left": ["C3", "CP3", "P3"],
    "parietal_midline": ["Cz", "CPz", "Pz"],
    "occipital": ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"],
}


# ---------------------------------------------------------------------- Carga
def load_dataset():
    """Carga las features base en log10, etiquetas, meta y canales EEG."""
    log_path = FEATURES_DIR / "features_X.npy"
    if not log_path.exists():
        raise FileNotFoundError(
            f"No se encontro {log_path}. Ejecuta primero epochs_to_features.py"
        )

    X_log = np.load(FEATURES_DIR / "features_X.npy")
    y = np.load(FEATURES_DIR / "features_y.npy")
    meta = pd.read_csv(FEATURES_DIR / "features_meta.csv")

    with open(FEATURES_DIR / "features_info.json") as f:
        info = json.load(f)

    ch_names = info.get("ch_names_eeg", [])
    if not ch_names:
        raise ValueError(
            "features_info.json no contiene 'ch_names_eeg'. "
            "Vuelve a ejecutar epochs_to_features.py"
        )

    return X_log, y, meta, ch_names


# ---------------------------------------------------------------------- Normalizacion within-fold
def build_feature_matrix(delta_2d, ch_names):
    """Construye bandpower Alpha/Beta y asimetria Alpha."""
    n_ch = len(ch_names)
    delta_3d = delta_2d.reshape(len(delta_2d), n_ch, N_BANDS)

    blocks = []
    alpha_idx = BAND_IDX["Alpha"]
    beta_idx = BAND_IDX["Beta"]

    bandpower = np.stack(
        [delta_3d[:, :, alpha_idx], delta_3d[:, :, beta_idx]],
        axis=2,
    )
    blocks.append(bandpower.reshape(len(delta_2d), -1))

    ch_idx_map = {ch: i for i, ch in enumerate(ch_names)}
    asym_cols = []
    for left, right in ALPHA_ASYMMETRY_PAIRS:
        if left in ch_idx_map and right in ch_idx_map:
            asym_cols.append(
                delta_3d[:, ch_idx_map[right], alpha_idx]
                - delta_3d[:, ch_idx_map[left], alpha_idx]
            )
    if asym_cols:
        blocks.append(np.column_stack(asym_cols))

    return np.concatenate(blocks, axis=1).astype(np.float32)


def fit_normalization_pipeline(X_log_train, y_train, ch_names):
    """Ajusta baseline Neutro + scaler usando solo train."""
    n_ch = len(ch_names)
    X_train = X_log_train.reshape(-1, n_ch, N_BANDS)

    neutro_mask = (y_train == LABEL_MAP["NEUTRO"])
    if neutro_mask.sum() == 0:
        baseline = X_train.mean(axis=0)
    else:
        baseline = X_train[neutro_mask].mean(axis=0)

    delta_train = X_train - baseline[np.newaxis, :, :]
    delta_train_2d = delta_train.reshape(len(delta_train), -1)

    scaler = StandardScaler()
    delta_train_2d = scaler.fit_transform(delta_train_2d)

    preprocessing = {
        "baseline": baseline,
        "scaler": scaler,
    }

    return build_feature_matrix(delta_train_2d, ch_names), preprocessing


def transform_normalization_pipeline(X_log, ch_names, preprocessing):
    """Aplica baseline + scaler ya ajustados."""
    n_ch = len(ch_names)
    X = X_log.reshape(-1, n_ch, N_BANDS)
    delta = X - preprocessing["baseline"][np.newaxis, :, :]
    delta_2d = delta.reshape(len(delta), -1)
    delta_2d = preprocessing["scaler"].transform(delta_2d)
    return build_feature_matrix(delta_2d, ch_names)


def apply_normalization_pipeline(X_log_train, y_train, X_log_test, ch_names):
    """Normaliza train/test sin usar test para ajustar parametros."""
    X_train_feat, preprocessing = fit_normalization_pipeline(X_log_train, y_train, ch_names)
    X_test_feat = transform_normalization_pipeline(X_log_test, ch_names, preprocessing)
    return X_train_feat, X_test_feat


# ---------------------------------------------------------------------- Features
def build_feature_names(ch_names):
    """Nombres en el mismo orden que build_feature_matrix."""
    feature_names = [f"{ch}_{band}" for ch in ch_names for band in SELECTED_BANDS]

    ch_idx_map = {ch: i for i, ch in enumerate(ch_names)}
    feature_names += [
        f"AlphaAsym_{right}-{left}"
        for left, right in ALPHA_ASYMMETRY_PAIRS
        if left in ch_idx_map and right in ch_idx_map
    ]

    return feature_names


def describe_feature(feature_name):
    """Devuelve tipo, canal, banda y par de una feature."""
    if feature_name.startswith("AlphaAsym_"):
        pair = feature_name.replace("AlphaAsym_", "", 1)
        return "AlphaAsym", "", "", pair

    if "_" in feature_name:
        channel, band = feature_name.rsplit("_", 1)
        if band in SELECTED_BANDS:
            return "bandpower", channel, band, ""

    return "Other", "", "", ""


def feature_block_counts(feature_names):
    """Cuenta features por bloque."""
    counts = {
        "bandpower": 0,
        "alpha_asymmetry": 0,
        "other": 0,
    }

    for feature_name in feature_names:
        feature_type, _, _, _ = describe_feature(feature_name)
        if feature_type == "bandpower":
            counts["bandpower"] += 1
        elif feature_type == "AlphaAsym":
            counts["alpha_asymmetry"] += 1
        else:
            counts["other"] += 1

    return counts


def feature_color(feature_name):
    """Color para graficos de importancia."""
    feature_type, _, band, _ = describe_feature(feature_name)
    if feature_type == "AlphaAsym":
        return BAND_COLORS["AlphaAsym"]
    if feature_type == "bandpower":
        return BAND_COLORS.get(band, BAND_COLORS["Other"])
    return BAND_COLORS["Other"]


def channel_cluster(channel):
    """Devuelve el primer cluster que contiene un canal."""
    for cluster_name, channels in EEG_CLUSTERS.items():
        if channel in channels:
            return cluster_name
    return "other"


def available_clusters(ch_names):
    """Lista canales disponibles por cluster para documentar la tanda."""
    available = {}
    ch_set = set(ch_names)
    for cluster_name, channels in EEG_CLUSTERS.items():
        available[cluster_name] = [ch for ch in channels if ch in ch_set]
    return available
