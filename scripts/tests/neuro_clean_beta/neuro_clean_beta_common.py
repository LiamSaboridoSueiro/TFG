"""
Funciones comunes para la tanda neuro_clean_beta.

Esta prueba mantiene Beta, elimina Gamma y excluye canales periféricos
potencialmente contaminados. Se usa tanto en entrenamiento como en SHAP.
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

# ---------------------------------------------------------------------- Configuración
PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEST_DIR = Path(__file__).resolve().parent
FEATURES_DIR = PROJECT_ROOT / "data/features"
OUTPUT_DIR = TEST_DIR / "results"
SHAP_DIR = OUTPUT_DIR / "shap_topomap"

CONDITIONS = ["JOY", "NEUTRO", "SAD"]
LABEL_MAP = {"JOY": 0, "NEUTRO": 1, "SAD": 2}
LABEL_INV = {label: condition for condition, label in LABEL_MAP.items()}

N_FOLDS = 5
RANDOM_STATE = 42

BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
N_BANDS = len(BANDS)
BAND_IDX = {band: idx for idx, band in enumerate(BANDS)}

EXCLUDED_BANDS = {"Gamma"}
PLOT_BANDS = [band for band in BANDS if band not in EXCLUDED_BANDS]

ALPHA_ASYMMETRY_PAIRS = [
    ("Fp1", "Fp2"), ("AF7", "AF8"), ("AF3", "AF4"),
    ("F7", "F8"),   ("F5", "F6"),   ("F3", "F4"),   ("F1", "F2"),
    ("FC5", "FC6"), ("FC3", "FC4"), ("FC1", "FC2"),
    ("T7", "T8"),   ("C5", "C6"),   ("C3", "C4"),   ("C1", "C2"),
    ("TP7", "TP8"), ("CP5", "CP6"), ("CP3", "CP4"), ("CP1", "CP2"),
    ("P7", "P8"),   ("P5", "P6"),   ("P3", "P4"),   ("P1", "P2"),
    ("PO7", "PO8"), ("PO5", "PO6"), ("PO3", "PO4"), ("O1", "O2"),
]

PERIPHERAL_CHANNELS = {
    "Fp1", "Fp2", "Fpz",
    "AF7", "AF8", "AF3", "AF4",
    "F7", "F8", "F5", "F6",
    "FT7", "FT8",
    "T7", "T8",
    "TP7", "TP8",
    "FC5", "FC6",
    "PO7", "PO8", "PO5", "PO6",
    "O1", "O2", "Oz",
}

BAND_COLORS = {
    "Delta": "#e74c3c",
    "Theta": "#e67e22",
    "Alpha": "#f1c40f",
    "Beta": "#2ecc71",
    "Gamma": "#3498db",
    "ThetaAlphaRatio": "#8e44ad",
    "AlphaAsym": "#34495e",
    "Other": "#95a5a6",
}


# ---------------------------------------------------------------------- Carga de datos
def load_dataset():
    """Carga X_log, y, meta y canales EEG."""
    log_path = FEATURES_DIR / "features_X.npy"
    if not log_path.exists():
        raise FileNotFoundError(
            f"No se encontró {log_path}. Ejecuta primero epochs_to_features.py"
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


# ---------------------------------------------------------------------- Normalización dentro del fold
def build_feature_matrix(delta_2d, ch_names):
    """Añade Theta/Alpha y asimetría Alpha al bloque bandpower."""
    n_ch = len(ch_names)
    delta_3d = delta_2d.reshape(len(delta_2d), n_ch, N_BANDS)

    blocks = [delta_2d]

    theta_idx = BAND_IDX["Theta"]
    alpha_idx = BAND_IDX["Alpha"]
    ratio = delta_3d[:, :, theta_idx] - delta_3d[:, :, alpha_idx]
    blocks.append(ratio)

    ch_idx_map = {ch: i for i, ch in enumerate(ch_names)}
    valid_pairs = []
    for left, right in ALPHA_ASYMMETRY_PAIRS:
        if left in ch_idx_map and right in ch_idx_map:
            valid_pairs.append((left, right, ch_idx_map[left], ch_idx_map[right]))

    if valid_pairs:
        asym_cols = []
        for _, _, left_idx, right_idx in valid_pairs:
            asym_cols.append(delta_3d[:, right_idx, alpha_idx] - delta_3d[:, left_idx, alpha_idx])
        blocks.append(np.column_stack(asym_cols))

    return np.concatenate(blocks, axis=1).astype(np.float32)


def fit_normalization_pipeline(X_log_train, y_train, ch_names):
    """Ajusta baseline + scaler usando solo train."""
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
    """Aplica una normalización ya ajustada."""
    n_ch = len(ch_names)
    X = X_log.reshape(-1, n_ch, N_BANDS)
    delta = X - preprocessing["baseline"][np.newaxis, :, :]
    delta_2d = delta.reshape(len(delta), -1)
    delta_2d = preprocessing["scaler"].transform(delta_2d)
    return build_feature_matrix(delta_2d, ch_names)


def apply_normalization_pipeline(X_log_train, y_train, X_log_test, ch_names):
    """Aplica el pipeline completo sin usar test para ajustar nada."""
    X_train_feat, preprocessing = fit_normalization_pipeline(X_log_train, y_train, ch_names)
    X_test_feat = transform_normalization_pipeline(X_log_test, ch_names, preprocessing)
    return X_train_feat, X_test_feat


# ---------------------------------------------------------------------- Features y selección
def build_feature_names(ch_names):
    """Construye nombres de features en el mismo orden que build_feature_matrix."""
    feature_names = [f"{ch}_{band}" for ch in ch_names for band in BANDS]
    feature_names += [f"ThetaAlphaRatio_{ch}" for ch in ch_names]

    ch_idx_map = {ch: i for i, ch in enumerate(ch_names)}
    feature_names += [
        f"AlphaAsym_{right}-{left}"
        for left, right in ALPHA_ASYMMETRY_PAIRS
        if left in ch_idx_map and right in ch_idx_map
    ]

    return feature_names


def describe_feature(feature_name):
    """Devuelve tipo, canal, banda y par de una feature."""
    if feature_name.startswith("ThetaAlphaRatio_"):
        channel = feature_name.replace("ThetaAlphaRatio_", "", 1)
        return "ThetaAlphaRatio", channel, "", ""

    if feature_name.startswith("AlphaAsym_"):
        pair = feature_name.replace("AlphaAsym_", "", 1)
        return "AlphaAsym", "", "", pair

    if "_" in feature_name:
        channel, band = feature_name.rsplit("_", 1)
        if band in BAND_IDX:
            return "bandpower", channel, band, ""

    return "Other", "", "", ""


def pair_has_excluded_channel(pair, excluded_channels):
    """Comprueba si una asimetría incluye un canal excluido."""
    if not pair:
        return False
    parts = pair.split("-")
    return any(ch in excluded_channels for ch in parts)


def should_keep_feature(feature_name):
    """Aplica el filtro neuro_clean_beta a una feature."""
    feature_type, channel, band, pair = describe_feature(feature_name)

    if feature_type == "bandpower":
        if band in EXCLUDED_BANDS:
            return False
        if channel in PERIPHERAL_CHANNELS:
            return False
        return True

    if feature_type == "ThetaAlphaRatio":
        return channel not in PERIPHERAL_CHANNELS

    if feature_type == "AlphaAsym":
        return not pair_has_excluded_channel(pair, PERIPHERAL_CHANNELS)

    return True


def build_selected_feature_indices(feature_names):
    """Devuelve índices y nombres de las features que sobreviven al filtro."""
    indices = []
    selected_names = []

    for idx, feature_name in enumerate(feature_names):
        if should_keep_feature(feature_name):
            indices.append(idx)
            selected_names.append(feature_name)

    return np.array(indices, dtype=int), selected_names


def feature_block_counts(feature_names):
    """Cuenta features por tipo para documentar la prueba."""
    counts = {
        "bandpower": 0,
        "theta_alpha_ratio": 0,
        "alpha_asymmetry": 0,
        "other": 0,
    }

    for feature_name in feature_names:
        feature_type, _, _, _ = describe_feature(feature_name)
        if feature_type == "bandpower":
            counts["bandpower"] += 1
        elif feature_type == "ThetaAlphaRatio":
            counts["theta_alpha_ratio"] += 1
        elif feature_type == "AlphaAsym":
            counts["alpha_asymmetry"] += 1
        else:
            counts["other"] += 1

    return counts


def feature_color(feature_name):
    """Color asociado a cada tipo/banda de feature."""
    feature_type, _, band, _ = describe_feature(feature_name)
    if feature_type == "ThetaAlphaRatio":
        return BAND_COLORS["ThetaAlphaRatio"]
    if feature_type == "AlphaAsym":
        return BAND_COLORS["AlphaAsym"]
    if feature_type == "bandpower":
        return BAND_COLORS.get(band, BAND_COLORS["Other"])
    return BAND_COLORS["Other"]
