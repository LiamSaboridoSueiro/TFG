"""
Script de explicabilidad within-subject:

    Modelos finales por sujeto -> SHAP lineal -> agregación por feature/canal/banda -> topomaps

    El análisis está pensado para los modelos LogReg guardados por train_within_subject.py.
    Para un modelo lineal, la contribución tipo SHAP se calcula exactamente como:

        shap_j = coef_j * (x_j - media_referencia_j)

    Es decir, cuánto empuja cada feature el logit de una clase respecto al perfil medio
    del propio sujeto. Esto evita depender del paquete externo shap y mantiene la
    interpretación alineada con la caja blanca del modelo.

Salida:
  results/shap_topomap/within_subject/shap_feature_importance_by_subject.csv
  results/shap_topomap/within_subject/shap_feature_importance.csv
  results/shap_topomap/within_subject/shap_channel_band_importance.csv
  results/shap_topomap/within_subject/shap_band_importance.csv
  results/shap_topomap/within_subject/shap_artifact_audit.csv
  results/shap_topomap/within_subject/explicability_summary.json
  results/shap_topomap/within_subject/01_top_features_global.png
  results/shap_topomap/within_subject/02_band_importance_by_class.png
  results/shap_topomap/within_subject/03_topomap_abs_<COND>.png
  results/shap_topomap/within_subject/04_topomap_signed_<COND>.png
  results/shap_topomap/within_subject/05_artifact_audit.png

"""

import os
import json
import warnings
from pathlib import Path

# MNE y Matplotlib pueden intentar escribir configuración/caché en $HOME.
# En algunos entornos del proyecto $HOME no es escribible, así que usamos /tmp.
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# ---------------------------------------------------------------------- Configuración
FEATURES_DIR = Path("data/features")
MODELS_DIR   = Path("models/within_subject")
RESULTS_DIR  = Path("results/shap_topomap/within_subject")

CONDITIONS = ["JOY", "NEUTRO", "SAD"]
LABEL_MAP  = {"JOY": 0, "NEUTRO": 1, "SAD": 2}
LABEL_INV  = {label: condition for condition, label in LABEL_MAP.items()}

BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
N_BANDS = len(BANDS)
BAND_IDX = {band: idx for idx, band in enumerate(BANDS)}

SFREQ = 250
N_TOP_FEATURES = 25
N_TOP_FEATURES_JSON = 12

FEATURE_COL_WIDTH = 32
METRIC_COL_WIDTH  = 12

BAND_COLORS = {
    "Delta": "#e74c3c",
    "Theta": "#e67e22",
    "Alpha": "#f1c40f",
    "Beta":  "#2ecc71",
    "Gamma": "#3498db",
    "ThetaAlphaRatio": "#8e44ad",
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

# Canales donde Beta/Gamma puede estar más mezclado con parpadeo, frente, mandíbula
# o actividad muscular periférica. No se excluyen: se auditan.
ARTIFACT_SUSPECT_CHANNELS = {
    "Fp1", "Fp2", "Fpz", "AF7", "AF8", "AF3", "AF4",
    "F7", "F8", "F5", "F6", "FT7", "FT8",
    "T7", "T8", "TP7", "TP8",
    "FC5", "FC6", "PO7", "PO8", "PO5", "PO6",
    "O1", "O2", "Oz",
}


# ---------------------------------------------------------------------- Carga de datos
def load_dataset():
    """Carga las features base en log10 y sus metadatos."""
    log_path = FEATURES_DIR / "features_X.npy"

    if not log_path.exists():
        raise FileNotFoundError(
            f"No se encontró {log_path}. Ejecuta primero epochs_to_features.py"
        )

    X_log = np.load(FEATURES_DIR / "features_X.npy")
    y     = np.load(FEATURES_DIR / "features_y.npy")
    meta  = pd.read_csv(FEATURES_DIR / "features_meta.csv")

    with open(FEATURES_DIR / "features_info.json") as f:
        info = json.load(f)

    ch_names = info.get("ch_names_eeg", [])
    if not ch_names:
        raise ValueError(
            "features_info.json no contiene 'ch_names_eeg'. "
            "Vuelve a ejecutar epochs_to_features.py"
        )

    return X_log, y, meta, ch_names


def load_subject_models():
    """Carga los modelos finales within-subject guardados por sujeto."""
    model_paths = sorted(MODELS_DIR.glob("*.joblib"))
    if not model_paths:
        raise FileNotFoundError(
            f"No se encontraron modelos en {MODELS_DIR}. "
            "Ejecuta primero train_within_subject.py"
        )

    models = {}
    for model_path in model_paths:
        model_data = joblib.load(model_path)
        subject_id = model_data.get("subject_id", model_path.stem)
        models[subject_id] = model_data

    return models


# ---------------------------------------------------------------------- Preprocesado guardado en cada modelo
def build_feature_matrix(delta_2d, ch_names):
    """Añade las features derivadas al bloque bandpower ya normalizado."""
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


def transform_saved_preprocessing(X_log, ch_names, preprocessing):
    """Aplica baseline + scaler guardados en el modelo de cada sujeto."""
    n_ch = len(ch_names)
    X = X_log.reshape(-1, n_ch, N_BANDS)

    delta = X - preprocessing["baseline"][np.newaxis, :, :]
    delta_2d = delta.reshape(len(delta), -1)
    delta_2d = preprocessing["scaler"].transform(delta_2d)

    return build_feature_matrix(delta_2d, ch_names)


def get_feature_names(model_data, ch_names):
    """Obtiene los nombres de features del modelo y valida su longitud básica."""
    feature_names = model_data.get("feature_names", [])
    if feature_names:
        return list(feature_names)

    names = [f"{ch}_{band}" for ch in ch_names for band in BANDS]
    names += [f"ThetaAlphaRatio_{ch}" for ch in ch_names]

    ch_idx_map = {ch: i for i, ch in enumerate(ch_names)}
    names += [
        f"AlphaAsym_{right}-{left}"
        for left, right in ALPHA_ASYMMETRY_PAIRS
        if left in ch_idx_map and right in ch_idx_map
    ]
    return names


# ---------------------------------------------------------------------- SHAP lineal
def compute_linear_shap_values(classifier, X_features):
    """
    Calcula contribuciones SHAP exactas para un modelo lineal.

    Para cada clase:
        logit_clase(x) = intercept + sum(coef_j * x_j)
        shap_j         = coef_j * (x_j - media_j)
    """
    if not hasattr(classifier, "coef_"):
        return None, None, None

    coef = np.asarray(classifier.coef_, dtype=float)
    intercept = np.asarray(classifier.intercept_, dtype=float)
    classes = np.asarray(getattr(classifier, "classes_", np.arange(coef.shape[0])))

    if coef.ndim != 2 or coef.shape[0] != len(classes):
        return None, None, None

    reference = X_features.mean(axis=0)
    centered = X_features - reference[np.newaxis, :]

    shap_values = centered[:, np.newaxis, :] * coef[np.newaxis, :, :]
    base_values = intercept + reference @ coef.T

    return shap_values, base_values, classes


def evaluate_model(classifier, X_features, y_subject):
    """Evalúa el modelo final sobre las épocas del propio sujeto."""
    y_pred = classifier.predict(X_features)
    acc = accuracy_score(y_subject, y_pred)
    f1 = f1_score(y_subject, y_pred, average="macro", zero_division=0)
    return float(acc), float(f1), y_pred


# ---------------------------------------------------------------------- Descripción de features
def describe_feature(feature_name):
    """Devuelve tipo, canal, banda y par hemisférico de una feature."""
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


def feature_color(feature_name):
    """Color asociado a cada tipo de feature para las figuras."""
    feature_type, _, band, _ = describe_feature(feature_name)
    if feature_type == "ThetaAlphaRatio":
        return BAND_COLORS["ThetaAlphaRatio"]
    if feature_type == "AlphaAsym":
        return BAND_COLORS["AlphaAsym"]
    if feature_type == "bandpower":
        return BAND_COLORS.get(band, BAND_COLORS["Other"])
    return BAND_COLORS["Other"]


# ---------------------------------------------------------------------- Agregación de explicabilidad
def build_subject_feature_rows(subject_id, y_subject, feature_names, shap_values, classes):
    """Construye una tabla larga de importancia por sujeto, clase y feature."""
    rows = []

    for class_pos, class_label in enumerate(classes):
        if int(class_label) not in LABEL_INV:
            continue

        condition = LABEL_INV[int(class_label)]
        mask = (y_subject == int(class_label))
        if mask.sum() == 0:
            continue

        class_shap = shap_values[mask, class_pos, :]
        mean_abs = np.abs(class_shap).mean(axis=0)
        mean_signed = class_shap.mean(axis=0)

        for feature_idx, feature_name in enumerate(feature_names):
            feature_type, channel, band, pair = describe_feature(feature_name)
            rows.append({
                "subject_id": subject_id,
                "class_label": int(class_label),
                "condition": condition,
                "feature_idx": int(feature_idx),
                "feature_name": feature_name,
                "feature_type": feature_type,
                "channel": channel,
                "band": band,
                "pair": pair,
                "mean_abs_shap": float(mean_abs[feature_idx]),
                "mean_signed_shap": float(mean_signed[feature_idx]),
                "n_epochs_class": int(mask.sum()),
            })

    return rows


def aggregate_feature_importance(feature_subject_df):
    """Promedia importancia por feature dando el mismo peso a cada sujeto."""
    group_cols = [
        "class_label", "condition", "feature_idx", "feature_name",
        "feature_type", "channel", "band", "pair",
    ]

    summary = (
        feature_subject_df
        .groupby(group_cols, dropna=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            std_abs_shap=("mean_abs_shap", "std"),
            mean_signed_shap=("mean_signed_shap", "mean"),
            std_signed_shap=("mean_signed_shap", "std"),
            n_subjects=("subject_id", "nunique"),
        )
        .reset_index()
    )

    summary["std_abs_shap"] = summary["std_abs_shap"].fillna(0.0)
    summary["std_signed_shap"] = summary["std_signed_shap"].fillna(0.0)
    return summary


def build_channel_band_importance(feature_summary_df, ch_names):
    """Extrae solo el bloque canal x banda para topomaps."""
    rows = []
    bandpower_df = feature_summary_df[feature_summary_df["feature_type"] == "bandpower"]

    for _, row in bandpower_df.iterrows():
        if row["channel"] not in ch_names or row["band"] not in BANDS:
            continue
        rows.append({
            "class_label": int(row["class_label"]),
            "condition": row["condition"],
            "channel": row["channel"],
            "band": row["band"],
            "mean_abs_shap": float(row["mean_abs_shap"]),
            "mean_signed_shap": float(row["mean_signed_shap"]),
            "n_subjects": int(row["n_subjects"]),
        })

    return pd.DataFrame(rows)


def build_topomap_values(channel_band_df, ch_names):
    """Convierte la tabla canal/banda a matrices por clase."""
    ch_idx = {ch: i for i, ch in enumerate(ch_names)}
    maps = {}

    for condition in CONDITIONS:
        maps[condition] = {
            "abs": np.zeros((len(ch_names), N_BANDS), dtype=float),
            "signed": np.zeros((len(ch_names), N_BANDS), dtype=float),
        }

    for _, row in channel_band_df.iterrows():
        condition = row["condition"]
        channel = row["channel"]
        band = row["band"]

        if condition not in maps or channel not in ch_idx or band not in BAND_IDX:
            continue

        i_ch = ch_idx[channel]
        i_band = BAND_IDX[band]
        maps[condition]["abs"][i_ch, i_band] = row["mean_abs_shap"]
        maps[condition]["signed"][i_ch, i_band] = row["mean_signed_shap"]

    return maps


def compute_band_importance(channel_band_df):
    """Calcula el peso relativo de cada banda por clase."""
    rows = []

    for condition in CONDITIONS:
        cond_df = channel_band_df[channel_band_df["condition"] == condition]
        total = cond_df["mean_abs_shap"].sum()
        if total <= 0:
            total = 1e-12

        for band in BANDS:
            value = cond_df[cond_df["band"] == band]["mean_abs_shap"].sum()
            rows.append({
                "condition": condition,
                "band": band,
                "mean_abs_shap": float(value),
                "percentage": float(value / total * 100),
            })

    return pd.DataFrame(rows)


def compute_artifact_audit(feature_subject_df):
    """
    Resume cuánto peso cae en Beta/Gamma y en canales periféricos.

    Este bloque no decide si hay artefacto: solo cuantifica el riesgo para poder
    discutirlo con criterio en la memoria.
    """
    rows = []
    bandpower_df = feature_subject_df[feature_subject_df["feature_type"] == "bandpower"]

    for (subject_id, condition), group in bandpower_df.groupby(["subject_id", "condition"]):
        total = group["mean_abs_shap"].sum()
        if total <= 0:
            total = 1e-12

        high_freq = group[group["band"].isin(["Beta", "Gamma"])]
        peripheral_high_freq = high_freq[high_freq["channel"].isin(ARTIFACT_SUSPECT_CHANNELS)]

        beta_gamma = high_freq["mean_abs_shap"].sum()
        peripheral_beta_gamma = peripheral_high_freq["mean_abs_shap"].sum()

        rows.append({
            "subject_id": subject_id,
            "condition": condition,
            "total_bandpower_abs": float(total),
            "beta_gamma_abs": float(beta_gamma),
            "peripheral_beta_gamma_abs": float(peripheral_beta_gamma),
            "beta_gamma_percentage": float(beta_gamma / total * 100),
            "peripheral_beta_gamma_percentage": float(peripheral_beta_gamma / total * 100),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- Plots
def plot_top_features(feature_summary_df):
    """Dibuja las features más importantes promediando clases y sujetos."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    global_importance = (
        feature_summary_df
        .groupby(["feature_name", "feature_type", "channel", "band", "pair"], dropna=False)
        .agg(mean_abs_shap=("mean_abs_shap", "mean"))
        .reset_index()
        .sort_values("mean_abs_shap", ascending=False)
        .head(N_TOP_FEATURES)
    )

    top_names = global_importance["feature_name"].tolist()
    top_vals = global_importance["mean_abs_shap"].values
    colors = [feature_color(name) for name in top_names]

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("SHAP medio absoluto (promedio clases y sujetos)")
    ax.set_title(f"Top {N_TOP_FEATURES} features — explicabilidad within-subject")
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_items = [
        ("Delta", BAND_COLORS["Delta"]),
        ("Theta", BAND_COLORS["Theta"]),
        ("Alpha", BAND_COLORS["Alpha"]),
        ("Beta", BAND_COLORS["Beta"]),
        ("Gamma", BAND_COLORS["Gamma"]),
        ("ThetaAlphaRatio", BAND_COLORS["ThetaAlphaRatio"]),
        ("AlphaAsym", BAND_COLORS["AlphaAsym"]),
    ]
    ax.legend(
        handles=[Patch(facecolor=color, label=name) for name, color in legend_items],
        fontsize=8,
        loc="lower right",
    )

    plt.tight_layout()
    output_path = RESULTS_DIR / "01_top_features_global.png"
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  {output_path.name}")


def plot_band_importance(band_importance_df):
    """Dibuja la importancia relativa de bandas por clase."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(CONDITIONS))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    for i, band in enumerate(BANDS):
        values = []
        for condition in CONDITIONS:
            row = band_importance_df[
                (band_importance_df["condition"] == condition) &
                (band_importance_df["band"] == band)
            ]
            values.append(float(row["percentage"].iloc[0]) if len(row) else 0.0)

        ax.bar(
            x + (i - 2) * width,
            values,
            width=width,
            label=band,
            color=BAND_COLORS[band],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, fontsize=11)
    ax.set_ylabel("% SHAP absoluto dentro de bandpower")
    ax.set_title("Importancia relativa por banda y clase")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = RESULTS_DIR / "02_band_importance_by_class.png"
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  {output_path.name}")


def create_topomap_info(ch_names):
    """Crea un objeto Info con montaje estándar para dibujar topomaps."""
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1005")
    info.set_montage(montage, match_case=False, on_missing="ignore")
    return info


def plot_topomap_compat(data, info, ax, cmap, vmin, vmax):
    """Compatibilidad entre versiones de MNE para plot_topomap."""
    try:
        im, _ = mne.viz.plot_topomap(
            data,
            info,
            axes=ax,
            show=False,
            cmap=cmap,
            contours=0,
            vlim=(vmin, vmax),
        )
    except TypeError:
        im, _ = mne.viz.plot_topomap(
            data,
            info,
            axes=ax,
            show=False,
            cmap=cmap,
            contours=0,
            vmin=vmin,
            vmax=vmax,
        )
    return im


def plot_topomap_grid(topomap_values, ch_names, mode):
    """Genera una figura de topomaps por clase y banda."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    info = create_topomap_info(ch_names)

    for condition in CONDITIONS:
        data = topomap_values[condition][mode]

        if mode == "abs":
            cmap = "Reds"
            vmin = 0.0
            vmax = float(np.nanmax(data))
            if vmax <= 0:
                vmax = 1.0
            title_prefix = "SHAP absoluto"
            output_name = f"03_topomap_abs_{condition}.png"
        else:
            cmap = "RdBu_r"
            max_abs = float(np.nanmax(np.abs(data)))
            if max_abs <= 0:
                max_abs = 1.0
            vmin = -max_abs
            vmax = max_abs
            title_prefix = "SHAP firmado"
            output_name = f"04_topomap_signed_{condition}.png"

        fig, axes = plt.subplots(1, len(BANDS), figsize=(16, 4))
        fig.patch.set_facecolor("#f8f9fa")
        fig.suptitle(
            f"{title_prefix} — {condition} — modelos within-subject",
            fontsize=13,
            fontweight="bold",
        )

        im = None
        for band_idx, band in enumerate(BANDS):
            ax = axes[band_idx]
            ax.set_facecolor("#ffffff")
            im = plot_topomap_compat(data[:, band_idx], info, ax, cmap, vmin, vmax)
            ax.set_title(band, fontsize=11)

        if im is not None:
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
            cbar.ax.tick_params(labelsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        output_path = RESULTS_DIR / output_name
        plt.savefig(output_path, dpi=120)
        plt.close(fig)
        print(f"  {output_path.name}")


def plot_artifact_audit(artifact_audit_df):
    """Dibuja el resumen de posible dependencia de Beta/Gamma periférico."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = (
        artifact_audit_df
        .groupby("condition")
        .agg(
            beta_gamma_percentage=("beta_gamma_percentage", "mean"),
            peripheral_beta_gamma_percentage=("peripheral_beta_gamma_percentage", "mean"),
        )
        .reindex(CONDITIONS)
        .reset_index()
    )

    x = np.arange(len(CONDITIONS))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    ax.bar(
        x - width / 2,
        summary["beta_gamma_percentage"],
        width=width,
        color="#3498db",
        alpha=0.85,
        label="Beta+Gamma",
    )
    ax.bar(
        x + width / 2,
        summary["peripheral_beta_gamma_percentage"],
        width=width,
        color="#e74c3c",
        alpha=0.85,
        label="Beta+Gamma periférico",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, fontsize=11)
    ax.set_ylabel("% SHAP absoluto dentro de bandpower")
    ax.set_title("Auditoría de posible contribución muscular")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = RESULTS_DIR / "05_artifact_audit.png"
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  {output_path.name}")


# ---------------------------------------------------------------------- Guardado
def save_tables(
    feature_subject_df,
    feature_summary_df,
    channel_band_df,
    band_importance_df,
    artifact_audit_df,
):
    """Guarda todas las tablas de explicabilidad."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "shap_feature_importance_by_subject.csv": feature_subject_df,
        "shap_feature_importance.csv": feature_summary_df,
        "shap_channel_band_importance.csv": channel_band_df,
        "shap_band_importance.csv": band_importance_df,
        "shap_artifact_audit.csv": artifact_audit_df,
    }

    for filename, df in paths.items():
        output_path = RESULTS_DIR / filename
        df.to_csv(output_path, index=False)
        print(f"  {output_path.name}")


def save_summary_json(
    subject_summary,
    feature_summary_df,
    band_importance_df,
    artifact_audit_df,
):
    """Guarda un resumen compacto para la memoria."""
    subject_df = pd.DataFrame(subject_summary)

    top_features = {}
    for condition in CONDITIONS:
        top_df = (
            feature_summary_df[feature_summary_df["condition"] == condition]
            .sort_values("mean_abs_shap", ascending=False)
            .head(N_TOP_FEATURES_JSON)
        )
        top_features[condition] = [
            {
                "feature_name": row["feature_name"],
                "feature_type": row["feature_type"],
                "channel": row["channel"],
                "band": row["band"],
                "pair": row["pair"],
                "mean_abs_shap": float(row["mean_abs_shap"]),
                "mean_signed_shap": float(row["mean_signed_shap"]),
            }
            for _, row in top_df.iterrows()
        ]

    band_percentages = {}
    for condition in CONDITIONS:
        cond_rows = band_importance_df[band_importance_df["condition"] == condition]
        band_percentages[condition] = {
            row["band"]: float(row["percentage"])
            for _, row in cond_rows.iterrows()
        }

    artifact_summary = {}
    artifact_grouped = artifact_audit_df.groupby("condition").agg(
        beta_gamma_percentage=("beta_gamma_percentage", "mean"),
        peripheral_beta_gamma_percentage=("peripheral_beta_gamma_percentage", "mean"),
    )
    for condition in CONDITIONS:
        if condition not in artifact_grouped.index:
            continue
        row = artifact_grouped.loc[condition]
        artifact_summary[condition] = {
            "beta_gamma_percentage": float(row["beta_gamma_percentage"]),
            "peripheral_beta_gamma_percentage": float(row["peripheral_beta_gamma_percentage"]),
        }

    summary = {
        "method": "linear_shap_for_logistic_regression",
        "explanation": "coef_j * (x_j - mean_subject_feature_j)",
        "n_subjects": int(subject_df["subject_id"].nunique()) if len(subject_df) else 0,
        "mean_subject_accuracy_final_models": (
            float(subject_df["accuracy"].mean()) if len(subject_df) else None
        ),
        "mean_subject_f1_final_models": (
            float(subject_df["f1_macro"].mean()) if len(subject_df) else None
        ),
        "top_features_by_class": top_features,
        "band_percentages": band_percentages,
        "artifact_audit": artifact_summary,
    }

    output_path = RESULTS_DIR / "explicability_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  {output_path.name}")


# ---------------------------------------------------------------------- Resumen de consola
def print_console_summary(subject_summary, feature_summary_df, band_importance_df, artifact_audit_df):
    """Imprime un resumen corto para revisar en terminal."""
    subject_df = pd.DataFrame(subject_summary)

    print("\n" + "=" * 75)
    print("RESUMEN EXPLICABILIDAD WITHIN-SUBJECT")
    print("=" * 75)

    if len(subject_df):
        print(
            f"\n  Sujetos explicados: {subject_df['subject_id'].nunique()}  "
            f"| Acc modelo final: {subject_df['accuracy'].mean():.3f}  "
            f"| F1 macro: {subject_df['f1_macro'].mean():.3f}"
        )

    print("\n  Top features por clase:")
    for condition in CONDITIONS:
        top_df = (
            feature_summary_df[feature_summary_df["condition"] == condition]
            .sort_values("mean_abs_shap", ascending=False)
            .head(5)
        )
        print(f"\n  {condition}")
        print(f"  {'Feature':<{FEATURE_COL_WIDTH}} {'SHAP abs':>{METRIC_COL_WIDTH}} {'SHAP sign':>{METRIC_COL_WIDTH}}")
        print("  " + "-" * 62)
        for _, row in top_df.iterrows():
            print(
                f"  {row['feature_name']:<{FEATURE_COL_WIDTH}} "
                f"{row['mean_abs_shap']:>{METRIC_COL_WIDTH}.5f} "
                f"{row['mean_signed_shap']:>{METRIC_COL_WIDTH}.5f}"
            )

    print("\n  Importancia por banda (% dentro de bandpower):")
    print(f"  {'Clase':<10} " + " ".join(f"{band:>9}" for band in BANDS))
    print("  " + "-" * 60)
    for condition in CONDITIONS:
        values = []
        for band in BANDS:
            row = band_importance_df[
                (band_importance_df["condition"] == condition) &
                (band_importance_df["band"] == band)
            ]
            values.append(float(row["percentage"].iloc[0]) if len(row) else 0.0)
        print(f"  {condition:<10} " + " ".join(f"{v:>8.1f}%" for v in values))

    artifact_summary = (
        artifact_audit_df
        .groupby("condition")
        .agg(
            beta_gamma_percentage=("beta_gamma_percentage", "mean"),
            peripheral_beta_gamma_percentage=("peripheral_beta_gamma_percentage", "mean"),
        )
        .reindex(CONDITIONS)
    )

    print("\n  Auditoría Beta/Gamma:")
    print(f"  {'Clase':<10} {'Beta+Gamma':>14} {'B/G periférico':>16}")
    print("  " + "-" * 46)
    for condition, row in artifact_summary.iterrows():
        print(
            f"  {condition:<10} "
            f"{row['beta_gamma_percentage']:>13.1f}% "
            f"{row['peripheral_beta_gamma_percentage']:>15.1f}%"
        )


# ---------------------------------------------------------------------- MAIN
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("EXPLICABILIDAD WITHIN-SUBJECT CON SHAP LINEAL!!!!!!!!!!")
    print(f"  Features: {FEATURES_DIR}")
    print(f"  Modelos:  {MODELS_DIR}")
    print(f"  Salida:   {RESULTS_DIR}")
    print("  Método:   SHAP lineal exacto para LogisticRegression")

    print("\n  Cargando datos...")
    X_log, y, meta, ch_names = load_dataset()

    if X_log.shape[1] != len(ch_names) * N_BANDS:
        raise ValueError(
            f"Shape inconsistente: X_log.shape[1]={X_log.shape[1]} "
            f"pero se esperaba {len(ch_names)} canales x {N_BANDS} bandas = {len(ch_names) * N_BANDS}."
        )

    print(f"  Épocas:  {len(y)}")
    print(f"  Sujetos: {meta['subject_id'].nunique()}")
    print(f"  Features base: {X_log.shape[1]}")

    print("\n  Cargando modelos within-subject...")
    models = load_subject_models()
    print(f"  Modelos encontrados: {len(models)}")

    print("\n  Calculando SHAP lineal por sujeto...")
    print(f"  {'Sujeto':<18} {'Modelo':>10} {'N_ep':>7} {'Acc':>8} {'F1':>8} Estado")
    print("  " + "-" * 65)

    feature_rows = []
    subject_summary = []
    skipped_subjects = []

    for subject_id in sorted(meta["subject_id"].unique()):
        if subject_id not in models:
            skipped_subjects.append(subject_id)
            print(f"  {subject_id:<18} {'-':>10} {'-':>7} {'-':>8} {'-':>8} sin_modelo")
            continue

        model_data = models[subject_id]
        classifier = model_data["classifier"]
        classifier_name = model_data.get("classifier_name", classifier.__class__.__name__)
        model_ch_names = model_data.get("ch_names_eeg", ch_names)
        feature_names = get_feature_names(model_data, model_ch_names)

        mask = (meta["subject_id"] == subject_id).values
        X_subject_log = X_log[mask]
        y_subject = y[mask]

        X_features = transform_saved_preprocessing(
            X_subject_log,
            model_ch_names,
            model_data["preprocessing"],
        )

        if X_features.shape[1] != len(feature_names):
            n = min(X_features.shape[1], len(feature_names))
            X_features = X_features[:, :n]
            feature_names = feature_names[:n]

        shap_values, base_values, classes = compute_linear_shap_values(classifier, X_features)

        if shap_values is None:
            skipped_subjects.append(subject_id)
            print(
                f"  {subject_id:<18} {classifier_name:>10} {len(y_subject):>7} "
                f"{'-':>8} {'-':>8} no_lineal"
            )
            continue

        acc, f1, y_pred = evaluate_model(classifier, X_features, y_subject)

        subject_summary.append({
            "subject_id": subject_id,
            "classifier_name": classifier_name,
            "n_epochs": int(len(y_subject)),
            "accuracy": acc,
            "f1_macro": f1,
        })

        feature_rows.extend(
            build_subject_feature_rows(
                subject_id,
                y_subject,
                feature_names,
                shap_values,
                classes,
            )
        )

        print(
            f"  {subject_id:<18} {classifier_name:>10} {len(y_subject):>7} "
            f"{acc:>8.3f} {f1:>8.3f} OK"
        )

    if not feature_rows:
        raise RuntimeError("No se pudo calcular SHAP para ningún sujeto.")

    print("\n  Agregando resultados...")
    feature_subject_df = pd.DataFrame(feature_rows)
    feature_summary_df = aggregate_feature_importance(feature_subject_df)
    channel_band_df = build_channel_band_importance(feature_summary_df, ch_names)
    topomap_values = build_topomap_values(channel_band_df, ch_names)
    band_importance_df = compute_band_importance(channel_band_df)
    artifact_audit_df = compute_artifact_audit(feature_subject_df)

    print("\n  Guardando tablas...")
    save_tables(
        feature_subject_df,
        feature_summary_df,
        channel_band_df,
        band_importance_df,
        artifact_audit_df,
    )
    save_summary_json(
        subject_summary,
        feature_summary_df,
        band_importance_df,
        artifact_audit_df,
    )

    print("\n  Generando figuras...")
    plot_top_features(feature_summary_df)
    plot_band_importance(band_importance_df)
    plot_topomap_grid(topomap_values, ch_names, mode="abs")
    plot_topomap_grid(topomap_values, ch_names, mode="signed")
    plot_artifact_audit(artifact_audit_df)

    print_console_summary(
        subject_summary,
        feature_summary_df,
        band_importance_df,
        artifact_audit_df,
    )

    if skipped_subjects:
        print(f"\n  Sujetos omitidos: {skipped_subjects}")

    print("\n" + "=" * 75)
    print("EXPLICABILIDAD COMPLETADA!!!!!!!!!!")
    print(f"Resultados en: {RESULTS_DIR}")
    print("=" * 75)
