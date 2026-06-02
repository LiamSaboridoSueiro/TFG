"""
Explicabilidad del test alpha_beta:

    CV temporal por sujeto -> SHAP lineal out-of-fold -> topomaps Alpha/Beta -> clusters

Salida:
  results/explicability/within_subject/shap_subject_summary.csv
  results/explicability/within_subject/shap_feature_importance_by_subject.csv
  results/explicability/within_subject/shap_feature_importance.csv
  results/explicability/within_subject/shap_channel_band_importance.csv
  results/explicability/within_subject/shap_band_importance.csv
  results/explicability/within_subject/shap_cluster_band_importance.csv
  results/explicability/within_subject/explicability_summary.json
  results/explicability/within_subject/01_top_features_global.png
  results/explicability/within_subject/02_band_importance_by_class.png
  results/explicability/within_subject/03_topomap_abs_<COND>.png
  results/explicability/within_subject/04_topomap_signed_<COND>.png
  results/explicability/within_subject/05_cluster_importance_by_class.png
  results/explicability/within_subject/06_cluster_band_heatmap.png
  results/explicability/within_subject/topomaps_subjects/topomap_abs_<SUBJECT>.png
  results/explicability/within_subject/topomaps_subjects/topomap_signed_<SUBJECT>.png
"""

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Permite ejecutar este script desde scripts/5_explicability importando
# utilidades compartidas desde scripts/.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils import (
    BAND_COLORS,
    CONDITIONS,
    EXCLUDED_BANDS,
    FEATURES_DIR,
    LABEL_INV,
    LABEL_MAP,
    EEG_CLUSTERS,
    N_BANDS,
    N_FOLDS,
    RANDOM_STATE,
    RESULTS_DIR,
    SELECTED_BANDS,
    SHAP_DIR,
    apply_normalization_pipeline,
    available_clusters,
    build_feature_names,
    channel_cluster,
    describe_feature,
    feature_block_counts,
    feature_color,
    load_dataset,
)


warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# ---------------------------------------------------------------------- Configuracion
SFREQ = 250
N_TOP_FEATURES = 25
N_TOP_FEATURES_JSON = 12

FEATURE_COL_WIDTH = 32
METRIC_COL_WIDTH = 12
SUBJECT_TOPOMAP_DIR = SHAP_DIR / "topomaps_subjects"


# ---------------------------------------------------------------------- SHAP lineal
def compute_linear_shap_values(classifier, X_background, X_explain):
    """
    Calcula SHAP exacto para LogReg:
        shap_j = coef_j * (x_test_j - mean_train_j)
    """
    if not hasattr(classifier, "coef_"):
        return None, None, None

    coef = np.asarray(classifier.coef_, dtype=float)
    intercept = np.asarray(classifier.intercept_, dtype=float)
    classes = np.asarray(getattr(classifier, "classes_", np.arange(coef.shape[0])))

    if coef.ndim != 2 or coef.shape[0] != len(classes):
        return None, None, None

    reference = X_background.mean(axis=0)
    centered = X_explain - reference[np.newaxis, :]
    shap_values = centered[:, np.newaxis, :] * coef[np.newaxis, :, :]
    base_values = intercept + reference @ coef.T

    return shap_values, base_values, classes


def build_temporal_stratified_folds(meta_subject, y_subject, n_splits=N_FOLDS):
    """
    Construye folds temporales con test contiguo dentro de cada condicion.

    Cada fold contiene un tramo de JOY, NEUTRO y SAD para mantener el test
    estratificado sin mezclar epocas temporalmente adyacentes al azar.
    """
    folds = []
    all_indices = np.arange(len(y_subject))

    chunks_by_label = {}
    for condition in CONDITIONS:
        label = LABEL_MAP[condition]
        label_positions = np.flatnonzero(y_subject == label)
        if len(label_positions) < n_splits:
            return []

        if "epoch_idx" in meta_subject.columns:
            order = np.argsort(meta_subject.iloc[label_positions]["epoch_idx"].to_numpy())
            label_positions = label_positions[order]

        chunks_by_label[label] = np.array_split(label_positions, n_splits)

    for fold_idx in range(n_splits):
        test_idx = np.concatenate([
            chunks_by_label[LABEL_MAP[condition]][fold_idx]
            for condition in CONDITIONS
        ])
        test_idx = np.sort(test_idx)
        train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=False)
        folds.append((train_idx, test_idx))

    return folds


def evaluate_subject_oof(X_log_subject, y_subject, meta_subject, ch_names):
    """Entrena folds temporales y devuelve metricas + SHAP solo sobre test."""
    folds = build_temporal_stratified_folds(meta_subject, y_subject)
    if not folds:
        return None

    classifier = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=1.0,
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    y_true_all = []
    y_pred_all = []
    shap_chunks = []
    classes_ref = None

    for train_idx, test_idx in folds:
        X_log_train = X_log_subject[train_idx]
        X_log_test = X_log_subject[test_idx]
        y_train = y_subject[train_idx]
        y_test = y_subject[test_idx]

        X_train, X_test = apply_normalization_pipeline(
            X_log_train,
            y_train,
            X_log_test,
            ch_names,
        )

        fold_clf = clone(classifier)
        fold_clf.fit(X_train, y_train)

        y_pred = fold_clf.predict(X_test)
        shap_values, _, classes = compute_linear_shap_values(fold_clf, X_train, X_test)

        if shap_values is None:
            return None

        if classes_ref is None:
            classes_ref = classes
        elif not np.array_equal(classes_ref, classes):
            raise ValueError("El orden de clases cambio entre folds.")

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        shap_chunks.append(shap_values)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    shap_oof = np.concatenate(shap_chunks, axis=0)

    return {
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "f1_macro": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
        "confusion": confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2]),
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "shap_values": shap_oof,
        "classes": classes_ref,
    }


# ---------------------------------------------------------------------- Agregacion
def build_subject_feature_rows(subject_id, y_subject, feature_names, shap_values, classes):
    """Construye tabla larga de importancia por sujeto, clase y feature."""
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
            cluster = channel_cluster(channel) if channel else ""
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
                "cluster": cluster,
                "mean_abs_shap": float(mean_abs[feature_idx]),
                "mean_signed_shap": float(mean_signed[feature_idx]),
                "n_epochs_class": int(mask.sum()),
            })

    return rows


def aggregate_feature_importance(feature_subject_df):
    """Promedia importancia por feature dando el mismo peso a cada sujeto."""
    group_cols = [
        "class_label", "condition", "feature_idx", "feature_name",
        "feature_type", "channel", "band", "pair", "cluster",
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
    """Extrae canal x banda para topomaps."""
    rows = []
    bandpower_df = feature_summary_df[feature_summary_df["feature_type"] == "bandpower"]

    for _, row in bandpower_df.iterrows():
        if row["channel"] not in ch_names or row["band"] not in SELECTED_BANDS:
            continue
        rows.append({
            "class_label": int(row["class_label"]),
            "condition": row["condition"],
            "channel": row["channel"],
            "band": row["band"],
            "cluster": row["cluster"],
            "mean_abs_shap": float(row["mean_abs_shap"]),
            "mean_signed_shap": float(row["mean_signed_shap"]),
            "n_subjects": int(row["n_subjects"]),
        })

    return pd.DataFrame(rows)


def build_topomap_values(channel_band_df, ch_names):
    """Convierte tabla canal/banda a matrices por clase."""
    ch_idx = {ch: i for i, ch in enumerate(ch_names)}
    band_idx = {band: i for i, band in enumerate(SELECTED_BANDS)}
    maps = {}

    for condition in CONDITIONS:
        maps[condition] = {
            "abs": np.zeros((len(ch_names), len(SELECTED_BANDS)), dtype=float),
            "signed": np.zeros((len(ch_names), len(SELECTED_BANDS)), dtype=float),
        }

    for _, row in channel_band_df.iterrows():
        condition = row["condition"]
        channel = row["channel"]
        band = row["band"]
        if condition not in maps or channel not in ch_idx or band not in band_idx:
            continue
        maps[condition]["abs"][ch_idx[channel], band_idx[band]] = row["mean_abs_shap"]
        maps[condition]["signed"][ch_idx[channel], band_idx[band]] = row["mean_signed_shap"]

    return maps


def compute_band_importance(channel_band_df):
    """Calcula peso relativo por banda y clase."""
    rows = []
    for condition in CONDITIONS:
        cond_df = channel_band_df[channel_band_df["condition"] == condition]
        total = cond_df["mean_abs_shap"].sum()
        if total <= 0:
            total = 1e-12
        for band in SELECTED_BANDS:
            value = cond_df[cond_df["band"] == band]["mean_abs_shap"].sum()
            rows.append({
                "condition": condition,
                "band": band,
                "mean_abs_shap": float(value),
                "percentage": float(value / total * 100),
            })
    return pd.DataFrame(rows)


def compute_cluster_band_importance(channel_band_df):
    """Agrega SHAP por condicion, cluster y banda."""
    rows = []
    clusters = list(EEG_CLUSTERS.keys()) + ["other"]

    for condition in CONDITIONS:
        cond_df = channel_band_df[channel_band_df["condition"] == condition]
        total = cond_df["mean_abs_shap"].sum()
        if total <= 0:
            total = 1e-12

        for cluster in clusters:
            for band in SELECTED_BANDS:
                subset = cond_df[
                    (cond_df["cluster"] == cluster) &
                    (cond_df["band"] == band)
                ]
                value = subset["mean_abs_shap"].sum()
                rows.append({
                    "condition": condition,
                    "cluster": cluster,
                    "band": band,
                    "mean_abs_shap": float(value),
                    "percentage_total_bandpower": float(value / total * 100),
                    "n_channels": int(subset["channel"].nunique()),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- Plots
def plot_top_features(feature_summary_df):
    """Dibuja las features mas importantes."""
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    global_importance = (
        feature_summary_df
        .groupby(["feature_name", "feature_type", "channel", "band", "pair", "cluster"], dropna=False)
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
    ax.set_xlabel("SHAP medio absoluto out-of-fold")
    ax.set_title(f"Top {N_TOP_FEATURES} features - Alpha/Beta")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    output_path = SHAP_DIR / "01_top_features_global.png"
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  {output_path.name}")


def plot_band_importance(band_importance_df):
    """Dibuja importancia relativa Alpha/Beta."""
    x = np.arange(len(CONDITIONS))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    for i, band in enumerate(SELECTED_BANDS):
        values = []
        for condition in CONDITIONS:
            row = band_importance_df[
                (band_importance_df["condition"] == condition) &
                (band_importance_df["band"] == band)
            ]
            values.append(float(row["percentage"].iloc[0]) if len(row) else 0.0)

        ax.bar(
            x + (i - 0.5) * width,
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
    output_path = SHAP_DIR / "02_band_importance_by_class.png"
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  {output_path.name}")


def create_topomap_info(ch_names):
    """Crea Info con montaje estandar."""
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1005")
    info.set_montage(montage, match_case=False, on_missing="ignore")
    return info


def plot_topomap_compat(data, info, ax, cmap, vmin, vmax):
    """Compatibilidad entre versiones de MNE."""
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


def add_topomap_colorbar(fig, im, layout="global"):
    """Anade la barra de color fuera de la cuadricula para no tapar mapas."""
    if layout == "subject":
        cbar_ax = fig.add_axes([0.91, 0.16, 0.025, 0.68])
    else:
        cbar_ax = fig.add_axes([0.91, 0.24, 0.025, 0.50])

    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=8)
    return cbar


def plot_topomap_grid(topomap_values, ch_names, mode):
    """Genera topomaps por clase y banda."""
    info = create_topomap_info(ch_names)

    for condition in CONDITIONS:
        data = topomap_values[condition][mode]

        if mode == "abs":
            cmap = "Reds"
            vmin = 0.0
            vmax = float(np.nanmax(data))
            if vmax <= 0:
                vmax = 1.0
            title_prefix = "SHAP absoluto OOF"
            output_name = f"03_topomap_abs_{condition}.png"
        else:
            cmap = "RdBu_r"
            max_abs = float(np.nanmax(np.abs(data)))
            if max_abs <= 0:
                max_abs = 1.0
            vmin = -max_abs
            vmax = max_abs
            title_prefix = "SHAP con signo OOF"
            output_name = f"04_topomap_signed_{condition}.png"

        fig, axes = plt.subplots(1, len(SELECTED_BANDS), figsize=(9, 4))
        fig.patch.set_facecolor("#f8f9fa")
        fig.suptitle(
            f"{title_prefix} - {condition} - Alpha/Beta",
            fontsize=13,
            fontweight="bold",
        )

        im = None
        for band_idx, band in enumerate(SELECTED_BANDS):
            ax = axes[band_idx]
            ax.set_facecolor("#ffffff")
            im = plot_topomap_compat(data[:, band_idx], info, ax, cmap, vmin, vmax)
            ax.set_title(band, fontsize=11)

        plt.tight_layout(rect=[0, 0, 0.88, 0.92])
        if im is not None:
            add_topomap_colorbar(fig, im, layout="global")

        output_path = SHAP_DIR / output_name
        plt.savefig(output_path, dpi=120)
        plt.close(fig)
        print(f"  {output_path.name}")


def build_subject_topomap_values(subject_df, ch_names):
    """Convierte SHAP de un sujeto a matrices condicion x canal x banda."""
    ch_idx = {ch: i for i, ch in enumerate(ch_names)}
    band_idx = {band: i for i, band in enumerate(SELECTED_BANDS)}
    maps = {}

    for condition in CONDITIONS:
        maps[condition] = {
            "abs": np.zeros((len(ch_names), len(SELECTED_BANDS)), dtype=float),
            "signed": np.zeros((len(ch_names), len(SELECTED_BANDS)), dtype=float),
        }

    bandpower_df = subject_df[subject_df["feature_type"] == "bandpower"]
    for _, row in bandpower_df.iterrows():
        condition = row["condition"]
        channel = row["channel"]
        band = row["band"]
        if condition not in maps or channel not in ch_idx or band not in band_idx:
            continue
        maps[condition]["abs"][ch_idx[channel], band_idx[band]] = row["mean_abs_shap"]
        maps[condition]["signed"][ch_idx[channel], band_idx[band]] = row["mean_signed_shap"]

    return maps


def plot_subject_topomap_grid(feature_subject_df, ch_names, mode):
    """Genera una imagen por sujeto con JOY/NEUTRO/SAD juntos."""
    SUBJECT_TOPOMAP_DIR.mkdir(parents=True, exist_ok=True)
    info = create_topomap_info(ch_names)
    n_created = 0

    for subject_id in sorted(feature_subject_df["subject_id"].unique()):
        subject_df = feature_subject_df[feature_subject_df["subject_id"] == subject_id]
        topomap_values = build_subject_topomap_values(subject_df, ch_names)

        data_all = np.stack([
            topomap_values[condition][mode]
            for condition in CONDITIONS
        ])

        if mode == "abs":
            cmap = "Reds"
            vmin = 0.0
            vmax = float(np.nanmax(data_all))
            if vmax <= 0:
                vmax = 1.0
            title_prefix = "SHAP absoluto OOF"
            output_name = f"topomap_abs_{subject_id}.png"
        else:
            cmap = "RdBu_r"
            max_abs = float(np.nanmax(np.abs(data_all)))
            if max_abs <= 0:
                max_abs = 1.0
            vmin = -max_abs
            vmax = max_abs
            title_prefix = "SHAP con signo OOF"
            output_name = f"topomap_signed_{subject_id}.png"

        fig, axes = plt.subplots(
            len(CONDITIONS),
            len(SELECTED_BANDS),
            figsize=(9, 10),
        )
        fig.patch.set_facecolor("#f8f9fa")
        fig.suptitle(
            f"{title_prefix} - {subject_id} - Alpha/Beta",
            fontsize=13,
            fontweight="bold",
        )

        im = None
        for row_idx, condition in enumerate(CONDITIONS):
            for col_idx, band in enumerate(SELECTED_BANDS):
                ax = axes[row_idx, col_idx]
                ax.set_facecolor("#ffffff")
                im = plot_topomap_compat(
                    topomap_values[condition][mode][:, col_idx],
                    info,
                    ax,
                    cmap,
                    vmin,
                    vmax,
                )
                if row_idx == 0:
                    ax.set_title(band, fontsize=11)
                if col_idx == 0:
                    ax.set_ylabel(condition, fontsize=11, fontweight="bold")

        plt.tight_layout(rect=[0, 0, 0.88, 0.94])
        if im is not None:
            add_topomap_colorbar(fig, im, layout="subject")

        output_path = SUBJECT_TOPOMAP_DIR / output_name
        plt.savefig(output_path, dpi=120)
        plt.close(fig)
        n_created += 1

    print(f"  topomaps_subjects ({mode}): {n_created} imagenes")


def plot_cluster_importance(cluster_band_df):
    """Barplot de importancia total por cluster y clase."""
    summary = (
        cluster_band_df
        .groupby(["condition", "cluster"], dropna=False)
        .agg(percentage_total_bandpower=("percentage_total_bandpower", "sum"))
        .reset_index()
    )

    clusters = list(EEG_CLUSTERS.keys()) + ["other"]
    x = np.arange(len(clusters))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    for i, condition in enumerate(CONDITIONS):
        values = []
        for cluster in clusters:
            row = summary[
                (summary["condition"] == condition) &
                (summary["cluster"] == cluster)
            ]
            values.append(float(row["percentage_total_bandpower"].iloc[0]) if len(row) else 0.0)
        ax.bar(x + (i - 1) * width, values, width=width, label=condition, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(clusters, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("% SHAP absoluto dentro de bandpower")
    ax.set_title("Importancia por cluster y clase")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = SHAP_DIR / "05_cluster_importance_by_class.png"
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  {output_path.name}")


def plot_cluster_band_heatmap(cluster_band_df):
    """Heatmap cluster x banda para cada clase."""
    clusters = list(EEG_CLUSTERS.keys()) + ["other"]
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(15, 7), sharey=True)
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle("Importancia cluster x banda", fontsize=13, fontweight="bold")

    vmax = max(float(cluster_band_df["percentage_total_bandpower"].max()), 1e-9)
    for ax, condition in zip(axes, CONDITIONS):
        matrix = np.zeros((len(clusters), len(SELECTED_BANDS)), dtype=float)
        for i, cluster in enumerate(clusters):
            for j, band in enumerate(SELECTED_BANDS):
                row = cluster_band_df[
                    (cluster_band_df["condition"] == condition) &
                    (cluster_band_df["cluster"] == cluster) &
                    (cluster_band_df["band"] == band)
                ]
                matrix[i, j] = float(row["percentage_total_bandpower"].iloc[0]) if len(row) else 0.0

        im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=vmax)
        ax.set_title(condition)
        ax.set_xticks(range(len(SELECTED_BANDS)))
        ax.set_xticklabels(SELECTED_BANDS)
        ax.set_yticks(range(len(clusters)))
        ax.set_yticklabels(clusters, fontsize=9)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="% SHAP")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_path = SHAP_DIR / "06_cluster_band_heatmap.png"
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  {output_path.name}")


# ---------------------------------------------------------------------- Guardado
def save_tables(subject_summary, feature_subject_df, feature_summary_df, channel_band_df, band_importance_df, cluster_band_df):
    """Guarda tablas de explicabilidad."""
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "shap_subject_summary.csv": pd.DataFrame(subject_summary),
        "shap_feature_importance_by_subject.csv": feature_subject_df,
        "shap_feature_importance.csv": feature_summary_df,
        "shap_channel_band_importance.csv": channel_band_df,
        "shap_band_importance.csv": band_importance_df,
        "shap_cluster_band_importance.csv": cluster_band_df,
    }
    for filename, df in paths.items():
        df.to_csv(SHAP_DIR / filename, index=False)
        print(f"  {filename}")


def save_summary_json(subject_summary, feature_summary_df, band_importance_df, cluster_band_df, feature_names, ch_names):
    """Guarda resumen compacto."""
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
                "cluster": row["cluster"],
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

    cluster_percentages = {}
    cluster_totals = (
        cluster_band_df
        .groupby(["condition", "cluster"], dropna=False)
        .agg(percentage_total_bandpower=("percentage_total_bandpower", "sum"))
        .reset_index()
    )
    for condition in CONDITIONS:
        cond_rows = cluster_totals[cluster_totals["condition"] == condition]
        cluster_percentages[condition] = {
            row["cluster"]: float(row["percentage_total_bandpower"])
            for _, row in cond_rows.iterrows()
        }

    cm_total = np.zeros((3, 3), dtype=int)
    for item in subject_summary:
        cm_total += np.asarray(item["confusion"], dtype=int)

    summary = {
        "test_name": "alpha_beta",
        "method": "out_of_fold_linear_shap_for_logistic_regression",
        "explanation": "coef_j * (x_test_j - mean_train_feature_j)",
        "cv": {
            "type": "contiguous_condition_chunks",
            "n_splits": N_FOLDS,
            "scope": "within_subject",
            "shuffle": False,
            "group_source": "condition + epoch_idx",
        },
        "selected_bands": SELECTED_BANDS,
        "excluded_bands": EXCLUDED_BANDS,
        "n_features_selected": len(feature_names),
        "selected_feature_blocks": feature_block_counts(feature_names),
        "clusters": EEG_CLUSTERS,
        "available_cluster_channels": available_clusters(ch_names),
        "n_subjects": int(subject_df["subject_id"].nunique()) if len(subject_df) else 0,
        "mean_subject_accuracy_oof": float(subject_df["accuracy"].mean()) if len(subject_df) else None,
        "mean_subject_f1_oof": float(subject_df["f1_macro"].mean()) if len(subject_df) else None,
        "confusion_matrix_oof": cm_total.tolist(),
        "top_features_by_class": top_features,
        "band_percentages": band_percentages,
        "cluster_percentages": cluster_percentages,
    }

    with open(SHAP_DIR / "explicability_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  explicability_summary.json")


# ---------------------------------------------------------------------- Consola
def print_console_summary(subject_summary, feature_summary_df, band_importance_df, cluster_band_df):
    """Imprime resumen de terminal."""
    subject_df = pd.DataFrame(subject_summary)

    print("RESUMEN EXPLICABILIDAD Alpha/Beta")

    if len(subject_df):
        print(
            f"\n  Sujetos explicados: {subject_df['subject_id'].nunique()}  "
            f"| Acc OOF: {subject_df['accuracy'].mean():.3f}  "
            f"| F1 macro OOF: {subject_df['f1_macro'].mean():.3f}"
        )

    print("\n  Top features por clase:")
    for condition in CONDITIONS:
        top_df = (
            feature_summary_df[feature_summary_df["condition"] == condition]
            .sort_values("mean_abs_shap", ascending=False)
            .head(5)
        )
        print(f"\n  {condition}")
        print(f"  {'Feature':<{FEATURE_COL_WIDTH}} {'Cluster':<24} {'SHAP abs':>{METRIC_COL_WIDTH}}")
        print("  " + "-" * 76)
        for _, row in top_df.iterrows():
            print(
                f"  {row['feature_name']:<{FEATURE_COL_WIDTH}} "
                f"{row['cluster']:<24} "
                f"{row['mean_abs_shap']:>{METRIC_COL_WIDTH}.5f}"
            )

    print("\n  Importancia por banda (% dentro de bandpower):")
    print(f"  {'Clase':<10} " + " ".join(f"{band:>9}" for band in SELECTED_BANDS))
    print("  " + "-" * 34)
    for condition in CONDITIONS:
        values = []
        for band in SELECTED_BANDS:
            row = band_importance_df[
                (band_importance_df["condition"] == condition) &
                (band_importance_df["band"] == band)
            ]
            values.append(float(row["percentage"].iloc[0]) if len(row) else 0.0)
        print(f"  {condition:<10} " + " ".join(f"{v:>8.1f}%" for v in values))

    cluster_totals = (
        cluster_band_df
        .groupby(["condition", "cluster"], dropna=False)
        .agg(percentage_total_bandpower=("percentage_total_bandpower", "sum"))
        .reset_index()
    )
    print("\n  Top clusters por clase:")
    for condition in CONDITIONS:
        top_clusters = (
            cluster_totals[cluster_totals["condition"] == condition]
            .sort_values("percentage_total_bandpower", ascending=False)
            .head(3)
        )
        values = ", ".join(
            f"{row['cluster']}={row['percentage_total_bandpower']:.1f}%"
            for _, row in top_clusters.iterrows()
        )
        print(f"  {condition:<10} {values}")


# ---------------------------------------------------------------------- MAIN
if __name__ == "__main__":
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    print("EXPLICABILIDAD Alpha/Beta CON SHAP LINEAL OOF!!!!!!!!!!")
    print(f"  Features: {FEATURES_DIR}")
    print(f"  Test:     {RESULTS_DIR}")
    print(f"  Salida:   {SHAP_DIR}")
    print(f"  Bandas:   {SELECTED_BANDS}")
    print(f"  Excluye:  {EXCLUDED_BANDS}")
    print("  Clusters: resumen post-hoc sobre SHAP por canal")

    print("\n  Cargando datos...")
    X_log, y, meta, ch_names = load_dataset()
    feature_names = build_feature_names(ch_names)

    if X_log.shape[1] != len(ch_names) * N_BANDS:
        raise ValueError(
            f"Shape inconsistente: X_log.shape[1]={X_log.shape[1]} "
            f"pero se esperaba {len(ch_names)} canales x {N_BANDS} bandas = {len(ch_names) * N_BANDS}."
        )

    print(f"  Epocas:  {len(y)}")
    print(f"  Sujetos: {meta['subject_id'].nunique()}")
    print(f"  Features seleccionadas: {len(feature_names)}")
    print(f"  Bloques seleccionados: {feature_block_counts(feature_names)}")

    print("\n  Calculando SHAP lineal out-of-fold por sujeto con CV temporal...")
    print(f"  {'Sujeto':<18} {'Modelo':>10} {'N_ep':>7} {'Acc':>8} {'F1':>8} Estado")
    print("  " + "-" * 65)

    feature_rows = []
    subject_summary = []
    skipped_subjects = []

    for subject_id in sorted(meta["subject_id"].unique()):
        mask = (meta["subject_id"] == subject_id).values
        X_subject_log = X_log[mask]
        y_subject = y[mask]
        meta_subject = meta.loc[mask].reset_index(drop=True)

        classes, counts = np.unique(y_subject, return_counts=True)
        min_class = counts.min()
        if len(classes) < len(CONDITIONS) or min_class < N_FOLDS:
            skipped_subjects.append(subject_id)
            print(
                f"  {subject_id:<18} {'LogReg':>10} {len(y_subject):>7} "
                f"{'-':>8} {'-':>8} pocas_epocas"
            )
            continue

        result = evaluate_subject_oof(X_subject_log, y_subject, meta_subject, ch_names)

        if result is None:
            skipped_subjects.append(subject_id)
            print(
                f"  {subject_id:<18} {'LogReg':>10} {len(y_subject):>7} "
                f"{'-':>8} {'-':>8} no_lineal"
            )
            continue

        subject_summary.append({
            "subject_id": subject_id,
            "classifier_name": "LogReg",
            "n_epochs": int(len(y_subject)),
            "accuracy": result["accuracy"],
            "f1_macro": result["f1_macro"],
            "confusion": result["confusion"].tolist(),
        })

        feature_rows.extend(
            build_subject_feature_rows(
                subject_id,
                result["y_true"],
                feature_names,
                result["shap_values"],
                result["classes"],
            )
        )

        print(
            f"  {subject_id:<18} {'LogReg':>10} {len(y_subject):>7} "
            f"{result['accuracy']:>8.3f} {result['f1_macro']:>8.3f} OK"
        )

    if not feature_rows:
        raise RuntimeError("No se pudo calcular SHAP para ningun sujeto.")

    print("\n  Agregando resultados...")
    feature_subject_df = pd.DataFrame(feature_rows)
    feature_summary_df = aggregate_feature_importance(feature_subject_df)
    channel_band_df = build_channel_band_importance(feature_summary_df, ch_names)
    topomap_values = build_topomap_values(channel_band_df, ch_names)
    band_importance_df = compute_band_importance(channel_band_df)
    cluster_band_df = compute_cluster_band_importance(channel_band_df)

    print("\n  Guardando tablas...")
    save_tables(
        subject_summary,
        feature_subject_df,
        feature_summary_df,
        channel_band_df,
        band_importance_df,
        cluster_band_df,
    )
    save_summary_json(
        subject_summary,
        feature_summary_df,
        band_importance_df,
        cluster_band_df,
        feature_names,
        ch_names,
    )

    print("\n  Generando figuras...")
    plot_top_features(feature_summary_df)
    plot_band_importance(band_importance_df)
    plot_topomap_grid(topomap_values, ch_names, mode="abs")
    plot_topomap_grid(topomap_values, ch_names, mode="signed")
    plot_subject_topomap_grid(feature_subject_df, ch_names, mode="abs")
    plot_subject_topomap_grid(feature_subject_df, ch_names, mode="signed")
    plot_cluster_importance(cluster_band_df)
    plot_cluster_band_heatmap(cluster_band_df)

    print_console_summary(
        subject_summary,
        feature_summary_df,
        band_importance_df,
        cluster_band_df,
    )

    if skipped_subjects:
        print(f"\n  Sujetos omitidos: {skipped_subjects}")

    print("EXPLICABILIDAD Alpha/Beta COMPLETADA!!!!!!!!!!")
    print(f"Resultados en: {SHAP_DIR}")
