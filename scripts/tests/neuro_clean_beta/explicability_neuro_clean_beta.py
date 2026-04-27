"""
Explicabilidad del test within-subject neuro_clean_beta:

    CV por sujeto -> SHAP lineal out-of-fold -> agregación por feature/canal/banda -> topomaps

Este script explica la tanda pedida por el tutor:
  - sin Gamma
  - con Delta, Theta, Alpha y Beta
  - sin canales periféricos sospechosos

Salida:
  scripts/tests/neuro_clean_beta/results/shap_topomap/shap_subject_summary.csv
  scripts/tests/neuro_clean_beta/results/shap_topomap/shap_feature_importance_by_subject.csv
  scripts/tests/neuro_clean_beta/results/shap_topomap/shap_feature_importance.csv
  scripts/tests/neuro_clean_beta/results/shap_topomap/shap_channel_band_importance.csv
  scripts/tests/neuro_clean_beta/results/shap_topomap/shap_band_importance.csv
  scripts/tests/neuro_clean_beta/results/shap_topomap/explicability_summary.json
  scripts/tests/neuro_clean_beta/results/shap_topomap/01_top_features_global.png
  scripts/tests/neuro_clean_beta/results/shap_topomap/02_band_importance_by_class.png
  scripts/tests/neuro_clean_beta/results/shap_topomap/03_topomap_abs_<COND>.png
  scripts/tests/neuro_clean_beta/results/shap_topomap/04_topomap_signed_<COND>.png
"""

import json
import os
import warnings

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
from sklearn.model_selection import StratifiedKFold

from neuro_clean_beta_common import (
    BAND_COLORS,
    CONDITIONS,
    EXCLUDED_BANDS,
    FEATURES_DIR,
    LABEL_INV,
    N_BANDS,
    N_FOLDS,
    OUTPUT_DIR,
    PERIPHERAL_CHANNELS,
    PLOT_BANDS,
    RANDOM_STATE,
    SHAP_DIR,
    apply_normalization_pipeline,
    build_feature_names,
    build_selected_feature_indices,
    describe_feature,
    feature_block_counts,
    feature_color,
    load_dataset,
)


warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# ---------------------------------------------------------------------- Configuración
SFREQ = 250
N_TOP_FEATURES = 25
N_TOP_FEATURES_JSON = 12

FEATURE_COL_WIDTH = 32
METRIC_COL_WIDTH = 12


# ---------------------------------------------------------------------- SHAP lineal
def compute_linear_shap_values(classifier, X_background, X_explain):
    """
    Calcula contribuciones SHAP exactas para un modelo lineal.

    Para cada clase:
        logit_clase(x) = intercept + sum(coef_j * x_j)
        shap_j         = coef_j * (x_j - media_train_j)
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


def evaluate_subject_oof(X_log_subject, y_subject, ch_names, selected_indices):
    """Entrena folds CV y devuelve métricas + SHAP solo sobre test."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
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

    for train_idx, test_idx in skf.split(X_log_subject, y_subject):
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

        X_train = X_train[:, selected_indices]
        X_test = X_test[:, selected_indices]

        fold_clf = clone(classifier)
        fold_clf.fit(X_train, y_train)

        y_pred = fold_clf.predict(X_test)
        shap_values, _, classes = compute_linear_shap_values(fold_clf, X_train, X_test)

        if shap_values is None:
            return None

        if classes_ref is None:
            classes_ref = classes
        elif not np.array_equal(classes_ref, classes):
            raise ValueError("El orden de clases cambió entre folds.")

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


# ---------------------------------------------------------------------- Agregación
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


def build_channel_band_importance(feature_summary_df, topomap_ch_names):
    """Extrae solo el bloque canal x banda para topomaps."""
    rows = []
    bandpower_df = feature_summary_df[feature_summary_df["feature_type"] == "bandpower"]

    for _, row in bandpower_df.iterrows():
        if row["channel"] not in topomap_ch_names or row["band"] not in PLOT_BANDS:
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


def build_topomap_values(channel_band_df, topomap_ch_names):
    """Convierte la tabla canal/banda a matrices por clase."""
    ch_idx = {ch: i for i, ch in enumerate(topomap_ch_names)}
    band_idx = {band: i for i, band in enumerate(PLOT_BANDS)}
    maps = {}

    for condition in CONDITIONS:
        maps[condition] = {
            "abs": np.zeros((len(topomap_ch_names), len(PLOT_BANDS)), dtype=float),
            "signed": np.zeros((len(topomap_ch_names), len(PLOT_BANDS)), dtype=float),
        }

    for _, row in channel_band_df.iterrows():
        condition = row["condition"]
        channel = row["channel"]
        band = row["band"]

        if condition not in maps or channel not in ch_idx or band not in band_idx:
            continue

        i_ch = ch_idx[channel]
        i_band = band_idx[band]
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

        for band in PLOT_BANDS:
            value = cond_df[cond_df["band"] == band]["mean_abs_shap"].sum()
            rows.append({
                "condition": condition,
                "band": band,
                "mean_abs_shap": float(value),
                "percentage": float(value / total * 100),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- Plots
def plot_top_features(feature_summary_df):
    """Dibuja las features más importantes promediando clases y sujetos."""
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

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
    ax.set_xlabel("SHAP medio absoluto out-of-fold")
    ax.set_title(f"Top {N_TOP_FEATURES} features - neuro_clean_beta within-subject")
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_items = []
    for band in PLOT_BANDS:
        if band in feature_summary_df["band"].values:
            legend_items.append((band, BAND_COLORS[band]))
    if "ThetaAlphaRatio" in feature_summary_df["feature_type"].values:
        legend_items.append(("ThetaAlphaRatio", BAND_COLORS["ThetaAlphaRatio"]))
    if "AlphaAsym" in feature_summary_df["feature_type"].values:
        legend_items.append(("AlphaAsym", BAND_COLORS["AlphaAsym"]))

    ax.legend(
        handles=[Patch(facecolor=color, label=name) for name, color in legend_items],
        fontsize=8,
        loc="lower right",
    )

    plt.tight_layout()
    output_path = SHAP_DIR / "01_top_features_global.png"
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  {output_path.name}")


def plot_band_importance(band_importance_df):
    """Dibuja la importancia relativa de bandas por clase."""
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(CONDITIONS))
    width = 0.18
    offsets = np.linspace(
        -width * (len(PLOT_BANDS) - 1) / 2,
        width * (len(PLOT_BANDS) - 1) / 2,
        len(PLOT_BANDS),
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    for i, band in enumerate(PLOT_BANDS):
        values = []
        for condition in CONDITIONS:
            row = band_importance_df[
                (band_importance_df["condition"] == condition) &
                (band_importance_df["band"] == band)
            ]
            values.append(float(row["percentage"].iloc[0]) if len(row) else 0.0)

        ax.bar(
            x + offsets[i],
            values,
            width=width,
            label=band,
            color=BAND_COLORS[band],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, fontsize=11)
    ax.set_ylabel("% SHAP absoluto dentro de bandpower")
    ax.set_title("Importancia relativa por banda y clase - neuro_clean_beta")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = SHAP_DIR / "02_band_importance_by_class.png"
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


def plot_topomap_grid(topomap_values, topomap_ch_names, mode):
    """Genera una figura de topomaps por clase y banda."""
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    info = create_topomap_info(topomap_ch_names)

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
            title_prefix = "SHAP firmado OOF"
            output_name = f"04_topomap_signed_{condition}.png"

        fig, axes = plt.subplots(1, len(PLOT_BANDS), figsize=(15, 4))
        fig.patch.set_facecolor("#f8f9fa")
        fig.suptitle(
            f"{title_prefix} - {condition} - neuro_clean_beta",
            fontsize=13,
            fontweight="bold",
        )

        im = None
        for band_idx, band in enumerate(PLOT_BANDS):
            ax = axes[band_idx]
            ax.set_facecolor("#ffffff")
            im = plot_topomap_compat(data[:, band_idx], info, ax, cmap, vmin, vmax)
            ax.set_title(band, fontsize=11)

        if im is not None:
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
            cbar.ax.tick_params(labelsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        output_path = SHAP_DIR / output_name
        plt.savefig(output_path, dpi=120)
        plt.close(fig)
        print(f"  {output_path.name}")


# ---------------------------------------------------------------------- Guardado
def save_tables(subject_summary, feature_subject_df, feature_summary_df, channel_band_df, band_importance_df):
    """Guarda todas las tablas de explicabilidad."""
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "shap_subject_summary.csv": pd.DataFrame(subject_summary),
        "shap_feature_importance_by_subject.csv": feature_subject_df,
        "shap_feature_importance.csv": feature_summary_df,
        "shap_channel_band_importance.csv": channel_band_df,
        "shap_band_importance.csv": band_importance_df,
    }

    for filename, df in paths.items():
        output_path = SHAP_DIR / filename
        df.to_csv(output_path, index=False)
        print(f"  {output_path.name}")


def save_summary_json(subject_summary, feature_summary_df, band_importance_df, selected_feature_names):
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

    cm_total = np.zeros((3, 3), dtype=int)
    for item in subject_summary:
        cm_total += np.asarray(item["confusion"], dtype=int)

    summary = {
        "test_name": "neuro_clean_beta",
        "method": "out_of_fold_linear_shap_for_logistic_regression",
        "explanation": "coef_j * (x_test_j - mean_train_feature_j)",
        "cv": {
            "type": "StratifiedKFold",
            "n_splits": N_FOLDS,
            "scope": "within_subject",
            "shap_scope": "test_fold_only",
        },
        "excluded_bands": sorted(EXCLUDED_BANDS),
        "excluded_channels": sorted(PERIPHERAL_CHANNELS),
        "n_features_selected": len(selected_feature_names),
        "selected_feature_blocks": feature_block_counts(selected_feature_names),
        "n_subjects": int(subject_df["subject_id"].nunique()) if len(subject_df) else 0,
        "mean_subject_accuracy_oof": (
            float(subject_df["accuracy"].mean()) if len(subject_df) else None
        ),
        "mean_subject_f1_oof": (
            float(subject_df["f1_macro"].mean()) if len(subject_df) else None
        ),
        "confusion_matrix_oof": cm_total.tolist(),
        "top_features_by_class": top_features,
        "band_percentages": band_percentages,
    }

    output_path = SHAP_DIR / "explicability_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  {output_path.name}")


# ---------------------------------------------------------------------- Resumen de consola
def print_console_summary(subject_summary, feature_summary_df, band_importance_df):
    """Imprime un resumen corto para revisar en terminal."""
    subject_df = pd.DataFrame(subject_summary)

    print("\n" + "=" * 75)
    print("RESUMEN EXPLICABILIDAD NEURO CLEAN + BETA")
    print("=" * 75)

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
        print(f"  {'Feature':<{FEATURE_COL_WIDTH}} {'SHAP abs':>{METRIC_COL_WIDTH}} {'SHAP sign':>{METRIC_COL_WIDTH}}")
        print("  " + "-" * 62)
        for _, row in top_df.iterrows():
            print(
                f"  {row['feature_name']:<{FEATURE_COL_WIDTH}} "
                f"{row['mean_abs_shap']:>{METRIC_COL_WIDTH}.5f} "
                f"{row['mean_signed_shap']:>{METRIC_COL_WIDTH}.5f}"
            )

    print("\n  Importancia por banda (% dentro de bandpower):")
    print(f"  {'Clase':<10} " + " ".join(f"{band:>9}" for band in PLOT_BANDS))
    print("  " + "-" * 52)
    for condition in CONDITIONS:
        values = []
        for band in PLOT_BANDS:
            row = band_importance_df[
                (band_importance_df["condition"] == condition) &
                (band_importance_df["band"] == band)
            ]
            values.append(float(row["percentage"].iloc[0]) if len(row) else 0.0)
        print(f"  {condition:<10} " + " ".join(f"{v:>8.1f}%" for v in values))


# ---------------------------------------------------------------------- MAIN
if __name__ == "__main__":
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    print("EXPLICABILIDAD TEST NEURO CLEAN + BETA CON SHAP LINEAL OOF!!!!!!!!!!")
    print(f"  Features: {FEATURES_DIR}")
    print(f"  Test:     {OUTPUT_DIR}")
    print(f"  Salida:   {SHAP_DIR}")
    print("  Método:   SHAP lineal exacto para LogisticRegression")
    print("  Validación: SHAP sobre test fold, referencia calculada con train fold")
    print(f"  Excluir bandas: {sorted(EXCLUDED_BANDS)}")
    print(f"  Excluir canales: {len(PERIPHERAL_CHANNELS)}")

    print("\n  Cargando datos...")
    X_log, y, meta, ch_names = load_dataset()

    if X_log.shape[1] != len(ch_names) * N_BANDS:
        raise ValueError(
            f"Shape inconsistente: X_log.shape[1]={X_log.shape[1]} "
            f"pero se esperaba {len(ch_names)} canales x {N_BANDS} bandas = {len(ch_names) * N_BANDS}."
        )

    full_feature_names = build_feature_names(ch_names)
    selected_indices, selected_feature_names = build_selected_feature_indices(full_feature_names)
    topomap_ch_names = [ch for ch in ch_names if ch not in PERIPHERAL_CHANNELS]

    print(f"  Épocas:  {len(y)}")
    print(f"  Sujetos: {meta['subject_id'].nunique()}")
    print(f"  Features completas: {len(full_feature_names)}")
    print(f"  Features seleccionadas: {len(selected_feature_names)}")
    print(f"  Bloques seleccionados: {feature_block_counts(selected_feature_names)}")
    print(f"  Canales en topomap: {len(topomap_ch_names)}")

    print("\n  Calculando SHAP lineal out-of-fold por sujeto...")
    print(f"  {'Sujeto':<18} {'Modelo':>10} {'N_ep':>7} {'Acc':>8} {'F1':>8} Estado")
    print("  " + "-" * 65)

    feature_rows = []
    subject_summary = []
    skipped_subjects = []

    for subject_id in sorted(meta["subject_id"].unique()):
        mask = (meta["subject_id"] == subject_id).values
        X_subject_log = X_log[mask]
        y_subject = y[mask]

        classes, counts = np.unique(y_subject, return_counts=True)
        min_class = counts.min()
        if len(classes) < len(CONDITIONS) or min_class < N_FOLDS:
            skipped_subjects.append(subject_id)
            print(
                f"  {subject_id:<18} {'LogReg':>10} {len(y_subject):>7} "
                f"{'-':>8} {'-':>8} pocas_epocas"
            )
            continue

        result = evaluate_subject_oof(
            X_subject_log,
            y_subject,
            ch_names,
            selected_indices,
        )

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
                selected_feature_names,
                result["shap_values"],
                result["classes"],
            )
        )

        print(
            f"  {subject_id:<18} {'LogReg':>10} {len(y_subject):>7} "
            f"{result['accuracy']:>8.3f} {result['f1_macro']:>8.3f} OK"
        )

    if not feature_rows:
        raise RuntimeError("No se pudo calcular SHAP para ningún sujeto.")

    print("\n  Agregando resultados...")
    feature_subject_df = pd.DataFrame(feature_rows)
    feature_summary_df = aggregate_feature_importance(feature_subject_df)
    channel_band_df = build_channel_band_importance(feature_summary_df, topomap_ch_names)
    topomap_values = build_topomap_values(channel_band_df, topomap_ch_names)
    band_importance_df = compute_band_importance(channel_band_df)

    print("\n  Guardando tablas...")
    save_tables(
        subject_summary,
        feature_subject_df,
        feature_summary_df,
        channel_band_df,
        band_importance_df,
    )
    save_summary_json(
        subject_summary,
        feature_summary_df,
        band_importance_df,
        selected_feature_names,
    )

    print("\n  Generando figuras...")
    plot_top_features(feature_summary_df)
    plot_band_importance(band_importance_df)
    plot_topomap_grid(topomap_values, topomap_ch_names, mode="abs")
    plot_topomap_grid(topomap_values, topomap_ch_names, mode="signed")

    print_console_summary(subject_summary, feature_summary_df, band_importance_df)

    if skipped_subjects:
        print(f"\n  Sujetos omitidos: {skipped_subjects}")

    print("\n" + "=" * 75)
    print("EXPLICABILIDAD NEURO CLEAN + BETA COMPLETADA!!!!!!!!!!")
    print(f"Resultados en: {SHAP_DIR}")
    print("=" * 75)
