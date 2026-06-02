"""
Analisis de separabilidad de las features EEG:

    Usa el mismo espacio de features que el pipeline principal:
    Alpha/Beta + AlphaAsym.

    - PCA global: condicion vs sujeto
    - LDA global: condicion vs sujeto
    - PCA por sujeto: condiciones dentro de cada sujeto
    - LDA por sujeto: condiciones dentro de cada sujeto
    - PCA/LDA individuales por sujeto con ejes, escalas y leyenda
    - Metricas de separabilidad en CSV y JSON

Salida:
  results/separability/01_pca_global_condition_subject.png
  results/separability/02_lda_global_condition_subject.png
  results/separability/03_pca_within_subject.png
  results/separability/04_lda_within_subject.png
  results/separability/by_subject/pca/*.png
  results/separability/by_subject/lda/*.png
  results/separability/05_separability_summary.png
  results/separability/separability_metrics.csv
  results/separability/separability_summary.json

"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score


warnings.filterwarnings("ignore")

# Permite ejecutar este script desde scripts/4_separability importando
# utilidades compartidas desde scripts/.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils import (
    CONDITIONS,
    EXCLUDED_BANDS,
    FEATURES_DIR,
    LABEL_MAP,
    N_BANDS,
    PROJECT_ROOT,
    RANDOM_STATE,
    SELECTED_BANDS,
    build_feature_names,
    feature_block_counts,
    fit_normalization_pipeline,
    load_dataset,
)


# ---------------------------------------------------------------------- Configuracion
RESULTS_DIR = PROJECT_ROOT / "results" / "separability"

CONDITION_COLORS = {
    "JOY": "#e74c3c",
    "NEUTRO": "#3498db",
    "SAD": "#2ecc71",
}


# ---------------------------------------------------------------------- Normalizacion
def fit_feature_matrix(X_log_train, y_train, ch_names):
    """
    Aplica el mismo preprocesado que los train.

    Para analisis visual se ajusta con los datos que se estan dibujando:
    global para all-subjects y por sujeto para within-subject.
    """
    X_feat, _ = fit_normalization_pipeline(X_log_train, y_train, ch_names)
    return X_feat


# ---------------------------------------------------------------------- Proyecciones
def compute_pca_2d(X):
    """Calcula PCA 2D y devuelve coordenadas + varianza explicada."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


def compute_lda_2d(X, labels):
    """Calcula LDA 2D; si solo hay una dimension, rellena la segunda con ceros."""
    n_classes = len(np.unique(labels))
    if n_classes < 2:
        return np.zeros((len(labels), 2), dtype=float)

    n_components = min(2, n_classes - 1)
    try:
        # Shrinkage ayuda a estabilizar LDA cuando hay muchas features EEG.
        lda = LinearDiscriminantAnalysis(
            n_components=n_components,
            solver="eigen",
            shrinkage="auto",
        )
        coords = lda.fit_transform(X, labels)
    except Exception:
        # Fallback compatible si la matriz no permite el solver con shrinkage.
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        coords = lda.fit_transform(X, labels)

    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])

    return coords


# ---------------------------------------------------------------------- Metricas
def safe_silhouette(X, labels):
    """Calcula silhouette evitando errores cuando no hay clases suficientes."""
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return np.nan

    try:
        return float(silhouette_score(X, labels))
    except ValueError:
        return np.nan


def centroid_ratio(X, labels):
    """
    Ratio separacion/interior:
        distancia media entre centroides / dispersion media dentro de clase.
    """
    labels = np.asarray(labels)
    centroids = []
    spreads = []

    for label in sorted(np.unique(labels)):
        X_label = X[labels == label]
        if len(X_label) == 0:
            continue
        centroid = X_label.mean(axis=0)
        centroids.append(centroid)
        spreads.append(np.mean(np.linalg.norm(X_label - centroid, axis=1)))

    if len(centroids) < 2:
        return np.nan, np.nan, np.nan

    distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            distances.append(np.linalg.norm(centroids[i] - centroids[j]))

    centroid_distance = float(np.mean(distances))
    within_spread = float(np.mean(spreads))
    ratio = centroid_distance / (within_spread + 1e-12)

    return float(ratio), centroid_distance, within_spread


def make_metric_row(scope, subject_id, space, label_type, X, labels, extra=None):
    """Crea una fila de metricas en formato tabular."""
    ratio, distance, spread = centroid_ratio(X, labels)
    row = {
        "scope": scope,
        "subject_id": subject_id,
        "space": space,
        "label_type": label_type,
        "n_samples": int(len(labels)),
        "n_labels": int(len(np.unique(labels))),
        "silhouette": safe_silhouette(X, labels),
        "centroid_ratio": ratio,
        "centroid_distance": distance,
        "within_spread": spread,
    }
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------------- Plots
def plot_condition_scatter(ax, coords, y, title):
    """Dibuja un scatter coloreado por condicion."""
    ax.set_facecolor("#ffffff")
    for cond in CONDITIONS:
        label = LABEL_MAP[cond]
        mask = (y == label)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.68,
            color=CONDITION_COLORS[cond],
            label=cond,
            edgecolors="none",
        )
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def plot_subject_scatter(ax, coords, subjects, title):
    """Dibuja un scatter coloreado por sujeto."""
    sujetos = sorted(pd.Series(subjects).unique())
    subject_to_code = {sid: idx for idx, sid in enumerate(sujetos)}
    codes = pd.Series(subjects).map(subject_to_code).values

    cmap = plt.get_cmap("tab20", len(sujetos))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=codes,
        cmap=cmap,
        s=12,
        alpha=0.75,
        edgecolors="none",
    )
    ax.set_facecolor("#ffffff")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    return sc, sujetos


def plot_global_pca(X_feat, y, meta):
    """PCA global coloreado por condicion y por sujeto."""
    coords, var = compute_pca_2d(X_feat)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle("PCA global all-subjects", fontsize=13, fontweight="bold")

    plot_condition_scatter(axes[0], coords, y, "Coloreado por condicion")
    axes[0].set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    axes[0].legend(fontsize=9)

    sc, sujetos = plot_subject_scatter(
        axes[1],
        coords,
        meta["subject_id"].values,
        "Coloreado por sujeto",
    )
    axes[1].set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")

    cbar = fig.colorbar(sc, ax=axes[1], ticks=np.arange(len(sujetos)))
    cbar.ax.set_yticklabels([sid.replace("211-000", "") for sid in sujetos])
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Sujeto", fontsize=9)

    plt.tight_layout()
    fname = RESULTS_DIR / "01_pca_global_condition_subject.png"
    plt.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  {fname.name}")

    return coords, var


def plot_global_lda(X_feat, y, meta):
    """LDA global supervisado por condicion y coloreado de dos formas."""
    coords = compute_lda_2d(X_feat, y)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle("LDA global all-subjects", fontsize=13, fontweight="bold")

    plot_condition_scatter(axes[0], coords, y, "Coloreado por condicion")
    axes[0].set_xlabel("LD1")
    axes[0].set_ylabel("LD2")
    axes[0].legend(fontsize=9)

    sc, sujetos = plot_subject_scatter(
        axes[1],
        coords,
        meta["subject_id"].values,
        "Coloreado por sujeto",
    )
    axes[1].set_xlabel("LD1")
    axes[1].set_ylabel("LD2")

    cbar = fig.colorbar(sc, ax=axes[1], ticks=np.arange(len(sujetos)))
    cbar.ax.set_yticklabels([sid.replace("211-000", "") for sid in sujetos])
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Sujeto", fontsize=9)

    plt.tight_layout()
    fname = RESULTS_DIR / "02_lda_global_condition_subject.png"
    plt.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  {fname.name}")

    return coords


def plot_within_grid(subject_coords, title, filename):
    """Dibuja una cuadricula con una proyeccion por sujeto."""
    sujetos = sorted(subject_coords.keys())
    n_cols = 4
    n_rows = int(np.ceil(len(sujetos) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    fig.patch.set_facecolor("#f8f9fa")
    axes = np.ravel(axes)

    for ax, sid in zip(axes, sujetos):
        coords, y_subject = subject_coords[sid]
        ax.set_facecolor("#ffffff")

        for cond in CONDITIONS:
            label = LABEL_MAP[cond]
            mask = (y_subject == label)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=12,
                alpha=0.75,
                color=CONDITION_COLORS[cond],
                edgecolors="none",
            )

        ax.set_title(sid.replace("211-000", ""), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.20)

    for ax in axes[len(sujetos):]:
        ax.axis("off")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", label=cond,
               markerfacecolor=CONDITION_COLORS[cond], markersize=7)
        for cond in CONDITIONS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fname = RESULTS_DIR / filename
    plt.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  {fname.name}")


def plot_single_subject_projection(subject_id, coords, y_subject, method, output_dir, pca_var=None):
    """Guarda una proyeccion individual de un sujeto con ejes, escala y leyenda."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    for cond in CONDITIONS:
        label = LABEL_MAP[cond]
        mask = (y_subject == label)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=28,
            alpha=0.78,
            color=CONDITION_COLORS[cond],
            label=cond,
            edgecolors="white",
            linewidths=0.25,
        )

    method_upper = method.upper()
    ax.set_title(
        f"{method_upper} within-subject - {subject_id}",
        fontsize=12,
        fontweight="bold",
    )

    if method == "pca" and pca_var is not None:
        ax.set_xlabel(f"PC1 ({pca_var[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca_var[1] * 100:.1f}%)")
    else:
        ax.set_xlabel("LD1")
        ax.set_ylabel("LD2")

    ax.grid(True, alpha=0.25)
    ax.legend(title="Condicion", fontsize=9, title_fontsize=9, loc="best")

    # Margen visual para que los puntos no queden pegados al borde.
    x_min, x_max = np.nanmin(coords[:, 0]), np.nanmax(coords[:, 0])
    y_min, y_max = np.nanmin(coords[:, 1]), np.nanmax(coords[:, 1])
    x_margin = max((x_max - x_min) * 0.08, 1e-6)
    y_margin = max((y_max - y_min) * 0.08, 1e-6)
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    safe_subject = subject_id.replace("/", "_")
    fname = output_dir / f"{method}_{safe_subject}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=140)
    plt.close(fig)


def plot_individual_subjects(pca_subject_coords, lda_subject_coords, pca_subject_var):
    """Guarda PCA y LDA individuales por sujeto en subcarpetas."""
    pca_dir = RESULTS_DIR / "by_subject" / "pca"
    lda_dir = RESULTS_DIR / "by_subject" / "lda"

    for subject_id in sorted(pca_subject_coords.keys()):
        pca_coords, y_subject = pca_subject_coords[subject_id]
        plot_single_subject_projection(
            subject_id,
            pca_coords,
            y_subject,
            method="pca",
            output_dir=pca_dir,
            pca_var=pca_subject_var[subject_id],
        )

        lda_coords, y_subject = lda_subject_coords[subject_id]
        plot_single_subject_projection(
            subject_id,
            lda_coords,
            y_subject,
            method="lda",
            output_dir=lda_dir,
        )

    print(f"  by_subject/pca/*.png ({len(pca_subject_coords)} sujetos)")
    print(f"  by_subject/lda/*.png ({len(lda_subject_coords)} sujetos)")


def plot_summary(metrics_df):
    """Resumen visual de separabilidad global y within-subject."""
    global_rows = metrics_df[
        (metrics_df["scope"] == "global") &
        (metrics_df["space"].isin(["original", "pca_2d", "lda_2d"]))
    ].copy()
    global_rows["name"] = global_rows["space"] + " / " + global_rows["label_type"]

    within_rows = metrics_df[
        (metrics_df["scope"] == "within_subject") &
        (metrics_df["label_type"] == "condition")
    ].copy()

    pivot = within_rows.pivot(index="subject_id", columns="space", values="centroid_ratio")
    pivot = pivot.sort_values("lda_2d", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle("Resumen de separabilidad", fontsize=13, fontweight="bold")

    axes[0].barh(
        global_rows["name"],
        global_rows["centroid_ratio"],
        color="#3498db",
        alpha=0.85,
    )
    axes[0].set_title("Global: ratio centroides / dispersion")
    axes[0].set_xlabel("Mayor = mas separable")
    axes[0].grid(True, axis="x", alpha=0.3)
    axes[0].set_facecolor("#ffffff")

    x = np.arange(len(pivot.index))
    width = 0.28
    colors = {
        "original": "#34495e",
        "pca_2d": "#9b59b6",
        "lda_2d": "#27ae60",
    }
    for i, space in enumerate(["original", "pca_2d", "lda_2d"]):
        if space not in pivot.columns:
            continue
        axes[1].bar(
            x + (i - 1) * width,
            pivot[space].values,
            width=width,
            label=space,
            color=colors[space],
            alpha=0.85,
        )

    axes[1].set_title("Within-subject por sujeto")
    axes[1].set_ylabel("Ratio centroides / dispersion")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([sid.replace("211-000", "") for sid in pivot.index], rotation=90, fontsize=8)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_facecolor("#ffffff")

    plt.tight_layout()
    fname = RESULTS_DIR / "05_separability_summary.png"
    plt.savefig(fname, dpi=120)
    plt.close(fig)
    print(f"  {fname.name}")


# ---------------------------------------------------------------------- Analisis principal
def analyze_global(X_log, y, meta, ch_names):
    """Analiza la separabilidad global all-subjects."""
    X_feat = fit_feature_matrix(X_log, y, ch_names)
    subject_labels = meta["subject_id"].values

    metrics = []

    pca_coords, pca_var = plot_global_pca(X_feat, y, meta)
    lda_coords = plot_global_lda(X_feat, y, meta)

    spaces = [
        ("original", X_feat, {}),
        ("pca_2d", pca_coords, {"pc1_var": float(pca_var[0]), "pc2_var": float(pca_var[1])}),
        ("lda_2d", lda_coords, {}),
    ]

    for space_name, X_space, extra in spaces:
        metrics.append(make_metric_row("global", "all", space_name, "condition", X_space, y, extra))
        metrics.append(make_metric_row("global", "all", space_name, "subject", X_space, subject_labels, extra))

    return metrics


def analyze_within_subject(X_log, y, meta, ch_names):
    """Analiza la separabilidad condicion a condicion dentro de cada sujeto."""
    metrics = []
    pca_subject_coords = {}
    lda_subject_coords = {}
    pca_subject_var = {}

    for subject_id in sorted(meta["subject_id"].unique()):
        mask = (meta["subject_id"] == subject_id).values
        X_subject = fit_feature_matrix(X_log[mask], y[mask], ch_names)
        y_subject = y[mask]

        pca_coords, pca_var = compute_pca_2d(X_subject)
        lda_coords = compute_lda_2d(X_subject, y_subject)

        pca_subject_coords[subject_id] = (pca_coords, y_subject)
        lda_subject_coords[subject_id] = (lda_coords, y_subject)
        pca_subject_var[subject_id] = pca_var

        metrics.append(make_metric_row(
            "within_subject",
            subject_id,
            "original",
            "condition",
            X_subject,
            y_subject,
        ))
        metrics.append(make_metric_row(
            "within_subject",
            subject_id,
            "pca_2d",
            "condition",
            pca_coords,
            y_subject,
            {"pc1_var": float(pca_var[0]), "pc2_var": float(pca_var[1])},
        ))
        metrics.append(make_metric_row(
            "within_subject",
            subject_id,
            "lda_2d",
            "condition",
            lda_coords,
            y_subject,
        ))

    plot_within_grid(
        pca_subject_coords,
        "PCA por sujeto - coloreado por condicion",
        "03_pca_within_subject.png",
    )
    plot_within_grid(
        lda_subject_coords,
        "LDA por sujeto - coloreado por condicion",
        "04_lda_within_subject.png",
    )
    plot_individual_subjects(pca_subject_coords, lda_subject_coords, pca_subject_var)

    return metrics


def save_outputs(metrics, feature_names):
    """Guarda metricas tabulares y resumen JSON."""
    metrics_df = pd.DataFrame(metrics)
    metrics_path = RESULTS_DIR / "separability_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  {metrics_path.name}")

    global_df = metrics_df[metrics_df["scope"] == "global"]
    within_df = metrics_df[metrics_df["scope"] == "within_subject"]

    summary = {
        "feature_config": {
            "selected_bands": SELECTED_BANDS,
            "excluded_bands": EXCLUDED_BANDS,
            "n_features": len(feature_names),
            "feature_blocks": feature_block_counts(feature_names),
            "normalization": "baseline NEUTRO + StandardScaler",
        },
        "global": {
            row["space"] + "_" + row["label_type"]: {
                "silhouette": None if pd.isna(row["silhouette"]) else float(row["silhouette"]),
                "centroid_ratio": None if pd.isna(row["centroid_ratio"]) else float(row["centroid_ratio"]),
            }
            for _, row in global_df.iterrows()
        },
        "within_subject_mean": {},
        "within_subject_top_lda": [],
    }

    for space in ["original", "pca_2d", "lda_2d"]:
        rows = within_df[within_df["space"] == space]
        summary["within_subject_mean"][space] = {
            "silhouette": float(rows["silhouette"].mean()),
            "centroid_ratio": float(rows["centroid_ratio"].mean()),
        }

    top_lda = within_df[within_df["space"] == "lda_2d"].sort_values("centroid_ratio", ascending=False)
    for _, row in top_lda.head(8).iterrows():
        summary["within_subject_top_lda"].append({
            "subject_id": row["subject_id"],
            "silhouette": float(row["silhouette"]),
            "centroid_ratio": float(row["centroid_ratio"]),
        })

    summary_path = RESULTS_DIR / "separability_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  {summary_path.name}")

    plot_summary(metrics_df)

    return metrics_df, summary


def print_console_summary(metrics_df):
    """Imprime un resumen corto para revisar en terminal."""
    print("\nRESUMEN GLOBAL")
    print("  Espacio             Etiqueta     silhouette   centroid_ratio")
    print("  ------------------------------------------------------------")
    global_df = metrics_df[metrics_df["scope"] == "global"]
    for _, row in global_df.iterrows():
        print(
            f"  {row['space']:<18} {row['label_type']:<10} "
            f"{row['silhouette']:>10.3f} {row['centroid_ratio']:>16.3f}"
        )

    print("\nTOP WITHIN-SUBJECT POR LDA")
    print("  Sujeto       silhouette   centroid_ratio")
    print("  ----------------------------------------")
    within_lda = metrics_df[
        (metrics_df["scope"] == "within_subject") &
        (metrics_df["space"] == "lda_2d")
    ].sort_values("centroid_ratio", ascending=False)
    for _, row in within_lda.head(10).iterrows():
        print(
            f"  {row['subject_id']:<12} "
            f"{row['silhouette']:>10.3f} {row['centroid_ratio']:>16.3f}"
        )


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("ANALISIS DE SEPARABILIDAD EEG!!!!!!!!!!")
    print(f"Features:  {FEATURES_DIR}")
    print(f"Resultados: {RESULTS_DIR}")
    print(f"Bandas:    {SELECTED_BANDS}")
    print(f"Excluye:   {EXCLUDED_BANDS}")

    X_log, y, meta, ch_names = load_dataset()

    if X_log.shape[1] != len(ch_names) * N_BANDS:
        raise ValueError(
            f"Shape inconsistente: X_log.shape[1]={X_log.shape[1]} "
            f"pero se esperaba {len(ch_names)} canales x {N_BANDS} bandas = {len(ch_names) * N_BANDS}."
        )

    print(f"\nEpocas: {len(y)}")
    print(f"Sujetos: {meta['subject_id'].nunique()}")
    print(f"Features base: {X_log.shape[1]}")
    feature_names = build_feature_names(ch_names)
    print(f"Features usadas: {len(feature_names)}")
    print(f"Bloques usados: {feature_block_counts(feature_names)}")

    print("\nGenerando figuras y metricas...")
    all_metrics = []
    all_metrics.extend(analyze_global(X_log, y, meta, ch_names))
    all_metrics.extend(analyze_within_subject(X_log, y, meta, ch_names))

    metrics_df, _ = save_outputs(all_metrics, feature_names)
    print_console_summary(metrics_df)

    print("\nTerminado - resultados en results/separability/")
