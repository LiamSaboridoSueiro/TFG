"""
Control temporal para alpha_beta_language.

Objetivo:
  comprobar si el rendimiento alto del CV aleatorio puede estar inflado por
  autocorrelacion entre epocas cercanas.

Estrategia:
  para cada sujeto y condicion, ordenar por epoch_idx y dividir en 5 tramos
  contiguos. En el fold k se testea el tramo k de JOY, NEUTRO y SAD, y se
  entrena con el resto.

Salida:
  scripts/tests/alpha_beta_language/results_temporal_cv/temporal_cv_results.json
  scripts/tests/alpha_beta_language/results_temporal_cv/temporal_cv_summary.csv
  scripts/tests/alpha_beta_language/results_temporal_cv/01_temporal_accuracy_<clf>.png
  scripts/tests/alpha_beta_language/results_temporal_cv/02_temporal_confusion_<clf>.png
  scripts/tests/alpha_beta_language/results_temporal_cv/03_random_vs_temporal_<clf>.png
"""

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.svm import SVC

from alpha_beta_language_common import (
    CONDITIONS,
    EXCLUDED_BANDS,
    FEATURES_DIR,
    LABEL_MAP,
    N_FOLDS,
    RANDOM_STATE,
    RESULTS_DIR,
    SELECTED_BANDS,
    TEST_DIR,
    apply_normalization_pipeline,
    build_feature_names,
    feature_block_counts,
    load_dataset,
)


warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------- Configuracion
TEMPORAL_RESULTS_DIR = TEST_DIR / "results_temporal_cv"

SUBJECT_COL_WIDTH = 18
METRIC_COL_WIDTH = 18
N_EP_COL_WIDTH = 6

CLASIFICADORES = {
    "LogReg": LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=1.0,
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "SVM_RBF": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
}


# ---------------------------------------------------------------------- Folds temporales
def format_metric(mean_value, std_value):
    """Formatea media y desviacion para tablas."""
    return f"{mean_value:.3f} +/- {std_value:.2f}"


def build_temporal_stratified_folds(meta_subject, y_subject, n_splits=N_FOLDS):
    """
    Construye folds con test contiguo dentro de cada clase.

    No usa shuffle. Cada fold contiene un tramo temporal de cada condicion para
    mantener las tres clases en test.
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


def evaluate_subject_temporal(X_log_subject, y_subject, meta_subject, clf, ch_names):
    """Evalua un sujeto con folds temporales contiguos."""
    folds = build_temporal_stratified_folds(meta_subject, y_subject)
    if not folds:
        return None

    accs, f1s = [], []
    y_true_all = []
    y_pred_all = []

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

        fold_clf = clone(clf)
        fold_clf.fit(X_train, y_train)
        y_pred = fold_clf.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    return (
        np.array(accs),
        np.array(f1s),
        np.array(y_true_all),
        np.array(y_pred_all),
    )


def classify_all_subjects_temporal(X_log, y, meta, ch_names):
    """Ejecuta clasificadores usando folds temporales."""
    subjects = sorted(meta["subject_id"].unique())
    results = {name: [] for name in CLASIFICADORES}

    print(f"\n  {'Sujeto':<{SUBJECT_COL_WIDTH}}", end="")
    for name in CLASIFICADORES:
        print(f"  {name:>{METRIC_COL_WIDTH}}", end="")
    print(f"  {'N_ep':>{N_EP_COL_WIDTH}}")
    print("  " + "-" * (SUBJECT_COL_WIDTH + (METRIC_COL_WIDTH + 2) * len(CLASIFICADORES) + N_EP_COL_WIDTH + 2))

    for subject_id in subjects:
        mask = (meta["subject_id"] == subject_id).values
        X_log_subject = X_log[mask]
        y_subject = y[mask]
        meta_subject = meta.loc[mask].reset_index(drop=True)
        n_epochs = len(y_subject)

        classes, counts = np.unique(y_subject, return_counts=True)
        min_class = counts.min()
        if len(classes) < len(CONDITIONS) or min_class < N_FOLDS:
            print(f"  {subject_id:<{SUBJECT_COL_WIDTH}}  saltando ({min_class} epocas en clase minima)")
            for name in CLASIFICADORES:
                results[name].append({
                    "subject_id": subject_id,
                    "acc_media": np.nan,
                    "acc_std": np.nan,
                    "f1_media": np.nan,
                    "f1_std": np.nan,
                    "n_epocas": n_epochs,
                    "y_true": np.array([]),
                    "y_pred": np.array([]),
                })
            continue

        print(f"  {subject_id:<{SUBJECT_COL_WIDTH}}", end="", flush=True)

        for name, clf in CLASIFICADORES.items():
            evaluated = evaluate_subject_temporal(
                X_log_subject,
                y_subject,
                meta_subject,
                clf,
                ch_names,
            )
            if evaluated is None:
                results[name].append({
                    "subject_id": subject_id,
                    "acc_media": np.nan,
                    "acc_std": np.nan,
                    "f1_media": np.nan,
                    "f1_std": np.nan,
                    "n_epocas": n_epochs,
                    "y_true": np.array([]),
                    "y_pred": np.array([]),
                })
                print(f"  {'-':>{METRIC_COL_WIDTH}}", end="", flush=True)
                continue

            accs, f1s, y_true, y_pred = evaluated
            results[name].append({
                "subject_id": subject_id,
                "acc_media": accs.mean(),
                "acc_std": accs.std(),
                "f1_media": f1s.mean(),
                "f1_std": f1s.std(),
                "n_epocas": n_epochs,
                "y_true": y_true,
                "y_pred": y_pred,
            })
            print(f"  {format_metric(accs.mean(), accs.std()):>{METRIC_COL_WIDTH}}", end="", flush=True)

        print(f"  {n_epochs:>{N_EP_COL_WIDTH}}")

    return results


def classifier_summary(results):
    """Imprime resumen global por clasificador."""
    print("=" * 75)
    print("RESUMEN TEMPORAL CV ALPHA + BETA LANGUAGE")
    print("=" * 75)
    print(f"\n{'Clasificador':<20} {'Acc media':>10} {'Acc std':>9} {'F1 macro':>10}  Mejora")
    print("-" * 75)

    best_name, best_acc = None, -np.inf
    for name, rows in results.items():
        accs = [row["acc_media"] for row in rows if not np.isnan(row["acc_media"])]
        f1s = [row["f1_media"] for row in rows if not np.isnan(row.get("f1_media", np.nan))]
        if not accs:
            continue

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_f1 = np.mean(f1s) if f1s else np.nan
        improvement = (mean_acc - 0.333) * 100
        bar = "#" * int(mean_acc * 35)
        print(
            f"{name:<20} {mean_acc:>10.3f} {std_acc:>9.3f} "
            f"{mean_f1:>10.3f}  {improvement:+.1f}pp  {bar}"
        )
        if mean_acc > best_acc:
            best_acc, best_name = mean_acc, name

    print(f"\nMejor clasificador temporal: {best_name} ({best_acc:.3f})")
    return best_name


# ---------------------------------------------------------------------- Figuras y guardado
def plot_accuracy(results, best_clf):
    """Genera figura de accuracy temporal por sujeto."""
    rows = results[best_clf]
    subjects = [row["subject_id"].replace("211-000", "") for row in rows]
    accs = [row["acc_media"] if not np.isnan(row["acc_media"]) else 0 for row in rows]
    stds = [row["acc_std"] if not np.isnan(row["acc_std"]) else 0 for row in rows]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")
    colors = ["#27ae60" if acc >= 0.60 else "#f39c12" if acc >= 0.333 else "#e74c3c" for acc in accs]
    mean_acc = np.nanmean(accs)

    ax.bar(subjects, accs, yerr=stds, color=colors, alpha=0.85, capsize=4)
    ax.axhline(0.333, color="red", ls="--", lw=1.5, label="chance")
    ax.axhline(mean_acc, color="navy", ls="-", lw=1.5, label=f"media ({mean_acc:.3f})")
    ax.set_title(f"Temporal CV por sujeto - {best_clf} - Alpha/Beta")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=90, fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = TEMPORAL_RESULTS_DIR / f"01_temporal_accuracy_{best_clf}.png"
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  {output_path.name}")


def plot_confusion(results, best_clf):
    """Genera matriz de confusion agregada."""
    cm_total = np.zeros((3, 3), dtype=int)
    for row in results[best_clf]:
        if len(row["y_true"]) == 0:
            continue
        cm_total += confusion_matrix(row["y_true"], row["y_pred"], labels=[0, 1, 2])

    cm_norm = cm_total.astype(float) / cm_total.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(f"Matriz de confusion temporal - {best_clf}", fontsize=12, fontweight="bold")

    for ax, cm, title in zip(axes, [cm_norm, cm_total], ["Normalizada", "Absoluta"]):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if title == "Normalizada" else None)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(CONDITIONS)
        ax.set_yticklabels(CONDITIONS)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        for i in range(3):
            for j in range(3):
                value = f"{cm[i, j]:.2f}" if title == "Normalizada" else str(cm[i, j])
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, value, ha="center", va="center", fontsize=11, color=color, fontweight="bold")

    plt.tight_layout()
    output_path = TEMPORAL_RESULTS_DIR / f"02_temporal_confusion_{best_clf}.png"
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  {output_path.name}")

    y_true_all = np.concatenate([row["y_true"] for row in results[best_clf] if len(row["y_true"]) > 0])
    y_pred_all = np.concatenate([row["y_pred"] for row in results[best_clf] if len(row["y_pred"]) > 0])
    report = classification_report(
        y_true_all,
        y_pred_all,
        target_names=CONDITIONS,
        digits=3,
        output_dict=True,
    )
    with open(TEMPORAL_RESULTS_DIR / f"02_temporal_classification_report_{best_clf}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  02_temporal_classification_report_{best_clf}.json")


def plot_random_vs_temporal(results, best_clf):
    """Compara el CV aleatorio ya guardado con este control temporal."""
    random_path = RESULTS_DIR / "within_subject_summary.csv"
    if not random_path.exists():
        print("  Comparativa random_vs_temporal no disponible")
        return

    random_df = pd.read_csv(random_path)
    random_df = random_df[random_df["classifier"] == best_clf]
    temporal_rows = [
        {
            "subject_id": row["subject_id"],
            "acc_temporal": row["acc_media"],
        }
        for row in results[best_clf]
        if not np.isnan(row["acc_media"])
    ]
    temporal_df = pd.DataFrame(temporal_rows)
    comp = random_df.merge(temporal_df, on="subject_id", how="inner")
    if len(comp) == 0:
        print("  Comparativa random_vs_temporal vacia")
        return

    comp["delta_random_minus_temporal"] = comp["acc_media"] - comp["acc_temporal"]
    comp.to_csv(TEMPORAL_RESULTS_DIR / f"random_vs_temporal_{best_clf}.csv", index=False)

    subjects = [sid.replace("211-000", "") for sid in comp["subject_id"]]
    x = np.arange(len(comp))
    width = 0.38

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")
    ax.bar(x - width / 2, comp["acc_media"], width=width, label="CV aleatorio", color="#3498db", alpha=0.85)
    ax.bar(x + width / 2, comp["acc_temporal"], width=width, label="CV temporal", color="#e67e22", alpha=0.85)
    ax.axhline(0.333, color="red", ls="--", lw=1.5, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=90, fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"CV aleatorio vs CV temporal - {best_clf}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = TEMPORAL_RESULTS_DIR / f"03_random_vs_temporal_{best_clf}.png"
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  {output_path.name}")


def save_results(results, best_clf, feature_names):
    """Guarda JSON y CSV resumen."""
    summary_rows = []
    json_summary = {
        "test_name": "alpha_beta_language_temporal_cv",
        "title": "TEMPORAL CV ALPHA + BETA LANGUAGE",
        "description": (
            "Control de leakage temporal: folds por tramos contiguos de epoch_idx "
            "dentro de cada condicion y sujeto."
        ),
        "selected_bands": SELECTED_BANDS,
        "excluded_bands": EXCLUDED_BANDS,
        "n_features_selected": len(feature_names),
        "selected_feature_blocks": feature_block_counts(feature_names),
        "cv": {
            "type": "contiguous_condition_chunks",
            "n_splits": N_FOLDS,
            "scope": "within_subject",
            "shuffle": False,
            "group_source": "condition + epoch_idx",
        },
        "best_classifier": best_clf,
        "classifiers": {},
    }

    for name, rows in results.items():
        accs = [row["acc_media"] for row in rows if not np.isnan(row["acc_media"])]
        f1s = [row["f1_media"] for row in rows if not np.isnan(row.get("f1_media", np.nan))]
        json_summary["classifiers"][name] = {
            "acc_media_global": float(np.nanmean(accs)) if accs else None,
            "acc_std_global": float(np.nanstd(accs)) if accs else None,
            "f1_media_global": float(np.nanmean(f1s)) if f1s else None,
            "por_sujeto": [],
        }

        for row in rows:
            item = {
                "classifier": name,
                "subject_id": row["subject_id"],
                "acc_media": float(row["acc_media"]) if not np.isnan(row["acc_media"]) else None,
                "acc_std": float(row["acc_std"]) if not np.isnan(row["acc_std"]) else None,
                "f1_media": float(row.get("f1_media", np.nan)) if not np.isnan(row.get("f1_media", np.nan)) else None,
                "f1_std": float(row.get("f1_std", np.nan)) if not np.isnan(row.get("f1_std", np.nan)) else None,
                "n_epocas": int(row["n_epocas"]),
            }
            json_summary["classifiers"][name]["por_sujeto"].append({
                k: v for k, v in item.items() if k != "classifier"
            })
            summary_rows.append(item)

    with open(TEMPORAL_RESULTS_DIR / "temporal_cv_results.json", "w") as f:
        json.dump(json_summary, f, indent=2)
    print("  temporal_cv_results.json")

    pd.DataFrame(summary_rows).to_csv(TEMPORAL_RESULTS_DIR / "temporal_cv_summary.csv", index=False)
    print("  temporal_cv_summary.csv")


# ---------------------------------------------------------------------- MAIN
if __name__ == "__main__":
    TEMPORAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("TEMPORAL CV ALPHA + BETA LANGUAGE!!!!!!!!!!")
    print(f"  Features: {FEATURES_DIR}")
    print(f"  Salida:   {TEMPORAL_RESULTS_DIR}")
    print(f"  Bandas:   {SELECTED_BANDS}")
    print("  Split:    tramos contiguos por condicion y sujeto, sin shuffle")

    print("\n  Cargando datos...")
    X_log, y, meta, ch_names = load_dataset()
    feature_names = build_feature_names(ch_names)

    print(f"  Epocas: {len(y)}")
    print(f"  Sujetos: {meta['subject_id'].nunique()}")
    print(f"  Features seleccionadas: {len(feature_names)}")
    print(f"  Bloques seleccionados: {feature_block_counts(feature_names)}")

    print("\n  Resultados por sujeto:")
    results = classify_all_subjects_temporal(X_log, y, meta, ch_names)
    best_clf = classifier_summary(results)

    print("\n  Guardando resultados...")
    save_results(results, best_clf, feature_names)

    print("\n  Generando figuras...")
    plot_accuracy(results, best_clf)
    plot_confusion(results, best_clf)
    plot_random_vs_temporal(results, best_clf)

    print("\n" + "=" * 75)
    print("CONTROL TEMPORAL COMPLETADO!!!!!!!!!!")
    print(f"Resultados en: {TEMPORAL_RESULTS_DIR}")
    print("=" * 75)
