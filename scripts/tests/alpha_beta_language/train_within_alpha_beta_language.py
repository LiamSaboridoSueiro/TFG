"""
Test within-subject alpha_beta_language.

Modelo principal defendible para la memoria:
  - Alpha y Beta
  - sin Delta, Theta ni Gamma
  - sin eliminar canales temporales/frontales relevantes

Salida:
  scripts/tests/alpha_beta_language/results/within_subject_results.json
  scripts/tests/alpha_beta_language/results/within_subject_summary.csv
  scripts/tests/alpha_beta_language/results/selected_features.csv
  scripts/tests/alpha_beta_language/results/01_accuracy_<clf>.png
  scripts/tests/alpha_beta_language/results/02_confusion_<clf>.png
  scripts/tests/alpha_beta_language/results/03_importancia_<clf>.png
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from alpha_beta_language_common import (
    CONDITIONS,
    EXCLUDED_BANDS,
    FEATURES_DIR,
    LANGUAGE_CLUSTERS,
    N_FOLDS,
    RANDOM_STATE,
    RESULTS_DIR,
    SELECTED_BANDS,
    apply_normalization_pipeline,
    available_clusters,
    build_feature_names,
    describe_feature,
    feature_block_counts,
    feature_color,
    load_dataset,
)


warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------- Configuracion
N_TOP_FEAT = 20
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


# ---------------------------------------------------------------------- CV
def format_metric(mean_value, std_value):
    """Formatea media y desviacion para tablas."""
    return f"{mean_value:.3f} +/- {std_value:.2f}"


def evaluate_subject(X_log_subject, y_subject, clf, ch_names):
    """Evalua un sujeto con StratifiedKFold."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s = [], []
    y_true_all = []
    y_pred_all = []
    importances = []

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

        fold_clf = clone(clf)
        fold_clf.fit(X_train, y_train)
        y_pred = fold_clf.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        if hasattr(fold_clf, "coef_"):
            importances.append(np.abs(fold_clf.coef_).mean(axis=0))
        elif hasattr(fold_clf, "feature_importances_"):
            importances.append(fold_clf.feature_importances_)

    imp_mean = np.mean(importances, axis=0) if importances else None

    return (
        np.array(accs),
        np.array(f1s),
        np.array(y_true_all),
        np.array(y_pred_all),
        imp_mean,
    )


def classify_all_subjects(X_log, y, meta, ch_names):
    """Ejecuta todos los clasificadores por sujeto."""
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
                    "importancias": None,
                })
            continue

        print(f"  {subject_id:<{SUBJECT_COL_WIDTH}}", end="", flush=True)

        for name, clf in CLASIFICADORES.items():
            accs, f1s, y_true, y_pred, imp = evaluate_subject(
                X_log_subject,
                y_subject,
                clf,
                ch_names,
            )
            results[name].append({
                "subject_id": subject_id,
                "acc_media": accs.mean(),
                "acc_std": accs.std(),
                "f1_media": f1s.mean(),
                "f1_std": f1s.std(),
                "n_epocas": n_epochs,
                "y_true": y_true,
                "y_pred": y_pred,
                "importancias": imp,
            })
            print(f"  {format_metric(accs.mean(), accs.std()):>{METRIC_COL_WIDTH}}", end="", flush=True)

        print(f"  {n_epochs:>{N_EP_COL_WIDTH}}")

    return results


def classifier_summary(results):
    """Imprime resumen global por clasificador y devuelve el mejor."""
    print("=" * 75)
    print("RESUMEN TEST WITHIN-SUBJECT ALPHA + BETA LANGUAGE")
    print("=" * 75)
    print(f"\n{'Clasificador':<20} {'Acc media':>10} {'Acc std':>9} {'F1 macro':>10}  Mejora")
    print("-" * 75)

    best_name, best_acc = None, -np.inf

    for name, rows in results.items():
        accs = [r["acc_media"] for r in rows if not np.isnan(r["acc_media"])]
        f1s = [r["f1_media"] for r in rows if not np.isnan(r.get("f1_media", np.nan))]

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

    print(f"\nMejor clasificador: {best_name} ({best_acc:.3f})")
    return best_name


# ---------------------------------------------------------------------- Plots y guardado
def plot_accuracy(results, best_clf):
    """Genera figura de accuracy por sujeto."""
    rows = results[best_clf]
    subjects = [r["subject_id"].replace("211-000", "") for r in rows]
    accs = [r["acc_media"] if not np.isnan(r["acc_media"]) else 0 for r in rows]
    stds = [r["acc_std"] if not np.isnan(r["acc_std"]) else 0 for r in rows]

    colors = [
        "#e74c3c" if acc < 0.333 else
        "#f39c12" if acc < 0.60 else
        "#27ae60"
        for acc in accs
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#f8f9fa")
    acc_mean = np.nanmean(accs)
    fig.suptitle(
        f"TEST WITHIN-SUBJECT ALPHA + BETA LANGUAGE - {best_clf}\n"
        f"Accuracy media: {acc_mean:.3f}  (chance = 0.333)",
        fontsize=13,
        fontweight="bold",
    )

    axes[0].bar(subjects, accs, yerr=stds, color=colors, alpha=0.85, capsize=4)
    axes[0].axhline(0.333, color="red", ls="--", lw=1.5, label="chance")
    axes[0].axhline(acc_mean, color="navy", ls="-", lw=1.5, label=f"media ({acc_mean:.3f})")
    axes[0].set_title("Accuracy por sujeto (+/- std de 5 folds)")
    axes[0].set_xticks(range(len(subjects)))
    axes[0].set_xticklabels(subjects, rotation=90, fontsize=8)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].set_facecolor("#ffffff")

    accs_valid = [acc for acc in accs if acc > 0]
    axes[1].hist(accs_valid, bins=10, color="#3498db", alpha=0.7, edgecolor="white")
    axes[1].axvline(0.333, color="red", ls="--", lw=1.5, label="chance")
    axes[1].axvline(np.mean(accs_valid), color="navy", ls="-", lw=1.5, label=f"media={np.mean(accs_valid):.3f}")
    axes[1].set_title("Distribucion de accuracies")
    axes[1].set_xlabel("Accuracy")
    axes[1].set_ylabel("N sujetos")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_facecolor("#ffffff")

    plt.tight_layout()
    output_path = RESULTS_DIR / f"01_accuracy_{best_clf}.png"
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
    fig.suptitle(
        f"Matriz de confusion - {best_clf} - ALPHA + BETA LANGUAGE",
        fontsize=12,
        fontweight="bold",
    )

    for ax, cm, title in zip(axes, [cm_norm, cm_total], ["Normalizada", "Absoluta"]):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if title == "Normalizada" else None)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(CONDITIONS, fontsize=11)
        ax.set_yticklabels(CONDITIONS, fontsize=11)
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
    output_path = RESULTS_DIR / f"02_confusion_{best_clf}.png"
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
    with open(RESULTS_DIR / f"02_classification_report_{best_clf}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  02_classification_report_{best_clf}.json")


def plot_feature_importance(results, best_clf, feature_names):
    """Genera top de importancias para el mejor clasificador."""
    importances = [
        row["importancias"] for row in results[best_clf]
        if row["importancias"] is not None
    ]

    if not importances:
        print("  Importancia no disponible")
        return

    global_importance = np.mean(importances, axis=0)
    top_idx = np.argsort(global_importance)[::-1][:N_TOP_FEAT]
    top_names = [feature_names[i] for i in top_idx]
    top_values = global_importance[top_idx]
    colors = [feature_color(name) for name in top_names]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")
    y_pos = range(len(top_names))
    ax.barh(y_pos, top_values[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importancia media (promedio sujetos)")
    ax.set_title(f"Top {N_TOP_FEAT} features - {best_clf} - ALPHA + BETA LANGUAGE")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    output_path = RESULTS_DIR / f"03_importancia_{best_clf}.png"
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  {output_path.name}")


def save_selected_features(feature_names):
    """Guarda features usadas por la tanda."""
    rows = []
    for idx, feature_name in enumerate(feature_names):
        feature_type, channel, band, pair = describe_feature(feature_name)
        rows.append({
            "feature_idx_selected": idx,
            "feature_name": feature_name,
            "feature_type": feature_type,
            "channel": channel,
            "band": band,
            "pair": pair,
        })

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "selected_features.csv", index=False)
    print("  selected_features.csv")


def save_results(results, best_clf, feature_names, ch_names):
    """Guarda JSON y CSV resumen."""
    summary_rows = []
    json_summary = {
        "test_name": "alpha_beta_language",
        "title": "TEST WITHIN-SUBJECT ALPHA + BETA LANGUAGE",
        "description": "Entrenamiento within-subject con Alpha/Beta y clusters de lenguaje para interpretacion.",
        "selected_bands": SELECTED_BANDS,
        "excluded_bands": EXCLUDED_BANDS,
        "n_features_selected": len(feature_names),
        "selected_feature_blocks": feature_block_counts(feature_names),
        "language_clusters": LANGUAGE_CLUSTERS,
        "available_cluster_channels": available_clusters(ch_names),
        "cv": {
            "type": "StratifiedKFold",
            "n_splits": N_FOLDS,
            "scope": "within_subject",
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

    with open(RESULTS_DIR / "within_subject_results.json", "w") as f:
        json.dump(json_summary, f, indent=2)
    print("  within_subject_results.json")

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "within_subject_summary.csv", index=False)
    print("  within_subject_summary.csv")


# ---------------------------------------------------------------------- MAIN
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("TEST WITHIN-SUBJECT ALPHA + BETA LANGUAGE!!!!!!!!!!")
    print(f"  Features: {FEATURES_DIR}")
    print(f"  Salida:   {RESULTS_DIR}")
    print(f"  Bandas:   {SELECTED_BANDS}")
    print(f"  Excluye:  {EXCLUDED_BANDS}")
    print("  Clusters: temporal/frontal/parietal/occipital para resumen SHAP")

    print("\n  Cargando datos...")
    X_log, y, meta, ch_names = load_dataset()
    feature_names = build_feature_names(ch_names)

    print(f"  Epocas: {len(y)}")
    print(f"  Sujetos: {meta['subject_id'].nunique()}")
    print(f"  Features seleccionadas: {len(feature_names)}")
    print(f"  Bloques seleccionados: {feature_block_counts(feature_names)}")

    print("\n  Resultados por sujeto:")
    results = classify_all_subjects(X_log, y, meta, ch_names)
    best_clf = classifier_summary(results)

    print("\n  Guardando resultados...")
    save_selected_features(feature_names)
    save_results(results, best_clf, feature_names, ch_names)

    print("\n  Generando figuras...")
    plot_accuracy(results, best_clf)
    plot_confusion(results, best_clf)
    plot_feature_importance(results, best_clf, feature_names)

    print("\n" + "=" * 75)
    print("TEST COMPLETADO!!!!!!!!!!")
    print(f"Resultados en: {RESULTS_DIR}")
    print("=" * 75)
