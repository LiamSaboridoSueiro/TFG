"""
Script de entrenamiento all-subjects alpha_beta:

    StratifiedGroupKFold (5 folds, grupos=sujeto) -> busqueda de hiperparametros -> Clasificador global

    Clasificadores:
    - LogisticRegression elastic-net
    - DecisionTreeClassifier

Salida:
  results/all_subjects/all_subjects_results.json             metricas e hiperparametros por clasificador
  results/all_subjects/logReg/01_cv_scores.png                metricas CV regresion logistica
  results/all_subjects/logReg/02_confusion.png                matriz de confusion regresion logistica
  results/all_subjects/logReg/03_importancia.png              importancia de features regresion logistica
  results/all_subjects/logReg/04_coeficientes_modelo_final.png
  results/all_subjects/decisionTree/01_cv_scores.png          metricas CV arbol de decision
  results/all_subjects/decisionTree/02_confusion.png          matriz de confusion arbol de decision
  results/all_subjects/decisionTree/03_importancia.png        importancia de features arbol de decision
  results/all_subjects/decisionTree/04_arbol_modelo_final.png
  models/all_subjects/logReg/global_model.joblib              modelo final global regresion logistica
  models/all_subjects/decisionTree/global_model.joblib        modelo final global arbol de decision

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
import warnings
from pathlib import Path

import joblib
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.base import clone


warnings.filterwarnings("ignore")

# Permite ejecutar este script desde scripts/3_train importando utilidades
# compartidas desde scripts/.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils import (
    CONDITIONS,
    EXCLUDED_BANDS,
    FEATURES_DIR,
    LABEL_MAP,
    N_BANDS,
    N_FOLDS,
    RANDOM_STATE,
    SELECTED_BANDS,
    PROJECT_ROOT,
    apply_normalization_pipeline,
    build_feature_names,
    feature_block_counts,
    feature_color,
    fit_normalization_pipeline,
    load_dataset,
)


# ---------------------------------------------------------------------- Configuracion
RESULTS_DIR = PROJECT_ROOT / "results" / "all_subjects"
MODELS_DIR = PROJECT_ROOT / "models" / "all_subjects"

N_TOP_FEAT = 20       # n de features que se muestran en el grafico de importancia
SAVE_MODELS = True    # si True guarda un modelo final por clasificador
SCORING    = "f1_macro"

CLF_COL_WIDTH    = 20
METRIC_COL_WIDTH = 10

CLASIFICADORES = {
    "logReg": LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "decisionTree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
}

PARAM_GRIDS = {
    "logReg": {
        "C": [0.01, 0.1, 1.0],
        "l1_ratio": [0.15, 0.85],
    },
    "decisionTree": {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, None],
        "min_samples_leaf": [1, 3, 5],
    },
}


def format_metric(mean_value, std_value):
    """Formatea media y desviacion para tablas de consola."""
    return f"{mean_value:.3f} +/- {std_value:.2f}"


# ---------------------------------------------------------------------- Busqueda de hiperparametros

def evaluate_params(X_log, y, groups, ch_names, clf, params, n_folds=N_FOLDS):
    """
    Evalua un conjunto de hiperparametros con StratifiedGroupKFold.

    Los grupos son subject_id, asi que cada fold evalua sujetos no vistos.
    """
    sgkf = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    accs, f1s     = [], []
    y_true_all    = []
    y_pred_all    = []
    fold_subjects = []
    importancias  = []

    for train_idx, test_idx in sgkf.split(X_log, y, groups):
        X_log_train = X_log[train_idx]
        X_log_test  = X_log[test_idx]
        y_train     = y[train_idx]
        y_test      = y[test_idx]

        X_train, X_test = apply_normalization_pipeline(
            X_log_train, y_train,
            X_log_test,  ch_names
        )

        fold_clf = clone(clf)
        fold_clf.set_params(**params)
        fold_clf.fit(X_train, y_train)
        y_pred = fold_clf.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        fold_subjects.append(sorted(np.unique(groups[test_idx]).tolist()))

        if hasattr(fold_clf, "coef_"):
            importancias.append(np.abs(fold_clf.coef_).mean(axis=0))
        elif hasattr(fold_clf, "feature_importances_"):
            importancias.append(fold_clf.feature_importances_)

    imp_media = np.mean(importancias, axis=0) if importancias else None

    return {
        "params": params,
        "accs": np.array(accs),
        "f1s": np.array(f1s),
        "acc_media": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_media": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "y_true": np.array(y_true_all),
        "y_pred": np.array(y_pred_all),
        "fold_subjects": fold_subjects,
        "importancias": imp_media,
    }


def hyperparameter_search(X_log, y, meta, ch_names):
    """
    Busca hiperparametros por clasificador y devuelve el mejor resultado de cada uno.
    """
    groups = meta["subject_id"].values
    resultados = {}

    print("\n  Busqueda de hiperparametros por clasificador:")
    for nombre, clf in CLASIFICADORES.items():
        grid = list(ParameterGrid(PARAM_GRIDS[nombre]))
        print(f"\n  {nombre}: {len(grid)} combinaciones")

        candidatos = []
        for idx, params in enumerate(grid, start=1):
            r = evaluate_params(X_log, y, groups, ch_names, clf, params)
            candidatos.append(r)

            print(
                f"    [{idx:02d}/{len(grid):02d}] "
                f"F1={format_metric(r['f1_media'], r['f1_std']):>14}  "
                f"Acc={format_metric(r['acc_media'], r['acc_std']):>14}  "
                f"{params}",
                flush=True,
            )

        mejor = max(candidatos, key=lambda r: (r["f1_media"], r["acc_media"]))
        resultados[nombre] = {
            "best": mejor,
            "candidates": candidatos,
        }

        print(f"  Mejor {nombre}: F1={mejor['f1_media']:.3f}  Acc={mejor['acc_media']:.3f}")
        print(f"  Params: {mejor['params']}")

    return resultados


# ---------------------------------------------------------------------- Resumen y modelo final

def classifier_summary(resultados):

    print("=" * 76)
    print("RESUMEN GLOBAL ALL-SUBJECTS  (GroupKFold por sujeto, sin mezclar sujetos)")
    print("=" * 76)
    print(f"\n{'Clasificador':<{CLF_COL_WIDTH}} {'F1 macro':>{METRIC_COL_WIDTH}} {'Acc media':>{METRIC_COL_WIDTH}}  Mejora")
    print("-" * 76)

    mejor_nombre, mejor_f1 = None, -np.inf

    for nombre, data in resultados.items():
        best = data["best"]
        mejora = (best["acc_media"] - 0.333) * 100      # mejora sobre baseline aleatorio (3 clases -> 33.3%)
        barra  = "█" * int(best["f1_media"] * 35)

        print(
            f"{nombre:<{CLF_COL_WIDTH}} "
            f"{format_metric(best['f1_media'], best['f1_std']):>{METRIC_COL_WIDTH + 7}} "
            f"{format_metric(best['acc_media'], best['acc_std']):>{METRIC_COL_WIDTH + 7}}  "
            f"{mejora:+.1f}pp  {barra}"
        )

        if best["f1_media"] > mejor_f1:
            mejor_f1, mejor_nombre = best["f1_media"], nombre

    best = resultados[mejor_nombre]["best"]
    print(f"\nMejor clasificador global: {mejor_nombre} (F1 macro={best['f1_media']:.3f}, Acc={best['acc_media']:.3f})")
    print(f"Mejores hiperparametros: {best['params']}")
    return mejor_nombre


def save_final_models(X_log, y, meta, ch_names, resultados):
    """Entrena y guarda un modelo final por clasificador."""
    X_train, preprocessing = fit_normalization_pipeline(X_log, y, ch_names)
    trained_models = {}

    for clf_name, data in resultados.items():
        best_params = data["best"]["params"]
        clf = clone(CLASIFICADORES[clf_name])
        clf.set_params(**best_params)
        clf.fit(X_train, y)

        trained_models[clf_name] = {
            "classifier": clf,
            "preprocessing": preprocessing,
            "best_params": best_params,
        }

        if not SAVE_MODELS:
            continue

        output_path = MODELS_DIR / clf_name / "global_model.joblib"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifier": clf,
                "preprocessing": preprocessing,
                "classifier_name": clf_name,
                "best_params": best_params,
                "ch_names_eeg": ch_names,
                "feature_names": build_feature_names(ch_names),
                "selected_bands": SELECTED_BANDS,
                "excluded_bands": EXCLUDED_BANDS,
                "label_map": LABEL_MAP,
                "conditions": CONDITIONS,
                "training_subjects": sorted(meta["subject_id"].unique().tolist()),
                "scoring": SCORING,
                "cv": {
                    "type": "StratifiedGroupKFold",
                    "n_splits": N_FOLDS,
                    "group": "subject_id",
                },
            },
            output_path,
        )
        print(f"  Modelo final global {clf_name}: {output_path}")

    return trained_models


# ---------------------------------------------------------------------- PLOTS

def plot_cv_scores(resultados, mejor_clf):
    datos = resultados[mejor_clf]["best"]
    folds = [f"Fold {i + 1}" for i in range(len(datos["accs"]))]
    output_dir = RESULTS_DIR / mejor_clf
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        f"All-subjects Alpha/Beta - {mejor_clf}\n"
        f"F1 macro medio: {datos['f1_media']:.3f}  |  Accuracy media: {datos['acc_media']:.3f}",
        fontsize=13,
        fontweight="bold",
    )

    axes[0].bar(folds, datos["accs"], color="#3498db", alpha=0.85)
    axes[0].axhline(0.333, color="red", ls="--", lw=1.5, label="chance")
    axes[0].axhline(datos["acc_media"], color="navy", lw=1.5, label=f"media ({datos['acc_media']:.3f})")
    axes[0].set_title("Accuracy por fold")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].set_facecolor("#ffffff")

    axes[1].bar(folds, datos["f1s"], color="#27ae60", alpha=0.85)
    axes[1].axhline(datos["f1_media"], color="navy", lw=1.5, label=f"media ({datos['f1_media']:.3f})")
    axes[1].set_title("F1 macro por fold")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("F1 macro")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_facecolor("#ffffff")

    plt.tight_layout()
    fname = output_dir / "01_cv_scores.png"
    plt.savefig(fname, dpi=120)
    print(f"  {fname.relative_to(RESULTS_DIR)}")
    plt.show()


def plot_confusion(resultados, mejor_clf):
    y_true = resultados[mejor_clf]["best"]["y_true"]
    y_pred = resultados[mejor_clf]["best"]["y_pred"]
    output_dir = RESULTS_DIR / mejor_clf
    output_dir.mkdir(parents=True, exist_ok=True)

    cm_total = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm_total.astype(float) / cm_total.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        f"Matriz de confusion - {mejor_clf} all-subjects Alpha/Beta",
        fontsize=12,
        fontweight="bold",
    )
    for ax, cm, titulo in zip(
        axes, [cm_norm, cm_total],
        ["Normalizada", "Absoluta (n epocas)"]
    ):
        im = ax.imshow(cm, cmap="Blues",
                       vmin=0, vmax=1 if "Norm" in titulo else None)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CONDITIONS, fontsize=11)
        ax.set_yticklabels(CONDITIONS, fontsize=11)
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
        ax.set_title(titulo)
        plt.colorbar(im, ax=ax)
        for i in range(3):
            for j in range(3):
                val   = f"{cm[i,j]:.2f}" if "Norm" in titulo else str(cm[i,j])
                color = "white" if cm_norm[i,j] > 0.5 else "black"
                ax.text(j, i, val, ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")

    plt.tight_layout()
    fname = output_dir / "02_confusion.png"
    plt.savefig(fname, dpi=120)
    print(f"  {fname.relative_to(RESULTS_DIR)}")
    plt.show()

    print("\n  Reporte de clasificacion:")
    print(classification_report(y_true, y_pred,
                                target_names=CONDITIONS, digits=3))


def plot_feature_importance(resultados, mejor_clf, ch_names, n_top=N_TOP_FEAT):
    imp_global = resultados[mejor_clf]["best"]["importancias"]
    if imp_global is None:
        print("  Importancia no disponible")
        return
    output_dir = RESULTS_DIR / mejor_clf
    output_dir.mkdir(parents=True, exist_ok=True)

    feat_names = build_feature_names(ch_names)

    # Truncar si hay discrepancia de longitud
    n = min(len(imp_global), len(feat_names))
    imp_global = imp_global[:n]
    feat_names = feat_names[:n]

    top_idx   = np.argsort(imp_global)[::-1][:n_top]
    top_names = [feat_names[i] for i in top_idx]
    top_vals  = imp_global[top_idx]
    colores   = [feature_color(nombre) for nombre in top_names]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")
    y_pos = range(len(top_names))
    ax.barh(y_pos, top_vals[::-1], color=colores[::-1], alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importancia media (promedio folds)")
    ax.set_title(f"Top {n_top} features - {mejor_clf}")
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#f1c40f", label="Alpha"),
        Patch(facecolor="#2ecc71", label="Beta"),
        Patch(facecolor="#34495e", label="AlphaAsym"),
    ], fontsize=8, loc="lower right")

    plt.tight_layout()
    fname = output_dir / "03_importancia.png"
    plt.savefig(fname, dpi=120)
    print(f"  {fname.relative_to(RESULTS_DIR)}")
    plt.show()


def plot_final_logreg_coefficients(trained_models, ch_names, n_top=N_TOP_FEAT):
    """Grafica los coeficientes medios absolutos del modelo final logReg."""
    model_data = trained_models.get("logReg") if trained_models else None
    if model_data is None:
        return

    clf = model_data["classifier"]
    if not hasattr(clf, "coef_"):
        return

    output_dir = RESULTS_DIR / "logReg"
    output_dir.mkdir(parents=True, exist_ok=True)

    feat_names = build_feature_names(ch_names)
    coef = np.asarray(clf.coef_, dtype=float)
    importance = np.abs(coef).mean(axis=0)

    n = min(len(importance), len(feat_names))
    importance = importance[:n]
    feat_names = feat_names[:n]

    top_idx = np.argsort(importance)[::-1][:n_top]
    top_names = [feat_names[i] for i in top_idx]
    top_vals = importance[top_idx]
    colores = [feature_color(nombre) for nombre in top_names]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")
    y_pos = range(len(top_names))
    ax.barh(y_pos, top_vals[::-1], color=colores[::-1], alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Coeficiente medio absoluto")
    ax.set_title(f"Top {n_top} coeficientes - modelo final logReg")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    fname = output_dir / "04_coeficientes_modelo_final.png"
    plt.savefig(fname, dpi=140)
    print(f"  {fname.relative_to(RESULTS_DIR)}")
    plt.show()


def plot_final_decision_tree(trained_models, ch_names):
    """Grafica los primeros niveles del arbol de decision final."""
    model_data = trained_models.get("decisionTree") if trained_models else None
    if model_data is None:
        return

    clf = model_data["classifier"]
    output_dir = RESULTS_DIR / "decisionTree"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        clf,
        feature_names=build_feature_names(ch_names),
        class_names=CONDITIONS,
        filled=True,
        rounded=True,
        max_depth=4,
        fontsize=7,
        proportion=True,
        ax=ax,
    )
    ax.set_title("Primeros niveles del modelo final decisionTree", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = output_dir / "04_arbol_modelo_final.png"
    plt.savefig(fname, dpi=160)
    print(f"  {fname.relative_to(RESULTS_DIR)}")
    plt.show()


# ---------------------------------------------------------------------- Guardado JSON

def json_safe_params(params):
    safe = {}
    for k, v in params.items():
        if isinstance(v, np.generic):
            safe[k] = v.item()
        else:
            safe[k] = v
    return safe


def save_selected_features(ch_names):
    feature_names = build_feature_names(ch_names)
    out = RESULTS_DIR / "selected_features.csv"
    pd.DataFrame({"feature_name": feature_names}).to_csv(out, index=False)
    print(f"  Features: {out.name}")


def save_json(resultados, mejor_clf, ch_names):
    feature_names = build_feature_names(ch_names)
    resumen = {
        "test_name": "alpha_beta_all_subjects",
        "cv": {
            "type": "StratifiedGroupKFold",
            "n_splits": N_FOLDS,
            "group": "subject_id",
            "scoring": SCORING,
        },
        "selected_bands": SELECTED_BANDS,
        "excluded_bands": EXCLUDED_BANDS,
        "n_features_selected": len(feature_names),
        "selected_feature_blocks": feature_block_counts(feature_names),
        "mejor_clasificador": mejor_clf,
        "clasificadores": {},
    }

    for nombre, data in resultados.items():
        best = data["best"]
        resumen["clasificadores"][nombre] = {
            "best_params": json_safe_params(best["params"]),
            "acc_media": float(best["acc_media"]),
            "acc_std": float(best["acc_std"]),
            "f1_media": float(best["f1_media"]),
            "f1_std": float(best["f1_std"]),
            "accs_por_fold": [float(v) for v in best["accs"]],
            "f1s_por_fold": [float(v) for v in best["f1s"]],
            "fold_subjects": best["fold_subjects"],
            "candidates": [
                {
                    "params": json_safe_params(c["params"]),
                    "acc_media": float(c["acc_media"]),
                    "acc_std": float(c["acc_std"]),
                    "f1_media": float(c["f1_media"]),
                    "f1_std": float(c["f1_std"]),
                }
                for c in data["candidates"]
            ],
        }

    out = RESULTS_DIR / "all_subjects_results.json"
    with open(out, "w") as f:
        json.dump(resumen, f, indent=2)
    print(f"\n  Resultados: {out.name}")


# ---------------------------------------------------------------------- MAIN
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("CLASIFICACION ALL-SUBJECTS Alpha/Beta!!!!!!!!!!!!!!!!")
    print(f"  Estrategia: StratifiedGroupKFold {N_FOLDS}-Fold por sujeto")
    print(f"  Scoring busqueda: {SCORING}")
    print(f"  Bandas: {SELECTED_BANDS}  |  Excluidas: {EXCLUDED_BANDS}")
    print(f"  Chance level: 0.333  (3 clases)        :o")

    # Cargar datos en escala log (sin normalizar)
    print("\n  Cargando datos...")
    X_log, y, meta, ch_names = load_dataset()

    # Verificar que el shape de X_log es coherente con los canales y bandas
    if len(ch_names) > 0 and X_log.shape[1] != len(ch_names) * N_BANDS:
        raise ValueError(
            f"Shape inconsistente: X_log.shape[1]={X_log.shape[1]} "
            f"pero se esperaba {len(ch_names)} canales x {N_BANDS} bandas = {len(ch_names) * N_BANDS}. "
            f"Vuelve a ejecutar epochs_to_features.py"
        )

    sujetos = sorted(meta["subject_id"].unique())
    feature_names = build_feature_names(ch_names)
    print(f"  Sujetos: {len(sujetos)}")
    print(f"  Epocas:  {len(y)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Bloques: {feature_block_counts(feature_names)}")
    print(f"  Clasificadores: {list(CLASIFICADORES.keys())}")

    resultados = hyperparameter_search(X_log, y, meta, ch_names)
    mejor_clf = classifier_summary(resultados)

    print("\n  Guardando modelos finales...")
    save_final_models(X_log, y, meta, ch_names, resultados)

    print("\n  Generando figuras por clasificador...")
    for clf_name in resultados:
        plot_cv_scores(resultados, clf_name)
        plot_confusion(resultados, clf_name)
        plot_feature_importance(resultados, clf_name, ch_names)
    save_selected_features(ch_names)
    save_json(resultados, mejor_clf, ch_names)

    # Tabla comparativa
    print("TABLA COMPARATIVA GLOBAL!!!!!")
    print(f"\n  {'Clasificador':<{CLF_COL_WIDTH}} {'F1 macro':>{METRIC_COL_WIDTH}} {'Acc media':>{METRIC_COL_WIDTH}}  Params")
    print("  " + "-" * 92)

    for nombre, data in resultados.items():
        best = data["best"]
        print(
            f"  {nombre:<{CLF_COL_WIDTH}} {best['f1_media']:>{METRIC_COL_WIDTH}.3f} {best['acc_media']:>{METRIC_COL_WIDTH}.3f}  "
            f"{best['params']}"
        )

    print("Terminado - resultados en results/all_subjects/")
