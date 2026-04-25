"""
Script de entrenamiento all-subjects:

    StratifiedGroupKFold (5 folds, grupos=sujeto) -> búsqueda de hiperparámetros -> Clasificador global

    Clasificadores:
    - LogisticRegression elastic-net  (interpretable con SHAP)
    - Random Forest
    - SVM RBF

Salida:
  results/all_subjects/all_subjects_results.json             métricas e hiperparámetros por clasificador
  results/all_subjects/01_cv_scores_<clf>.png                 métricas CV del mejor clasificador
  results/all_subjects/02_confusion_<clf>.png                 matriz de confusión agregada
  results/all_subjects/03_importancia_<clf>.png               importancia de features
  models/all_subjects/global_model.joblib                     modelo final global

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import warnings
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.base import clone


warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------- Configuración
FEATURES_DIR = Path("data/features")
RESULTS_DIR  = Path("results/all_subjects")
FINAL_MODEL_PATH = Path("models/all_subjects/global_model.joblib")
CONDITIONS   = ["JOY", "NEUTRO", "SAD"]
LABEL_MAP    = {"JOY": 0, "NEUTRO": 1, "SAD": 2}

LABEL_INV = {}
for label, cond in LABEL_MAP.items():
    LABEL_INV[cond] = label

N_FOLDS      = 5
RANDOM_STATE = 42       # semilla de aleatoriedad para reproducibilidad
N_TOP_FEAT   = 20       # n de features que se muestran en el gráfico de importancia 20/398
SAVE_MODEL   = True     # si True guarda el mejor modelo final entrenado con todos los sujetos
SCORING      = "f1_macro"

CLASIFICADORES = {
    "LogReg": LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "RandomForest": RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "SVM_RBF": SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
}

PARAM_GRIDS = {
    "LogReg": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.15, 0.5, 0.85],
    },
    "RandomForest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt"],
    },
    "SVM_RBF": {
        "C": [0.1, 1.0, 10.0, 30.0],
        "gamma": ["scale", 0.01, 0.001],
    },
}

# Bandas y configuración de features derivadas
BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

N_BANDS  = len(BANDS)
BAND_IDX = {}
for i, b in enumerate(BANDS):
    BAND_IDX[b] = i


INCLUDE_THETA_ALPHA_RATIO = True
INCLUDE_ALPHA_ASYMMETRY   = True

ALPHA_ASYMMETRY_PAIRS = [
    ("Fp1","Fp2"), ("AF7","AF8"), ("AF3","AF4"),
    ("F7","F8"),   ("F5","F6"),   ("F3","F4"),   ("F1","F2"),
    ("FC5","FC6"), ("FC3","FC4"), ("FC1","FC2"),
    ("T7","T8"),   ("C5","C6"),   ("C3","C4"),   ("C1","C2"),
    ("TP7","TP8"), ("CP5","CP6"), ("CP3","CP4"), ("CP1","CP2"),
    ("P7","P8"),   ("P5","P6"),   ("P3","P4"),   ("P1","P2"),
    ("PO7","PO8"), ("PO5","PO6"), ("PO3","PO4"), ("O1","O2"),
]


# ---------------------------------------------------------------------- Carga de datos en escala log

def load_dataset():
    log_path = FEATURES_DIR / "features_X.npy"

    if not log_path.exists():
        raise FileNotFoundError(
            f"No se encontró {log_path}. Ejecuta primero epochs_to_features.py"
        )

    X_log    = np.load(FEATURES_DIR / "features_X.npy")
    y        = np.load(FEATURES_DIR / "features_y.npy")
    meta     = pd.read_csv(FEATURES_DIR / "features_meta.csv")
    with open(FEATURES_DIR / "features_info.json") as f:
        info = json.load(f)
    ch_names = info.get("ch_names_eeg", [])
    if not ch_names:
        raise ValueError(
            "features_info.json no contiene 'ch_names_eeg'. "
            "Vuelve a ejecutar epochs_to_features.py con el guardado de canales actualizado."
        )

    return X_log, y, meta, ch_names


# ---------------------------------------------------------------------- Normalización dentro del fold
def _build_feature_matrix(delta_2d, ch_names):
    """Añade las features derivadas al bloque bandpower ya normalizado."""

    n_ch = len(ch_names)
    delta_3d = delta_2d.reshape(len(delta_2d), n_ch, N_BANDS)

    bloques = [delta_2d]

    if INCLUDE_THETA_ALPHA_RATIO and ch_names:
        theta_idx = BAND_IDX["Theta"]
        alpha_idx = BAND_IDX["Alpha"]
        ratio = delta_3d[:, :, theta_idx] - delta_3d[:, :, alpha_idx]
        bloques.append(ratio)

    if INCLUDE_ALPHA_ASYMMETRY and ch_names:
        ch_idx_map = {}
        for i, ch in enumerate(ch_names):
            ch_idx_map[ch] = i

        pares_validos = []
        for l, r in ALPHA_ASYMMETRY_PAIRS:
            if l in ch_idx_map and r in ch_idx_map:
                pares_validos.append((l, r, ch_idx_map[l], ch_idx_map[r]))

        if pares_validos:
            alpha_idx = BAND_IDX["Alpha"]
            asym_cols = []
            for _, _, li, ri in pares_validos:
                asym_cols.append(delta_3d[:, ri, alpha_idx] - delta_3d[:, li, alpha_idx])
            bloques.append(np.column_stack(asym_cols))

    return np.concatenate(bloques, axis=1).astype(np.float32)


def fit_normalization_pipeline(X_log_train, y_train, ch_names):
    """Ajusta baseline + scaler usando solo los datos recibidos."""
    n_ch = len(ch_names)

    # Reshape a (n_ep, n_ch, n_bands) para operar por canal y banda
    X_tr = X_log_train.reshape(-1, n_ch, N_BANDS)

    # Baseline Neutro global, calculado solo con el train del fold ---------------
    neutro_mask = (y_train == LABEL_MAP["NEUTRO"])
    if neutro_mask.sum() == 0:
        baseline = X_tr.mean(axis=0)               # fallback: media de todo train
    else:
        baseline = X_tr[neutro_mask].mean(axis=0)  # (n_ch, n_bands)

    # Resta el baseline para ver la diferencia de Neutro -------------------------
    delta_tr = X_tr - baseline[np.newaxis, :, :]

    # Estandarizar (Z-score) ------------------------------------------------------
    delta_tr_2d = delta_tr.reshape(len(delta_tr), -1)   # (n_ep, n_ch*n_bands)

    scaler      = StandardScaler()
    delta_tr_2d = scaler.fit_transform(delta_tr_2d)

    preprocessing = {
        "baseline": baseline,
        "scaler": scaler,
    }

    return _build_feature_matrix(delta_tr_2d, ch_names), preprocessing


def transform_normalization_pipeline(X_log, ch_names, preprocessing):
    """Aplica una normalización ya ajustada a nuevas épocas."""
    n_ch = len(ch_names)
    X = X_log.reshape(-1, n_ch, N_BANDS)
    delta = X - preprocessing["baseline"][np.newaxis, :, :]
    delta_2d = delta.reshape(len(delta), -1)
    delta_2d = preprocessing["scaler"].transform(delta_2d)
    return _build_feature_matrix(delta_2d, ch_names)


def apply_normalization_pipeline(X_log_train, y_train, X_log_test, ch_names):
    """
    Aplica el pipeline completo de normalización usando SOLO datos de train.

    Return:
        X_train_feat y X_test_feat ya normalizados.
    """
    X_train_feat, preprocessing = fit_normalization_pipeline(X_log_train, y_train, ch_names)
    X_test_feat = transform_normalization_pipeline(X_log_test, ch_names, preprocessing)
    return X_train_feat, X_test_feat


def build_feature_names(ch_names):
    feat_names = [f"{ch}_{b}" for ch in ch_names for b in BANDS]
    if INCLUDE_THETA_ALPHA_RATIO:
        feat_names += [f"ThetaAlphaRatio_{ch}" for ch in ch_names]
    if INCLUDE_ALPHA_ASYMMETRY:
        ch_idx_map = {ch: i for i, ch in enumerate(ch_names)}
        feat_names += [
            f"AlphaAsym_{r}-{l}"
            for l, r in ALPHA_ASYMMETRY_PAIRS
            if l in ch_idx_map and r in ch_idx_map
        ]
    return feat_names


# ---------------------------------------------------------------------- Búsqueda de hiperparámetros

def evaluate_params(X_log, y, groups, ch_names, clf, params, n_folds=N_FOLDS):
    """
    Evalúa un conjunto de hiperparámetros con StratifiedGroupKFold.

    Los grupos son subject_id, así que cada fold evalúa sujetos no vistos.
    """
    sgkf = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    accs, f1s    = [], []
    y_true_all   = []
    y_pred_all   = []
    fold_subjects = []
    importancias = []

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
    Busca hiperparámetros por clasificador y devuelve el mejor resultado de cada uno.
    """
    groups = meta["subject_id"].values
    resultados = {}

    print("\n  Búsqueda de hiperparámetros por clasificador:")
    for nombre, clf in CLASIFICADORES.items():
        grid = list(ParameterGrid(PARAM_GRIDS[nombre]))
        print(f"\n  {nombre}: {len(grid)} combinaciones")

        candidatos = []
        for idx, params in enumerate(grid, start=1):
            r = evaluate_params(X_log, y, groups, ch_names, clf, params)
            candidatos.append(r)

            print(
                f"    [{idx:02d}/{len(grid):02d}] "
                f"F1={r['f1_media']:.3f}±{r['f1_std']:.2f}  "
                f"Acc={r['acc_media']:.3f}±{r['acc_std']:.2f}  "
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
    print(f"\n{'Clasificador':<20} {'F1 macro':>10} {'F1 std':>8} {'Acc media':>10} {'Acc std':>8}  Mejora")
    print("-" * 76)

    mejor_nombre, mejor_f1 = None, -np.inf

    for nombre, data in resultados.items():
        best = data["best"]
        mejora = (best["acc_media"] - 0.333) * 100      # mejora sobre baseline aleatorio (3 clases -> 33.3%)
        barra  = "█" * int(best["f1_media"] * 35)

        print(
            f"{nombre:<20} {best['f1_media']:>10.3f} {best['f1_std']:>8.3f} "
            f"{best['acc_media']:>10.3f} {best['acc_std']:>8.3f}  {mejora:+.1f}pp  {barra}"
        )

        if best["f1_media"] > mejor_f1:
            mejor_f1, mejor_nombre = best["f1_media"], nombre

    best = resultados[mejor_nombre]["best"]
    print(f"\nMejor clasificador global: {mejor_nombre} (F1 macro={best['f1_media']:.3f}, Acc={best['acc_media']:.3f})")
    print(f"Mejores hiperparámetros: {best['params']}")
    return mejor_nombre


def save_final_model(X_log, y, meta, ch_names, mejor_clf, best_params):
    """Entrena el mejor modelo con todos los sujetos y lo guarda en un único .joblib."""
    if not SAVE_MODEL:
        return

    X_train, preprocessing = fit_normalization_pipeline(X_log, y, ch_names)
    clf = clone(CLASIFICADORES[mejor_clf])
    clf.set_params(**best_params)
    clf.fit(X_train, y)

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": clf,
            "preprocessing": preprocessing,
            "classifier_name": mejor_clf,
            "best_params": best_params,
            "ch_names_eeg": ch_names,
            "feature_names": build_feature_names(ch_names),
            "bands": BANDS,
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
        FINAL_MODEL_PATH,
    )
    print(f"\n  Modelo final global: {FINAL_MODEL_PATH}")


# ---------------------------------------------------------------------- PLOTS

def plot_cv_scores(resultados, mejor_clf):
    datos = resultados[mejor_clf]["best"]
    folds = [f"Fold {i + 1}" for i in range(len(datos["accs"]))]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        f"All-subjects — {mejor_clf}\n"
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
    fname = RESULTS_DIR / f"01_cv_scores_{mejor_clf}.png"
    plt.savefig(fname, dpi=120)
    print(f"  {fname.name}")
    plt.show()


def plot_confusion(resultados, mejor_clf):
    y_true = resultados[mejor_clf]["best"]["y_true"]
    y_pred = resultados[mejor_clf]["best"]["y_pred"]

    cm_total = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    cm_norm = cm_total.astype(float) / cm_total.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        f"Matriz de confusión — {mejor_clf} all-subjects",
        fontsize=12,
        fontweight="bold",
    )
    for ax, cm, titulo in zip(
        axes, [cm_norm, cm_total],
        ["Normalizada", "Absoluta (nº épocas)"]
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
    fname = RESULTS_DIR / f"02_confusion_{mejor_clf}.png"
    plt.savefig(fname, dpi=120)
    print(f"  {fname.name}")
    plt.show()

    print("\n  Reporte de clasificación:")
    print(classification_report(y_true, y_pred,
                                 target_names=CONDITIONS, digits=3))


def plot_feature_importance(resultados, mejor_clf, ch_names, n_top=N_TOP_FEAT):
    imp_global = resultados[mejor_clf]["best"]["importancias"]
    if imp_global is None:
        print("  Importancia no disponible")
        return

    feat_names = build_feature_names(ch_names)

    # Truncar si hay discrepancia de longitud
    n = min(len(imp_global), len(feat_names))
    imp_global = imp_global[:n]
    feat_names = feat_names[:n]

    top_idx   = np.argsort(imp_global)[::-1][:n_top]
    top_names = [feat_names[i] for i in top_idx]
    top_vals  = imp_global[top_idx]

    def color_feat(nombre):
        if nombre.startswith("ThetaAlphaRatio"): return "#8e44ad"
        if nombre.startswith("AlphaAsym"):        return "#34495e"
        banda = nombre.rsplit("_", 1)[-1]
        return {"Delta":"#e74c3c","Theta":"#e67e22","Alpha":"#f1c40f",
                "Beta":"#2ecc71","Gamma":"#3498db"}.get(banda, "#95a5a6")

    colores = [color_feat(n) for n in top_names]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")
    y_pos = range(len(top_names))
    ax.barh(y_pos, top_vals[::-1], color=colores[::-1], alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importancia media (promedio folds)")
    ax.set_title(f"Top {n_top} features — {mejor_clf}")
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=c, label=b)
        for b, c in [("Delta","#e74c3c"),("Theta","#e67e22"),
                     ("Alpha","#f1c40f"),("Beta","#2ecc71"),
                     ("Gamma","#3498db"),("ThetaAlphaRatio","#8e44ad"),
                     ("AlphaAsym","#34495e")]
    ], fontsize=8, loc="lower right")

    plt.tight_layout()
    fname = RESULTS_DIR / f"03_importancia_{mejor_clf}.png"
    plt.savefig(fname, dpi=120)
    print(f"  {fname.name}")
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


def save_json(resultados, mejor_clf):
    resumen = {
        "cv": {
            "type": "StratifiedGroupKFold",
            "n_splits": N_FOLDS,
            "group": "subject_id",
            "scoring": SCORING,
        },
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

    print("CLASIFICACIÓN ALL-SUBJECTS!!!!!!!!!!!!!!!!")
    print(f"  Estrategia: StratifiedGroupKFold {N_FOLDS}-Fold por sujeto")
    print(f"  Scoring búsqueda: {SCORING}")
    print(f"  Chance level: 0.333  (3 clases)        :o")

    # Cargar datos en escala log (sin normalizar)
    print("\n  Cargando datos...")
    X_log, y, meta, ch_names = load_dataset()

    # Verificar que el shape de X_log es coherente con los canales y bandas
    if len(ch_names) > 0 and X_log.shape[1] != len(ch_names) * N_BANDS:
        raise ValueError(
            f"Shape inconsistente: X_log.shape[1]={X_log.shape[1]} "
            f"pero se esperaba {len(ch_names)} canales × {N_BANDS} bandas = {len(ch_names) * N_BANDS}. "
            f"Vuelve a ejecutar epochs_to_features.py"
        )

    sujetos = sorted(meta["subject_id"].unique())
    print(f"  Sujetos: {len(sujetos)}")
    print(f"  Épocas:  {len(y)}")
    print(f"  Clasificadores: {list(CLASIFICADORES.keys())}")

    resultados = hyperparameter_search(X_log, y, meta, ch_names)
    mejor_clf = classifier_summary(resultados)

    best_params = resultados[mejor_clf]["best"]["params"]
    save_final_model(X_log, y, meta, ch_names, mejor_clf, best_params)

    print("\n  Generando figuras...")
    plot_cv_scores(resultados, mejor_clf)
    plot_confusion(resultados, mejor_clf)
    plot_feature_importance(resultados, mejor_clf, ch_names)
    save_json(resultados, mejor_clf)

    # Tabla comparativa
    print("TABLA COMPARATIVA GLOBAL!!!!!")
    print(f"\n  {'Clasificador':<20} {'F1 macro':>10} {'Acc media':>10}  Params")
    print("  " + "-" * 92)

    for nombre, data in resultados.items():
        best = data["best"]
        print(
            f"  {nombre:<20} {best['f1_media']:>10.3f} {best['acc_media']:>10.3f}  "
            f"{best['params']}"
        )

    print("Terminado — resultados en results/all_subjects/")
