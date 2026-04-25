"""
Script de entrenamiento within-subject: 

    StratifiedKFold (5 folds) por sujeto -> Clasificador

    Clasificadores:
    - LogisticRegression elastic-net  (interpretable con SHAP)
    - Random Forest
    - SVM RBF

Salida:
  results/within_subject/within_subject_results.json             métricas por sujeto y clasificador
  results/within_subject/01_accuracy_<clf>.png                   accuracy por sujeto (mejor clasificador)
  results/within_subject/02_confusion_<clf>.png                  matriz de confusión agregada
  results/within_subject/03_importancia_<clf>.png                importancia de features
  models/within_subject/<sujeto>.joblib                          modelo final por sujeto

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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.base import clone


warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------- Configuración
FEATURES_DIR = Path("data/features")
RESULTS_DIR  = Path("results/within_subject")
FINAL_MODELS_DIR = Path("models/within_subject")
CONDITIONS   = ["JOY", "NEUTRO", "SAD"]
LABEL_MAP    = {"JOY": 0, "NEUTRO": 1, "SAD": 2}

LABEL_INV = {}
for label, cond in LABEL_MAP.items():
    LABEL_INV[cond] = label

N_FOLDS      = 5
RANDOM_STATE = 42       # semilla de aleatoriedad para reproducibilidad
N_TOP_FEAT   = 20       # n de features que se muestran en el gráfico de importancia 20/398
SAVE_MODELS  = True     # si True guarda un modelo final por sujeto del mejor clasificador

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

    # Baseline Neutro -----------------------------------------------------------
    neutro_mask = (y_train == LABEL_MAP["NEUTRO"])
    if neutro_mask.sum() == 0:
        baseline = X_tr.mean(axis=0)               # fallback: media de todo train
    else:
        baseline = X_tr[neutro_mask].mean(axis=0)  # (n_ch, n_bands)

    # Resta el baseline para ver la diferencia de Neutro ------------------------
    delta_tr = X_tr - baseline[np.newaxis, :, :]

    # Estandarizar (Z-score) -----------------------------------------------------
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


def save_subject_model(X_log_suj, y_suj, clf, ch_names, clf_name, sid):
    """Entrena con todas las épocas del sujeto y guarda modelo + preprocesado."""
    X_train, preprocessing = fit_normalization_pipeline(X_log_suj, y_suj, ch_names)
    final_clf = clone(clf)
    final_clf.fit(X_train, y_suj)

    FINAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": final_clf,
            "preprocessing": preprocessing,
            "subject_id": sid,
            "classifier_name": clf_name,
            "ch_names_eeg": ch_names,
            "feature_names": build_feature_names(ch_names),
            "bands": BANDS,
            "label_map": LABEL_MAP,
            "conditions": CONDITIONS,
        },
        FINAL_MODELS_DIR / f"{sid}.joblib",
    )


def save_best_subject_models(X_log, y, meta, ch_names, clf_name):
    """Guarda un único modelo final por sujeto usando el mejor clasificador global."""
    if not SAVE_MODELS:
        return

    clf = CLASIFICADORES[clf_name]
    sujetos = sorted(meta["subject_id"].unique())

    for sid in sujetos:
        mask = (meta["subject_id"] == sid).values
        y_suj = y[mask]

        clases, counts = np.unique(y_suj, return_counts=True)
        if len(clases) < len(CONDITIONS) or counts.min() < N_FOLDS:
            continue

        save_subject_model(X_log[mask], y_suj, clf, ch_names, clf_name, sid)

    print(f"Modelos finales por sujeto: {FINAL_MODELS_DIR}")



# Cross-validation por cada sujeto

def evaluate_subject(X_log_suj, y_suj, clf, ch_names, clf_name="", sid="", n_folds=N_FOLDS):
    """
    Evalúa un clasificador con Stratified K-Fold
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    accs, f1s    = [], []
    y_true_all   = []
    y_pred_all   = []
    importancias = []

    for train_idx, test_idx in skf.split(X_log_suj, y_suj):
        X_log_train = X_log_suj[train_idx]
        X_log_test  = X_log_suj[test_idx]
        y_train     = y_suj[train_idx]
        y_test      = y_suj[test_idx]

        # Pipeline completo dentro de cada fold
        X_train, X_test = apply_normalization_pipeline(
            X_log_train, y_train,
            X_log_test,  ch_names
        )

        fold_clf = clone(clf)
        fold_clf.fit(X_train, y_train)
        y_pred = fold_clf.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        if hasattr(fold_clf, "coef_"):
            importancias.append(np.abs(fold_clf.coef_).mean(axis=0))
        elif hasattr(fold_clf, "feature_importances_"):
            importancias.append(fold_clf.feature_importances_)

    imp_media = np.mean(importancias, axis=0) if importancias else None

    return (np.array(accs), np.array(f1s), np.array(y_true_all), np.array(y_pred_all), imp_media)


# Análisis completo de los resultados de cada clasificador en cada sujeto

def classify_all_subjects(X_log, y, meta, ch_names):
    """
    Para cada sujeto y cada clasificador, ejecuta evaluate_subject
    y acumula los resultados en un diccionario.

    Imprime una tabla progresiva con el accuracy de cada clasificador
    por sujeto conforme va terminando.

    Return:
        resultados[nombre_clf] = lista de dicts con métricas por sujeto
    """
    sujetos    = sorted(meta["subject_id"].unique())
    resultados = {nombre: [] for nombre in CLASIFICADORES}

    # cabecera de la tabla
    print(f"\n  {'Sujeto':<20}", end="")
    for nombre in CLASIFICADORES:
        print(f"  {nombre:>17}", end="")
    print(f"  {'N_ep':>6}")
    print("  " + "-" * (20 + 19 * len(CLASIFICADORES) + 8))

    for sid in sujetos:
        mask      = (meta["subject_id"] == sid).values
        X_log_suj = X_log[mask]                         # épocas de este sujeto
        y_suj     = y[mask]
        n_ep      = len(y_suj)

        # Sujetos con menos épocas que N_FOLDS en alguna clase se saltan.
        clases, counts = np.unique(y_suj, return_counts=True)
        min_clase = counts.min()
        if min_clase < N_FOLDS:
            print(f"  {sid:<20}  saltando ({min_clase} épocas en clase mínima)")
            for nombre in CLASIFICADORES:
                resultados[nombre].append({
                    "subject_id": sid, "acc_media": np.nan,
                    "acc_std": np.nan, "f1_media": np.nan,
                    "n_epocas": n_ep,  "y_true": np.array([]),
                    "y_pred": np.array([]), "importancias": None,
                })
            continue

        print(f"  {sid:<20}", end="", flush=True)

        for nombre, clf in CLASIFICADORES.items():
            accs, f1s, y_true, y_pred, imp = evaluate_subject(
                X_log_suj, y_suj, clf, ch_names, clf_name=nombre, sid=sid
            )
            resultados[nombre].append({
                "subject_id":  sid,
                "acc_media":   accs.mean(),
                "acc_std":     accs.std(),
                "f1_media":    f1s.mean(),
                "f1_std":      f1s.std(),
                "n_epocas":    n_ep,
                "y_true":      y_true,
                "y_pred":      y_pred,
                "importancias": imp,
            })
            print(f"  {accs.mean():>6.3f}±{accs.std():.2f}", end="", flush=True)

        print(f"  {n_ep:>6}")

    return resultados

# Resumen
def classifier_summary(resultados):

    print("=" * 65)
    print("RESUMEN POR CLASIFICADOR  (sin leakage)")
    print("=" * 65)
    print(f"\n{'Clasificador':<20} {'Acc media':>10} {'Acc std':>9} {'F1 macro':>10}  Mejora")
    print("-" * 65)

    mejor_nombre, mejor_acc = None, 0

    for nombre, lista in resultados.items():
        accs = [r["acc_media"] for r in lista if not np.isnan(r["acc_media"])]
        f1s  = [r["f1_media"]  for r in lista if not np.isnan(r.get("f1_media", np.nan))]

        if not accs:
            continue

        media  = np.mean(accs)
        std    = np.std(accs)
        f1     = np.mean(f1s) if f1s else np.nan
        mejora = (media - 0.333) * 100          # mejora sobre baseline aleatorio (3 clases → 33.3%)
        barra  = "█" * int(media * 35)

        print(f"{nombre:<20} {media:>10.3f} {std:>9.3f} {f1:>10.3f}  {mejora:+.1f}pp  {barra}")

        if media > mejor_acc:
            mejor_acc, mejor_nombre = media, nombre

    print(f"\nMejor clasificador: {mejor_nombre} ({mejor_acc:.3f})")
    return mejor_nombre



# PLOTS

def plot_accuracy(resultados, mejor_clf):
    datos   = resultados[mejor_clf]
    sujetos = [d["subject_id"].replace("211-000", "") for d in datos]
    accs    = [d["acc_media"] if not np.isnan(d["acc_media"]) else 0
               for d in datos]
    stds    = [d["acc_std"] if not np.isnan(d["acc_std"]) else 0
               for d in datos]
    colores = ["#e74c3c" if a < 0.333 else
               "#f39c12" if a < 0.60  else
               "#27ae60" for a in accs]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#f8f9fa")
    acc_media = np.nanmean(accs)
    fig.suptitle(
        f"Within-subject  — {mejor_clf}\n"
        f"Accuracy media: {acc_media:.3f}  (chance = 0.333)",
        fontsize=13, fontweight="bold"
    )
    axes[0].bar(sujetos, accs, yerr=stds, color=colores, alpha=0.85, capsize=4)
    axes[0].axhline(0.333,     color="red",  ls="--", lw=1.5, label="chance")
    axes[0].axhline(acc_media, color="navy", ls="-",  lw=1.5,
                    label=f"media ({acc_media:.3f})")
    axes[0].set_title("Accuracy por sujeto (±std de 5 folds)")
    axes[0].set_xticklabels(sujetos, rotation=90, fontsize=8)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].set_facecolor("#ffffff")

    accs_v = [a for a in accs if a > 0]
    axes[1].hist(accs_v, bins=10, color="#3498db", alpha=0.7, edgecolor="white")
    axes[1].axvline(0.333,         color="red",  ls="--", lw=1.5, label="chance")
    axes[1].axvline(np.mean(accs_v), color="navy", ls="-", lw=1.5,
                    label=f"media={np.mean(accs_v):.3f}")
    axes[1].set_title("Distribución de accuracies")
    axes[1].set_xlabel("Accuracy")
    axes[1].set_ylabel("Nº sujetos")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_facecolor("#ffffff")

    plt.tight_layout()
    fname = RESULTS_DIR / f"01_accuracy_{mejor_clf}.png"
    plt.savefig(fname, dpi=120)
    print(f"\n {fname.name}")
    plt.show()


def plot_confusion(resultados, mejor_clf):
    cm_total = np.zeros((3, 3), dtype=int)
    for d in resultados[mejor_clf]:
        if len(d["y_true"]) == 0:
            continue
        cm_total += confusion_matrix(d["y_true"], d["y_pred"], labels=[0,1,2])

    cm_norm = cm_total.astype(float) / cm_total.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        f"Matriz de confusión — {mejor_clf} within-subject ",
        fontsize=12, fontweight="bold"
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
    print(f" {fname.name}")
    plt.show()

    y_true_all = np.concatenate([d["y_true"] for d in resultados[mejor_clf]
                                  if len(d["y_true"]) > 0])
    y_pred_all = np.concatenate([d["y_pred"] for d in resultados[mejor_clf]
                                  if len(d["y_pred"]) > 0])
    print("\n  Reporte de clasificación:")
    print(classification_report(y_true_all, y_pred_all,
                                 target_names=CONDITIONS, digits=3))


def plot_feature_importance(resultados, mejor_clf, ch_names, n_top=N_TOP_FEAT):
    imps = [d["importancias"] for d in resultados[mejor_clf]
            if d["importancias"] is not None]
    if not imps:
        print("  ⚠ Importancia no disponible")
        return

    imp_global = np.mean(imps, axis=0)

    # Construir nombres de features en el mismo orden que apply_normalization_pipeline
    n_ch = len(ch_names)
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
    ax.set_xlabel("Importancia media (promedio sujetos)")
    ax.set_title(f"Top {n_top} features — {mejor_clf} ")
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
    print(f" {fname.name}")
    plt.show()


def save_json(resultados):
    resumen = {}
    for nombre, lista in resultados.items():
        accs = [r["acc_media"] for r in lista if not np.isnan(r["acc_media"])]
        f1s  = [r["f1_media"]  for r in lista
                if not np.isnan(r.get("f1_media", np.nan))]
        resumen[nombre] = {
            "acc_media_global": float(np.nanmean(accs)),
            "acc_std_global":   float(np.nanstd(accs)),
            "f1_media_global":  float(np.nanmean(f1s)) if f1s else None,
            "por_sujeto": [
                {
                    "subject_id": r["subject_id"],
                    "acc_media":  float(r["acc_media"]) if not np.isnan(r["acc_media"]) else None,
                    "acc_std":    float(r["acc_std"])   if not np.isnan(r["acc_std"])   else None,
                    "f1_media":   float(r.get("f1_media", np.nan))
                                  if not np.isnan(r.get("f1_media", np.nan)) else None,
                    "n_epocas":   int(r["n_epocas"]),
                }
                for r in lista
            ]
        }
    out = RESULTS_DIR / "within_subject_results.json"
    with open(out, "w") as f:
        json.dump(resumen, f, indent=2)
    print(f"\n Resultados: {out.name}")


# MAIN
if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("CLASIFICACIÓN WITHIN-SUBJECT!!!!!!!!!!!!!!!!")
    print(f"  Estrategia: Stratified {N_FOLDS}-Fold CV por sujeto")
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

    print(f"\n  Clasificadores: {list(CLASIFICADORES.keys())}")
    print("\n  Resultados por sujeto:")
    resultados = classify_all_subjects(X_log, y, meta, ch_names)

    mejor_clf = classifier_summary(resultados)
    save_best_subject_models(X_log, y, meta, ch_names, mejor_clf)

    print("\n  Generando figuras...")
    plot_accuracy(resultados, mejor_clf)
    plot_confusion(resultados, mejor_clf)
    plot_feature_importance(resultados, mejor_clf, ch_names)
    save_json(resultados)

    # Tabla comparativa
    print("TABLA COMPARATIVA POR SUJETO!!!!!")
    sujetos = sorted(meta["subject_id"].unique())
    nombres = list(CLASIFICADORES.keys())
    print(f"\n  {'Sujeto':<20}", end="")
    for n in nombres:
        print(f"  {n:>17}", end="")
    print(f"  {'N_ep':>6}  Mejor")
    print("  " + "-" * (20 + 19*len(nombres) + 16))

    for sid in sujetos:
        print(f"  {sid.replace('211-000',''):<20}", end="")
        accs_suj, n_ep = {}, 0
        for nombre in nombres:
            r = next((r for r in resultados[nombre]
                      if r["subject_id"] == sid), None)
            if r and not np.isnan(r["acc_media"]):
                accs_suj[nombre] = r["acc_media"]
                n_ep = r["n_epocas"]
                print(f"  {r['acc_media']:>6.3f}±{r['acc_std']:.2f}", end="")
            else:
                print(f"  {'—':>17}", end="")
        if accs_suj:
            print(f"  {n_ep:>6}  {max(accs_suj, key=accs_suj.get)}")
        else:
            print(f"  {n_ep:>6}  —")

    print("Terminado — resultados en data/results/")
