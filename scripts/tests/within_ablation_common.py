"""
Funciones comunes para pruebas within-subject con subconjuntos de features.

Estas pruebas sirven para auditar si el rendimiento alto del modelo within-subject
depende de bandas altas o de canales periféricos potencialmente contaminados.
Cada script lanzador define una configuración y este módulo ejecuta el mismo
pipeline de CV que train_within_subject.py, pero seleccionando columnas después
de aplicar el preprocesado dentro de cada fold.
"""

import os
import json
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------- Configuración
FEATURES_DIR = Path("data/features")
TESTS_DIR = Path("scripts/tests")

CONDITIONS = ["JOY", "NEUTRO", "SAD"]
LABEL_MAP = {"JOY": 0, "NEUTRO": 1, "SAD": 2}

LABEL_INV = {}
for label, cond in LABEL_MAP.items():
    LABEL_INV[cond] = label

N_FOLDS = 5
RANDOM_STATE = 42
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

BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
N_BANDS = len(BANDS)
BAND_IDX = {}
for i, band in enumerate(BANDS):
    BAND_IDX[band] = i

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

    ch_idx_map = {}
    for i, ch in enumerate(ch_names):
        ch_idx_map[ch] = i

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


def should_keep_feature(feature_name, excluded_bands, excluded_channels):
    """Aplica el filtro de bandas/canales a una feature."""
    feature_type, channel, band, pair = describe_feature(feature_name)

    if feature_type == "bandpower":
        if band in excluded_bands:
            return False
        if channel in excluded_channels:
            return False
        return True

    if feature_type == "ThetaAlphaRatio":
        return channel not in excluded_channels

    if feature_type == "AlphaAsym":
        return not pair_has_excluded_channel(pair, excluded_channels)

    return True


def build_selected_feature_indices(feature_names, excluded_bands=None, excluded_channels=None):
    """Devuelve índices y nombres de las features que sobreviven al filtro."""
    excluded_bands = set(excluded_bands or [])
    excluded_channels = set(excluded_channels or [])

    indices = []
    selected_names = []

    for idx, feature_name in enumerate(feature_names):
        if should_keep_feature(feature_name, excluded_bands, excluded_channels):
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


# ---------------------------------------------------------------------- Entrenamiento CV
def format_metric(mean_value, std_value):
    """Formatea media y desviación para tablas."""
    return f"{mean_value:.3f} +/- {std_value:.2f}"


def evaluate_subject(X_log_subject, y_subject, clf, ch_names, selected_indices):
    """Evalúa un sujeto con StratifiedKFold y selección de features."""
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

        X_train = X_train[:, selected_indices]
        X_test = X_test[:, selected_indices]

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


def classify_all_subjects(X_log, y, meta, ch_names, selected_indices):
    """Ejecuta todos los clasificadores para todos los sujetos."""
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
            print(f"  {subject_id:<{SUBJECT_COL_WIDTH}}  saltando ({min_class} épocas en clase mínima)")
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
                selected_indices,
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


def classifier_summary(results, test_title):
    """Imprime resumen global por clasificador y devuelve el mejor."""
    print("=" * 75)
    print(f"RESUMEN {test_title}  (within-subject)")
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
        bar = "█" * int(mean_acc * 35)

        print(
            f"{name:<20} {mean_acc:>10.3f} {std_acc:>9.3f} "
            f"{mean_f1:>10.3f}  {improvement:+.1f}pp  {bar}"
        )

        if mean_acc > best_acc:
            best_acc, best_name = mean_acc, name

    print(f"\nMejor clasificador: {best_name} ({best_acc:.3f})")
    return best_name


# ---------------------------------------------------------------------- Plots
def plot_accuracy(results, best_clf, output_dir, test_title):
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
        f"{test_title} — {best_clf}\n"
        f"Accuracy media: {acc_mean:.3f}  (chance = 0.333)",
        fontsize=13,
        fontweight="bold",
    )

    axes[0].bar(subjects, accs, yerr=stds, color=colors, alpha=0.85, capsize=4)
    axes[0].axhline(0.333, color="red", ls="--", lw=1.5, label="chance")
    axes[0].axhline(acc_mean, color="navy", ls="-", lw=1.5, label=f"media ({acc_mean:.3f})")
    axes[0].set_title("Accuracy por sujeto (±std de 5 folds)")
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
    axes[1].set_title("Distribución de accuracies")
    axes[1].set_xlabel("Accuracy")
    axes[1].set_ylabel("Nº sujetos")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_facecolor("#ffffff")

    plt.tight_layout()
    output_path = output_dir / f"01_accuracy_{best_clf}.png"
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  {output_path.name}")


def plot_confusion(results, best_clf, output_dir, test_title):
    """Genera matriz de confusión agregada."""
    cm_total = np.zeros((3, 3), dtype=int)
    for row in results[best_clf]:
        if len(row["y_true"]) == 0:
            continue
        cm_total += confusion_matrix(row["y_true"], row["y_pred"], labels=[0, 1, 2])

    cm_norm = cm_total.astype(float) / cm_total.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#f8f9fa")
    fig.suptitle(
        f"Matriz de confusión — {best_clf} — {test_title}",
        fontsize=12,
        fontweight="bold",
    )

    for ax, cm, title in zip(
        axes,
        [cm_norm, cm_total],
        ["Normalizada", "Absoluta (nº épocas)"],
    ):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if "Norm" in title else None)
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
                value = f"{cm[i, j]:.2f}" if "Norm" in title else str(cm[i, j])
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, value, ha="center", va="center", fontsize=11, color=color, fontweight="bold")

    plt.tight_layout()
    output_path = output_dir / f"02_confusion_{best_clf}.png"
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  {output_path.name}")

    y_true_all = np.concatenate([
        row["y_true"] for row in results[best_clf]
        if len(row["y_true"]) > 0
    ])
    y_pred_all = np.concatenate([
        row["y_pred"] for row in results[best_clf]
        if len(row["y_pred"]) > 0
    ])

    report = classification_report(
        y_true_all,
        y_pred_all,
        target_names=CONDITIONS,
        digits=3,
        output_dict=True,
    )
    with open(output_dir / f"02_classification_report_{best_clf}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  02_classification_report_{best_clf}.json")


def plot_feature_importance(results, best_clf, selected_feature_names, output_dir, test_title):
    """Genera top de importancias para el mejor clasificador."""
    importances = [
        row["importancias"] for row in results[best_clf]
        if row["importancias"] is not None
    ]

    if not importances:
        print("  Importancia no disponible")
        return

    global_importance = np.mean(importances, axis=0)
    n = min(len(global_importance), len(selected_feature_names))
    global_importance = global_importance[:n]
    feature_names = selected_feature_names[:n]

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
    ax.set_title(f"Top {N_TOP_FEAT} features — {best_clf} — {test_title}")
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_items = [
        ("Delta", "#e74c3c"),
        ("Theta", "#e67e22"),
        ("Alpha", "#f1c40f"),
        ("Beta", "#2ecc71"),
        ("Gamma", "#3498db"),
        ("ThetaAlphaRatio", "#8e44ad"),
        ("AlphaAsym", "#34495e"),
    ]
    ax.legend(
        handles=[Patch(facecolor=color, label=label) for label, color in legend_items],
        fontsize=8,
        loc="lower right",
    )

    plt.tight_layout()
    output_path = output_dir / f"03_importancia_{best_clf}.png"
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  {output_path.name}")


# ---------------------------------------------------------------------- Guardado
def save_selected_features(selected_feature_names, output_dir):
    """Guarda la lista de features usadas."""
    rows = []
    for idx, feature_name in enumerate(selected_feature_names):
        feature_type, channel, band, pair = describe_feature(feature_name)
        rows.append({
            "feature_idx_selected": idx,
            "feature_name": feature_name,
            "feature_type": feature_type,
            "channel": channel,
            "band": band,
            "pair": pair,
        })

    out = output_dir / "selected_features.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  {out.name}")


def save_results(results, best_clf, config, selected_feature_names, output_dir):
    """Guarda JSON y CSV resumen."""
    summary_rows = []
    json_summary = {
        "test_name": config["test_name"],
        "title": config["title"],
        "description": config["description"],
        "excluded_bands": sorted(config.get("excluded_bands", [])),
        "excluded_channels": sorted(config.get("excluded_channels", [])),
        "n_features_selected": len(selected_feature_names),
        "selected_feature_blocks": feature_block_counts(selected_feature_names),
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

    with open(output_dir / "within_subject_results.json", "w") as f:
        json.dump(json_summary, f, indent=2)
    print("  within_subject_results.json")

    pd.DataFrame(summary_rows).to_csv(output_dir / "within_subject_summary.csv", index=False)
    print("  within_subject_summary.csv")


# ---------------------------------------------------------------------- Ejecución principal
def run_within_ablation(config):
    """Ejecuta una prueba completa de ablación within-subject."""
    output_dir = TESTS_DIR / config["test_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{config['title']}!!!!!!!!!!")
    print(f"  Features: {FEATURES_DIR}")
    print(f"  Salida:   {output_dir}")
    print(f"  Chance level: 0.333  (3 clases)")
    print(f"  Excluir bandas: {sorted(config.get('excluded_bands', []))}")
    print(f"  Excluir canales: {len(config.get('excluded_channels', []))}")

    print("\n  Cargando datos...")
    X_log, y, meta, ch_names = load_dataset()

    if X_log.shape[1] != len(ch_names) * N_BANDS:
        raise ValueError(
            f"Shape inconsistente: X_log.shape[1]={X_log.shape[1]} "
            f"pero se esperaba {len(ch_names)} canales x {N_BANDS} bandas = {len(ch_names) * N_BANDS}."
        )

    full_feature_names = build_feature_names(ch_names)
    selected_indices, selected_feature_names = build_selected_feature_indices(
        full_feature_names,
        excluded_bands=config.get("excluded_bands", []),
        excluded_channels=config.get("excluded_channels", []),
    )

    if len(selected_indices) == 0:
        raise ValueError("La selección de features se quedó vacía.")

    print(f"  Épocas: {len(y)}")
    print(f"  Sujetos: {meta['subject_id'].nunique()}")
    print(f"  Features completas: {len(full_feature_names)}")
    print(f"  Features seleccionadas: {len(selected_feature_names)}")
    print(f"  Bloques seleccionados: {feature_block_counts(selected_feature_names)}")

    print("\n  Resultados por sujeto:")
    results = classify_all_subjects(X_log, y, meta, ch_names, selected_indices)
    best_clf = classifier_summary(results, config["title"])

    print("\n  Guardando resultados...")
    save_selected_features(selected_feature_names, output_dir)
    save_results(results, best_clf, config, selected_feature_names, output_dir)

    print("\n  Generando figuras...")
    plot_accuracy(results, best_clf, output_dir, config["title"])
    plot_confusion(results, best_clf, output_dir, config["title"])
    plot_feature_importance(results, best_clf, selected_feature_names, output_dir, config["title"])

    print("\nTABLA COMPARATIVA POR SUJETO!!!!!")
    subjects = sorted(meta["subject_id"].unique())
    names = list(CLASIFICADORES.keys())
    print(f"\n  {'Sujeto':<{SUBJECT_COL_WIDTH}}", end="")
    for name in names:
        print(f"  {name:>{METRIC_COL_WIDTH}}", end="")
    print(f"  {'N_ep':>{N_EP_COL_WIDTH}}  Mejor")
    print("  " + "-" * (SUBJECT_COL_WIDTH + (METRIC_COL_WIDTH + 2) * len(names) + N_EP_COL_WIDTH + 10))

    for subject_id in subjects:
        print(f"  {subject_id.replace('211-000', ''):<{SUBJECT_COL_WIDTH}}", end="")
        accs_subject = {}
        n_epochs = 0
        for name in names:
            row = next((r for r in results[name] if r["subject_id"] == subject_id), None)
            if row and not np.isnan(row["acc_media"]):
                accs_subject[name] = row["acc_media"]
                n_epochs = row["n_epocas"]
                print(f"  {format_metric(row['acc_media'], row['acc_std']):>{METRIC_COL_WIDTH}}", end="")
            else:
                print(f"  {'-':>{METRIC_COL_WIDTH}}", end="")
        if accs_subject:
            print(f"  {n_epochs:>{N_EP_COL_WIDTH}}  {max(accs_subject, key=accs_subject.get)}")
        else:
            print(f"  {n_epochs:>{N_EP_COL_WIDTH}}  -")

    print("\n" + "=" * 75)
    print("TEST COMPLETADO!!!!!!!!!!")
    print(f"Resultados en: {output_dir}")
    print("=" * 75)
