"""
Visualizacion de modelos within-subject ya entrenados.

Salida:
  results/explicability/within_subject/model_figures/logReg/<SUBJECT>_coeficientes.png
  results/explicability/within_subject/model_figures/decisionTree/<SUBJECT>_arbol.png
  results/explicability/within_subject/model_figures/logreg_top_coefficients.csv
  results/explicability/within_subject/model_figures/decision_tree_summary.csv
"""

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# Permite ejecutar este script desde scripts/5_explicability importando
# utilidades compartidas desde scripts/.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from utils import LABEL_INV, PROJECT_ROOT


warnings.filterwarnings("ignore")

MODELS_DIR = PROJECT_ROOT / "models" / "within_subject"
OUTPUT_DIR = PROJECT_ROOT / "results" / "explicability" / "within_subject" / "model_figures"
LOGREG_DIR = OUTPUT_DIR / "logReg"
TREE_DIR = OUTPUT_DIR / "decisionTree"

TOP_N_COEFFICIENTS = 15
TREE_FIGSIZE = (24, 14)
TREE_FONT_SIZE = 7


def ensure_output_dirs():
    """Crea las carpetas de salida necesarias."""
    LOGREG_DIR.mkdir(parents=True, exist_ok=True)
    TREE_DIR.mkdir(parents=True, exist_ok=True)


def load_model_files(classifier_name):
    """Devuelve los modelos within-subject disponibles para un clasificador."""
    model_dir = MODELS_DIR / classifier_name
    if not model_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de modelos: {model_dir}")

    return sorted(model_dir.glob("*.joblib"))


def class_names(model_data, classifier):
    """Obtiene los nombres de clase en el mismo orden que el clasificador."""
    label_map = model_data.get("label_map", {})
    inv = {value: key for key, value in label_map.items()} if label_map else LABEL_INV
    names = []
    for label in classifier.classes_:
        try:
            lookup_label = int(label)
        except (TypeError, ValueError):
            lookup_label = label
        names.append(str(inv.get(lookup_label, label)))
    return names


def feature_names(model_data, classifier):
    """Obtiene los nombres de features guardados junto al modelo."""
    names = model_data.get("feature_names", [])
    if names:
        names = [str(name) for name in names]
        n_features = getattr(classifier, "n_features_in_", len(names))
        if len(names) == n_features:
            return names

    n_features = getattr(classifier, "n_features_in_", 0)
    return [f"feature_{idx}" for idx in range(n_features)]


def plot_logreg_coefficients(model_path):
    """Genera una figura con los coeficientes mas relevantes por clase."""
    model_data = joblib.load(model_path)
    classifier = model_data["classifier"]
    subject_id = str(model_data.get("subject_id", model_path.stem))

    if not hasattr(classifier, "coef_"):
        return []

    coef = np.asarray(classifier.coef_, dtype=float)
    names = np.asarray(feature_names(model_data, classifier), dtype=object)
    classes = class_names(model_data, classifier)

    n_classes = coef.shape[0]
    fig, axes = plt.subplots(
        n_classes,
        1,
        figsize=(13, max(4, 3.8 * n_classes)),
        squeeze=False,
    )

    rows = []
    for class_idx, class_name in enumerate(classes):
        ax = axes[class_idx, 0]
        class_coef = coef[class_idx]
        top_idx = np.argsort(np.abs(class_coef))[-TOP_N_COEFFICIENTS:][::-1]
        plot_idx = top_idx[np.argsort(class_coef[top_idx])]

        values = class_coef[plot_idx]
        labels = names[plot_idx]
        colors = np.where(values >= 0, "#2b8cbe", "#d95f0e")

        ax.barh(labels, values, color=colors)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_title(f"{subject_id} - clase {class_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Coeficiente del modelo")
        ax.grid(axis="x", alpha=0.25)

        for rank, feature_idx in enumerate(top_idx, start=1):
            rows.append(
                {
                    "subject_id": subject_id,
                    "class": class_name,
                    "rank": rank,
                    "feature": str(names[feature_idx]),
                    "coefficient": float(class_coef[feature_idx]),
                    "abs_coefficient": float(abs(class_coef[feature_idx])),
                }
            )

    fig.suptitle(
        f"Coeficientes principales de regresion logistica - {subject_id}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = LOGREG_DIR / f"{subject_id}_coeficientes.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return rows


def plot_decision_tree(model_path):
    """Genera una figura del arbol de decision entrenado para un sujeto."""
    model_data = joblib.load(model_path)
    classifier = model_data["classifier"]
    subject_id = str(model_data.get("subject_id", model_path.stem))

    if not hasattr(classifier, "tree_"):
        return None

    names = feature_names(model_data, classifier)
    classes = class_names(model_data, classifier)

    fig, ax = plt.subplots(figsize=TREE_FIGSIZE)
    plot_tree(
        classifier,
        ax=ax,
        feature_names=names,
        class_names=classes,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        fontsize=TREE_FONT_SIZE,
    )
    ax.set_title(
        f"Arbol de decision within-subject - {subject_id}",
        fontsize=16,
        fontweight="bold",
    )

    output_path = TREE_DIR / f"{subject_id}_arbol.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "subject_id": subject_id,
        "node_count": int(classifier.tree_.node_count),
        "max_depth": int(classifier.tree_.max_depth),
        "n_features_in": int(getattr(classifier, "n_features_in_", len(names))),
        "output_file": str(output_path.relative_to(PROJECT_ROOT)),
    }


def main():
    ensure_output_dirs()

    print("FIGURAS DE MODELOS WITHIN-SUBJECT")
    print(f"  Modelos: {MODELS_DIR}")
    print(f"  Salida:  {OUTPUT_DIR}")

    coef_rows = []
    for model_path in load_model_files("logReg"):
        coef_rows.extend(plot_logreg_coefficients(model_path))

    if coef_rows:
        pd.DataFrame(coef_rows).to_csv(
            OUTPUT_DIR / "logreg_top_coefficients.csv",
            index=False,
        )

    tree_rows = []
    for model_path in load_model_files("decisionTree"):
        summary = plot_decision_tree(model_path)
        if summary is not None:
            tree_rows.append(summary)

    if tree_rows:
        pd.DataFrame(tree_rows).to_csv(
            OUTPUT_DIR / "decision_tree_summary.csv",
            index=False,
        )

    print(f"  Figuras logReg:       {len(list(LOGREG_DIR.glob('*.png')))}")
    print(f"  Figuras decisionTree: {len(list(TREE_DIR.glob('*.png')))}")
    print("Hecho.")


if __name__ == "__main__":
    main()
