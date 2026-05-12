"""
Script de diagnóstico del entrenamiento within-subject:

    Comprueba si las accuracies altas pueden explicarse por fuga de información
    o por dependencia entre épocas muy parecidas.

Pruebas:
    - CV aleatoria original: mismo esquema que train_within_subject.py
    - CV por bloques temporales: reduce la mezcla de épocas cercanas train/test
    - Permutation test: repite el entrenamiento con etiquetas barajadas
    - Baseline mayoritario: clasificador trivial por fold
    - Duplicados exactos: comprueba si hay filas repetidas en X_log

Salida:
  results/within_subject_tests/within_subject_tests.json
  results/within_subject_tests/within_subject_tests_summary.csv
  results/within_subject_tests/01_sanity_<clf>.png

"""

import importlib.util
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------- CONFIGURACIÓN
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path("scripts/3_train")
TRAIN_SCRIPT = SCRIPT_DIR / "train_within_subject.py"
RESULTS_DIR = Path("results/within_subject_tests")

N_FOLDS = 5
RANDOM_STATE = 42
N_PERMUTATIONS = 20

# LogReg es suficiente para diagnosticar la sospecha principal y tarda poco.
# Si quieres auditar todos, cambia a ["LogReg", "RandomForest", "SVM_RBF"].
CLASSIFIERS_TO_TEST = ["LogReg"]

SUBJECT_COL_WIDTH = 18
METRIC_COL_WIDTH = 16


# ---------------------------------------------------------------------- Carga del entrenamiento original
def load_train_module():
    """Carga train_within_subject.py para reutilizar exactamente su pipeline."""
    spec = importlib.util.spec_from_file_location("train_within_subject", TRAIN_SCRIPT)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    return train_module


def format_metric(mean_value, std_value):
    """Formatea media y desviación para tablas de consola."""
    return f"{mean_value:.3f} +/- {std_value:.2f}"


def to_float_list(values):
    """Convierte arrays numpy a listas JSON-friendly."""
    return [float(v) for v in values]


# ---------------------------------------------------------------------- Splits y métricas
def get_valid_subjects(meta, y):
    """Devuelve sujetos con suficientes épocas por clase para N_FOLDS."""
    valid_subjects = []
    skipped_subjects = []

    for subject_id in sorted(meta["subject_id"].unique()):
        mask = (meta["subject_id"] == subject_id).values
        y_subject = y[mask]
        classes, counts = np.unique(y_subject, return_counts=True)

        if len(classes) < 3 or counts.min() < N_FOLDS:
            skipped_subjects.append(subject_id)
        else:
            valid_subjects.append(subject_id)

    return valid_subjects, skipped_subjects


def iter_random_stratified_splits(y_subject):
    """Genera los folds aleatorios originales de within-subject."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    yield from skf.split(np.zeros(len(y_subject)), y_subject)


def iter_blocked_stratified_splits(meta_subject, y_subject):
    """
    Genera folds por bloques de epoch_idx dentro de cada condición.

    Cada fold recibe un bloque temporal de JOY, NEUTRO y SAD. Así se evita que
    train y test estén formados por épocas aleatorias potencialmente vecinas.
    """
    fold_indices = [[] for _ in range(N_FOLDS)]

    for label in sorted(np.unique(y_subject)):
        label_indices = np.where(y_subject == label)[0]
        label_meta = meta_subject.iloc[label_indices]
        order = np.argsort(label_meta["epoch_idx"].values)
        label_indices = label_indices[order]

        chunks = np.array_split(label_indices, N_FOLDS)
        for fold_idx, chunk in enumerate(chunks):
            fold_indices[fold_idx].extend(chunk.tolist())

    all_indices = np.arange(len(y_subject))
    for fold_idx in range(N_FOLDS):
        test_idx = np.array(sorted(fold_indices[fold_idx]), dtype=int)
        train_idx = np.setdiff1d(all_indices, test_idx)
        yield train_idx, test_idx


def evaluate_splits(train_module, X_log_subject, y_subject, ch_names, clf, splits):
    """Evalúa un clasificador usando una lista de folds ya definida."""
    accs, f1s = [], []

    for train_idx, test_idx in splits:
        X_log_train = X_log_subject[train_idx]
        X_log_test = X_log_subject[test_idx]
        y_train = y_subject[train_idx]
        y_test = y_subject[test_idx]

        X_train, X_test = train_module.apply_normalization_pipeline(
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

    return np.array(accs), np.array(f1s)


def evaluate_majority_baseline(y_subject, splits):
    """Evalúa un baseline que predice siempre la clase mayoritaria del train."""
    accs = []

    for train_idx, test_idx in splits:
        y_train = y_subject[train_idx]
        y_test = y_subject[test_idx]

        values, counts = np.unique(y_train, return_counts=True)
        majority_label = values[np.argmax(counts)]
        y_pred = np.full_like(y_test, majority_label)
        accs.append(accuracy_score(y_test, y_pred))

    return np.array(accs)


def permutation_test(train_module, X_log_subject, y_subject, ch_names, clf):
    """
    Baraja etiquetas dentro del sujeto y repite la CV.

    Si esta prueba queda muy por encima de chance (~0.333), hay sospecha fuerte
    de fuga de información o estructura espuria.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    permutation_accs = []

    for _ in range(N_PERMUTATIONS):
        y_perm = rng.permutation(y_subject)
        splits = list(iter_random_stratified_splits(y_perm))
        accs, _ = evaluate_splits(
            train_module,
            X_log_subject,
            y_perm,
            ch_names,
            clf,
            splits,
        )
        permutation_accs.append(float(accs.mean()))

    return np.array(permutation_accs)


# ---------------------------------------------------------------------- Checks de datos
def exact_duplicate_report(X_log_subject, y_subject):
    """
    Busca filas de X_log exactamente repetidas.

    Return:
        número de grupos duplicados y grupos duplicados con etiquetas distintas.
    """
    _, inverse, counts = np.unique(X_log_subject, axis=0, return_inverse=True, return_counts=True)
    duplicate_groups = np.where(counts > 1)[0]

    mixed_label_groups = 0
    for group_idx in duplicate_groups:
        labels = np.unique(y_subject[inverse == group_idx])
        if len(labels) > 1:
            mixed_label_groups += 1

    return int(len(duplicate_groups)), int(mixed_label_groups)


def make_decision(random_acc, blocked_acc, permutation_acc, baseline_acc):
    """Clasifica el resultado del diagnóstico con reglas simples."""
    flags = []

    if permutation_acc - baseline_acc > 0.10:
        flags.append("permutation_alta")

    if random_acc - blocked_acc > 0.15:
        flags.append("bloques_bajan_mucho")

    if random_acc - baseline_acc < 0.10:
        flags.append("cerca_del_baseline")

    if not flags:
        return "OK", []

    return "REVISAR", flags


# ---------------------------------------------------------------------- Diagnóstico principal
def run_subject_diagnostics(train_module, X_log, y, meta, ch_names, classifier_name):
    """Ejecuta todas las pruebas para un clasificador."""
    clf = train_module.CLASIFICADORES[classifier_name]
    valid_subjects, skipped_subjects = get_valid_subjects(meta, y)

    rows = []
    details = {
        "classifier": classifier_name,
        "n_permutations": N_PERMUTATIONS,
        "subjects": {},
        "skipped_subjects": skipped_subjects,
    }

    print(f"\nDIAGNÓSTICO WITHIN-SUBJECT - {classifier_name}")
    print(f"Permutaciones por sujeto: {N_PERMUTATIONS}")
    print()
    print(
        f"  {'Sujeto':<{SUBJECT_COL_WIDTH}}"
        f"  {'CV random':>{METRIC_COL_WIDTH}}"
        f"  {'CV bloques':>{METRIC_COL_WIDTH}}"
        f"  {'Permutado':>{METRIC_COL_WIDTH}}"
        f"  {'Baseline':>{METRIC_COL_WIDTH}}"
        f"  Estado"
    )
    print("  " + "-" * 105)

    for subject_id in valid_subjects:
        mask = (meta["subject_id"] == subject_id).values
        X_log_subject = X_log[mask]
        y_subject = y[mask]
        meta_subject = meta[mask].reset_index(drop=True)

        random_splits = list(iter_random_stratified_splits(y_subject))
        blocked_splits = list(iter_blocked_stratified_splits(meta_subject, y_subject))

        random_accs, random_f1s = evaluate_splits(
            train_module,
            X_log_subject,
            y_subject,
            ch_names,
            clf,
            random_splits,
        )
        blocked_accs, blocked_f1s = evaluate_splits(
            train_module,
            X_log_subject,
            y_subject,
            ch_names,
            clf,
            blocked_splits,
        )
        baseline_accs = evaluate_majority_baseline(y_subject, random_splits)
        permutation_accs = permutation_test(
            train_module,
            X_log_subject,
            y_subject,
            ch_names,
            clf,
        )

        duplicate_groups, mixed_label_duplicate_groups = exact_duplicate_report(
            X_log_subject,
            y_subject,
        )

        status, flags = make_decision(
            random_acc=float(random_accs.mean()),
            blocked_acc=float(blocked_accs.mean()),
            permutation_acc=float(permutation_accs.mean()),
            baseline_acc=float(baseline_accs.mean()),
        )

        row = {
            "subject_id": subject_id,
            "n_epochs": int(len(y_subject)),
            "random_acc_mean": float(random_accs.mean()),
            "random_acc_std": float(random_accs.std()),
            "random_f1_mean": float(random_f1s.mean()),
            "blocked_acc_mean": float(blocked_accs.mean()),
            "blocked_acc_std": float(blocked_accs.std()),
            "blocked_f1_mean": float(blocked_f1s.mean()),
            "permutation_acc_mean": float(permutation_accs.mean()),
            "permutation_acc_std": float(permutation_accs.std()),
            "baseline_acc_mean": float(baseline_accs.mean()),
            "duplicate_groups": duplicate_groups,
            "mixed_label_duplicate_groups": mixed_label_duplicate_groups,
            "status": status,
            "flags": ",".join(flags),
        }
        rows.append(row)

        details["subjects"][subject_id] = {
            **row,
            "random_accs": to_float_list(random_accs),
            "random_f1s": to_float_list(random_f1s),
            "blocked_accs": to_float_list(blocked_accs),
            "blocked_f1s": to_float_list(blocked_f1s),
            "permutation_accs": to_float_list(permutation_accs),
            "baseline_accs": to_float_list(baseline_accs),
        }

        print(
            f"  {subject_id:<{SUBJECT_COL_WIDTH}}"
            f"  {format_metric(row['random_acc_mean'], row['random_acc_std']):>{METRIC_COL_WIDTH}}"
            f"  {format_metric(row['blocked_acc_mean'], row['blocked_acc_std']):>{METRIC_COL_WIDTH}}"
            f"  {format_metric(row['permutation_acc_mean'], row['permutation_acc_std']):>{METRIC_COL_WIDTH}}"
            f"  {row['baseline_acc_mean']:>{METRIC_COL_WIDTH}.3f}"
            f"  {status}"
        )

    return pd.DataFrame(rows), details


# ---------------------------------------------------------------------- Plots y guardado
def plot_sanity(summary_df, classifier_name):
    """Genera una figura comparando CV real, bloques, permutación y baseline."""
    if summary_df.empty:
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    subjects = [s.replace("211-000", "") for s in summary_df["subject_id"]]
    x = np.arange(len(summary_df))
    width = 0.22

    fig, ax = plt.subplots(figsize=(max(14, len(summary_df) * 0.6), 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    ax.bar(x - 1.5 * width, summary_df["random_acc_mean"], width, label="CV random", color="#3498db", alpha=0.85)
    ax.bar(x - 0.5 * width, summary_df["blocked_acc_mean"], width, label="CV bloques", color="#27ae60", alpha=0.85)
    ax.bar(x + 0.5 * width, summary_df["permutation_acc_mean"], width, label="Permutado", color="#e67e22", alpha=0.85)
    ax.bar(x + 1.5 * width, summary_df["baseline_acc_mean"], width, label="Baseline", color="#95a5a6", alpha=0.85)

    ax.axhline(0.333, color="red", ls="--", lw=1.5, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=90, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Diagnóstico within-subject - {classifier_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = RESULTS_DIR / f"01_sanity_{classifier_name}.png"
    plt.savefig(output_path, dpi=120)
    print(f"\nFigura guardada: {output_path}")
    plt.close(fig)


def save_results(all_summaries, all_details):
    """Guarda CSV y JSON con todos los diagnósticos."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.concat(all_summaries, ignore_index=True)
    csv_path = RESULTS_DIR / "within_subject_tests_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    json_path = RESULTS_DIR / "within_subject_tests.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "criteria": {
                    "n_folds": N_FOLDS,
                    "n_permutations": N_PERMUTATIONS,
                    "permutation_minus_baseline_warning_threshold": 0.10,
                    "random_minus_blocked_warning_threshold": 0.15,
                    "random_minus_baseline_warning_threshold": 0.10,
                },
                "classifiers": all_details,
            },
            f,
            indent=2,
        )

    print(f"Resumen CSV:  {csv_path}")
    print(f"Resumen JSON: {json_path}")


# ---------------------------------------------------------------------- MAIN
if __name__ == "__main__":
    print("TESTS DE ENTRENAMIENTO WITHIN-SUBJECT!!!!!!!!!!")
    print(f"Script base: {TRAIN_SCRIPT}")
    print(f"Resultados: {RESULTS_DIR}")

    train_module = load_train_module()
    X_log, y, meta, ch_names = train_module.load_dataset()

    if X_log.shape[1] != len(ch_names) * train_module.N_BANDS:
        raise ValueError(
            f"Shape inconsistente: X_log.shape[1]={X_log.shape[1]} "
            f"pero se esperaban {len(ch_names) * train_module.N_BANDS} features base"
        )

    if np.isnan(X_log).any() or np.isinf(X_log).any():
        raise ValueError("X_log contiene NaN o Inf")

    print(f"\nÉpocas: {len(y)}")
    print(f"Sujetos: {meta['subject_id'].nunique()}")
    print(f"Features base: {X_log.shape[1]}")
    print(f"Clasificadores auditados: {CLASSIFIERS_TO_TEST}")

    summaries = []
    details = {}

    for classifier_name in CLASSIFIERS_TO_TEST:
        summary_df, classifier_details = run_subject_diagnostics(
            train_module,
            X_log,
            y,
            meta,
            ch_names,
            classifier_name,
        )
        summary_df.insert(0, "classifier", classifier_name)
        summaries.append(summary_df)
        details[classifier_name] = classifier_details
        plot_sanity(summary_df, classifier_name)

    save_results(summaries, details)

    print("\nINTERPRETACIÓN RÁPIDA")
    print("  - Permutado cerca del baseline: no hay fuga obvia de etiqueta.")
    print("  - CV bloques mucho menor que CV random: posible dependencia temporal entre épocas.")
    print("  - Duplicados con etiquetas distintas: revisar features o ensamblado.")
    print("\nTESTS COMPLETADOS!!!!!!!!!!")
