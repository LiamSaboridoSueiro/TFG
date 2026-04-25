"""
Script de checks: sujetos válidos para extracción de features
    Completitud por sujeto -> Balance de épocas -> Check espectral automático -> Lista de sujetos aptos

Checks:
    - Completitud: todos los sujetos deben tener JOY / NEUTRO / SAD
    - Balance: avisa si una condición tiene menos del 50% de épocas que la condición máxima
    - Espectro: clasifica automáticamente sujetos con PSD anómala

Salida:
  data/checks/valid_subjects.txt               lista de sujetos aptos para epochs_to_features.py
  data/checks/invalid_subjects.txt             lista de sujetos descartados y motivo
  results/checks/check_valid_subjects_results.json resumen automático del check
  results/checks/01_balance_epochs.png         balance de épocas por sujeto
  results/checks/02_spectrum_<cond>.png        espectro PSD por sujeto y condición

"""

import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np


warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# ---------------------------------------------------------------------- CONFIGURACIÓN INICIAL
EPOCHS_DIR  = Path("data/processed/epochs")
STATS_DIR   = Path("data/processed/stats")
RESULTS_DIR = Path("results/checks")

RAW_DIRS = {
    "JOY":    Path("data/raw/edf/Joy"),
    "NEUTRO": Path("data/raw/edf/Neutro"),
    "SAD":    Path("data/raw/edf/Sad"),
}

VALID_SUBJECTS_FILE = RESULTS_DIR / "valid_subjects.txt"
INVALID_SUBJECTS_FILE = RESULTS_DIR / "invalid_subjects.txt"

CONDITIONS = ["JOY", "NEUTRO", "SAD"]

SFREQ = 250
TMIN  = -1.5
TMAX  =  1.0

N_FFT     = int((TMAX - TMIN) * SFREQ)     # 625 muestras por época
N_OVERLAP = N_FFT // 2                     # 50% solapamiento

BALANCE_THRESHOLD = 0.50                    # condición mínima >= 50% de la máxima
SPECTRUM_N_COLS   = 4
ALPHA_PEAK_RATIO  = 1.30                    # pico alpha mínimo frente a beta
SLOPE_RATIO       = 3.00                    # caída mínima entre 1 Hz y 40 Hz

COLORS = {
    "JOY":    "#27ae60",
    "NEUTRO": "#2980b9",
    "SAD":    "#c0392b",
}


# ---------------------------------------------------------------------- Carga de archivos disponibles
def extract_subject_id(filename: str) -> str:
    """
    Extrae el ID de sujeto desde un archivo EDF, FIF o JSON.

    Ejemplos:
        211-000532-02_JOY.edf       -> 211-000532
        211-000532-02_JOY-epo.fif   -> 211-000532
        211-000532-02_JOY_stats.json -> 211-000532
    """
    parts = filename.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return filename.split("_")[0]


def load_available_epochs() -> dict:
    """Devuelve dict: subject_id -> {condition: fif_path}."""
    available = defaultdict(dict)

    for condition in CONDITIONS:
        condition_dir = EPOCHS_DIR / condition
        if not condition_dir.exists():
            continue

        for fif_path in sorted(condition_dir.glob("*-epo.fif")):
            subject_id = extract_subject_id(fif_path.name)
            available[subject_id][condition] = fif_path

    return dict(available)


def load_available_raws() -> dict:
    """Devuelve dict: subject_id -> {condition: edf_path}."""
    available = defaultdict(dict)

    for condition, condition_dir in RAW_DIRS.items():
        if not condition_dir.exists():
            continue

        for edf_path in sorted(condition_dir.glob("*.edf")):
            subject_id = extract_subject_id(edf_path.name)
            available[subject_id][condition] = edf_path

    return dict(available)


def load_preprocessing_stats() -> dict:
    """Devuelve dict: subject_id -> {condition: stats_dict}."""
    stats = defaultdict(dict)

    if not STATS_DIR.exists():
        return {}

    for stats_path in sorted(STATS_DIR.glob("*_stats.json")):
        with open(stats_path) as f:
            stats_data = json.load(f)

        subject_id = extract_subject_id(stats_path.name)
        condition = stats_data.get("emotion", "").upper()

        if condition in CONDITIONS:
            stats[subject_id][condition] = stats_data

    return dict(stats)


# ---------------------------------------------------------------------- CHECK 1: Completitud
def check_completeness(available_epochs, available_raws):
    """
    Comprueba que cada sujeto tenga las tres condiciones procesadas.

    Return:
        complete_subjects e incomplete_subjects.
    """
    print("CHECK 1 - COMPLETITUD POR SUJETO")
    print(f"  Condiciones requeridas: {CONDITIONS}\n")

    all_subjects = sorted(set(available_epochs.keys()) | set(available_raws.keys()))

    complete_subjects = []
    incomplete_subjects = []
    table = []

    required_conditions = set(CONDITIONS)

    for subject_id in all_subjects:
        epoch_conditions = set(available_epochs.get(subject_id, {}).keys())
        raw_conditions = set(available_raws.get(subject_id, {}).keys())

        missing_epochs = required_conditions - epoch_conditions
        missing_raws = required_conditions - raw_conditions
        raw_without_epoch = raw_conditions - epoch_conditions

        if not missing_epochs:
            complete_subjects.append(subject_id)
            status = "COMPLETO"
        else:
            incomplete_subjects.append(subject_id)
            if raw_without_epoch:
                status = f"CRUDO OK, falta epoch: {sorted(raw_without_epoch)}"
            elif missing_raws:
                status = f"Falta crudo: {sorted(missing_raws)}"
            else:
                status = f"Falta epoch: {sorted(missing_epochs)}"

        table.append((subject_id, status, epoch_conditions, raw_conditions))

    print(f"  {'Sujeto':<20} {'Estado':<45} {'Epochs':<25} {'Crudos'}")
    print("  " + "-" * 110)
    for subject_id, status, epoch_conditions, raw_conditions in table:
        print(
            f"  {subject_id:<20} {status:<45} "
            f"{str(sorted(epoch_conditions)):<25} {sorted(raw_conditions)}"
        )

    print(f"\n  Sujetos completos:   {len(complete_subjects)}")
    print(f"  Sujetos incompletos: {len(incomplete_subjects)}")

    if incomplete_subjects:
        print("\n  Sujetos a revisar:")
        for subject_id in incomplete_subjects:
            epoch_conditions = set(available_epochs.get(subject_id, {}).keys())
            raw_conditions = set(available_raws.get(subject_id, {}).keys())
            raw_without_epoch = raw_conditions - epoch_conditions
            missing_conditions = required_conditions - epoch_conditions

            if raw_without_epoch:
                print(f"    {subject_id}: reejecutar edf_to_epochs.py para {sorted(raw_without_epoch)}")
            else:
                print(f"    {subject_id}: no hay epochs para {sorted(missing_conditions)}")

    return complete_subjects, incomplete_subjects


# ---------------------------------------------------------------------- CHECK 2: Balance de épocas
def read_epoch_count(available_epochs, stats_data, subject_id, condition) -> int:
    """Lee el número de épocas desde stats; si falta, lo calcula desde el FIF."""
    if condition in stats_data.get(subject_id, {}):
        return int(stats_data[subject_id][condition]["epochs_final"])

    fif_path = available_epochs.get(subject_id, {}).get(condition)
    if fif_path is None:
        return 0

    epochs = mne.read_epochs(fif_path, preload=False, verbose=False)
    return len(epochs)


def check_epoch_balance(available_epochs, stats_data, complete_subjects):
    """
    Revisa si cada sujeto tiene un número razonablemente balanceado de épocas.

    Return:
        unbalanced_subjects y balance_table.
    """
    print("CHECK 2 - BALANCE DE EPOCAS POR SUJETO")
    print(f"  Umbral: condición mínima >= {BALANCE_THRESHOLD * 100:.0f}% de la máxima\n")

    unbalanced_subjects = []
    balance_table = []

    for subject_id in complete_subjects:
        epoch_counts = {}
        for condition in CONDITIONS:
            epoch_counts[condition] = read_epoch_count(
                available_epochs, stats_data, subject_id, condition
            )

        values = list(epoch_counts.values())
        max_epochs = max(values)
        min_epochs = min(values)
        ratio = min_epochs / max_epochs if max_epochs > 0 else 0

        is_unbalanced = ratio < BALANCE_THRESHOLD
        if is_unbalanced:
            unbalanced_subjects.append(subject_id)
            note = f"DESEQUILIBRIO ({ratio:.0%})"
        else:
            note = ""

        balance_table.append((subject_id, epoch_counts, ratio, note))

    print(f"  {'Sujeto':<20} {'JOY':>6} {'NEUTRO':>8} {'SAD':>6} {'Ratio min/max':>14}  Nota")
    print("  " + "-" * 82)
    for subject_id, epoch_counts, ratio, note in balance_table:
        print(
            f"  {subject_id:<20} "
            f"{epoch_counts.get('JOY', 0):>6} "
            f"{epoch_counts.get('NEUTRO', 0):>8} "
            f"{epoch_counts.get('SAD', 0):>6} "
            f"{ratio:>13.0%}  {note}"
        )

    print(f"\n  Sujetos equilibrados:    {len(complete_subjects) - len(unbalanced_subjects)}")
    print(f"  Sujetos desequilibrados: {len(unbalanced_subjects)}")

    if unbalanced_subjects:
        print("  Nota: el desequilibrio no excluye automáticamente, pero conviene revisarlo.")

    return unbalanced_subjects, balance_table


def plot_epoch_balance(complete_subjects, balance_table):
    """Genera una figura con el balance de épocas por sujeto y condición."""
    if not complete_subjects:
        print("\n  No hay sujetos completos para generar figura de balance")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(14, len(complete_subjects) * 0.6), 5))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    x = np.arange(len(complete_subjects))
    width = 0.28
    short_ids = [subject_id.replace("211-000", "") for subject_id in complete_subjects]

    for i, condition in enumerate(CONDITIONS):
        values = []
        for subject_id in complete_subjects:
            row = next((r for r in balance_table if r[0] == subject_id), None)
            values.append(row[1].get(condition, 0) if row else 0)

        ax.bar(
            x + i * width,
            values,
            width,
            label=condition,
            color=COLORS[condition],
            alpha=0.85,
        )

    ax.axhline(10, color="red", ls="--", lw=1, label="mínimo recomendado (10)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(short_ids, rotation=45, ha="right", fontsize=8)
    ax.set_title("Balance de épocas por sujeto y condición", fontsize=12, fontweight="bold")
    ax.set_ylabel("Nº épocas válidas")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = RESULTS_DIR / "01_balance_epochs.png"
    plt.savefig(output_path, dpi=120)
    print(f"\n  Figura guardada: {output_path}")
    plt.show()


# ---------------------------------------------------------------------- CHECK 3: Espectro individual
def compute_mean_spectrum(fif_path: Path):
    """Calcula el espectro PSD medio de un archivo de epochs."""
    epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
    spectrum = epochs.compute_psd(
        method="welch",
        fmin=1,
        fmax=40,
        n_fft=N_FFT,
        n_overlap=N_OVERLAP,
        picks="eeg",
        verbose=False,
    )

    psd = spectrum.get_data()          # formato: (n_ep, n_ch, n_freq)
    freqs = spectrum.freqs
    mean_psd = psd.mean(axis=(0, 1)) * 1e12

    return freqs, mean_psd


def detect_spectrum_status(freqs, mean_psd) -> dict:
    """
    Detecta automáticamente si el espectro pasa los criterios mínimos.

    Return:
        dict con métricas, criterios y decisión final.
    """
    alpha_mask = (freqs >= 8) & (freqs <= 12)
    beta_mask = (freqs >= 15) & (freqs <= 25)

    alpha_power = float(mean_psd[alpha_mask].max())
    beta_power = float(mean_psd[beta_mask].mean())
    low_freq_power = float(mean_psd[0])
    high_freq_power = float(mean_psd[-1])

    alpha_ratio = alpha_power / beta_power if beta_power > 0 else np.inf
    slope_ratio = low_freq_power / high_freq_power if high_freq_power > 0 else np.inf

    has_alpha_peak = alpha_ratio >= ALPHA_PEAK_RATIO
    has_slope = slope_ratio >= SLOPE_RATIO
    is_valid = has_alpha_peak and has_slope

    reasons = []
    if not has_alpha_peak:
        reasons.append("sin_pico_alpha_suficiente")
    if not has_slope:
        reasons.append("sin_pendiente_1f_suficiente")

    return {
        "is_valid": bool(is_valid),
        "has_alpha_peak": bool(has_alpha_peak),
        "has_slope": bool(has_slope),
        "alpha_power": alpha_power,
        "beta_power": beta_power,
        "alpha_beta_ratio": float(alpha_ratio),
        "low_freq_power": low_freq_power,
        "high_freq_power": high_freq_power,
        "slope_ratio": float(slope_ratio),
        "reasons": reasons,
    }


def check_individual_spectrum(available_epochs, complete_subjects):
    """
    Ejecuta el check espectral automático por condición.

    Return:
        spectrum_outliers y spectrum_report.
    """
    print("CHECK 3 - ESPECTRO INDIVIDUAL POR SUJETO")
    print("  Criterios automáticos:")
    print(f"  - Pico alpha/beta >= {ALPHA_PEAK_RATIO:.2f}")
    print(f"  - Pendiente 1/f >= {SLOPE_RATIO:.2f}\n")

    if not complete_subjects:
        print("  No hay sujetos completos para calcular espectros")
        return [], {}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_subjects = len(complete_subjects)
    n_cols = SPECTRUM_N_COLS
    n_rows = int(np.ceil(n_subjects / n_cols))
    spectrum_outliers = set()
    spectrum_report = {}

    for condition in CONDITIONS:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 4.5, n_rows * 3.2),
            sharex=True,
        )
        fig.patch.set_facecolor("#f8f9fa")
        fig.suptitle(f"Espectro PSD individual - {condition}", fontsize=13, fontweight="bold")

        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        condition_outliers = []
        spectrum_report[condition] = {}

        for i, subject_id in enumerate(complete_subjects):
            ax = axes_flat[i]
            ax.set_facecolor("#ffffff")

            fif_path = available_epochs[subject_id].get(condition)
            if fif_path is None:
                ax.set_visible(False)
                continue

            freqs, mean_psd = compute_mean_spectrum(fif_path)
            ax.semilogy(freqs, mean_psd, color=COLORS[condition], linewidth=1.2, alpha=0.9)

            alpha_mask = (freqs >= 8) & (freqs <= 12)
            ax.fill_between(freqs[alpha_mask], mean_psd[alpha_mask], alpha=0.25, color="#f39c12")

            spectrum_status = detect_spectrum_status(freqs, mean_psd)
            spectrum_report[condition][subject_id] = spectrum_status
            is_outlier = not spectrum_status["is_valid"]

            if is_outlier:
                spectrum_outliers.add(subject_id)
                condition_outliers.append(subject_id)
                ax.set_facecolor("#fff0f0")
                ax.set_title(
                    f"{subject_id.replace('211-000', '')}\nEXCLUIDO",
                    fontsize=8,
                    color="#c0392b",
                    fontweight="bold",
                )
            else:
                ax.set_title(
                    f"{subject_id.replace('211-000', '')}\nOK",
                    fontsize=8,
                    color="#27ae60",
                )

            ax.set_xlim(1, 40)
            ax.grid(True, alpha=0.3, linestyle=":")
            ax.tick_params(labelsize=7)
            ax.set_xlabel("Hz", fontsize=7)
            ax.set_ylabel("pV²/Hz", fontsize=7)

        for j in range(len(complete_subjects), len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.tight_layout()
        output_path = RESULTS_DIR / f"02_spectrum_{condition}.png"
        plt.savefig(output_path, dpi=110)
        print(f"  Figura guardada: {output_path}")

        if condition_outliers:
            print(f"    Sujetos excluidos por espectro en {condition}: {condition_outliers}")
        else:
            print(f"    Todos los sujetos de {condition} pasan el check espectral")

        plt.show()

    return sorted(spectrum_outliers), spectrum_report


# ---------------------------------------------------------------------- RESUMEN FINAL
def save_valid_subjects(valid_subjects):
    """Guarda la lista de sujetos aptos para que la use epochs_to_features.py."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(VALID_SUBJECTS_FILE, "w") as f:
        f.write("# Sujetos aptos para el modelo - generado por check_valid_subjects.py\n")
        f.write("# Formato: subject_id\n")
        for subject_id in valid_subjects:
            f.write(subject_id + "\n")

    print(f"\n  Lista guardada en: {VALID_SUBJECTS_FILE}")


def build_invalid_subjects_report(incomplete_subjects, spectrum_outliers, spectrum_report):
    """Construye un diccionario con los sujetos descartados y sus motivos."""
    invalid_report = {}

    for subject_id in incomplete_subjects:
        invalid_report.setdefault(subject_id, []).append({
            "reason": "incomplete_conditions",
            "detail": "faltan una o más condiciones procesadas",
        })

    for subject_id in spectrum_outliers:
        failed_conditions = []

        for condition, condition_report in spectrum_report.items():
            subject_report = condition_report.get(subject_id)
            if subject_report and not subject_report["is_valid"]:
                failed_conditions.append({
                    "condition": condition,
                    "reasons": subject_report["reasons"],
                    "alpha_beta_ratio": subject_report["alpha_beta_ratio"],
                    "slope_ratio": subject_report["slope_ratio"],
                })

        invalid_report.setdefault(subject_id, []).append({
            "reason": "spectrum_outlier",
            "detail": failed_conditions,
        })

    return invalid_report


def save_invalid_subjects(invalid_report):
    """Guarda la lista de sujetos descartados con sus motivos."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(INVALID_SUBJECTS_FILE, "w") as f:
        f.write("# Sujetos descartados - generado por check_valid_subjects.py\n")
        f.write("# Formato: subject_id | motivo | detalle\n")

        if not invalid_report:
            f.write("# Ningún sujeto descartado\n")
        else:
            for subject_id in sorted(invalid_report):
                for item in invalid_report[subject_id]:
                    reason = item["reason"]
                    detail = item["detail"]

                    if reason == "spectrum_outlier":
                        detail_text = []
                        for failed in detail:
                            detail_text.append(
                                f"{failed['condition']}:"
                                f"{','.join(failed['reasons'])}"
                                f" alpha_beta={failed['alpha_beta_ratio']:.3f}"
                                f" slope={failed['slope_ratio']:.3f}"
                            )
                        detail = "; ".join(detail_text)

                    f.write(f"{subject_id} | {reason} | {detail}\n")

    print(f"  Lista de descartados guardada en: {INVALID_SUBJECTS_FILE}")


def save_results_json(
    complete_subjects,
    incomplete_subjects,
    unbalanced_subjects,
    spectrum_outliers,
    valid_subjects,
    invalid_report,
    spectrum_report,
):
    """Guarda un resumen automático del check en results/checks."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "criteria": {
            "required_conditions": CONDITIONS,
            "balance_threshold": BALANCE_THRESHOLD,
            "alpha_peak_ratio": ALPHA_PEAK_RATIO,
            "slope_ratio": SLOPE_RATIO,
            "excluded_if_incomplete": True,
            "excluded_if_spectrum_outlier": True,
            "excluded_if_unbalanced": False,
        },
        "counts": {
            "complete_subjects": len(complete_subjects),
            "incomplete_subjects": len(incomplete_subjects),
            "unbalanced_subjects": len(unbalanced_subjects),
            "spectrum_outliers": len(spectrum_outliers),
            "valid_subjects": len(valid_subjects),
        },
        "subjects": {
            "complete": complete_subjects,
            "incomplete": incomplete_subjects,
            "unbalanced": unbalanced_subjects,
            "spectrum_outliers": spectrum_outliers,
            "valid": valid_subjects,
            "invalid": sorted(invalid_report.keys()),
        },
        "invalid_report": invalid_report,
        "spectrum_report": spectrum_report,
    }

    output_path = RESULTS_DIR / "check_valid_subjects_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Resumen JSON guardado en: {output_path}")


def final_summary(
    complete_subjects,
    incomplete_subjects,
    unbalanced_subjects,
    spectrum_outliers,
    spectrum_report,
):
    """
    Imprime el resumen final y guarda valid_subjects.txt.

    Return:
        valid_subjects.
    """
    print("\n" + "=" * 70)
    print("RESUMEN FINAL - SUJETOS APTOS PARA ENTRENAR")
    print("=" * 70)

    excluded = set(incomplete_subjects) | set(spectrum_outliers)
    valid_subjects = [subject_id for subject_id in complete_subjects if subject_id not in excluded]
    invalid_report = build_invalid_subjects_report(
        incomplete_subjects,
        spectrum_outliers,
        spectrum_report,
    )
    warning_subjects = [
        subject_id
        for subject_id in complete_subjects
        if subject_id in set(unbalanced_subjects) and subject_id not in excluded
    ]

    print(f"\n  APTOS: {len(valid_subjects)}")
    for subject_id in valid_subjects:
        print(f"      {subject_id}")

    if warning_subjects:
        print(f"\n  APTOS CON AVISO (épocas desbalanceadas): {len(warning_subjects)}")
        for subject_id in warning_subjects:
            print(f"      {subject_id}")

    if incomplete_subjects:
        print(f"\n  INCOMPLETOS (excluir): {len(incomplete_subjects)}")
        for subject_id in incomplete_subjects:
            print(f"      {subject_id}")

    if spectrum_outliers:
        print(f"\n  EXCLUIDOS POR ESPECTRO AUTOMÁTICO: {len(spectrum_outliers)}")
        for subject_id in spectrum_outliers:
            print(f"      {subject_id}")

    print(f"\n  Dataset recomendado: {len(valid_subjects)} sujetos x 3 condiciones")
    print(f"  Figuras guardadas en: {RESULTS_DIR}")

    save_valid_subjects(valid_subjects)
    save_invalid_subjects(invalid_report)
    save_results_json(
        complete_subjects,
        incomplete_subjects,
        unbalanced_subjects,
        spectrum_outliers,
        valid_subjects,
        invalid_report,
        spectrum_report,
    )

    return valid_subjects


# ---------------------------------------------------------------------- MAIN
if __name__ == "__main__":
    print("CHECK DE INTEGRIDAD DEL DATASET EEG!!!!!!!!!")

    available_epochs = load_available_epochs()
    available_raws   = load_available_raws()
    stats_data       = load_preprocessing_stats()

    #To Do: print how many subjects total

    complete_subjects, incomplete_subjects = check_completeness(
        available_epochs,
        available_raws,
    )
    unbalanced_subjects, balance_table = check_epoch_balance(
        available_epochs,
        stats_data,
        complete_subjects,
    )
    plot_epoch_balance(complete_subjects, balance_table)

    spectrum_outliers, spectrum_report = check_individual_spectrum(
        available_epochs,
        complete_subjects,
    )

    valid_subjects = final_summary(
        complete_subjects,
        incomplete_subjects,
        unbalanced_subjects,
        spectrum_outliers,
        spectrum_report,
    )

    print("\nCHECK COMPLETADO!!!!!!!!!")
