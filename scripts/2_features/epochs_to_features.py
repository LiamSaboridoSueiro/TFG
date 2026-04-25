"""
Script de preprocesamiento: Epochs -> Features 
    Welch PSD por época -> Integración por bandas -> Linealiza la distribución 1/f -> Normalización inter-sujeto ->  Features derivadas opcionales

Salida:
  data/features/features_X.npy                  (n_épocas, n_features)
  data/features/features_y.npy                  (n_épocas,)        0=JOY 1=NEUTRO 2=SAD
  data/features/features_meta.csv               columnas: subject_id, condition, epoch_idx
  data/features/features_info.json              parámetros usados y estadísticas del dataset

"""

import mne
import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# ---------------------------------------------------------------------- CONFIGURACIÓN INICIAL
EPOCHS_DIR   = Path("data/processed/epochs")
FEATURES_DIR = Path("data/features")
SUJETOS_FILE = Path("data/checks/valid_subjects.txt")   # generado por check_valid_subjects.py

CONDITIONS  = ["JOY", "NEUTRO", "SAD"]
LABEL_MAP   = {"JOY": 0, "NEUTRO": 1, "SAD": 2}

SFREQ     = 250                           # muestras por seg
TMIN      = -1.5
TMAX      =  1.0
N_FFT     = int((TMAX - TMIN) * SFREQ)    # 625 muestras por época
N_OVERLAP = N_FFT // 2                    # 50% solapamiento, cada ventana comparte la mitad de sus muestras con la siguiente

# Bandas de actividad de más lento a más rápido
BANDS = {
    "Delta": (1,  4),
    "Theta": (4,  8),
    "Alpha": (8,  12),
    "Beta":  (12, 30),
    "Gamma": (30, 40),
}

N_BANDS = len(BANDS)

BAND_TO_IDX = {}
for idx, band in enumerate(BANDS.keys()):
    BAND_TO_IDX[band] = idx

# Estrategia de normalización por sujeto
# "all"            -> media/desviación estándar                  sobre todas las épocas del sujeto
# "neutral"        -> media/desviación estándar                  solo sobre NEUTRO
# "neutral_robust" -> mediana/desviación absoluta mediana        solo sobre NEUTRO
ZSCORE_MODE = "neutral"

# Features derivadas. Mantienen interpretabilidad y suelen ser más estables
# cross-subject que la potencia bruta por banda
INCLUDE_THETA_ALPHA_RATIO = True
INCLUDE_ALPHA_ASYMMETRY   = True

# Pares homólogos para asimetría Alpha
# La convención es derecha - izquierda
ALPHA_ASYMMETRY_PAIRS = [
    ("Fp1", "Fp2"),
    ("AF7", "AF8"),
    ("AF3", "AF4"),
    ("F7", "F8"),
    ("F5", "F6"),
    ("F3", "F4"),
    ("F1", "F2"),
    ("FC5", "FC6"),
    ("FC3", "FC4"),
    ("FC1", "FC2"),
    ("T7", "T8"),
    ("C5", "C6"),
    ("C3", "C4"),
    ("C1", "C2"),
    ("TP7", "TP8"),
    ("CP5", "CP6"),
    ("CP3", "CP4"),
    ("CP1", "CP2"),
    ("P7", "P8"),
    ("P5", "P6"),
    ("P3", "P4"),
    ("P1", "P2"),
    ("PO7", "PO8"),
    ("PO5", "PO6"),
    ("PO3", "PO4"),
    ("O1", "O2"),
]

def load_valid_subjects() -> list[str]:
    """Lee la lista de sujetos aptos generada por check_valid_subjects.py."""
    if not SUJETOS_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró {SUJETOS_FILE}. "
            "Ejecuta primero check_valid_subjects.py"
        )
    sujetos = []
    with open(SUJETOS_FILE) as f:
        for linea in f:
            linea = linea.strip()
            if linea and not linea.startswith("#"):
                sujetos.append(linea)
    return sujetos


def find_fif(sujeto_id: str, condicion: str) -> Path | None:
    """Busca el archivo .fif de un sujeto y condición."""
    carpeta = EPOCHS_DIR / condicion
    patron  = f"{sujeto_id}*_{condicion}-epo.fif"
    archivos = list(carpeta.glob(patron))
    if not archivos:
        return None
    return archivos[0]


def compute_band_psd(epochs: mne.Epochs) -> np.ndarray:
    """
    Calcula la potencia media por banda para todas las épocas de un archivo.

    Return:
        array shape (n_epocas, n_canales, n_bandas)  en escala lineal (V²/Hz)
    """
    spectrum = epochs.compute_psd(
        method="welch",
        fmin=1.0, fmax=40.0,
        n_fft=N_FFT,
        n_overlap=N_OVERLAP,
        picks="eeg",
        verbose=False,
    )
    psd   = spectrum.get_data()   # formato: (n_ep, n_ch, n_freq)
    freqs = spectrum.freqs

    n_ep, n_ch, _ = psd.shape
    result = np.zeros((n_ep, n_ch, N_BANDS), dtype=np.float64)

    for b_idx, (band_name, limites_banda) in enumerate(BANDS.items()):

        fmin, fmax = limites_banda
        frecuencias_de_la_banda = (freqs >= fmin) & (freqs < fmax)

        psd_de_la_banda = psd[:, :, frecuencias_de_la_banda]        # nos quedamos solo con la PSD de esas frecuencias

        potencia_media_de_la_banda = psd_de_la_banda.mean(axis=2)   # hacemos la media sobre el eje de frecuencias

        result[:, :, b_idx] = potencia_media_de_la_banda            # guardamos el resultado en la posición de esta banda

    return result


def normalize_subject_features(features_por_cond: dict, mode: str = ZSCORE_MODE) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Normaliza features de un sujeto usando una referencia configurable.

    "neutral" es la opción que recomiendo aquí:
      - evita que JOY/SAD influyan en la escala de normalización
      - reduce la dependencia de cuántas épocas emocionales sobrevivieron
    """
    if "NEUTRO" not in features_por_cond:
        raise ValueError("Falta NEUTRO para normalizar al sujeto")

    if mode == "all":
        referencia = np.concatenate(list(features_por_cond.values()), axis=0)
        mu = referencia.mean(axis=0)
        sigma = referencia.std(axis=0)
    elif mode == "neutral":
        referencia = features_por_cond["NEUTRO"]
        mu = referencia.mean(axis=0)
        sigma = referencia.std(axis=0)
    elif mode == "neutral_robust":
        referencia = features_por_cond["NEUTRO"]
        mu = np.median(referencia, axis=0)
        mad = np.median(np.abs(referencia - mu[np.newaxis, ...]), axis=0)
        sigma = 1.4826 * mad
    else:
        raise ValueError(f"ZSCORE_MODE no reconocido: {mode}")

    sigma = np.where(sigma < 1e-8, 1e-8, sigma)

    features_z = {}
    for cond, arr in features_por_cond.items():
        features_z[cond] = (arr - mu) / sigma

    return features_z, mu, sigma


def get_alpha_asymmetry_pairs(ch_names: list[str]) -> list[tuple[str, str, int, int]]:
    """Devuelve pares homólogos presentes en el montage del sujeto."""
    idx_map = {ch: idx for idx, ch in enumerate(ch_names)}
    pares = []
    for left, right in ALPHA_ASYMMETRY_PAIRS:
        if left in idx_map and right in idx_map:
            pares.append((left, right, idx_map[left], idx_map[right]))
    return pares


def build_feature_blocks(deltas: dict, ch_names: list[str]) -> list[tuple[str, dict, list[str]]]:
    """
    Construye bloques de features 2D a partir de los deltas por banda.

    Cada bloque se normaliza de forma independiente para no mezclar
    distribuciones distintas (potencias, ratios y asimetrías).
    """
    bloques = []

    # Bloque 1: potencia por canal x banda
    bandpower = {}
    for cond, delta in deltas.items():

        n_ep = delta.shape[0]                           # número de épocas de esta condición
        bandpower[cond] = delta.reshape(n_ep, -1)       # aplanamos (n_ep, n_ch, n_bands) → (n_ep, n_ch*n_bands)

    # construimos los nombres de columna en el mismo orden que el reshape
    nombres_bandpower = []
    for ch in ch_names:
        for band in BANDS.keys():
            nombres_bandpower.append(f"{ch}_{band}")    # ej: "Fp1_Alpha", "Fp1_Beta", ...

    bloques.append(("bandpower", bandpower, nombres_bandpower))

    # Bloque 2: ratio Theta/Alpha por canal (opcional)
    if INCLUDE_THETA_ALPHA_RATIO:
        ratio_ta = {}
        for cond, delta in deltas.items():

            theta = delta[:, :, BAND_TO_IDX["Theta"]]  # (n_ep, n_ch) — potencia Theta de cada canal
            alpha = delta[:, :, BAND_TO_IDX["Alpha"]]  # (n_ep, n_ch) — potencia Alpha de cada canal

            ratio_ta[cond] = theta - alpha              # diferencia en escala log: equivale a log(Theta/Alpha)

        nombres_ratio = []
        for ch in ch_names:
            nombres_ratio.append(f"ThetaAlphaRatio_{ch}")

        bloques.append(("theta_alpha_ratio", ratio_ta, nombres_ratio))

    # Bloque 3: asimetría Alpha hemisférica (opcional)
    if INCLUDE_ALPHA_ASYMMETRY:
        pares = get_alpha_asymmetry_pairs(ch_names)  # lista de (left, right, left_idx, right_idx)
        if pares:
            alpha_asym = {}
            for cond, delta in deltas.items():

                cols = []
                for _, _, left_idx, right_idx in pares:
                    alpha_derecha   = delta[:, right_idx, BAND_TO_IDX["Alpha"]]  # (n_ep,)
                    alpha_izquierda = delta[:, left_idx,  BAND_TO_IDX["Alpha"]]  # (n_ep,)
                    asimetria = alpha_derecha - alpha_izquierda                  # positivo → dominancia derecha
                    cols.append(asimetria)

                alpha_asym[cond] = np.column_stack(cols)    # (n_ep, n_pares)

            nombres_asym = []
            for left, right, _, _ in pares:
                nombres_asym.append(f"AlphaAsym_{right}-{left}")

            bloques.append(("alpha_asymmetry", alpha_asym, nombres_asym))

    return bloques



# ---------------------------------------------------------------------- PSD + bandas para todos los sujetos
def extract_all_psd(sujetos: list[str]) -> dict:
    """
    Para cada sujeto y condición, extrae PSD por bandas.

    Return dict:
        datos[sujeto][condicion] = {
            "psd_lineal": array (n_ep, n_ch, n_bands),
            "n_canales":  int,
            "ch_names":   list[str],
            "n_epocas":   int,
        }
    """
    datos = {}
    n_total = len(sujetos) * len(CONDITIONS)
    contador = 0

    for sujeto in sujetos:
        datos[sujeto] = {}

        for cond in CONDITIONS:
            contador += 1
            fif_path = find_fif(sujeto, cond)
            
            if fif_path is None:
                print(f"[{contador}/{n_total}] {sujeto}/{cond}: archivo no encontrado, saltando")
                continue

            ep = mne.read_epochs(fif_path, preload=True, verbose=False)

            ep_eeg   = ep.copy().pick("eeg")    # descartamos canales no-EEG (EOG, etc.)
            n_ep     = len(ep_eeg)
            ch_names = ep_eeg.ch_names

            psd_lin = compute_band_psd(ep_eeg)   # (n_ep, n_ch, n_bands) en escala lineal V²/Hz

            datos[sujeto][cond] = {
                "psd_lineal": psd_lin,
                "n_canales":  len(ch_names),
                "ch_names":   ch_names,
                "n_epocas":   n_ep,
            }

            print(f"[{contador}/{n_total}] {sujeto}/{cond}: {n_ep} épocas, {len(ch_names)} canales")

    return datos


# ---------------------------------------------------------------------- Log10
def apply_log10(psd_lineal: np.ndarray) -> np.ndarray:
    """
    Convierte PSD a escala logarítmica: log10(PSD).

    Se añade un pequeño epsilon para evitar log(0) en épocas con potencia nula.
    El epsilon es 1e-30 V²/Hz, varios órdenes de magnitud por debajo de
    la señal EEG real (~1e-12 V²/Hz), así que no afecta a ningún valor real.
    """
    eps = 1e-30
    return np.log10(psd_lineal + eps)   # (n_ep, n_ch, n_bands)


# ---------------------------------------------------------------------- Δ log-ratio vs NEUTRO
def compute_delta(datos_sujeto: dict) -> dict:
    """
    Para un sujeto, calcula el Δ log-ratio de cada época respecto al
    baseline NEUTRO del mismo sujeto.

    Δ(época_emocion) = log10(PSD_emocion) - mean(log10(PSD_neutro))

    El baseline es la media de TODAS las épocas NEUTRO del sujeto,
    calculada por canal y por banda.

    Return dict condicion -> array (n_ep, n_ch, n_bands) con los deltas.
    """
    if "NEUTRO" not in datos_sujeto:
        raise ValueError("El sujeto no tiene condición NEUTRO — no se puede calcular baseline")

    # Baseline: media sobre épocas NEUTRO  ->  shape (n_ch, n_bands)
    psd_neutro_log = apply_log10(datos_sujeto["NEUTRO"]["psd_lineal"])
    baseline = psd_neutro_log.mean(axis=0)   # (n_ch, n_bands)

    deltas = {}
    for cond in CONDITIONS:
        if cond not in datos_sujeto:
            continue
        psd_log = apply_log10(datos_sujeto[cond]["psd_lineal"])
        deltas[cond] = psd_log - baseline[np.newaxis, :, :]         # Broadcasting: (n_ep, n_ch, n_bands) - (n_ch, n_bands)

    return deltas, baseline


# ---------------------------------------------------------------------- Z-score por sujeto y ensamblado de matrices X, y, meta
def assemble_dataset(sujetos: list[str], datos: dict) -> tuple:
    """
    Para cada sujeto aplica log10 -> delta -> zscore y ensambla las matrices
    globales X, y y el DataFrame de metadatos.

    Shape final de X_norm: (n_epocas_total, n_features_totales)
    Shape final de X_log:  (n_epocas_total, n_canales * n_bandas)  — sin normalizar
    """
    X_log_lista = []   # log10(PSD) sin delta ni zscore — para el CV sin leakage
    X_lista    = []
    y_lista    = []
    meta_lista = []

    zscore_params      = {}     # mu/sigma por sujeto y bloque, para poder invertir la normalización
    feature_names_ref  = None   # nombres de columna de referencia (se fijan con el primer sujeto)
    feature_blocks_ref = None
    ch_names_ref       = None

    for sujeto in sujetos:
        if sujeto not in datos or "NEUTRO" not in datos[sujeto]:
            print(f"{sujeto}: sin condición NEUTRO, saltando")
            continue

        # log10 + delta respecto al baseline NEUTRO
        try:
            deltas, _ = compute_delta(datos[sujeto])
        except ValueError as e:
            print(f"{sujeto}: error al calcular delta — {e}")
            continue

        # construir bloques de features
        ch_names = datos[sujeto]["NEUTRO"]["ch_names"]
        bloques  = build_feature_blocks(deltas, ch_names)

        # recopilamos los nombres de feature de todos los bloques en orden
        feature_names_sujeto = []
        for _, _, nombres in bloques:
            for nombre in nombres:
                feature_names_sujeto.append(nombre)

        if feature_names_ref is None:
            # primer sujeto: fijamos la referencia de nombres y bloques
            feature_names_ref  = feature_names_sujeto
            ch_names_ref       = list(ch_names)
            feature_blocks_ref = []
            for bloque, _, nombres in bloques:
                feature_blocks_ref.append({"name": bloque, "n_features": len(nombres)})
        elif feature_names_sujeto != feature_names_ref:
            raise RuntimeError(
                f"Las features del sujeto {sujeto} no coinciden con las de referencia"
            )

        # z-score independiente por bloque
        bloques_z             = {}
        zscore_params[sujeto] = {}
        for block_name, block_cond, block_names in bloques:
            block_z, mu, sigma = normalize_subject_features(block_cond, mode=ZSCORE_MODE)
            bloques_z[block_name]             = block_z
            zscore_params[sujeto][block_name] = {
                "n_features": len(block_names),
                "mu":    np.asarray(mu).tolist(),
                "sigma": np.asarray(sigma).tolist(),
            }

        # Ensamblado: concatenar bloques y acumular épocas
        for cond in CONDITIONS:
            if cond not in deltas:
                continue

            # log10(PSD) sin normalizar — (n_ep, n_ch * n_bands)
            psd_log = apply_log10(datos[sujeto][cond]["psd_lineal"])
            n_ep    = psd_log.shape[0]
            X_log_lista.append(psd_log.reshape(n_ep, -1))

            # juntamos todos los bloques z-scored de esta condición
            bloques_cond = []
            for block_name, _, _ in bloques:
                bloques_cond.append(bloques_z[block_name][cond])

            features_2d = np.concatenate(bloques_cond, axis=1)     # (n_ep, n_features_totales)
            label       = LABEL_MAP[cond]

            X_lista.append(features_2d)
            y_lista.append(np.full(n_ep, label, dtype=np.int8))

            for ep_idx in range(n_ep):
                meta_lista.append({
                    "subject_id": sujeto,
                    "condition":  cond,
                    "epoch_idx":  ep_idx,
                    "label":      label,
                })

        n_ep_sujeto = 0
        for cond in CONDITIONS:
            if cond in deltas:
                n_ep_sujeto += deltas[cond].shape[0]        # sumamos épocas de cada condición disponible

        print(f"{sujeto}: {n_ep_sujeto} épocas procesadas")

    X_log = np.concatenate(X_log_lista, axis=0).astype(np.float32)
    X     = np.concatenate(X_lista,     axis=0).astype(np.float32)
    y     = np.concatenate(y_lista,     axis=0)
    meta  = pd.DataFrame(meta_lista)

    return X_log, X, y, meta, zscore_params, feature_names_ref, feature_blocks_ref, ch_names_ref


# ---------------------------------------------------------------------- Guardado
def save_results(
    X_log,
    X_norm,
    y,
    meta,
    zscore_params,
    sujetos,
    n_canales,
    ch_names_eeg,
    feature_names,
    feature_blocks,
):
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Arrays y metadatos
    np.save(FEATURES_DIR / "features_X.npy",      X_log)   # log10(PSD) sin normalizar — para CV
    np.save(FEATURES_DIR / "features_X_norm.npy", X_norm)  # log10 + delta + zscore — para exploración
    np.save(FEATURES_DIR / "features_y.npy", y)
    meta.to_csv(FEATURES_DIR / "features_meta.csv", index=False)

    # JSON
    distribucion_y = {}
    for cond, lbl in LABEL_MAP.items():
        distribucion_y[cond] = int((y == lbl).sum())    # épocas por clase tras el ensamblado

    bandas_lista = {}
    for k, v in BANDS.items():
        bandas_lista[k] = list(v)

    info = {
        "n_epocas":   int(X_norm.shape[0]),
        "n_features": int(X_norm.shape[1]),
        "n_canales":  n_canales,
        "ch_names_eeg": ch_names_eeg,
        "n_bandas":   N_BANDS,
        "bandas":     bandas_lista,
        "label_map":  LABEL_MAP,
        "distribucion_y": distribucion_y,
        "sujetos":    sujetos,
        "n_sujetos":  len(sujetos),
        "parametros_welch": {
            "n_fft":         N_FFT,
            "n_overlap":     N_OVERLAP,
            "sfreq":         SFREQ,
            "resolucion_hz": round(SFREQ / N_FFT, 4),  # Hz por bin de frecuencia
        },
        "zscore_mode": ZSCORE_MODE,
        "normalizacion": [
            "log10(PSD)",
            "delta_log_ratio_vs_NEUTRO",
            f"zscore_por_sujeto_{ZSCORE_MODE}",
        ],
        "feature_blocks": feature_blocks,
        "feature_names":  feature_names,
        "zscore_params":  zscore_params,
    }

    with open(FEATURES_DIR / "features_info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Resumen de archivos generados
    print(f"features_X.npy      {X_log.shape}   float32  (log10 sin normalizar — para CV)")
    print(f"features_X_norm.npy {X_norm.shape}  float32  (log10 + delta + zscore — para exploración)")
    print(f"features_y.npy      {y.shape}  int8")
    print(f"features_meta.csv   {len(meta)} filas")
    print(f"features_info.json  {len(feature_names)} features, {len(sujetos)} sujetos")



# ---------------------------------------------------------------------- Diagnóstico rápido post-extracción
def feature_diagnostics(X_norm, y, meta):
    """Imprime estadísticas básicas para verificar que las features tienen sentido."""


    print("DIAGNÓSTICO DE FEATURES!!!!!!!!!")

    # Dimensiones
    print(f"Shape X: {X_norm.shape}   dtype: {X_norm.dtype}")
    print(f"Shape y: {y.shape}   valores únicos: {np.unique(y)}")
    print(f"Normalización: {ZSCORE_MODE}")

    # Distribución de clases
    print("\nDistribución de clases:")
    for cond, lbl in LABEL_MAP.items():
        n = (y == lbl).sum()
        print(f"  {cond} (label={lbl}): {n} épocas  ({n / len(y) * 100:.1f} %)")

    # Épocas por sujeto 
    print("\nÉpocas por sujeto:")
    for sid in meta["subject_id"].unique():
        mask     = meta["subject_id"] == sid
        n_tot    = mask.sum()
        por_cond = meta[mask]["condition"].value_counts()
        detalle  = "  ".join(f"{c}:{por_cond.get(c, 0)}" for c in CONDITIONS)
        print(f"  {sid}: {n_tot:3d} total  ({detalle})")

    # Estadísticas de X por condición 
    print("\nEstadísticas de X por condición:")
    for cond, lbl in LABEL_MAP.items():
        x_cond = X[y == lbl]
        print(f"  {cond}: media={x_cond.mean():+.4f}  std={x_cond.std():.4f}  "
              f"min={x_cond.min():.3f}  max={x_cond.max():.3f}")

    # Comprobación de NaN / Inf 
    n_nan = np.isnan(X).sum()
    n_inf = np.isinf(X).sum()
    print()
    if n_nan > 0 or n_inf > 0:
        print(f"ADVERTENCIA: {n_nan} NaN y {n_inf} Inf en X — revisar épocas con potencia cero")
    else:
        print("Sin NaN ni Inf en X")

    #  Verificación del delta NEUTRO 
    # Δ(NEUTRO - baseline_NEUTRO) ≈ 0 antes del z-score, y ~0 después
    x_neutro     = X[y == LABEL_MAP["NEUTRO"]]
    media_neutro = x_neutro.mean()
    print(f"Media de features NEUTRO: {media_neutro:+.4f}  (esperado: ~0.0)")
    if abs(media_neutro) > 0.5:
        print("ADVERTENCIA: media NEUTRO alejada de 0 — revisar cálculo del delta")
    else:
        print("Media NEUTRO coherente con el delta log-ratio")



# ---------------------------------------------------------------------- MAIN
if __name__ == "__main__":

    print("EXTRACCIÓN DE FEATURES EEG!!!!!!!!!")
    print(f"Epochs dir:   {EPOCHS_DIR.resolve()}")
    print(f"Features dir: {FEATURES_DIR.resolve()}")
    print(f"Welch n_fft={N_FFT}, n_overlap={N_OVERLAP}, resolución={SFREQ/N_FFT:.3f} Hz")
    print(f"Bandas: {list(BANDS.keys())}")
    print(f"Z-score mode: {ZSCORE_MODE}")
    print(f"Theta/Alpha ratio: {'sí' if INCLUDE_THETA_ALPHA_RATIO else 'no'}")
    print(f"Alpha asymmetry:   {'sí' if INCLUDE_ALPHA_ASYMMETRY else 'no'}")

    sujetos = load_valid_subjects()
    print(f"\nSujetos aptos: {len(sujetos)}")

    #  Paso 1-2: PSD + integración por bandas 
    print("\n Calculando PSD Welch por época y banda...")
    datos = extract_all_psd(sujetos)

    # tomamos n_canales del primer sujeto/condición disponible
    n_canales = 0
    for s in datos.values():
        for c in s.values():
            n_canales = c["n_canales"]
            break
        if n_canales:
            break

    print(f"Canales EEG: {n_canales}")
    print(f"Features base por época: {n_canales} × {N_BANDS} = {n_canales * N_BANDS}")

    #  Pasos 3-5: log10 -> delta -> zscore -> ensamblar 
    print("\n[2/3] Aplicando log10 -> delta log-ratio vs NEUTRO -> z-score...")
    X_log, X_norm, y, meta, zscore_params, feature_names, feature_blocks, ch_names_eeg = assemble_dataset(sujetos, datos)

    #  Guardado
    print("\n[3/3] Guardando resultados...")
    save_results(
        X_log, X_norm, y, meta, zscore_params,
        sujetos, n_canales, ch_names_eeg, feature_names, feature_blocks
    )

    feature_diagnostics(X_norm, y, meta)

    print()

    print("EXTRACCIÓN COMPLETADA!!!!!!!!")
    print(f"Archivos en: {FEATURES_DIR.resolve()}")

