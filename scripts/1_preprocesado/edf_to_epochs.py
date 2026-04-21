"""
Script de preprocesamiento: EDF -> Epochs
    filtrado -> downsample -> detectar malos -> re-ref (excluyendo bads) -> ICA -> interpolar

Salida:
    /data/processed/epochs/<emocion>/<nombre_archivo>-epo.fif                 épocas limpias
    /data/processed/preprocessing_stats/<nombre_archivo>_stats.json           estadísticas del preprocesado por archivo

"""

import mne
from mne.preprocessing import ICA
import sys
from pathlib import Path
import numpy as np
import json
from scipy.stats import kurtosis

ROOT = Path(__file__).resolve().parents[2]


def procesar_edf(edf_path: Path):

    print(f"Preprocesando EEG: {edf_path.name}")

    # Cargando archivo
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)


    # ---------------------------------------------------------------------- EXTRACCIÓN Y LIMPIEZA DE EVENTOS

    # Extraemos eventos del canal STIM
    events = mne.find_events(raw, stim_channel="STIM", verbose=False)

    if len(events) == 0:
        raise RuntimeError(f"No se encontraron eventos STIM en {edf_path.name}")

    event_times = events[:, 0] / raw.info["sfreq"]

    # Limpiamos rebotes de trigger (<100ms)
    clean_events = [events[0]]

    #
    for i in range(1, len(events)):
        tiempo_entre_eventos = event_times[i] - event_times[i - 1]
        if tiempo_entre_eventos > 0.1:
            clean_events.append(events[i])

    # Eventos limpios (Frases y reacciones)
    clean_events = np.array(clean_events)


    # ---------------------------------------------------------------------- SELECCIÓN DE EVENTOS (Solo finales de frase)

    # Identificamos eventos de FIN de frase
    clean_event_times = clean_events[:, 0] / raw.info["sfreq"]
    intervals = np.diff(clean_event_times)

    short_intervals = intervals[intervals < 2.0]
    long_intervals = intervals[intervals >= 2.0]

    # Si no hay intervalos largos, todos los triggers son fin de frase
    # (protocolo sin trigger de inicio grabado)
    if len(long_intervals) == 0:
        fin_mask = np.ones(len(clean_events), dtype=bool)
    else:
        fin_mask = np.concatenate([[False], intervals < 2.0])

    # Eventos seleccionados
    clean_events_fin = clean_events[fin_mask]


    # ----------------------------------------------------------------------  SELECCIÓN DE CANALES EEG

    valid_channels = []
    for letter in "ABCDEFGH":
        for num in range(1, 9):
            valid_channels.append(f"{letter}{num}")

    raw_ch_upper = {}
    for ch in raw.ch_names:
        raw_ch_upper[ch.upper()] = ch

    channels_to_pick = []
    for ch in valid_channels:
        if ch in raw_ch_upper:
            channels_to_pick.append(raw_ch_upper[ch])

    raw_eeg64 = raw.copy().pick_channels(channels_to_pick, verbose=False)


    # ---------------------------------------------------------------------- RENOMBRAR CANALES Y ASIGNAR MONTAJE ESTÁNDAR 10-05

    rename_dict = {
        "A1": "O2",   "A2": "O1",   "A3": "Oz",   "A4": "Pz",   "A5": "P4",  "A6": "CP4",  "A7": "P8",   "A8": "C4",
        "B1": "TP8",  "B2": "T8",   "B3": "P7",   "B4": "P3",   "B5": "CP3", "B6": "CPz",  "B7": "Cz",   "B8": "FC4",
        "C1": "FT8",  "C2": "TP7",  "C3": "C3",   "C4": "FCz",  "C5": "Fz",  "C6": "F4",   "C7": "F8",   "C8": "T7",
        "D1": "FT7",  "D2": "FC3",  "D3": "F3",   "D4": "Fp2",  "D5": "F7",  "D6": "Fp1",  "D7": "Heor", "D8": "Veol",
        "E1": "PO5",  "E2": "PO3",  "E3": "P1",   "E4": "POz",  "E5": "P2",  "E6": "PO4",  "E7": "CP2",  "E8": "P6",
        "F1": "PO6",  "F2": "CP6",  "F3": "C6",   "F4": "PO8",  "F5": "PO7", "F6": "P5",   "F7": "CP5",  "F8": "CP1",
        "G1": "C1",   "G2": "C2",   "G3": "FC2",  "G4": "FC6",  "G5": "C5",  "G6": "FC1",  "G7": "F2",   "G8": "F6",
        "H1": "FC5",  "H2": "F1",   "H3": "AF4",  "H4": "AF8",  "H5": "F5",  "H6": "AF7",  "H7": "AF3",  "H8": "Fpz",
    }

    # Renombramos los canales presentes
    present_rename_dict = {}
    for k, v in rename_dict.items():
        if k in raw_eeg64.ch_names:
            present_rename_dict[k] = v

    raw_eeg64.rename_channels(present_rename_dict, verbose=False)

    # Marcamos los canales oculares para que MNE los trate como EOG
    canales_eog = ["Heor", "Veol"]
    eog_dict = {}
    for canal in canales_eog:
        if canal in raw_eeg64.ch_names:
            eog_dict[canal] = "eog"

    if eog_dict:
        raw_eeg64.set_channel_types(eog_dict)

    # Asignamos el montaje estándar de posiciones anatómicas
    montage = mne.channels.make_standard_montage("standard_1005")
    raw_eeg64.set_montage(montage, match_case=False, on_missing="warn", verbose=False)


    # ---------------------------------------------------------------------- FILTRADO

    # Filtramos antes de detectar canales malos y rereferenciar para quitar ruido de red y deriva lenta de la señal
    raw_eeg64.notch_filter(freqs=50, verbose=False)
    raw_eeg64.filter(l_freq=0.5, h_freq=40, fir_design="firwin", verbose=False)


    # ---------------------------------------------------------------------- DOWNSAMPLING

    # Guardamos la frecuencia original antes de hacer downsampling
    sfreq_orig = raw_eeg64.info["sfreq"]

    # Reducimos a 250 Hz solo si la señal viene con una frecuencia mayor
    if sfreq_orig > 250:
        raw_eeg64.resample(250, verbose=False)

    # Leemos la nueva frecuencia tras el remuestreo
    sfreq_new = raw_eeg64.info["sfreq"]

    # Si hubo cambio de frecuencia, reajustamos las muestras de los eventos
    # para que sigan apuntando al mismo instante temporal
    if sfreq_new != sfreq_orig:
        clean_events_fin_resampled = clean_events_fin.copy()
        clean_events_fin_resampled[:, 0] = np.round(
            clean_events_fin[:, 0] * (sfreq_new / raw.info["sfreq"])
        ).astype(int)
    else:
        clean_events_fin_resampled = clean_events_fin


    # ---------------------------------------------------------------------- DETECCIÓN DE CANALES MALOS

    # Detectamos canales problemáticos antes de la rereferenciación,
    # para que la referencia promedio no se vea afectada por canales rotos

    # Seleccionamos solo los canales EEG, excluyendo los EOG
    picks_eeg = mne.pick_types(raw_eeg64.info, eeg=True, eog=False, exclude=[])

    # Guardamos los nombres de esos canales EEG
    ch_names = []
    for i in picks_eeg:
        ch_names.append(raw_eeg64.ch_names[i])

    # Extraemos la señal de los canales EEG seleccionados
    raw_data = raw_eeg64.get_data(picks=picks_eeg)


    def zscore_internal(x, eps=1e-12):
        """Z-score interno para detección de outliers (no se aplica de momento)."""
        return (x - np.mean(x)) / (np.std(x) + eps)


    # Diccionario donde iremos guardando las marcas de cada canal
    # Ejemplo: flags["Fz"] = ["flatline", "kurtosis alta"]
    flags = {}
    for ch in ch_names:
        flags[ch] = []

    # Calculamos métricas básicas por canal:
    ptp = np.ptp(raw_data, axis=1) # amplitud pico a pico
    var = np.var(raw_data, axis=1) # varianza

    # Calculamos las medianas globales para comparar cada canal
    # con el comportamiento típico del resto
    ptp_med = np.median(ptp)
    var_med = np.median(var)

    # Umbrales para detectar canales planos o con varianza demasiado baja
    ABS_PTP_MIN = 1e-6
    REL_PTP_MIN = 0.02
    REL_VAR_MIN = 0.02

    # Marcamos como flatline los canales con amplitud o varianza anormalmente bajas
    flatline_idx = np.where(
        (ptp < max(ABS_PTP_MIN, REL_PTP_MIN * ptp_med)) |
        (var < REL_VAR_MIN * var_med)
    )[0]

    for i in flatline_idx:
        flags[ch_names[i]].append("flatline")

    # Trabajamos con el logaritmo de la varianza para detectar canales
    # con energía anormalmente alta o baja respecto al resto
    logvar = np.log10(var + 1e-20)
    z_logvar = zscore_internal(logvar)
    Z_LOGVAR_THR = 3.5

    # Marcamos canales con varianza anormal
    for i in np.where(np.abs(z_logvar) > Z_LOGVAR_THR)[0]:
        flags[ch_names[i]].append("varianza anormal")

    # Calculamos la kurtosis de cada canal para detectar distribuciones
    # con picos o colas anómalas
    kurt = kurtosis(raw_data, axis=1, fisher=False, bias=False)
    z_kurt = zscore_internal(kurt)
    Z_KURT_THR = 5.0

    # Marcamos canales con kurtosis anormalmente alta.
    for i in np.where(z_kurt > Z_KURT_THR)[0]:
        flags[ch_names[i]].append("kurtosis alta")

    # Un canal se considera malo si:
    # - está marcado como flatline
    # - acumula dos o más señales de problema
    bad_channels = []
    for ch in ch_names:
        if "flatline" in flags[ch] or len(flags[ch]) >= 2:
            bad_channels.append(ch)

    # Guardamos la lista de canales malos en la estructura de MNE.
    raw_eeg64.info["bads"] = bad_channels


    # ---------------------------------------------------------------------- RE-REFERENCIACIÓN

    # MNE excluye automáticamente los canales en info["bads"] al calcular el promedio
    raw_eeg64.set_eeg_reference("average", projection=False, verbose=False)


    # ---------------------------------------------------------------------- ICA

    # Ajustamos la ICA sobre los canales EEG
    # MNE excluye automáticamente los canales marcados como malos en info["bads"] cuando usamos picks="eeg"
    ica = ICA(n_components=20, random_state=97, method="fastica", max_iter=500)
    ica.fit(raw_eeg64, picks="eeg", verbose=False)

    componentes_excluir = []

    # Detectamos canales de componentes oculares (EOG)
    eog_chs = []
    for ch in raw_eeg64.ch_names:
        indice_canal = raw_eeg64.ch_names.index(ch)
        tipo_canal = mne.channel_type(raw_eeg64.info, indice_canal)
        if tipo_canal == "eog":
            eog_chs.append(ch)

    # Si existen canales EOG, usamos esos canales como referencia para
    # detectar componentes relacionados con parpadeos y movimientos oculares
    if eog_chs:
        eog_inds, _ = ica.find_bads_eog(
            raw_eeg64, ch_name=eog_chs, threshold=3.0, verbose=False
        )
        componentes_excluir.extend(eog_inds)

    # Detección de componentes musculares (EMG)
    # Intentamos detectar componentes con patrón muscular automáticamente
    try:
        muscle_inds, _ = ica.find_bads_muscle(
            raw_eeg64, threshold=0.5, verbose=False
        )

        # Añadimos solo los componentes nuevos para no repetir
        # los que ya se habían marcado como oculares
        muscle_nuevos = []
        for i in muscle_inds:
            if i not in componentes_excluir:
                muscle_nuevos.append(i)

        componentes_excluir.extend(muscle_nuevos)
    except Exception as e:
        print(f" find_bads_muscle no disponible: {e}")

    # Limitamos el número máximo de componentes eliminados para evitar
    # quitar demasiada señal neuronal útil
    MAX_COMPONENTES = 5
    if len(componentes_excluir) > MAX_COMPONENTES:
        componentes_excluir = componentes_excluir[:MAX_COMPONENTES]

    # Indicamos a la ICA qué componentes deben eliminarse
    ica.exclude = componentes_excluir

    # Aplicamos la corrección ICA sobre una copia de la señal
    raw_eeg64_ica = raw_eeg64.copy()
    ica.apply(raw_eeg64_ica, verbose=False)


    # ---------------------------------------------------------------------- INTERPOLACIÓN de canales malos

    if raw_eeg64_ica.info["bads"]:
        raw_eeg64_ica.interpolate_bads(reset_bads=True, verbose=False)


    # ---------------------------------------------------------------------- CREAR EPOCHS

    epochs = mne.Epochs(
        raw_eeg64_ica,
        clean_events_fin_resampled,
        event_id={"frase": 1},
        tmin=-1.5,
        tmax=1.0,
        baseline=None,
        preload=True,
        verbose=False
    )


    # ---------------------------------------------------------------------- RECHAZO DE EPOCHS

    data = epochs.get_data()               # shape: (n_epochs, n_canales, n_tiempos)
    peak_to_peak = np.ptp(data, axis=2)    # shape: (n_epochs, n_canales)

    AMP_THR    = 80e-6      # 80 µV (umbral estándar en EEG para señal en µV sin normalizar)
    MAX_BAD_CH = 3          # máximo de canales con artefacto para aceptar la época

    bad_epoch_mask = (peak_to_peak > AMP_THR).sum(axis=1) > MAX_BAD_CH
    n_rejected = bad_epoch_mask.sum()

    epochs = epochs[np.where(~bad_epoch_mask)[0]]

    if len(epochs) == 0:
        raise RuntimeError(f"Sin epochs válidos tras rechazo en {edf_path.name}")


    # ---------------------------------------------------------------------- GUARDAR RESULTADOS

    emotion = edf_path.stem.split("_")[-1]

    output_dir = ROOT / "data/processed/epochs" / emotion
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{edf_path.stem}-epo.fif"
    epochs.save(output_path, overwrite=True, verbose=False)

    stats = {
        "file": edf_path.stem,
        "emotion": emotion,
        "epochs_final": int(len(epochs)),
        "epochs_rejected": int(n_rejected),
        "bad_channels": bad_channels,
        "ica_components_excluded": [int(x) for x in ica.exclude],
        "sampling_rate_hz": float(raw_eeg64_ica.info["sfreq"]),
    }

    stats_dir = ROOT / "data/processed/stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    stats_path = stats_dir / f"{edf_path.stem}_stats.json"

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


    print(f"Epochs guardados: {output_path}")
    print(f"Estadísticas:     {stats_path}")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        procesar_edf(Path(sys.argv[1]).resolve())
    else:
        RAW_BASE_DIR = ROOT / "data/raw/edf"
        EMOTIONS = ["Joy", "Sad", "Neutro"]

        print("Iniciando preprocesado de EDF a epochs\n")

        total_files = 0
        processed = 0
        failed = 0

        for emotion in EMOTIONS:
            emotion_dir = RAW_BASE_DIR / emotion

            if not emotion_dir.exists():
                print(f"ERROR ----------------------------------------- Carpeta no encontrada: {emotion_dir}")
                continue

            edf_files = sorted(emotion_dir.glob("*.edf"))
            total_files += len(edf_files)

            for edf_path in edf_files:

                try:
                    procesar_edf(edf_path)
                    processed += 1

                except Exception:
                    failed += 1
                    print("ERROR ----------------------------------------- procesando este archivo")

        print("\n---------------- Todos los archivos se han preprocesado ----------------")
        print(f"Total EDF encontrados : {total_files}")
        print(f"Procesados correctamente: {processed}")
        print(f"Fallidos               : {failed}")
