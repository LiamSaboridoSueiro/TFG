"""
Script para procesar todos los ficheros edf a épocas usando edf_to_epochs.py
"""
from pathlib import Path
import subprocess
import sys

# Config de paths
SCRIPT_PREPROCESS = Path("edf_to_epochs.py").resolve()
RAW_BASE_DIR = Path("../../data/raw/edf").resolve()
EMOTIONS = ["Joy", "Sad", "Neutro"]

def main():

    if not SCRIPT_PREPROCESS.exists():
        raise RuntimeError(f"No se encuentra {SCRIPT_PREPROCESS}")

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
                result = subprocess.run(
                    [sys.executable, str(SCRIPT_PREPROCESS), str(edf_path)],
                    check=True
                )
                processed += 1

            except subprocess.CalledProcessError:
                failed += 1
                print("ERROR ----------------------------------------- procesando este archivo")

    print("\n---------------- Todos los archivos se han preprocesado ----------------")
    print(f"Total EDF encontrados : {total_files}")
    print(f"Procesados correctamente: {processed}")
    print(f"Fallidos               : {failed}")


if __name__ == "__main__":
    main()
