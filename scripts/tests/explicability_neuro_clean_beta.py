"""
Explicabilidad del test within-subject neuro_clean_beta:

    CV por sujeto -> SHAP lineal out-of-fold -> agregación por feature/canal/banda -> topomaps

Este script explica el modelo pedido por el tutor:
  - sin Gamma
  - con Delta, Theta, Alpha y Beta
  - sin canales periféricos sospechosos

Reutiliza las funciones de explicability_neuro_clean.py para mantener exactamente
el mismo cálculo SHAP, cambiando solo la configuración de bandas/canales.

Salida:
  scripts/tests/neuro_clean_beta/shap_topomap/shap_subject_summary.csv
  scripts/tests/neuro_clean_beta/shap_topomap/shap_feature_importance_by_subject.csv
  scripts/tests/neuro_clean_beta/shap_topomap/shap_feature_importance.csv
  scripts/tests/neuro_clean_beta/shap_topomap/shap_channel_band_importance.csv
  scripts/tests/neuro_clean_beta/shap_topomap/shap_band_importance.csv
  scripts/tests/neuro_clean_beta/shap_topomap/explicability_summary.json
  scripts/tests/neuro_clean_beta/shap_topomap/01_top_features_global.png
  scripts/tests/neuro_clean_beta/shap_topomap/02_band_importance_by_class.png
  scripts/tests/neuro_clean_beta/shap_topomap/03_topomap_abs_<COND>.png
  scripts/tests/neuro_clean_beta/shap_topomap/04_topomap_signed_<COND>.png
"""

import numpy as np
import pandas as pd

import explicability_neuro_clean as base


# ---------------------------------------------------------------------- Configuración
base.TEST_NAME = "neuro_clean_beta"
base.OUTPUT_DIR = base.TESTS_DIR / base.TEST_NAME / "shap_topomap"
base.EXCLUDED_BANDS = {"Gamma"}
base.EXCLUDED_CHANNELS = base.PERIPHERAL_CHANNELS
base.PLOT_BANDS = [band for band in base.BANDS if band not in base.EXCLUDED_BANDS]


# ---------------------------------------------------------------------- MAIN
if __name__ == "__main__":
    base.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("EXPLICABILIDAD TEST NEURO CLEAN + BETA CON SHAP LINEAL OOF!!!!!!!!!!")
    print(f"  Features: {base.FEATURES_DIR}")
    print(f"  Salida:   {base.OUTPUT_DIR}")
    print("  Método:   SHAP lineal exacto para LogisticRegression")
    print("  Validación: SHAP sobre test fold, referencia calculada con train fold")
    print(f"  Excluir bandas: {sorted(base.EXCLUDED_BANDS)}")
    print(f"  Excluir canales: {len(base.EXCLUDED_CHANNELS)}")

    print("\n  Cargando datos...")
    X_log, y, meta, ch_names = base.load_dataset()

    if X_log.shape[1] != len(ch_names) * base.N_BANDS:
        raise ValueError(
            f"Shape inconsistente: X_log.shape[1]={X_log.shape[1]} "
            f"pero se esperaba {len(ch_names)} canales x {base.N_BANDS} bandas = {len(ch_names) * base.N_BANDS}."
        )

    full_feature_names = base.build_feature_names(ch_names)
    selected_indices, selected_feature_names = base.build_selected_feature_indices(
        full_feature_names,
        excluded_bands=base.EXCLUDED_BANDS,
        excluded_channels=base.EXCLUDED_CHANNELS,
    )
    topomap_ch_names = [ch for ch in ch_names if ch not in base.EXCLUDED_CHANNELS]

    print(f"  Épocas:  {len(y)}")
    print(f"  Sujetos: {meta['subject_id'].nunique()}")
    print(f"  Features completas: {len(full_feature_names)}")
    print(f"  Features seleccionadas: {len(selected_feature_names)}")
    print(f"  Bloques seleccionados: {base.feature_block_counts(selected_feature_names)}")
    print(f"  Canales en topomap: {len(topomap_ch_names)}")

    print("\n  Calculando SHAP lineal out-of-fold por sujeto...")
    print(f"  {'Sujeto':<18} {'Modelo':>10} {'N_ep':>7} {'Acc':>8} {'F1':>8} Estado")
    print("  " + "-" * 65)

    feature_rows = []
    subject_summary = []
    skipped_subjects = []

    for subject_id in sorted(meta["subject_id"].unique()):
        mask = (meta["subject_id"] == subject_id).values
        X_subject_log = X_log[mask]
        y_subject = y[mask]

        classes, counts = np.unique(y_subject, return_counts=True)
        min_class = counts.min()
        if len(classes) < len(base.CONDITIONS) or min_class < base.N_FOLDS:
            skipped_subjects.append(subject_id)
            print(
                f"  {subject_id:<18} {'LogReg':>10} {len(y_subject):>7} "
                f"{'-':>8} {'-':>8} pocas_epocas"
            )
            continue

        result = base.evaluate_subject_oof(
            X_subject_log,
            y_subject,
            ch_names,
            selected_indices,
        )

        if result is None:
            skipped_subjects.append(subject_id)
            print(
                f"  {subject_id:<18} {'LogReg':>10} {len(y_subject):>7} "
                f"{'-':>8} {'-':>8} no_lineal"
            )
            continue

        subject_summary.append({
            "subject_id": subject_id,
            "classifier_name": "LogReg",
            "n_epochs": int(len(y_subject)),
            "accuracy": result["accuracy"],
            "f1_macro": result["f1_macro"],
            "confusion": result["confusion"].tolist(),
        })

        feature_rows.extend(
            base.build_subject_feature_rows(
                subject_id,
                result["y_true"],
                selected_feature_names,
                result["shap_values"],
                result["classes"],
            )
        )

        print(
            f"  {subject_id:<18} {'LogReg':>10} {len(y_subject):>7} "
            f"{result['accuracy']:>8.3f} {result['f1_macro']:>8.3f} OK"
        )

    if not feature_rows:
        raise RuntimeError("No se pudo calcular SHAP para ningún sujeto.")

    print("\n  Agregando resultados...")
    feature_subject_df = pd.DataFrame(feature_rows)
    feature_summary_df = base.aggregate_feature_importance(feature_subject_df)
    channel_band_df = base.build_channel_band_importance(feature_summary_df, topomap_ch_names)
    topomap_values = base.build_topomap_values(channel_band_df, topomap_ch_names)
    band_importance_df = base.compute_band_importance(channel_band_df)

    print("\n  Guardando tablas...")
    base.save_tables(
        subject_summary,
        feature_subject_df,
        feature_summary_df,
        channel_band_df,
        band_importance_df,
    )
    base.save_summary_json(
        subject_summary,
        feature_summary_df,
        band_importance_df,
        selected_feature_names,
    )

    print("\n  Generando figuras...")
    base.plot_top_features(feature_summary_df)
    base.plot_band_importance(band_importance_df)
    base.plot_topomap_grid(topomap_values, topomap_ch_names, mode="abs")
    base.plot_topomap_grid(topomap_values, topomap_ch_names, mode="signed")

    base.print_console_summary(subject_summary, feature_summary_df, band_importance_df)

    if skipped_subjects:
        print(f"\n  Sujetos omitidos: {skipped_subjects}")

    print("\n" + "=" * 75)
    print("EXPLICABILIDAD NEURO CLEAN + BETA COMPLETADA!!!!!!!!!!")
    print(f"Resultados en: {base.OUTPUT_DIR}")
    print("=" * 75)
