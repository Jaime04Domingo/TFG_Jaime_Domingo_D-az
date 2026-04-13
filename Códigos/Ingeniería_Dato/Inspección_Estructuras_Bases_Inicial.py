#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:31:44 2026

@author: jaime
"""

# -*- coding: utf-8 -*-
"""
00_inspeccion_estructura_bases.py

Objetivo:
- Entender cómo están realmente estructuradas las bases de datos del TFG
- Ver nombres reales de columnas
- Detectar columnas de fecha, hora, estación, magnitudes, etc.
- Ver tipos, nulos, ejemplos de valores y columnas candidatas para joins
- Generar un informe TXT y varios CSV de apoyo

NO modifica los archivos originales.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. RUTAS
# =========================================================

BASE_DIR = Path("/Users/jaime/Documents/Universidad/TFG")

TRAFFIC_PATH = BASE_DIR / "Trafico_Aforos_Final.csv"
CLIMATE_PATH = BASE_DIR / "Clima_Final.csv"
POLLUTION_PATH = BASE_DIR / "Contaminacion_Definitivo.csv"
TRAFFIC_META_PATH = BASE_DIR / "aforos_trafico_permanentes" / "300233-5-aforo-trafico-permanentes.csv"

OUTPUT_DIR = BASE_DIR / "Inspeccion_Estructura_Bases"
OUTPUT_DIR.mkdir(exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "informe_estructura.txt"
EXCEL_PATH = OUTPUT_DIR / "inspeccion_estructura.xlsx"


# =========================================================
# 2. UTILIDADES
# =========================================================

def log(msg, f=None):
    print(msg)
    if f:
        f.write(msg + "\n")

def probar_lecturas_csv(path, nombre):
    """
    Intenta varias combinaciones razonables de lectura para adivinar
    separador y encoding.
    """
    intentos = [
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": ",", "encoding": "latin1"},
        {"sep": ";", "encoding": "latin1"},
    ]

    mejor_df = None
    mejor_conf = None
    max_cols = -1

    for cfg in intentos:
        try:
            df = pd.read_csv(path, nrows=200, **cfg)
            ncols = df.shape[1]
            if ncols > max_cols:
                max_cols = ncols
                mejor_df = df
                mejor_conf = cfg
        except Exception:
            continue

    if mejor_df is None:
        raise ValueError(f"No se pudo leer {nombre}: {path}")

    return mejor_df, mejor_conf

def cargar_completo(path, config):
    return pd.read_csv(path, **config)

def limpiar_nombres_columnas(columns):
    cols = []
    for c in columns:
        c2 = str(c).replace("\ufeff", "").strip()
        cols.append(c2)
    return cols

def detectar_columnas_especiales(df):
    cols = [str(c) for c in df.columns]
    cols_upper = [c.upper() for c in cols]

    especiales = {
        "fecha_candidatas": [],
        "hora_candidatas": [],
        "horarias_HOR": [],
        "horarias_Hxx": [],
        "validez_Vxx": [],
        "estacion_candidatas": [],
        "sentido_candidatas": [],
        "magnitud_candidatas": [],
        "coord_candidatas": [],
        "nulos_totales": [],
        "columnas_unnamed": [],
    }

    for c, cu in zip(cols, cols_upper):
        if any(x in cu for x in ["FECHA", "DATE", "FDIA", "ANO", "AÑO", "MES", "DIA", "DAY", "YEAR", "DOY"]):
            especiales["fecha_candidatas"].append(c)

        if any(x in cu for x in ["HORA", "HOUR"]):
            especiales["hora_candidatas"].append(c)

        if cu.startswith("HOR"):
            especiales["horarias_HOR"].append(c)

        if len(cu) == 3 and cu.startswith("H") and cu[1:].isdigit():
            especiales["horarias_Hxx"].append(c)

        if len(cu) == 3 and cu.startswith("V") and cu[1:].isdigit():
            especiales["validez_Vxx"].append(c)

        if any(x in cu for x in ["ESTACION", "STATION", "FEST", "Nº", "N°", "NUM"]):
            especiales["estacion_candidatas"].append(c)

        if any(x in cu for x in ["FSEN", "SENTIDO", "SENSE", "DIRECCION", "DIRECTION"]):
            especiales["sentido_candidatas"].append(c)

        if any(x in cu for x in ["MAGNITUD", "MAG", "CONTAMINANTE", "POLLUT"]):
            especiales["magnitud_candidatas"].append(c)

        if any(x in cu for x in ["LAT", "LON", "LONG", "COORD"]):
            especiales["coord_candidatas"].append(c)

        if cu.startswith("UNNAMED"):
            especiales["columnas_unnamed"].append(c)

        if df[c].isna().all():
            especiales["nulos_totales"].append(c)

    return especiales

def resumen_columnas(df, dataset_name):
    filas = []
    for c in df.columns:
        s = df[c]
        non_null = s.notna().sum()
        nulls = s.isna().sum()
        pct_null = round(nulls / len(df) * 100, 2) if len(df) > 0 else np.nan
        nunique = s.nunique(dropna=True)
        ejemplos = s.dropna().astype(str).head(5).tolist()

        filas.append({
            "dataset": dataset_name,
            "columna": c,
            "dtype": str(s.dtype),
            "nulos": int(nulls),
            "pct_nulos": pct_null,
            "no_nulos": int(non_null),
            "n_unique": int(nunique),
            "ejemplos": " | ".join(ejemplos)
        })
    return pd.DataFrame(filas)

def resumen_numericas(df, dataset_name):
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    out = num.describe().T.reset_index().rename(columns={"index": "columna"})
    out.insert(0, "dataset", dataset_name)
    return out

def probar_parseo_fechas(df, candidatas):
    resultados = []
    for c in candidatas:
        s = df[c]
        if s.dtype.kind in "biufcM":
            # numérica o datetime ya
            try:
                parsed = pd.to_datetime(s, errors="coerce")
                ratio = parsed.notna().mean()
                resultados.append({
                    "columna": c,
                    "parse_ok_ratio": round(float(ratio), 4),
                    "min_fecha": parsed.min(),
                    "max_fecha": parsed.max()
                })
            except Exception:
                resultados.append({
                    "columna": c,
                    "parse_ok_ratio": 0,
                    "min_fecha": None,
                    "max_fecha": None
                })
        else:
            mejor_ratio = 0
            mejor_min = None
            mejor_max = None
            for dayfirst in [True, False]:
                try:
                    parsed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
                    ratio = parsed.notna().mean()
                    if ratio > mejor_ratio:
                        mejor_ratio = ratio
                        mejor_min = parsed.min()
                        mejor_max = parsed.max()
                except Exception:
                    pass

            resultados.append({
                "columna": c,
                "parse_ok_ratio": round(float(mejor_ratio), 4),
                "min_fecha": mejor_min,
                "max_fecha": mejor_max
            })

    return pd.DataFrame(resultados).sort_values("parse_ok_ratio", ascending=False)

def analizar_dataset(path, nombre, f):
    log("\n" + "="*100, f)
    log(f"DATASET: {nombre}", f)
    log("="*100, f)
    log(f"Ruta: {path}", f)

    if not path.exists():
        log("Archivo NO encontrado.", f)
        return None

    # Lectura parcial para adivinar estructura
    muestra, config = probar_lecturas_csv(path, nombre)
    muestra.columns = limpiar_nombres_columnas(muestra.columns)

    log(f"Configuración de lectura elegida: {config}", f)
    log(f"Shape de muestra: {muestra.shape}", f)
    log(f"Columnas detectadas ({len(muestra.columns)}):", f)
    for i, c in enumerate(muestra.columns, 1):
        log(f"  {i:02d}. {c}", f)

    # Carga completa
    df = cargar_completo(path, config)
    df.columns = limpiar_nombres_columnas(df.columns)

    log(f"\nShape completo: {df.shape}", f)
    log(f"Duplicados completos: {df.duplicated().sum()}", f)

    especiales = detectar_columnas_especiales(df)

    log("\nColumnas especiales detectadas:", f)
    for k, v in especiales.items():
        log(f"- {k}: {v}", f)

    # Fechas
    parseo_fechas = pd.DataFrame()
    if especiales["fecha_candidatas"]:
        parseo_fechas = probar_parseo_fechas(df, especiales["fecha_candidatas"])
        log("\nPrueba de parseo de fechas:", f)
        if not parseo_fechas.empty:
            log(parseo_fechas.to_string(index=False), f)

    # Resumen rápido de variables “clave”
    for grupo in ["estacion_candidatas", "sentido_candidatas", "magnitud_candidatas"]:
        cols = especiales[grupo]
        if cols:
            log(f"\nValores de ejemplo para {grupo}:", f)
            for c in cols:
                vals = df[c].dropna().astype(str).unique()[:15]
                log(f"  {c}: {' | '.join(vals)}", f)

    if especiales["horarias_HOR"]:
        log(f"\nDetectadas columnas tipo HORx: {len(especiales['horarias_HOR'])}", f)
    if especiales["horarias_Hxx"]:
        log(f"Detectadas columnas tipo H01-H24: {len(especiales['horarias_Hxx'])}", f)
    if especiales["validez_Vxx"]:
        log(f"Detectadas columnas tipo V01-V24: {len(especiales['validez_Vxx'])}", f)

    # Estadísticos simples
    resumen_cols = resumen_columnas(df, nombre)
    resumen_num = resumen_numericas(df, nombre)

    # Primeras filas
    preview = df.head(10).copy()

    return {
        "df": df,
        "config": config,
        "resumen_columnas": resumen_cols,
        "resumen_numericas": resumen_num,
        "parseo_fechas": parseo_fechas,
        "preview": preview,
        "especiales": especiales,
    }


# =========================================================
# 3. MAIN
# =========================================================

def main():
    resultados = {}

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        log("INSPECCIÓN DE ESTRUCTURA DE LAS BASES DEL TFG", f)
        log("Este script NO modifica ningún archivo original.", f)

        resultados["trafico"] = analizar_dataset(TRAFFIC_PATH, "trafico", f)
        resultados["clima"] = analizar_dataset(CLIMATE_PATH, "clima", f)
        resultados["contaminacion"] = analizar_dataset(POLLUTION_PATH, "contaminacion", f)
        resultados["meta_trafico"] = analizar_dataset(TRAFFIC_META_PATH, "meta_trafico", f)

        log("\n" + "="*100, f)
        log("FIN DE INSPECCIÓN", f)
        log("="*100, f)
        log(f"Informe guardado en: {REPORT_PATH}", f)
        log(f"Excel guardado en: {EXCEL_PATH}", f)

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        for nombre, res in resultados.items():
            if res is None:
                continue

            if res["resumen_columnas"] is not None and not res["resumen_columnas"].empty:
                res["resumen_columnas"].to_excel(writer, sheet_name=f"{nombre[:20]}_cols", index=False)

            if res["resumen_numericas"] is not None and not res["resumen_numericas"].empty:
                res["resumen_numericas"].to_excel(writer, sheet_name=f"{nombre[:20]}_num", index=False)

            if res["parseo_fechas"] is not None and not res["parseo_fechas"].empty:
                res["parseo_fechas"].to_excel(writer, sheet_name=f"{nombre[:20]}_fechas", index=False)

            if res["preview"] is not None and not res["preview"].empty:
                res["preview"].to_excel(writer, sheet_name=f"{nombre[:20]}_preview", index=False)

    print("\nProceso terminado.")
    print(f"Revisa el TXT: {REPORT_PATH}")
    print(f"Revisa el Excel: {EXCEL_PATH}")


if __name__ == "__main__":
    main()