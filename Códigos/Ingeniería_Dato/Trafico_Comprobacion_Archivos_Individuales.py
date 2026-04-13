#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:50:40 2026

@author: jaime
"""

### Comparación de tamaño de archivos
 
import pandas as pd
import os
 
RUTA_GRANDE = "/Users/jaime/Documents/Universidad/TFG/BD_Trafico_Nueva/2024-3-aforo-trafico-permanentes-csv.csv"
RUTA_NORMAL = "/Users/jaime/Documents/Universidad/TFG/BD_Trafico_Nueva/2022-28-aforo-trafico-permanentes-csv.csv"
 
def analizar(ruta, nombre):
    print(f"\n{'='*65}")
    print(f"  {nombre}")
    print(f"  {os.path.basename(ruta)}")
    print(f"  Tamaño: {os.path.getsize(ruta)/1024/1024:.2f} MB")
    print(f"{'='*65}")
 
    df = None
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        for sep in [";", ","]:
            try:
                tmp = pd.read_csv(ruta, sep=sep, encoding=enc, low_memory=False)
                if tmp.shape[1] > 1:
                    df = tmp
                    print(f"  Encoding: {enc} | Sep: '{sep}'")
                    break
            except Exception:
                continue
        if df is not None:
            break
 
    if df is None:
        print("  ⚠ No se pudo cargar")
        return
 
    df.columns = df.columns.str.strip().str.replace("ï»¿","",regex=False).str.replace(r"^\ufeff","",regex=True)
 
    print(f"  Filas totales  : {len(df):,}")
    print(f"  Columnas ({df.shape[1]}): {list(df.columns)}")
 
    # Filas con todos NaN (excepto posibles columnas de texto)
    filas_vacias = df.isnull().all(axis=1).sum()
    filas_datos  = len(df) - filas_vacias
    print(f"  Filas con datos: {filas_datos:,}")
    print(f"  Filas vacías   : {filas_vacias:,}")
 
    # Primeras filas
    print(f"\n  PRIMERAS 5 FILAS:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.head(5).to_string())
 
    # Últimas filas
    print(f"\n  ÚLTIMAS 5 FILAS:")
    print(df.tail(5).to_string())
 
    # Valores únicos en columnas clave
    for col in ["FDIA", "FEST", "FSEN"]:
        if col in df.columns:
            unicos = df[col].dropna().unique()
            print(f"\n  '{col}' → {df[col].nunique()} únicos | muestra: {list(unicos[:8])}")
 
    # Rango de fechas
    if "FDIA" in df.columns:
        fechas = pd.to_datetime(df["FDIA"].dropna(), dayfirst=True, errors="coerce")
        print(f"\n  Rango fechas: {fechas.min().date()} → {fechas.max().date()}")
        print(f"  Días distintos: {fechas.nunique()}")
 
    # Estadísticas de HOR
    cols_hor = [c for c in df.columns if c.startswith("HOR")]
    if cols_hor:
        for col in cols_hor:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"\n  Valores máximos por columna horaria:")
        print(f"  {df[cols_hor].max().to_dict()}")
        n_9999 = (df[cols_hor] == 9999).sum().sum()
        print(f"  Valores 9999 (centinela): {n_9999:,}")
 
analizar(RUTA_GRANDE, "ARCHIVO GRANDE (18 MB)")
analizar(RUTA_NORMAL, "ARCHIVO NORMAL (<1 MB)")
 
print("\n\nFIN — Pega el output en el chat.")
 