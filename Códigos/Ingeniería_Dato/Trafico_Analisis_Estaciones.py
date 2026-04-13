#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:55:47 2026

@author: jaime
"""

"""
Lectura del archivo de metadatos de estaciones de aforos permanentes
"""
 
import pandas as pd
 
RUTA = "/Users/jaime/Documents/Universidad/TFG/aforos_trafico_permanentes/300233-5-aforo-trafico-permanentes.csv"
 
print("=" * 65)
print("  METADATOS DE ESTACIONES DE AFOROS")
print("=" * 65)
 
df = None
for enc in ["utf-8", "utf-8-sig", "latin-1"]:
    for sep in [";", ","]:
        try:
            tmp = pd.read_csv(RUTA, sep=sep, encoding=enc, low_memory=False)
            if tmp.shape[1] > 1:
                df = tmp
                print(f"  Encoding: {enc} | Sep: '{sep}'")
                break
        except Exception:
            continue
    if df is not None:
        break
 
if df is None:
    print("No se pudo cargar.")
    exit()
 
df.columns = df.columns.str.strip().str.replace("ï»¿","",regex=False).str.replace(r"^\ufeff","",regex=True)
 
print(f"  Filas: {len(df)} | Columnas: {list(df.columns)}")
print("\n--- CONTENIDO COMPLETO ---")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.max_rows", None)
print(df.to_string())
 
# Buscar específicamente ES20 y ES11
print("\n--- ESTACIONES PROBLEMÁTICAS ---")
for est in ["ES20", "ES11", "ES27", "ES44"]:
    col_est = next((c for c in df.columns if "est" in c.lower() or "fest" in c.lower() or "nº" in c.lower()), None)
    if col_est:
        fila = df[df[col_est].astype(str).str.contains(est, na=False)]
        if not fila.empty:
            print(f"\n{est}:")
            print(fila.to_string())
 
print("\nFIN")
 