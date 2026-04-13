#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 19:53:41 2026

@author: jaime
"""

### Intento MERGE DEFINITIVO
"""
Merge definitivo - Aforos de tráfico permanentes Madrid
Formato: FDIA, FEST, FSEN (4 sentidos × 2 bloques horarios), HOR1-HOR12
- HOR1-HOR12 en filas "-"  → horas 01:00-12:00
- HOR1-HOR12 en filas "="  → horas 13:00-24:00
- 9999 = dato no disponible (centinela oficial)
- Unnamed: 15 = columna vacía residual del exportador
"""
 
import pandas as pd
import numpy as np
import os
import glob
 
CARPETA     = "/Users/jaime/Documents/Universidad/TFG/BD_Trafico_Nueva"
RUTA_SALIDA = "/Users/jaime/Documents/Universidad/TFG/Trafico_Aforos_Definitivo.csv"
 
archivos = sorted(glob.glob(os.path.join(CARPETA, "*.csv")))
print(f"Archivos encontrados: {len(archivos)}")
print(f"Primeros: {[os.path.basename(a) for a in archivos[:4]]}")
print(f"Últimos : {[os.path.basename(a) for a in archivos[-4:]]}")
 
trozos  = []
errores = []
resumen = []
 
for i, arch in enumerate(archivos):
    nombre = os.path.basename(arch)
    print(f"[{i+1:>3}/{len(archivos)}] {nombre}...", end=" ")
 
    try:
        df = pd.read_csv(arch, sep=";", encoding="utf-8", low_memory=False)
    except Exception as e:
        print(f"⚠ ERROR: {e}")
        errores.append(nombre)
        continue
 
    # Limpiar nombre de columnas
    df.columns = (df.columns.str.strip()
                  .str.replace("ï»¿", "", regex=False)
                  .str.replace(r"^\ufeff", "", regex=True))
 
    # Eliminar columna vacía residual
    if "Unnamed: 15" in df.columns:
        df.drop(columns=["Unnamed: 15"], inplace=True)
 
    # Eliminar filas completamente vacías (el relleno del exportador)
    filas_antes = len(df)
    df.dropna(how="all", inplace=True)
    filas_datos = len(df)
 
    # Sustituir 9999 por NaN (centinela de dato no disponible)
    cols_hor = [c for c in df.columns if c.startswith("HOR")]
    for col in cols_hor:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == 9999, col] = np.nan
 
    # Parsear fecha
    df["FDIA"] = pd.to_datetime(df["FDIA"], dayfirst=True, errors="coerce")
    nulos_fecha = df["FDIA"].isnull().sum()
 
    resumen.append({
        "archivo": nombre,
        "filas_brutas": filas_antes,
        "filas_datos": filas_datos,
        "filas_vacias_eliminadas": filas_antes - filas_datos,
        "fecha_min": df["FDIA"].min(),
        "fecha_max": df["FDIA"].max(),
        "nulos_fecha": nulos_fecha
    })
 
    print(f"{filas_antes:>9,} → {filas_datos:>6,} filas | "
          f"{df['FDIA'].min().date() if df['FDIA'].notna().any() else '?'} "
          f"→ {df['FDIA'].max().date() if df['FDIA'].notna().any() else '?'}")
 
    trozos.append(df)
    del df
 
# ── CONCATENAR ───────────────────────────────────────────────────────
print(f"\nConcatenando {len(trozos)} archivos...")
df_final = pd.concat(trozos, ignore_index=True)
del trozos
 
print(f"Filas totales: {len(df_final):,}")
 
# ── DUPLICADOS ───────────────────────────────────────────────────────
antes = len(df_final)
df_final.drop_duplicates(inplace=True)
print(f"Duplicados eliminados: {antes - len(df_final):,} | Restantes: {len(df_final):,}")
 
# ── ORDENAR ──────────────────────────────────────────────────────────
print("Ordenando por fecha y estación...")
df_final.sort_values(["FDIA", "FEST", "FSEN"], inplace=True)
df_final.reset_index(drop=True, inplace=True)
 
# ── RESUMEN FINAL ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESUMEN FINAL")
print("=" * 65)
print(f"  Filas    : {len(df_final):,}")
print(f"  Columnas : {list(df_final.columns)}")
print(f"  Rango    : {df_final['FDIA'].min().date()} → {df_final['FDIA'].max().date()}")
print(f"  Estaciones únicas : {df_final['FEST'].nunique()}")
print(f"  Sentidos únicos   : {sorted(df_final['FSEN'].dropna().unique())}")
 
print(f"\n  Cobertura por año y mes:")
df_final["_anyo"] = df_final["FDIA"].dt.year
df_final["_mes"]  = df_final["FDIA"].dt.month
cob = df_final.groupby(["_anyo","_mes"]).size().unstack(fill_value=0)
print(cob.to_string())
df_final.drop(columns=["_anyo","_mes"], inplace=True)
 
print(f"\n  Nulos por columna horaria:")
cols_hor = [c for c in df_final.columns if c.startswith("HOR")]
for col in cols_hor:
    n = df_final[col].isnull().sum()
    print(f"    {col}: {n:,} ({n/len(df_final)*100:.1f}%)")
 
if errores:
    print(f"\n  ⚠ Archivos con error: {errores}")
 
# ── GUARDAR ──────────────────────────────────────────────────────────
print(f"\nGuardando en:\n  {RUTA_SALIDA}")
df_final.to_csv(RUTA_SALIDA, sep=";", index=False, encoding="utf-8-sig")
tam = os.path.getsize(RUTA_SALIDA) / (1024**2)
print(f"✓ Guardado ({tam:.1f} MB)")
 
# ── RESUMEN POR ARCHIVO ──────────────────────────────────────────────
print(f"\n  DETALLE POR ARCHIVO:")
print(f"  {'Archivo':<55} {'Datos':>7} {'Vacías':>9}")
print(f"  {'-'*73}")
for r in resumen:
    print(f"  {r['archivo']:<55} {r['filas_datos']:>7,} {r['filas_vacias_eliminadas']:>9,}")
 
print("\nFIN — Pega el output en el chat.")