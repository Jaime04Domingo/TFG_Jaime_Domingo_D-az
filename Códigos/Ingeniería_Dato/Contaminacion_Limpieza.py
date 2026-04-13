#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:02:53 2026

@author: jaime
"""

### Análisis Contaminación

"""
Limpieza del dataset de Contaminación - TFG
Sin aplicar umbrales de valores extremos — primero analizamos
"""
 
import pandas as pd
import numpy as np
import os
 
RUTA_ENTRADA = "/Users/jaime/Documents/Universidad/TFG/Contaminación.csv"
RUTA_SALIDA  = "/Users/jaime/Documents/Universidad/TFG/Contaminacion_Final.csv"
 
print("=" * 65)
print("  LIMPIEZA CONTAMINACIÓN - TFG")
print("=" * 65)
 
# ── 1. CARGA ─────────────────────────────────────────────────────────
print("\n[1] Cargando archivo...")
df = pd.read_csv(RUTA_ENTRADA, sep=",", encoding="utf-8-sig", low_memory=False)
df.columns = df.columns.str.strip().str.replace("ï»¿","",regex=False).str.replace(r"^\ufeff","",regex=True)
print(f"  Filas: {len(df):,} | Columnas: {df.shape[1]}")
 
# ── 2. ELIMINAR COLUMNAS PROVINCIA Y ARCHIVO_ORIGEN ──────────────────
print("\n[2] Eliminando columnas innecesarias...")
cols_eliminar = [c for c in df.columns if "PROVINCIA" in c.upper()]
if "__archivo_origen" in df.columns:
    cols_eliminar.append("__archivo_origen")
df.drop(columns=cols_eliminar, inplace=True, errors="ignore")
print(f"  Eliminadas: {cols_eliminar}")
 
# ── 3. CONSTRUIR FECHA ────────────────────────────────────────────────
print("\n[3] Construyendo fecha desde ANO + MES + DIA...")
df["fecha"] = pd.to_datetime(
    df[["ANO","MES","DIA"]].rename(columns={"ANO":"year","MES":"month","DIA":"day"}),
    errors="coerce"
)
print(f"  Rango: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
print(f"  Nulas: {df['fecha'].isnull().sum():,}")
df.drop(columns=["ANO","MES","DIA"], inplace=True)
 
# ── 4. IDENTIFICAR COLUMNAS H y V ────────────────────────────────────
cols_h = [c for c in df.columns if c.startswith("H") and c[1:].isdigit()]
cols_v = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
cols_h = sorted(cols_h, key=lambda x: int(x[1:]))
cols_v = sorted(cols_v, key=lambda x: int(x[1:]))
print(f"\n[4] Columnas H: {len(cols_h)} | Columnas V: {len(cols_v)}")
 
# ── 5. FILTRAR MEDICIONES INVÁLIDAS CON COLUMNAS V ───────────────────
print("\n[5] Filtrando mediciones inválidas (V='N')...")
mediciones_invalidas = 0
for h_col, v_col in zip(cols_h, cols_v):
    mask = df[v_col] == "N"
    n = mask.sum()
    mediciones_invalidas += n
    df.loc[mask, h_col] = np.nan
 
total_mediciones = len(df) * len(cols_h)
print(f"  Total mediciones: {total_mediciones:,}")
print(f"  Inválidas → NaN : {mediciones_invalidas:,} ({mediciones_invalidas/total_mediciones*100:.2f}%)")
 
# Eliminar columnas V
df.drop(columns=cols_v, inplace=True)
print(f"  Columnas V eliminadas")
 
# ── 6. ANÁLISIS DE VALORES POR MAGNITUD (sin aplicar límites) ────────
print("\n[6] ESTADÍSTICAS POR MAGNITUD (sin filtrar extremos)")
print("-" * 65)
magnitudes = sorted(df["MAGNITUD"].unique())
for mag in magnitudes:
    df_mag = df[df["MAGNITUD"] == mag][cols_h]
    todos = df_mag.values.flatten()
    todos = todos[~np.isnan(todos)]
    if len(todos) == 0:
        continue
    negativos = (todos < 0).sum()
    print(f"\n  MAGNITUD {mag}:")
    print(f"    N valores  : {len(todos):,}")
    print(f"    Min        : {todos.min():.2f}")
    print(f"    Max        : {todos.max():.2f}")
    print(f"    Media      : {todos.mean():.2f}")
    print(f"    Mediana    : {np.median(todos):.2f}")
    print(f"    P95        : {np.percentile(todos, 95):.2f}")
    print(f"    P99        : {np.percentile(todos, 99):.2f}")
    print(f"    P99.9      : {np.percentile(todos, 99.9):.2f}")
    print(f"    Negativos  : {negativos:,}")
 
# ── 7. REORGANIZAR Y ORDENAR ─────────────────────────────────────────
print("\n[7] Reorganizando y ordenando...")
cols_inicio = ["fecha","MUNICIPIO","ESTACION","MAGNITUD","PUNTO_MUESTREO"]
cols_inicio = [c for c in cols_inicio if c in df.columns]
cols_resto  = [c for c in df.columns if c not in cols_inicio]
df = df[cols_inicio + cols_resto]
df.sort_values(["fecha","ESTACION","MAGNITUD"], inplace=True)
df.reset_index(drop=True, inplace=True)
 
# ── 8. VALIDACIÓN ────────────────────────────────────────────────────
print("\n[8] Validación...")
errores = []
 
nulos_fecha = df["fecha"].isnull().sum()
print(f"  Fechas nulas    : {nulos_fecha} {'✓' if nulos_fecha==0 else '✗'}")
if nulos_fecha > 0: errores.append(f"{nulos_fecha} fechas nulas")
 
dups = df.duplicated().sum()
print(f"  Duplicados      : {dups} {'✓' if dups==0 else '✗'}")
if dups > 0: errores.append(f"{dups} duplicados")
 
cols_v_rest = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
print(f"  Columnas V rest.: {len(cols_v_rest)} {'✓' if len(cols_v_rest)==0 else '✗'}")
 
df["_anyo"] = df["fecha"].dt.year
df["_mes"]  = df["fecha"].dt.month
cob = df.groupby(["_anyo","_mes"]).size().unstack(fill_value=0)
print(f"\n  Cobertura por año y mes:")
print(cob.to_string())
df.drop(columns=["_anyo","_mes"], inplace=True)
 
# ── 9. RESUMEN ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESUMEN FINAL")
print("=" * 65)
print(f"  Filas      : {len(df):,}")
print(f"  Columnas   : {df.shape[1]}")
print(f"  Rango      : {df['fecha'].min().date()} → {df['fecha'].max().date()}")
print(f"  Estaciones : {df['ESTACION'].nunique()}")
print(f"  Magnitudes : {sorted(df['MAGNITUD'].unique())}")
print(f"  Mediciones inválidas → NaN: {mediciones_invalidas:,}")
 
if not errores:
    print(f"\n  ✅ VALIDACIÓN CORRECTA")
else:
    print(f"\n  ❌ PROBLEMAS: {errores}")
 
# ── 10. GUARDAR ──────────────────────────────────────────────────────
print(f"\n[10] Guardando en:\n  {RUTA_SALIDA}")
df.to_csv(RUTA_SALIDA, sep=";", index=False, encoding="utf-8-sig")
tam = os.path.getsize(RUTA_SALIDA) / (1024**2)
print(f"  ✓ Guardado ({tam:.1f} MB)")
 
print("\nFIN — Pega el output en el chat.")