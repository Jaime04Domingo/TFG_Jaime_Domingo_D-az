#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:16:22 2026

@author: jaime
"""

import pandas as pd
import numpy as np

# Ruta del archivo problemático
RUTA = "/Users/jaime/Documents/Universidad/TFG/Tráfico.csv"

print("=" * 80)
print("DIAGNÓSTICO DEL DATASET DE TRÁFICO PROBLEMÁTICO")
print("=" * 80)

# Intento de lectura flexible
df = None
errores = []

for sep, enc in [(";", "utf-8-sig"), (",", "utf-8-sig"), (";", "latin1"), (",", "latin1")]:
    try:
        df = pd.read_csv(RUTA, sep=sep, encoding=enc, low_memory=False)
        print(f"\nLectura correcta con sep='{sep}' y encoding='{enc}'")
        break
    except Exception as e:
        errores.append((sep, enc, str(e)))

if df is None:
    print("\nNo se pudo leer el archivo. Errores:")
    for e in errores:
        print(e)
    raise SystemExit

print("\n" + "-" * 80)
print("1. DIMENSIONES GENERALES")
print("-" * 80)
print(f"Filas totales: {len(df):,}")
print(f"Columnas totales: {len(df.columns)}")

print("\n" + "-" * 80)
print("2. NOMBRES DE COLUMNAS")
print("-" * 80)
for i, col in enumerate(df.columns, 1):
    print(f"{i:02d}. {repr(col)}")

print("\n" + "-" * 80)
print("3. DUPLICADOS")
print("-" * 80)
duplicados_totales = df.duplicated().sum()
porc_duplicados = duplicados_totales / len(df) * 100 if len(df) > 0 else 0
print(f"Filas duplicadas exactas: {duplicados_totales:,} ({porc_duplicados:.2f}%)")

# Duplicados ignorando columnas claramente vacías o problemáticas
cols_utiles = [c for c in df.columns if not str(c).startswith("Unnamed")]
if cols_utiles:
    duplicados_sin_unnamed = df[cols_utiles].duplicated().sum()
    porc_dup2 = duplicados_sin_unnamed / len(df) * 100 if len(df) > 0 else 0
    print(f"Duplicados ignorando columnas 'Unnamed': {duplicados_sin_unnamed:,} ({porc_dup2:.2f}%)")

print("\n" + "-" * 80)
print("4. VALORES NULOS")
print("-" * 80)
nulos = df.isna().sum().sort_values(ascending=False)
nulos_pct = (df.isna().mean() * 100).sort_values(ascending=False)

resumen_nulos = pd.DataFrame({
    "nulos": nulos,
    "% nulos": nulos_pct.round(2)
})

print(resumen_nulos.head(15))

print("\nColumnas completamente nulas:")
cols_100 = resumen_nulos[resumen_nulos["% nulos"] == 100].index.tolist()
print(cols_100 if cols_100 else "Ninguna")

print("\n" + "-" * 80)
print("5. COLUMNAS SOSPECHOSAS")
print("-" * 80)
cols_bom = [c for c in df.columns if "ï»¿" in str(c) or "\ufeff" in str(c)]
cols_prov = [c for c in df.columns if "PROVINCIA" in str(c).upper()]
cols_unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]

print(f"Columnas con BOM/codificación rara: {cols_bom if cols_bom else 'Ninguna'}")
print(f"Columnas tipo PROVINCIA: {cols_prov if cols_prov else 'Ninguna'}")
print(f"Columnas Unnamed: {cols_unnamed if cols_unnamed else 'Ninguna'}")

print("\n" + "-" * 80)
print("6. COMPARACIÓN ENTRE COLUMNAS PROVINCIA (si existen varias)")
print("-" * 80)
if len(cols_prov) >= 2:
    for c in cols_prov:
        print(f"\nColumna: {repr(c)}")
        print(f"  Nulos: {df[c].isna().sum():,} ({df[c].isna().mean()*100:.2f}%)")
        print(f"  Valores únicos no nulos: {df[c].dropna().unique()[:10]}")
else:
    print("No hay varias columnas PROVINCIA para comparar.")

print("\n" + "-" * 80)
print("7. TIPO DE DATOS")
print("-" * 80)
print(df.dtypes)

print("\n" + "-" * 80)
print("8. MUESTRA DE FILAS")
print("-" * 80)
print(df.head(10))

print("\n" + "-" * 80)
print("9. RESUMEN AUTOMÁTICO DE POSIBLES PROBLEMAS")
print("-" * 80)

problemas = []

if len(df) > 500000:
    problemas.append(f"volumen muy elevado de filas ({len(df):,}), potencialmente inflado por errores de integración")
if duplicados_totales > 0:
    problemas.append(f"{duplicados_totales:,} filas duplicadas exactas ({porc_duplicados:.2f}%)")
if cols_100:
    problemas.append(f"columnas completamente vacías: {cols_100}")
if cols_bom:
    problemas.append(f"columnas con problemas de codificación/BOM: {cols_bom}")
if len(cols_prov) >= 2:
    problemas.append(f"existencia de varias columnas PROVINCIA: {cols_prov}")
if cols_unnamed:
    problemas.append(f"columnas residuales tipo Unnamed: {cols_unnamed}")

if problemas:
    for p in problemas:
        print(f"- {p}")
else:
    print("No se detectaron problemas graves automáticos.")

print("\n" + "=" * 80)
print("FIN DEL DIAGNÓSTICO")
print("=" * 80)