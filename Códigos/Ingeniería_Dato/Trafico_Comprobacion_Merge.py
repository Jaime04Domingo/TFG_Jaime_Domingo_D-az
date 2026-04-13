#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:34:15 2026

@author: jaime
"""

### COMPROBACIÓN DE QUE SE HA HECHO CORRECTAMENTE EL MERGE 

"""
Validación del dataset Trafico_Aforos_Definitivo.csv
Comprueba que todas las transformaciones del merge salieron correctamente
"""
 
import pandas as pd
import numpy as np
 
RUTA = "/Users/jaime/Documents/Universidad/TFG/Trafico_Aforos_Definitivo.csv"
 
print("=" * 65)
print("  VALIDACIÓN - TRAFICO_AFOROS_DEFINITIVO")
print("=" * 65)
 
df = pd.read_csv(RUTA, sep=";", encoding="utf-8-sig", low_memory=False)
df["FDIA"] = pd.to_datetime(df["FDIA"], errors="coerce")
cols_hor = [c for c in df.columns if c.startswith("HOR")]
 
errores = []
 
# ── 1. COLUMNA UNNAMED ELIMINADA ─────────────────────────────────────
print("\n[1] Columna 'Unnamed: 15' eliminada")
if "Unnamed: 15" in df.columns:
    print("  ✗ FALLO — la columna sigue presente")
    errores.append("Unnamed: 15 no eliminada")
else:
    print("  ✓ OK — no está presente")
 
# ── 2. SIN FILAS COMPLETAMENTE VACÍAS ────────────────────────────────
print("\n[2] Sin filas completamente vacías")
vacias = df.isnull().all(axis=1).sum()
if vacias > 0:
    print(f"  ✗ FALLO — quedan {vacias:,} filas completamente vacías")
    errores.append(f"{vacias} filas vacías")
else:
    print(f"  ✓ OK — ninguna fila completamente vacía")
 
# ── 3. SIN VALORES 9999 ───────────────────────────────────────────────
print("\n[3] Sin valores centinela 9999 en columnas horarias")
n_9999 = 0
for col in cols_hor:
    n = (df[col] == 9999).sum()
    n_9999 += n
if n_9999 > 0:
    print(f"  ✗ FALLO — quedan {n_9999:,} valores 9999")
    errores.append(f"{n_9999} valores 9999")
else:
    print(f"  ✓ OK — ningún valor 9999 encontrado")
 
# ── 4. FECHAS PARSEADAS CORRECTAMENTE ───────────────────────────────
print("\n[4] Fechas parseadas correctamente")
nulos_fecha = df["FDIA"].isnull().sum()
if nulos_fecha > 0:
    print(f"  ✗ FALLO — {nulos_fecha:,} fechas no parseadas (NaT)")
    muestra = df[df["FDIA"].isnull()].index[:5].tolist()
    print(f"    Índices ejemplo: {muestra}")
    errores.append(f"{nulos_fecha} fechas nulas")
else:
    print(f"  ✓ OK — todas las fechas parseadas")
    print(f"    Rango: {df['FDIA'].min().date()} → {df['FDIA'].max().date()}")
 
# ── 5. ORDENACIÓN CRONOLÓGICA ────────────────────────────────────────
print("\n[5] Ordenación cronológica correcta")
fechas = df["FDIA"].dropna()
esta_ordenado = (fechas.diff().dropna() >= pd.Timedelta(0)).all()
if not esta_ordenado:
    n_desordenadas = (fechas.diff().dropna() < pd.Timedelta(0)).sum()
    print(f"  ✗ FALLO — {n_desordenadas:,} saltos fuera de orden")
    errores.append("ordenación incorrecta")
else:
    print(f"  ✓ OK — dataset ordenado cronológicamente")
 
# ── 6. SIN DUPLICADOS ────────────────────────────────────────────────
print("\n[6] Sin duplicados")
dups = df.duplicated().sum()
dups_clave = df.duplicated(subset=["FDIA","FEST","FSEN"]).sum()
if dups > 0:
    print(f"  ✗ FALLO — {dups:,} filas 100% duplicadas")
    errores.append(f"{dups} duplicados")
else:
    print(f"  ✓ OK — sin duplicados exactos")
if dups_clave > 0:
    print(f"  ⚠ AVISO — {dups_clave:,} duplicados por clave (FDIA+FEST+FSEN)")
else:
    print(f"  ✓ OK — sin duplicados por clave (FDIA+FEST+FSEN)")
 
# ── 7. COBERTURA COMPLETA POR AÑO Y MES ─────────────────────────────
print("\n[7] Cobertura temporal completa")
df["_anyo"] = df["FDIA"].dt.year
df["_mes"]  = df["FDIA"].dt.month
cob = df.groupby(["_anyo","_mes"]).size().unstack(fill_value=0)
 
# Filas esperadas: 59 estaciones × 4 sentidos × días del mes
# (algunos sentidos pueden no existir, por eso usamos los valores reales como referencia)
# Comprobamos que ningún mes completo tiene 0 registros
huecos = []
for anyo in cob.index:
    for mes in range(1, 13):
        if anyo == 2025 and mes > 3:
            continue  # datos solo hasta marzo 2025
        if mes not in cob.columns or cob.loc[anyo, mes] == 0:
            huecos.append(f"{int(anyo)}-{mes:02d}")
 
if huecos:
    print(f"  ✗ FALLO — meses sin datos: {huecos}")
    errores.append(f"huecos en {huecos}")
else:
    print(f"  ✓ OK — cobertura completa 2021-01 → 2025-03")
 
print(f"\n  Cobertura por año y mes:")
print(cob.to_string())
df.drop(columns=["_anyo","_mes"], inplace=True)
 
# ── 8. ESTACIONES Y SENTIDOS ─────────────────────────────────────────
print("\n[8] Estaciones y sentidos")
n_estaciones = df["FEST"].nunique()
sentidos = sorted(df["FSEN"].dropna().unique())
print(f"  Estaciones únicas : {n_estaciones} (esperadas: 59)")
print(f"  Sentidos únicos   : {sentidos}")
if n_estaciones != 59:
    print(f"  ⚠ AVISO — se esperaban 59 estaciones, hay {n_estaciones}")
 
# ── 9. RANGO DE VALORES HORARIOS ─────────────────────────────────────
print("\n[9] Rango de valores en columnas horarias (sin centinelas)")
print(f"  {'Columna':<8} {'Min':>8} {'Max':>8} {'Media':>8} {'Nulos':>8} {'%Nulos':>8}")
print(f"  {'-'*52}")
for col in cols_hor:
    vals = df[col].dropna()
    n_nulos = df[col].isnull().sum()
    pct = n_nulos / len(df) * 100
    # Avisar si hay negativos (no deberían existir en conteos de vehículos)
    n_neg = (vals < 0).sum()
    flag = " ⚠ negativos" if n_neg > 0 else ""
    print(f"  {col:<8} {vals.min():>8.0f} {vals.max():>8.0f} {vals.mean():>8.1f} {n_nulos:>8,} {pct:>7.1f}%{flag}")
 
# ── 10. RESUMEN FINAL ────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESULTADO DE LA VALIDACIÓN")
print("=" * 65)
if not errores:
    print(f"  ✅ TODAS LAS COMPROBACIONES PASADAS CORRECTAMENTE")
    print(f"  El dataset está listo para la fase de análisis.")
else:
    print(f"  ❌ {len(errores)} PROBLEMA(S) DETECTADO(S):")
    for e in errores:
        print(f"    - {e}")
 
print(f"\n  Filas totales : {len(df):,}")
print(f"  Columnas      : {list(df.columns)}")
print("\nFIN — Pega el output en el chat.")
 