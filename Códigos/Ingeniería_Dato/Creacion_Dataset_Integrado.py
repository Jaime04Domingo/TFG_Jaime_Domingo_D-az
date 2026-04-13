#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:27:43 2026

@author: jaime
"""

### PRUEBA DATASET INTEGRADO

"""
Construcción del Dataset Integrado Diario - TFG
Combina Tráfico + Contaminación + Clima en granularidad diaria
"""

import pandas as pd
import numpy as np
import os

RUTA_TRAFICO  = "/Users/jaime/Documents/Universidad/TFG/Trafico_Aforos_Final_2.csv"
RUTA_CONTAM   = "/Users/jaime/Documents/Universidad/TFG/Contaminacion_Definitivo_2.csv"
RUTA_CLIMA    = "/Users/jaime/Documents/Universidad/TFG/Clima_Final.csv"
RUTA_SALIDA   = "/Users/jaime/Documents/Universidad/TFG/Dataset_Diario_Integrado.csv"

# Magnitudes a conservar (relacionadas con tráfico)
MAGS_TRAFICO = {
    8:  "NO2",
    7:  "NO",
    12: "NOx",
    10: "PM10",
    9:  "PM25",
    6:  "CO",
    14: "O3",
}

print("=" * 65)
print("  CONSTRUCCIÓN DATASET DIARIO INTEGRADO - TFG")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════════
# BLOQUE 1: TRÁFICO → AGREGACIÓN DIARIA
# ═══════════════════════════════════════════════════════════════════
print("\n[1] Procesando tráfico...")
df_t = pd.read_csv(RUTA_TRAFICO, sep=";", encoding="utf-8-sig", low_memory=False)
df_t["FDIA"] = pd.to_datetime(df_t["FDIA"], errors="coerce")
print(f"  Cargado: {len(df_t):,} filas | {df_t['FEST'].nunique()} estaciones")

cols_hor = [f"HOR{i}" for i in range(1, 13)]

# Paso 1: Asignar hora real a cada columna según FSEN
# FSEN con "-" → HOR1-HOR12 = horas 1 a 12
# FSEN con "=" → HOR1-HOR12 = horas 13 a 24
registros = []
for _, row in df_t.iterrows():
    fecha = row["FDIA"]
    estacion = row["FEST"]
    fsen = str(row["FSEN"])
    offset = 12 if "=" in fsen else 0

    for i, col in enumerate(cols_hor, 1):
        hora_real = i + offset
        valor = row[col]
        if pd.notna(valor):
            registros.append({
                "fecha": fecha,
                "estacion": estacion,
                "hora": hora_real,
                "intensidad": float(valor)
            })

df_t_long = pd.DataFrame(registros)
print(f"  Registros horarios válidos: {len(df_t_long):,}")

# Paso 2: Media de las 24 horas por estación y día
# (sumamos sentidos porque ya están en filas separadas, luego promediamos horas)
df_t_est_dia = (
    df_t_long
    .groupby(["fecha", "estacion", "hora"])["intensidad"]
    .sum()   # suma sentidos para esa hora
    .reset_index()
    .groupby(["fecha", "estacion"])["intensidad"]
    .mean()  # media de las 24 horas → intensidad media horaria diaria por estación
    .reset_index()
)
df_t_est_dia.columns = ["fecha", "estacion", "intensidad_media_horaria"]

# Paso 3: Media entre todas las estaciones → un valor diario para Madrid
df_trafico_diario = (
    df_t_est_dia
    .groupby("fecha")
    .agg(
        trafico_medio=("intensidad_media_horaria", "mean"),
        n_estaciones=("estacion", "nunique")
    )
    .reset_index()
)

print(f"  Dataset tráfico diario: {len(df_trafico_diario):,} días")
print(f"  Rango: {df_trafico_diario['fecha'].min().date()} → {df_trafico_diario['fecha'].max().date()}")
print(f"  Media de estaciones por día: {df_trafico_diario['n_estaciones'].mean():.1f}")
del df_t, df_t_long, df_t_est_dia

# ═══════════════════════════════════════════════════════════════════
# BLOQUE 2: CONTAMINACIÓN → AGREGACIÓN DIARIA
# ═══════════════════════════════════════════════════════════════════
print("\n[2] Procesando contaminación...")
df_c = pd.read_csv(RUTA_CONTAM, sep=";", encoding="utf-8-sig", low_memory=False)
df_c["fecha"] = pd.to_datetime(df_c["fecha"], errors="coerce")
print(f"  Cargado: {len(df_c):,} filas")

# Filtrar solo magnitudes relacionadas con tráfico
df_c = df_c[df_c["MAGNITUD"].isin(MAGS_TRAFICO.keys())].copy()
print(f"  Tras filtrar magnitudes {list(MAGS_TRAFICO.keys())}: {len(df_c):,} filas")

cols_h = [f"H{i:02d}" for i in range(1, 25)]
# Verificar qué columnas H existen realmente
cols_h = [c for c in cols_h if c in df_c.columns]

# Paso 1: Media diaria de H01–H24 por estación y magnitud
df_c["media_diaria"] = df_c[cols_h].mean(axis=1, skipna=True)

# Paso 2: Media entre todas las estaciones por magnitud y día
df_contam_diario = (
    df_c
    .groupby(["fecha", "MAGNITUD"])["media_diaria"]
    .mean()
    .reset_index()
)

# Paso 3: Pivotar → una columna por contaminante
df_contam_pivot = df_contam_diario.pivot(
    index="fecha", columns="MAGNITUD", values="media_diaria"
).reset_index()

# Renombrar columnas con nombres legibles
df_contam_pivot.columns.name = None
rename_map = {"fecha": "fecha"}
for mag, nombre in MAGS_TRAFICO.items():
    if mag in df_contam_pivot.columns:
        rename_map[mag] = nombre
df_contam_pivot.rename(columns=rename_map, inplace=True)

print(f"  Dataset contaminación diario: {len(df_contam_pivot):,} días")
print(f"  Columnas: {list(df_contam_pivot.columns)}")
print(f"  Rango: {df_contam_pivot['fecha'].min().date()} → {df_contam_pivot['fecha'].max().date()}")
del df_c, df_contam_diario

# ═══════════════════════════════════════════════════════════════════
# BLOQUE 3: CLIMA → YA DIARIO
# ═══════════════════════════════════════════════════════════════════
print("\n[3] Procesando clima...")
df_cl = pd.read_csv(RUTA_CLIMA, sep=";", encoding="utf-8-sig", low_memory=False)
df_cl["fecha"] = pd.to_datetime(df_cl["fecha"], errors="coerce")
print(f"  Cargado: {len(df_cl):,} días")
print(f"  Columnas: {list(df_cl.columns)}")
print(f"  Rango: {df_cl['fecha'].min().date()} → {df_cl['fecha'].max().date()}")

# ═══════════════════════════════════════════════════════════════════
# BLOQUE 4: INTEGRACIÓN
# ═══════════════════════════════════════════════════════════════════
print("\n[4] Integrando las tres fuentes...")

# Merge tráfico + contaminación
df = pd.merge(df_trafico_diario, df_contam_pivot, on="fecha", how="inner")
print(f"  Tráfico ∩ Contaminación: {len(df):,} días")

# Merge con clima
df = pd.merge(df, df_cl, on="fecha", how="inner")
print(f"  + Clima: {len(df):,} días")

# Ordenar por fecha
df.sort_values("fecha", inplace=True)
df.reset_index(drop=True, inplace=True)

# ═══════════════════════════════════════════════════════════════════
# BLOQUE 5: AÑADIR VARIABLES ÚTILES
# ═══════════════════════════════════════════════════════════════════
print("\n[5] Añadiendo variables temporales...")
df["anyo"]          = df["fecha"].dt.year
df["mes"]           = df["fecha"].dt.month
df["dia_semana"]    = df["fecha"].dt.dayofweek   # 0=lunes, 6=domingo
df["es_fin_semana"] = df["fecha"].dt.dayofweek >= 5
df["nombre_dia"]    = df["fecha"].dt.day_name()

# Estación del año
def estacion(mes):
    if mes in [12, 1, 2]:  return "invierno"
    elif mes in [3, 4, 5]:  return "primavera"
    elif mes in [6, 7, 8]:  return "verano"
    else:                    return "otoño"

df["estacion_anyo"] = df["mes"].apply(estacion)
print("  ✓ anyo, mes, dia_semana, es_fin_semana, nombre_dia, estacion_anyo")

# ═══════════════════════════════════════════════════════════════════
# BLOQUE 6: VALIDACIÓN
# ═══════════════════════════════════════════════════════════════════
print("\n[6] Validación...")
print(f"  Filas totales  : {len(df):,}")
print(f"  Columnas       : {df.shape[1]}")
print(f"  Rango          : {df['fecha'].min().date()} → {df['fecha'].max().date()}")

print(f"\n  Columnas del dataset:")
for col in df.columns:
    n_nulos = int(df[col].isnull().sum())
    pct = round(n_nulos / len(df) * 100, 1)
    print(f"    {col:<25} | nulos: {n_nulos} ({pct}%)")

print(f"\n  Cobertura por año:")
print(df.groupby("anyo").size().to_string())

print(f"\n  Estadísticas principales:")
cols_stats = ["trafico_medio", "NO2", "PM10", "O3", "T2M_MAX", "PRECTOTCORR", "WS10M"]
cols_stats = [c for c in cols_stats if c in df.columns]
print(df[cols_stats].describe().round(3).to_string())

# ═══════════════════════════════════════════════════════════════════
# BLOQUE 7: GUARDAR
# ═══════════════════════════════════════════════════════════════════
print(f"\n[7] Guardando en:\n  {RUTA_SALIDA}")
df.to_csv(RUTA_SALIDA, sep=";", index=False, encoding="utf-8-sig")
tam = os.path.getsize(RUTA_SALIDA) / (1024**2)
print(f"  ✓ Guardado ({tam:.2f} MB)")

print("\n" + "=" * 65)
print("  DATASET INTEGRADO COMPLETADO")
print("=" * 65)
print(f"\n  Estructura final:")
print(f"    fecha              — Fecha del día")
print(f"    trafico_medio      — Intensidad media horaria (veh/h) entre estaciones")
print(f"    n_estaciones       — Nº estaciones con dato ese día")
print(f"    NO2, NO, NOx       — Media diaria µg/m³ entre estaciones")
print(f"    PM10, PM25         — Media diaria µg/m³ entre estaciones")
print(f"    CO                 — Media diaria mg/m³ entre estaciones")
print(f"    O3                 — Media diaria µg/m³ entre estaciones")
print(f"    T2M_MAX/MIN/RANGE  — Temperatura °C")
print(f"    RH2M               — Humedad relativa %")
print(f"    WS10M, WD10M       — Viento m/s y dirección °")
print(f"    ALLSKY_SFC_SW_DWN  — Radiación solar MJ/m²/día")
print(f"    PRECTOTCORR        — Precipitación mm/día")
print(f"    PS                 — Presión kPa")
print(f"    anyo, mes, dia_semana, es_fin_semana, nombre_dia, estacion_anyo")
print("\nFIN — Pega el output en el chat.")