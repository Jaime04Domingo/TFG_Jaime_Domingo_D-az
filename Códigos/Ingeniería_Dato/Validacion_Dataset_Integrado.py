#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 19:41:36 2026

@author: jaime
"""

"""
Validación completa + visualización exploratoria
Dataset_Diario_Integrado.csv - TFG
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

RUTA    = "/Users/jaime/Documents/Universidad/TFG/Dataset_Diario_Integrado.csv"
CARPETA = "/Users/jaime/Documents/Universidad/TFG/Graficos_Validacion/"
os.makedirs(CARPETA, exist_ok=True)

print("=" * 65)
print("  VALIDACIÓN DATASET INTEGRADO DIARIO")
print("=" * 65)

# ── CARGA ────────────────────────────────────────────────────────────
df = pd.read_csv(RUTA, sep=";", encoding="utf-8-sig", low_memory=False)
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
df = df.sort_values("fecha").reset_index(drop=True)

print(f"\n  Filas    : {len(df):,}")
print(f"  Columnas : {df.shape[1]}")
print(f"  Rango    : {df['fecha'].min().date()} → {df['fecha'].max().date()}")

errores = []

# ── 1. ESTRUCTURA BÁSICA ─────────────────────────────────────────────
print("\n[1] ESTRUCTURA BÁSICA")
print("-" * 50)
print(f"  Columnas: {list(df.columns)}")
print(f"  Tipos:")
for col in df.columns:
    print(f"    {col:<25} : {str(df[col].dtype)}")

# ── 2. NULOS ─────────────────────────────────────────────────────────
print("\n[2] NULOS POR COLUMNA")
print("-" * 50)
for col in df.columns:
    n = int(df[col].isnull().sum())
    pct = round(n / len(df) * 100, 2)
    flag = " ⚠" if n > 0 else " ✓"
    print(f"  {col:<25} : {n:>4} ({pct:.2f}%){flag}")

# ── 3. DUPLICADOS ────────────────────────────────────────────────────
print("\n[3] DUPLICADOS")
print("-" * 50)
dups = int(df.duplicated().sum())
dups_fecha = int(df.duplicated(subset=["fecha"]).sum())
print(f"  Filas 100% duplicadas : {dups} {'✓' if dups == 0 else '✗'}")
print(f"  Fechas duplicadas     : {dups_fecha} {'✓' if dups_fecha == 0 else '✗'}")
if dups > 0: errores.append(f"{dups} duplicados")

# ── 4. COBERTURA TEMPORAL ────────────────────────────────────────────
print("\n[4] COBERTURA TEMPORAL")
print("-" * 50)
df["anyo"] = df["fecha"].dt.year
df["mes"]  = df["fecha"].dt.month
cob = df.groupby(["anyo","mes"]).size().unstack(fill_value=0)
print(cob.to_string())

# Días esperados vs reales
for anyo in df["anyo"].unique():
    dias_reales   = int((df["anyo"] == anyo).sum())
    dias_esperados = 366 if anyo == 2024 else 365
    if anyo == 2025: dias_esperados = 90  # hasta marzo
    if anyo == 2021: dias_esperados = 365
    ok = "✓" if dias_reales == dias_esperados else f"⚠ esperados {dias_esperados}"
    print(f"  {anyo}: {dias_reales} días {ok}")

# Gaps (días faltantes)
idx_completo = pd.date_range(df["fecha"].min(), df["fecha"].max(), freq="D")
faltantes = idx_completo.difference(df["fecha"])
print(f"\n  Días faltantes en el rango: {len(faltantes)}")
if len(faltantes) > 0:
    print(f"  Detalle: {list(faltantes[:10])}")
    errores.append(f"{len(faltantes)} días faltantes")

# ── 5. RANGOS DE VALORES ─────────────────────────────────────────────
print("\n[5] RANGOS Y VALIDACIÓN DE VALORES")
print("-" * 50)

# Límites físicamente razonables
LIMITES = {
    "trafico_medio": (0, 3000,   "veh/h"),
    "NO2":           (0, 400,    "µg/m³"),
    "NO":            (0, 800,    "µg/m³"),
    "NOx":           (0, 1500,   "µg/m³"),
    "PM10":          (0, 800,    "µg/m³"),   # alto por sahariano
    "PM25":          (0, 300,    "µg/m³"),
    "CO":            (0, 5,      "mg/m³"),
    "O3":            (0, 200,    "µg/m³"),
    "T2M_MAX":       (-20, 50,   "°C"),
    "T2M_MIN":       (-25, 45,   "°C"),
    "T2M_RANGE":     (0, 40,     "°C"),
    "RH2M":          (0, 100,    "%"),
    "WS10M":         (0, 30,     "m/s"),
    "WD10M":         (0, 360,    "°"),
    "ALLSKY_SFC_SW_DWN": (0, 50, "MJ/m²"),
    "PRECTOTCORR":   (0, 200,    "mm"),
    "PS":            (85, 110,   "kPa"),
}

print(f"  {'Variable':<25} {'Min':>10} {'Media':>10} {'Median':>10} {'Max':>10} {'Nulos':>6} {'Fuera rango':>12} {'Unidad'}")
print(f"  {'-'*95}")
for col, (lo, hi, unidad) in LIMITES.items():
    if col not in df.columns:
        continue
    vals = df[col].dropna()
    if len(vals) == 0:
        continue
    vmin   = float(vals.min())
    vmedia = float(vals.mean())
    vmed   = float(vals.median())
    vmax   = float(vals.max())
    nulos  = int(df[col].isnull().sum())
    fuera  = int(((vals < lo) | (vals > hi)).sum())
    flag   = " ⚠" if fuera > 0 else ""
    print(f"  {col:<25} {vmin:>10.3f} {vmedia:>10.3f} {vmed:>10.3f} {vmax:>10.3f} {nulos:>6} {fuera:>10}{flag}  {unidad}")
    if fuera > 0:
        errores.append(f"{col}: {fuera} valores fuera del rango [{lo}, {hi}]")

# ── 6. VALORES NEGATIVOS ─────────────────────────────────────────────
print("\n[6] VALORES NEGATIVOS EN VARIABLES QUE NO DEBEN TENERLOS")
print("-" * 50)
cols_no_neg = ["trafico_medio","NO2","NO","NOx","PM10","PM25","CO","O3",
               "RH2M","WS10M","ALLSKY_SFC_SW_DWN","PRECTOTCORR"]
for col in cols_no_neg:
    if col in df.columns:
        n_neg = int((df[col] < 0).sum())
        if n_neg > 0:
            print(f"  ✗ {col}: {n_neg} valores negativos | min={float(df[col].min()):.3f}")
            errores.append(f"{col}: {n_neg} negativos")
        else:
            print(f"  ✓ {col}: sin negativos")

# ── 7. CONSISTENCIA INTERNA ──────────────────────────────────────────
print("\n[7] CONSISTENCIA INTERNA")
print("-" * 50)

# T2M_MIN <= T2M_MAX
inconsist_temp = int((df["T2M_MIN"] > df["T2M_MAX"]).sum())
print(f"  T2M_MIN > T2M_MAX    : {inconsist_temp} casos {'✓' if inconsist_temp == 0 else '✗'}")

# T2M_RANGE ≈ T2M_MAX - T2M_MIN
if "T2M_RANGE" in df.columns:
    diff_range = abs(df["T2M_RANGE"] - (df["T2M_MAX"] - df["T2M_MIN"]))
    inconsist_range = int((diff_range > 0.5).sum())
    print(f"  T2M_RANGE inconsist. : {inconsist_range} casos {'✓' if inconsist_range == 0 else '⚠'}")

# NOx >= NO + NO2 (aproximadamente)
if all(c in df.columns for c in ["NOx","NO","NO2"]):
    inconsist_nox = int((df["NOx"] < df["NO"] + df["NO2"] - 5).sum())
    print(f"  NOx < NO+NO2-5       : {inconsist_nox} casos {'✓' if inconsist_nox == 0 else '⚠'}")

# n_estaciones debe ser 58 siempre
n_est_min = int(df["n_estaciones"].min())
n_est_max = int(df["n_estaciones"].max())
print(f"  Estaciones tráfico   : min={n_est_min} max={n_est_max} {'✓' if n_est_min == n_est_max == 58 else '⚠'}")

# Tráfico en festivos/fines de semana debe ser menor
media_lab = float(df[df["dia_semana"] < 5]["trafico_medio"].mean())
media_fds = float(df[df["dia_semana"] >= 5]["trafico_medio"].mean())
print(f"  Tráfico laborable vs FdS: {media_lab:.1f} vs {media_fds:.1f} veh/h "
      f"{'✓ FdS menor' if media_fds < media_lab else '⚠ inesperado'}")

# ── 8. ESTADÍSTICAS COMPLETAS ────────────────────────────────────────
print("\n[8] ESTADÍSTICAS COMPLETAS")
print("-" * 50)
cols_num = ["trafico_medio","NO2","NO","NOx","PM10","PM25","CO","O3",
            "T2M_MAX","T2M_MIN","RH2M","WS10M","PRECTOTCORR"]
cols_num = [c for c in cols_num if c in df.columns]
print(df[cols_num].describe().round(3).to_string())

# ── 9. PRIMERAS Y ÚLTIMAS FILAS ──────────────────────────────────────
print("\n[9] PRIMERAS 5 FILAS")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(df[["fecha","trafico_medio","NO2","PM10","O3","T2M_MAX","PRECTOTCORR"]].head(5).to_string())
print("\n  ÚLTIMAS 5 FILAS")
print(df[["fecha","trafico_medio","NO2","PM10","O3","T2M_MAX","PRECTOTCORR"]].tail(5).to_string())

# ── RESUMEN DE VALIDACIÓN ────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESULTADO DE LA VALIDACIÓN")
print("=" * 65)
if not errores:
    print("  ✅ TODAS LAS COMPROBACIONES CORRECTAS")
else:
    print(f"  ⚠ {len(errores)} advertencias:")
    for e in errores:
        print(f"    - {e}")

# ═══════════════════════════════════════════════════════════════════
# VISUALIZACIÓN: TRÁFICO vs NO₂ evolución temporal
# ═══════════════════════════════════════════════════════════════════
print("\nGenerando visualización tráfico vs NO₂...")

# Media mensual para suavizar
df["anyo_mes"] = df["fecha"].dt.to_period("M")
mensual = df.groupby("anyo_mes").agg(
    trafico=("trafico_medio", "mean"),
    NO2=("NO2", "mean"),
    PM10=("PM10", "mean"),
    O3=("O3", "mean"),
    fecha=("fecha", "first")
).reset_index()
mensual["fecha"] = pd.to_datetime(mensual["fecha"])

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle("Evolución temporal — Dataset Integrado Diario\n"
             "Medias mensuales · Madrid 2021–2025",
             fontsize=13, fontweight="bold")

# Panel 1: Tráfico
ax1 = axes[0]
ax1.plot(mensual["fecha"], mensual["trafico"],
         color="#1E88E5", linewidth=2, marker="o", markersize=3)
ax1.fill_between(mensual["fecha"], mensual["trafico"],
                 alpha=0.15, color="#1E88E5")
ax1.set_ylabel("Intensidad media\n(veh/hora)", fontsize=10)
ax1.set_title("Tráfico (intensidad media horaria entre estaciones)",
              fontsize=10, fontweight="bold", color="#1E88E5")
ax1.grid(True, alpha=0.3, linestyle="--")

# Panel 2: NO₂
ax2 = axes[1]
ax2.plot(mensual["fecha"], mensual["NO2"],
         color="#E53935", linewidth=2, marker="o", markersize=3)
ax2.fill_between(mensual["fecha"], mensual["NO2"],
                 alpha=0.15, color="#E53935")
ax2.axhline(40, color="darkred", linestyle="--", linewidth=1.5,
            alpha=0.7, label="Límite anual UE (40 µg/m³)")
ax2.axhline(10, color="orange", linestyle="--", linewidth=1.5,
            alpha=0.7, label="Guía OMS (10 µg/m³)")
ax2.set_ylabel("NO₂ (µg/m³)", fontsize=10)
ax2.set_title("Concentración de NO₂",
              fontsize=10, fontweight="bold", color="#E53935")
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(True, alpha=0.3, linestyle="--")

# Panel 3: O₃ (efecto inverso al tráfico)
ax3 = axes[2]
ax3.plot(mensual["fecha"], mensual["O3"],
         color="#00897B", linewidth=2, marker="o", markersize=3)
ax3.fill_between(mensual["fecha"], mensual["O3"],
                 alpha=0.15, color="#00897B")
ax3.axhline(120, color="darkgreen", linestyle="--", linewidth=1.5,
            alpha=0.7, label="Valor objetivo UE (120 µg/m³)")
ax3.set_ylabel("O₃ (µg/m³)", fontsize=10)
ax3.set_title("Concentración de O₃ (efecto inverso al NO₂)",
              fontsize=10, fontweight="bold", color="#00897B")
ax3.legend(fontsize=8, loc="upper right")
ax3.grid(True, alpha=0.3, linestyle="--")
ax3.set_xlabel("Fecha", fontsize=10)

# Zonas estacionales de fondo
for ax in axes:
    for anyo in [2021, 2022, 2023, 2024, 2025]:
        for mes_ini, mes_fin, color_est in [(6, 8, "#FFF9C4"), (12, 12, "#E3F2FD")]:
            try:
                ini = pd.Timestamp(f"{anyo}-{mes_ini:02d}-01")
                fin = pd.Timestamp(f"{anyo}-{mes_fin:02d}-28")
                ax.axvspan(ini, fin, alpha=0.08, color=color_est)
            except Exception:
                pass

plt.tight_layout()
ruta_graf = CARPETA + "VAL_trafico_NO2_evolucion.png"
plt.savefig(ruta_graf, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ {ruta_graf}")

# Gráfico 2: Scatter tráfico vs NO₂ con color por año
fig2, ax = plt.subplots(figsize=(10, 7))
anyos_unicos = sorted(df["anyo"].unique())
colores_anyo = ["#1E88E5","#43A047","#FB8C00","#E53935","#8E24AA"]
for i, anyo in enumerate(anyos_unicos):
    sub = df[df["anyo"] == anyo]
    ax.scatter(sub["trafico_medio"], sub["NO2"],
               alpha=0.3, s=15,
               color=colores_anyo[i % len(colores_anyo)],
               label=str(anyo))

# Línea de tendencia global
x = df["trafico_medio"].dropna()
y = df.loc[x.index, "NO2"].dropna()
idx_comun = x.index.intersection(y.index)
x_c, y_c = x[idx_comun].values, y[idx_comun].values
z = np.polyfit(x_c, y_c, 1)
p = np.poly1d(z)
x_line = np.linspace(float(x_c.min()), float(x_c.max()), 100)
ax.plot(x_line, p(x_line), "k--", linewidth=2, alpha=0.6,
        label=f"Tendencia global (r={np.corrcoef(x_c,y_c)[0,1]:.2f})")

ax.set_xlabel("Tráfico medio diario (veh/hora)", fontsize=11)
ax.set_ylabel("NO₂ medio diario (µg/m³)", fontsize=11)
ax.set_title("Relación entre tráfico y NO₂\n(un punto = un día)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ruta_scatter = CARPETA + "VAL_scatter_trafico_NO2.png"
plt.savefig(ruta_scatter, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ {ruta_scatter}")

print("\n" + "=" * 65)
print("  GRÁFICOS GUARDADOS EN:")
print(f"  {CARPETA}")
print("  VAL_trafico_NO2_evolucion.png")
print("  VAL_scatter_trafico_NO2.png")
print("=" * 65)
print("\nFIN — Pega el output en el chat.")