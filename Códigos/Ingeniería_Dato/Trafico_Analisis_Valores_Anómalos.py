#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:42:46 2026

@author: jaime
"""

"""
Análisis de valores anómalos en columnas horarias
Trafico_Aforos_Definitivo.csv
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
 
RUTA    = "/Users/jaime/Documents/Universidad/TFG/Trafico_Aforos_Definitivo.csv"
CARPETA = "/Users/jaime/Documents/Universidad/TFG/"
 
print("=" * 65)
print("  ANÁLISIS DE VALORES ANÓMALOS - AFOROS")
print("=" * 65)
 
df = pd.read_csv(RUTA, sep=";", encoding="utf-8-sig", low_memory=False)
df["FDIA"] = pd.to_datetime(df["FDIA"], errors="coerce")
cols_hor = [c for c in df.columns if c.startswith("HOR")]
 
# Apilar todas las columnas horarias en una sola serie para análisis global
todos_valores = df[cols_hor].values.flatten()
todos_valores = todos_valores[~np.isnan(todos_valores)]
 
print(f"\nTotal valores horarios: {len(todos_valores):,}")
 
# ── 1. DISTRIBUCIÓN GENERAL ──────────────────────────────────────────
print("\n[1] DISTRIBUCIÓN DE VALORES POR TRAMOS")
tramos = [(0,100), (100,500), (500,1000), (1000,2000),
          (2000,5000), (5000,8000), (8000,9000), (9000,9998), (9999,9999)]
for lo, hi in tramos:
    if lo == hi:
        n = (todos_valores == lo).sum()
    else:
        n = ((todos_valores > lo) & (todos_valores <= hi)).sum()
    pct = n / len(todos_valores) * 100
    print(f"  {lo:>5} – {hi:>5}: {n:>10,}  ({pct:.3f}%)")
 
# ── 2. VALORES SOSPECHOSOS (>= 5000) ────────────────────────────────
print("\n[2] DETALLE DE VALORES >= 5000")
UMBRAL = 5000
df_anomalos = df.copy()
df_anomalos["_max_hora"] = df[cols_hor].max(axis=1)
anomalos = df_anomalos[df_anomalos["_max_hora"] >= UMBRAL].copy()
 
print(f"  Filas con al menos un valor >= {UMBRAL}: {len(anomalos):,}")
print(f"  Porcentaje sobre total: {len(anomalos)/len(df)*100:.2f}%")
 
print(f"\n  Por estación (FEST):")
print(anomalos["FEST"].value_counts().to_string())
 
print(f"\n  Por sentido (FSEN):")
print(anomalos["FSEN"].value_counts().to_string())
 
print(f"\n  Por año:")
anomalos["_anyo"] = anomalos["FDIA"].dt.year
print(anomalos["_anyo"].value_counts().sort_index().to_string())
 
print(f"\n  Por columna horaria:")
for col in cols_hor:
    n = (df[col] >= UMBRAL).sum()
    if n > 0:
        print(f"    {col}: {n:,} valores >= {UMBRAL} | max={df[col].max():.0f}")
 
print(f"\n  Muestra de 10 filas anómalas:")
print(anomalos[["FDIA","FEST","FSEN"] + cols_hor].head(10).to_string())
 
# ── 3. GRÁFICOS ──────────────────────────────────────────────────────
print("\n[3] Generando gráficos...")
 
fig = plt.figure(figsize=(16, 14))
fig.suptitle("Análisis de valores anómalos — Aforos de tráfico permanentes\nTrafico_Aforos_Definitivo.csv",
             fontsize=13, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
 
# ── Gráfico 1: Histograma de TODOS los valores (escala log) ──────────
ax1 = fig.add_subplot(gs[0, :])
bins = np.logspace(np.log10(1), np.log10(10000), 80)
ax1.hist(todos_valores[todos_valores > 0], bins=bins, color="#2196F3", edgecolor="white", linewidth=0.3)
ax1.axvline(5000, color="red", linestyle="--", linewidth=1.5, label="Umbral 5.000")
ax1.axvline(9000, color="orange", linestyle="--", linewidth=1.5, label="Zona centinela (≥9.000)")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Valor (vehículos/hora, escala logarítmica)")
ax1.set_ylabel("Frecuencia (log)")
ax1.set_title("Distribución de todos los valores horarios (escala log–log)")
ax1.legend()
ax1.grid(True, alpha=0.3)
 
# ── Gráfico 2: Zoom en valores > 2000 ────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
vals_altos = todos_valores[todos_valores >= 2000]
ax2.hist(vals_altos, bins=60, color="#FF5722", edgecolor="white", linewidth=0.3)
ax2.axvline(5000, color="red", linestyle="--", linewidth=1.5, label="Umbral 5.000")
ax2.axvline(9000, color="orange", linestyle="--", linewidth=1.5, label="Zona centinela")
ax2.set_xlabel("Valor (vehículos/hora)")
ax2.set_ylabel("Frecuencia")
ax2.set_title(f"Zoom: valores ≥ 2.000\n({len(vals_altos):,} valores)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
 
# ── Gráfico 3: Anómalos por estación ─────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
por_estacion = anomalos["FEST"].value_counts().sort_values(ascending=True)
ax3.barh(por_estacion.index, por_estacion.values, color="#9C27B0", edgecolor="white")
ax3.set_xlabel("Nº de filas con valor ≥ 5.000")
ax3.set_title(f"Anómalos (≥5.000) por estación")
ax3.grid(True, alpha=0.3, axis="x")
 
# ── Gráfico 4: Anómalos por año ───────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
por_anyo = anomalos["_anyo"].value_counts().sort_index()
ax4.bar(por_anyo.index.astype(str), por_anyo.values, color="#4CAF50", edgecolor="white")
ax4.set_xlabel("Año")
ax4.set_ylabel("Nº filas con valor ≥ 5.000")
ax4.set_title("Anómalos (≥5.000) por año")
ax4.grid(True, alpha=0.3, axis="y")
 
# ── Gráfico 5: Anómalos por columna horaria ──────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
por_col = {}
for col in cols_hor:
    por_col[col] = (df[col] >= UMBRAL).sum()
ax5.bar(por_col.keys(), por_col.values(), color="#FF9800", edgecolor="white")
ax5.set_xlabel("Columna horaria")
ax5.set_ylabel("Nº valores ≥ 5.000")
ax5.set_title("Anómalos (≥5.000) por franja horaria")
ax5.grid(True, alpha=0.3, axis="y")
ax5.tick_params(axis="x", rotation=45)
 
ruta_graf = os.path.join(CARPETA, "anomalos_aforos.png")
plt.savefig(ruta_graf, dpi=150, bbox_inches="tight")
print(f"  ✓ Gráfico guardado en: {ruta_graf}")
plt.show()
 
print("\nFIN — Pega el output en el chat.")