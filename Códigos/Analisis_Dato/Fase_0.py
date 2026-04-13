#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 22:23:02 2026

@author: jaime
"""

### FASE 0 Análisis del dato. Este script es el primero de la fase de análisis del dato. 




"""
FASE 0 — Análisis de Series Temporales
Dataset Integrado Diario - TFG
Variables principales: NO2 (contaminación) y trafico_medio
"""
import sys
import subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "statsmodels"], capture_output=False)
 
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import os
 
RUTA    = "/Users/jaime/Documents/Universidad/TFG/Dataset_Diario_Integrado.csv"
CARPETA = "/Users/jaime/Documents/Universidad/TFG/Graficos_Fase0/"
os.makedirs(CARPETA, exist_ok=True)
 
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})
 
print("=" * 65)
print("  FASE 0 — ANÁLISIS DE SERIES TEMPORALES")
print("=" * 65)
 
# ── CARGA ────────────────────────────────────────────────────────────
df = pd.read_csv(RUTA, sep=";", encoding="utf-8-sig", low_memory=False)
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)
df = df.set_index("fecha")
 
print(f"\n  Dataset: {len(df):,} días | {df.index.min().date()} → {df.index.max().date()}")
 
# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 1: EVOLUCIÓN TEMPORAL CON MEDIA MÓVIL (NO2 y TRÁFICO)
# ═══════════════════════════════════════════════════════════════════
print("\n[1] Generando evolución temporal con media móvil...")
 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle("Evolución temporal de NO₂ y Tráfico\n"
             "Serie diaria + Media móvil 30 días · Madrid 2021–2025",
             fontsize=13, fontweight="bold")
 
# NO2
no2_mm30 = df["NO2"].rolling(30, center=True).mean()
ax1.plot(df.index, df["NO2"], color="#E53935", alpha=0.25, linewidth=0.7, label="Diario")
ax1.plot(df.index, no2_mm30, color="#B71C1C", linewidth=2.5, label="Media móvil 30 días")
ax1.axhline(40, color="darkred", linestyle="--", linewidth=1.5, alpha=0.8, label="Límite UE (40 µg/m³)")
ax1.axhline(10, color="orange", linestyle="--", linewidth=1.5, alpha=0.8, label="Guía OMS (10 µg/m³)")
ax1.set_ylabel("NO₂ (µg/m³)", fontsize=11)
ax1.set_title("NO₂ — Tendencia bajista de 2021 a 2025", fontsize=11, fontweight="bold", color="#B71C1C")
ax1.legend(fontsize=9, loc="upper right")
 
# Anotar medias anuales
for anyo in [2021, 2022, 2023, 2024]:
    media = float(df[df.index.year == anyo]["NO2"].mean())
    ax1.annotate(f"{anyo}\n{media:.1f}",
                 xy=(pd.Timestamp(f"{anyo}-07-01"), media + 12),
                 fontsize=8, ha="center", color="#B71C1C",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
 
# Tráfico
traf_mm30 = df["trafico_medio"].rolling(30, center=True).mean()
ax2.plot(df.index, df["trafico_medio"], color="#1E88E5", alpha=0.25, linewidth=0.7, label="Diario")
ax2.plot(df.index, traf_mm30, color="#0D47A1", linewidth=2.5, label="Media móvil 30 días")
ax2.set_ylabel("Tráfico (veh/hora)", fontsize=11)
ax2.set_xlabel("Fecha", fontsize=11)
ax2.set_title("Tráfico — Intensidad media horaria entre estaciones", fontsize=11, fontweight="bold", color="#0D47A1")
ax2.legend(fontsize=9, loc="upper right")
 
# Anotar medias anuales
for anyo in [2021, 2022, 2023, 2024]:
    media = float(df[df.index.year == anyo]["trafico_medio"].mean())
    ax2.annotate(f"{anyo}\n{media:.0f}",
                 xy=(pd.Timestamp(f"{anyo}-07-01"), media - 200),
                 fontsize=8, ha="center", color="#0D47A1",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
 
plt.tight_layout()
plt.savefig(CARPETA + "F0_01_evolucion_media_movil.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ F0_01_evolucion_media_movil.png")
 
# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 2: DESCOMPOSICIÓN STL de NO2
# ═══════════════════════════════════════════════════════════════════
print("\n[2] Descomposición de serie temporal (NO₂)...")
 
no2_series = df["NO2"].dropna()
# Período 365 días (anual)
result_no2 = seasonal_decompose(no2_series, model="additive", period=365, extrapolate_trend="freq")
 
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle("Descomposición de la serie temporal de NO₂\n"
             "Componentes: Tendencia · Estacionalidad · Residuo · Madrid 2021–2025",
             fontsize=13, fontweight="bold")
 
# Serie original
axes[0].plot(no2_series.index, no2_series.values, color="#E53935", linewidth=0.8, alpha=0.7)
axes[0].set_ylabel("NO₂ (µg/m³)", fontsize=10)
axes[0].set_title("Serie original", fontsize=10, fontweight="bold")
 
# Tendencia
axes[1].plot(result_no2.trend.index, result_no2.trend.values, color="#B71C1C", linewidth=2)
axes[1].set_ylabel("Tendencia (µg/m³)", fontsize=10)
axes[1].set_title("Tendencia — Componente de largo plazo", fontsize=10, fontweight="bold")
# Marcar media por año sobre la tendencia
for anyo in [2021, 2022, 2023, 2024, 2025]:
    sub = result_no2.trend[result_no2.trend.index.year == anyo]
    if len(sub) > 0:
        media_tend = float(sub.mean())
        axes[1].axhline(media_tend, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        axes[1].text(sub.index[0], media_tend + 0.3, f"  {anyo}: {media_tend:.1f}", fontsize=7, color="gray")
 
# Estacionalidad
axes[2].plot(result_no2.seasonal.index, result_no2.seasonal.values, color="#FB8C00", linewidth=0.8)
axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[2].set_ylabel("Estacionalidad (µg/m³)", fontsize=10)
axes[2].set_title("Estacionalidad — Patrón anual repetido (alto en invierno, bajo en verano)",
                  fontsize=10, fontweight="bold")
 
# Residuo
axes[3].plot(result_no2.resid.index, result_no2.resid.values, color="#43A047", linewidth=0.6, alpha=0.7)
axes[3].axhline(0, color="black", linewidth=0.8, linestyle="--")
# Marcar outliers (residuos > 2 desviaciones)
resid_std = float(result_no2.resid.std())
outliers = result_no2.resid[abs(result_no2.resid) > 2 * resid_std]
axes[3].scatter(outliers.index, outliers.values, color="red", s=15, zorder=5, label=f"Outliers (|resid|>2σ): {len(outliers)}")
axes[3].set_ylabel("Residuo (µg/m³)", fontsize=10)
axes[3].set_xlabel("Fecha", fontsize=10)
axes[3].set_title("Residuo — Variación no explicada por tendencia ni estacionalidad",
                  fontsize=10, fontweight="bold")
axes[3].legend(fontsize=8)
 
plt.tight_layout()
plt.savefig(CARPETA + "F0_02_descomposicion_NO2.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ F0_02_descomposicion_NO2.png")
 
# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 3: DESCOMPOSICIÓN STL de TRÁFICO
# ═══════════════════════════════════════════════════════════════════
print("\n[3] Descomposición de serie temporal (Tráfico)...")
 
traf_series = df["trafico_medio"].dropna()
result_traf = seasonal_decompose(traf_series, model="additive", period=365, extrapolate_trend="freq")
 
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle("Descomposición de la serie temporal de Tráfico\n"
             "Componentes: Tendencia · Estacionalidad · Residuo · Madrid 2021–2025",
             fontsize=13, fontweight="bold")
 
axes[0].plot(traf_series.index, traf_series.values, color="#1E88E5", linewidth=0.8, alpha=0.7)
axes[0].set_ylabel("Tráfico (veh/h)", fontsize=10)
axes[0].set_title("Serie original", fontsize=10, fontweight="bold")
 
axes[1].plot(result_traf.trend.index, result_traf.trend.values, color="#0D47A1", linewidth=2)
axes[1].set_ylabel("Tendencia (veh/h)", fontsize=10)
axes[1].set_title("Tendencia — Componente de largo plazo", fontsize=10, fontweight="bold")
 
axes[2].plot(result_traf.seasonal.index, result_traf.seasonal.values, color="#FB8C00", linewidth=0.8)
axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[2].set_ylabel("Estacionalidad (veh/h)", fontsize=10)
axes[2].set_title("Estacionalidad — Patrón anual repetido", fontsize=10, fontweight="bold")
 
axes[3].plot(result_traf.resid.index, result_traf.resid.values, color="#43A047", linewidth=0.6, alpha=0.7)
axes[3].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[3].set_ylabel("Residuo (veh/h)", fontsize=10)
axes[3].set_xlabel("Fecha", fontsize=10)
axes[3].set_title("Residuo — Variación no explicada por tendencia ni estacionalidad",
                  fontsize=10, fontweight="bold")
 
plt.tight_layout()
plt.savefig(CARPETA + "F0_03_descomposicion_trafico.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ F0_03_descomposicion_trafico.png")
 
# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 4: ACF Y PACF de NO2
# ═══════════════════════════════════════════════════════════════════
print("\n[4] Autocorrelación (ACF/PACF) de NO₂...")
 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("Autocorrelación del NO₂\n"
             "ACF y PACF · Madrid 2021–2025",
             fontsize=13, fontweight="bold")
 
plot_acf(no2_series.dropna(), lags=60, ax=ax1, color="#E53935", alpha=0.05)
ax1.set_title("ACF — Autocorrelación: ¿cuánto se parece el NO₂ de hoy al de días anteriores?",
              fontsize=11, fontweight="bold")
ax1.set_xlabel("Lag (días)", fontsize=10)
ax1.set_ylabel("Correlación", fontsize=10)
 
plot_pacf(no2_series.dropna(), lags=60, ax=ax2, color="#E53935", alpha=0.05, method="ols")
ax2.set_title("PACF — Autocorrelación parcial: efecto directo de cada rezago",
              fontsize=11, fontweight="bold")
ax2.set_xlabel("Lag (días)", fontsize=10)
ax2.set_ylabel("Correlación parcial", fontsize=10)
 
plt.tight_layout()
plt.savefig(CARPETA + "F0_04_ACF_PACF_NO2.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ F0_04_ACF_PACF_NO2.png")
 
# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 5: BOXPLOT ESTACIONAL (NO2 y TRÁFICO por mes y por día de semana)
# ═══════════════════════════════════════════════════════════════════
print("\n[5] Boxplot estacional...")
 
df_reset = df.reset_index()
meses_es = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
dias_es  = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
 
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Patrones estacionales de NO₂ y Tráfico\n"
             "Por mes y por día de la semana · Madrid 2021–2025",
             fontsize=13, fontweight="bold")
 
# NO2 por mes
data_mes_no2 = [df_reset[df_reset["mes"]==m]["NO2"].dropna().values for m in range(1,13)]
bp1 = axes[0,0].boxplot(data_mes_no2, patch_artist=True, showfliers=False,
                         medianprops={"color":"white","linewidth":2},
                         boxprops={"linewidth":1})
for patch in bp1["boxes"]:
    patch.set_facecolor("#E53935")
    patch.set_alpha(0.7)
axes[0,0].set_xticklabels(meses_es, fontsize=9)
axes[0,0].set_ylabel("NO₂ (µg/m³)", fontsize=10)
axes[0,0].set_title("NO₂ por mes — Patrón estacional invernal", fontsize=10, fontweight="bold")
 
# NO2 por día de semana
data_dia_no2 = [df_reset[df_reset["dia_semana"]==d]["NO2"].dropna().values for d in range(7)]
bp2 = axes[0,1].boxplot(data_dia_no2, patch_artist=True, showfliers=False,
                         medianprops={"color":"white","linewidth":2},
                         boxprops={"linewidth":1})
for patch in bp2["boxes"]:
    patch.set_facecolor("#E53935")
    patch.set_alpha(0.7)
axes[0,1].set_xticklabels(dias_es, fontsize=9)
axes[0,1].set_ylabel("NO₂ (µg/m³)", fontsize=10)
axes[0,1].set_title("NO₂ por día de semana — Huella del tráfico laboral", fontsize=10, fontweight="bold")
 
# Tráfico por mes
data_mes_traf = [df_reset[df_reset["mes"]==m]["trafico_medio"].dropna().values for m in range(1,13)]
bp3 = axes[1,0].boxplot(data_mes_traf, patch_artist=True, showfliers=False,
                         medianprops={"color":"white","linewidth":2},
                         boxprops={"linewidth":1})
for patch in bp3["boxes"]:
    patch.set_facecolor("#1E88E5")
    patch.set_alpha(0.7)
axes[1,0].set_xticklabels(meses_es, fontsize=9)
axes[1,0].set_ylabel("Tráfico (veh/hora)", fontsize=10)
axes[1,0].set_title("Tráfico por mes — Caídas en agosto y diciembre", fontsize=10, fontweight="bold")
 
# Tráfico por día de semana
data_dia_traf = [df_reset[df_reset["dia_semana"]==d]["trafico_medio"].dropna().values for d in range(7)]
bp4 = axes[1,1].boxplot(data_dia_traf, patch_artist=True, showfliers=False,
                         medianprops={"color":"white","linewidth":2},
                         boxprops={"linewidth":1})
for patch in bp4["boxes"]:
    patch.set_facecolor("#1E88E5")
    patch.set_alpha(0.7)
axes[1,1].set_xticklabels(dias_es, fontsize=9)
axes[1,1].set_ylabel("Tráfico (veh/hora)", fontsize=10)
axes[1,1].set_title("Tráfico por día de semana — Caída clara en fin de semana", fontsize=10, fontweight="bold")
 
plt.tight_layout()
plt.savefig(CARPETA + "F0_05_boxplot_estacional.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ F0_05_boxplot_estacional.png")
 
# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 6: EVOLUCIÓN ANUAL COMPARATIVA (medias por año)
# ═══════════════════════════════════════════════════════════════════
print("\n[6] Evolución anual comparativa...")
 
anyos = [2021, 2022, 2023, 2024]  # 2025 incompleto
vars_plot = ["NO2", "PM10", "O3", "trafico_medio"]
titulos  = ["NO₂ (µg/m³)", "PM10 (µg/m³)", "O₃ (µg/m³)", "Tráfico (veh/h)"]
colores  = ["#E53935", "#8E24AA", "#00897B", "#1E88E5"]
 
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Evolución de las medias anuales · Madrid 2021–2025\n"
             "(2025 solo enero–marzo, no incluido en tendencia)",
             fontsize=13, fontweight="bold")
axes = axes.flatten()
 
for i, (var, titulo, color) in enumerate(zip(vars_plot, titulos, colores)):
    medias = []
    for anyo in anyos:
        m = float(df_reset[df_reset["anyo"] == anyo][var].mean())
        medias.append(m)
    axes[i].bar([str(a) for a in anyos], medias, color=color, alpha=0.8, edgecolor="white")
    for j, (anyo, m) in enumerate(zip(anyos, medias)):
        axes[i].text(j, m + max(medias)*0.02, f"{m:.1f}", ha="center", fontsize=10, fontweight="bold")
    # Línea de tendencia
    z = np.polyfit(range(len(anyos)), medias, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, len(anyos)-1, 100)
    axes[i].plot(x_line, p(x_line), "--", color="black", linewidth=1.5, alpha=0.6,
                 label=f"Tendencia: {z[0]:+.2f}/año")
    axes[i].set_title(titulo, fontsize=11, fontweight="bold", color=color)
    axes[i].legend(fontsize=8)
    axes[i].set_ylim(0, max(medias) * 1.2)
 
plt.tight_layout()
plt.savefig(CARPETA + "F0_06_medias_anuales.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ F0_06_medias_anuales.png")
 
# ═══════════════════════════════════════════════════════════════════
# TEST DE ESTACIONARIEDAD (ADF)
# ═══════════════════════════════════════════════════════════════════
print("\n[7] Tests de estacionariedad (Dickey-Fuller aumentado)...")
print("-" * 55)
 
for var, nombre in [("NO2", "NO₂"), ("trafico_medio", "Tráfico")]:
    serie = df[var].dropna()
    resultado = adfuller(serie, autolag="AIC")
    estadistico = float(resultado[0])
    pvalor      = float(resultado[1])
    lags_usados = int(resultado[2])
    criticos    = resultado[4]
    es_estac    = "NO ESTACIONARIA" if pvalor > 0.05 else "ESTACIONARIA"
    print(f"\n  {nombre}:")
    print(f"    Estadístico ADF : {estadistico:.4f}")
    print(f"    p-valor         : {pvalor:.6f}  →  {es_estac}")
    print(f"    Lags usados     : {lags_usados}")
    print(f"    Valores críticos: 1%={float(criticos['1%']):.3f}  5%={float(criticos['5%']):.3f}  10%={float(criticos['10%']):.3f}")
 
# ═══════════════════════════════════════════════════════════════════
# CORRELACIÓN ENTRE VARIABLES (HEATMAP)
# ═══════════════════════════════════════════════════════════════════
print("\n[8] Matriz de correlaciones...")
 
vars_corr = ["trafico_medio","NO2","NO","NOx","PM10","PM25","O3",
             "T2M_MAX","T2M_MIN","RH2M","WS10M","PRECTOTCORR"]
vars_corr = [v for v in vars_corr if v in df.columns]
corr = df[vars_corr].corr().round(2)
 
# Imprimir correlaciones con NO2
print("\n  Correlaciones con NO₂:")
print("-" * 40)
for var in vars_corr:
    if var != "NO2":
        r = float(corr.loc["NO2", var])
        bar = "█" * int(abs(r) * 20)
        signo = "+" if r > 0 else "-"
        print(f"  {var:<20} {signo}{abs(r):.2f}  {bar}")
 
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks(range(len(vars_corr)))
ax.set_yticks(range(len(vars_corr)))
etq = ["Tráfico","NO₂","NO","NOₓ","PM10","PM2.5","O₃",
       "T_max","T_min","Humedad","Viento","Lluvia"]
ax.set_xticklabels(etq[:len(vars_corr)], rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(etq[:len(vars_corr)], fontsize=9)
 
for i in range(len(vars_corr)):
    for j in range(len(vars_corr)):
        val = float(corr.values[i, j])
        color_txt = "white" if abs(val) > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7.5, color=color_txt, fontweight="bold")
 
ax.set_title("Matriz de correlaciones entre variables del dataset integrado\n"
             "Verde = correlación positiva · Rojo = correlación negativa",
             fontsize=12, fontweight="bold", pad=20)
 
plt.tight_layout()
plt.savefig(CARPETA + "F0_07_matriz_correlaciones.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ F0_07_matriz_correlaciones.png")
 
# ═══════════════════════════════════════════════════════════════════
# RESUMEN ESTADÍSTICO PARA LA MEMORIA
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  RESUMEN ESTADÍSTICO PARA LA MEMORIA")
print("=" * 65)
 
print("\n  Medias anuales NO₂ (µg/m³):")
for anyo in [2021, 2022, 2023, 2024]:
    m = float(df_reset[df_reset["anyo"] == anyo]["NO2"].mean())
    print(f"    {anyo}: {m:.2f}")
 
print("\n  Medias anuales Tráfico (veh/h):")
for anyo in [2021, 2022, 2023, 2024]:
    m = float(df_reset[df_reset["anyo"] == anyo]["trafico_medio"].mean())
    print(f"    {anyo}: {m:.1f}")
 
print("\n  NO₂ por tipo de día:")
media_lab = float(df_reset[df_reset["dia_semana"] < 5]["NO2"].mean())
media_fds = float(df_reset[df_reset["dia_semana"] >= 5]["NO2"].mean())
print(f"    Laborable (L-V): {media_lab:.2f} µg/m³")
print(f"    Fin de semana  : {media_fds:.2f} µg/m³")
print(f"    Diferencia     : {media_lab - media_fds:.2f} µg/m³ ({(media_lab-media_fds)/media_lab*100:.1f}%)")
 
print("\n  Tendencia lineal anual (media anual, pendiente por año):")
medias_no2 = [float(df_reset[df_reset["anyo"]==a]["NO2"].mean()) for a in [2021,2022,2023,2024]]
z_no2 = np.polyfit([2021,2022,2023,2024], medias_no2, 1)
print(f"    NO₂:     {z_no2[0]:+.3f} µg/m³/año")
medias_traf = [float(df_reset[df_reset["anyo"]==a]["trafico_medio"].mean()) for a in [2021,2022,2023,2024]]
z_traf = np.polyfit([2021,2022,2023,2024], medias_traf, 1)
print(f"    Tráfico: {z_traf[0]:+.1f} veh/h/año")
 
print(f"\n  Gráficos guardados en: {CARPETA}")
print("  F0_01_evolucion_media_movil.png")
print("  F0_02_descomposicion_NO2.png")
print("  F0_03_descomposicion_trafico.png")
print("  F0_04_ACF_PACF_NO2.png")
print("  F0_05_boxplot_estacional.png")
print("  F0_06_medias_anuales.png")
print("  F0_07_matriz_correlaciones.png")
print("\nFIN — Pega el output en el chat.")