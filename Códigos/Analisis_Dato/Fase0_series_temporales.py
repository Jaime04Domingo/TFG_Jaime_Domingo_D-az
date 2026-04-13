#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 23:07:32 2026

@author: jaime
"""

"""
FASE 0 — Análisis de Series Temporales
Sin statsmodels — solo pandas y numpy
TFG Madrid Tráfico & Contaminación
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
print(f"\n  Dataset: {len(df):,} días | {df['fecha'].min().date()} → {df['fecha'].max().date()}")

# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 1: EVOLUCIÓN TEMPORAL CON MEDIA MÓVIL
# ═══════════════════════════════════════════════════════════════════
print("\n[1] Evolución temporal con media móvil...")

no2_mm30  = df["NO2"].rolling(30, center=True).mean()
traf_mm30 = df["trafico_medio"].rolling(30, center=True).mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle("Evolución temporal de NO₂ y Tráfico\n"
             "Serie diaria + Media móvil 30 días · Madrid 2021–2025",
             fontsize=13, fontweight="bold")

ax1.plot(df["fecha"], df["NO2"], color="#E53935", alpha=0.2, linewidth=0.6, label="Diario")
ax1.plot(df["fecha"], no2_mm30, color="#B71C1C", linewidth=2.5, label="Media móvil 30 días")
ax1.axhline(40, color="darkred", linestyle="--", linewidth=1.5, alpha=0.8, label="Límite UE (40 µg/m³)")
ax1.axhline(10, color="orange", linestyle="--", linewidth=1.5, alpha=0.8, label="Guía OMS (10 µg/m³)")
ax1.set_ylabel("NO₂ (µg/m³)", fontsize=11)
ax1.set_title("NO₂ — Tendencia bajista de 2021 a 2025", fontsize=11, fontweight="bold", color="#B71C1C")
ax1.legend(fontsize=9, loc="upper right")

for anyo in [2021, 2022, 2023, 2024]:
    m = float(df[df["fecha"].dt.year == anyo]["NO2"].mean())
    ax1.annotate(f"{anyo}: {m:.1f}",
                 xy=(pd.Timestamp(f"{anyo}-07-01"), m + 10),
                 fontsize=8, ha="center", color="#B71C1C",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

ax2.plot(df["fecha"], df["trafico_medio"], color="#1E88E5", alpha=0.2, linewidth=0.6, label="Diario")
ax2.plot(df["fecha"], traf_mm30, color="#0D47A1", linewidth=2.5, label="Media móvil 30 días")
ax2.set_ylabel("Tráfico (veh/hora)", fontsize=11)
ax2.set_xlabel("Fecha", fontsize=11)
ax2.set_title("Tráfico — Intensidad media horaria entre estaciones", fontsize=11, fontweight="bold", color="#0D47A1")
ax2.legend(fontsize=9)

for anyo in [2021, 2022, 2023, 2024]:
    m = float(df[df["fecha"].dt.year == anyo]["trafico_medio"].mean())
    ax2.annotate(f"{anyo}: {m:.0f}",
                 xy=(pd.Timestamp(f"{anyo}-07-01"), m - 220),
                 fontsize=8, ha="center", color="#0D47A1",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

plt.tight_layout()
plt.savefig(CARPETA + "F0_01_evolucion_media_movil.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ F0_01_evolucion_media_movil.png")

# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 2: DESCOMPOSICIÓN MANUAL DE NO2
# Tendencia = media móvil 365 días
# Estacionalidad = promedio del residuo por día del año
# Residuo = original - tendencia - estacionalidad
# ═══════════════════════════════════════════════════════════════════
print("\n[2] Descomposición serie NO₂...")

no2 = df["NO2"].copy()
# Tendencia: media móvil 365 días
tendencia = no2.rolling(window=365, center=True, min_periods=180).mean()
# Estacionalidad: media por día del año sobre los residuos tendencia
resid_tend = no2 - tendencia
df["doy"] = df["fecha"].dt.dayofyear
estac_media = resid_tend.groupby(df["doy"]).transform("mean")
# Residuo final
residuo = no2 - tendencia - estac_media

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle("Descomposición de la serie temporal de NO₂\n"
             "Tendencia · Estacionalidad · Residuo · Madrid 2021–2025",
             fontsize=13, fontweight="bold")

axes[0].plot(df["fecha"], no2, color="#E53935", linewidth=0.7, alpha=0.7)
axes[0].set_ylabel("µg/m³", fontsize=10)
axes[0].set_title("Serie original", fontsize=10, fontweight="bold")

axes[1].plot(df["fecha"], tendencia, color="#B71C1C", linewidth=2.2)
axes[1].set_ylabel("µg/m³", fontsize=10)
axes[1].set_title("Tendencia — componente de largo plazo (media móvil 365 días)", fontsize=10, fontweight="bold")
for anyo in [2021,2022,2023,2024]:
    sub_tend = tendencia[df["fecha"].dt.year == anyo].dropna()
    if len(sub_tend) > 0:
        m = float(sub_tend.mean())
        axes[1].axhline(m, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        fecha_mid = df["fecha"][df["fecha"].dt.year == anyo].iloc[len(df["fecha"][df["fecha"].dt.year == anyo])//2]
        axes[1].text(fecha_mid, m + 0.3, f"  {anyo}: {m:.1f}", fontsize=7.5, color="gray")

axes[2].plot(df["fecha"], estac_media, color="#FB8C00", linewidth=0.9)
axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[2].set_ylabel("µg/m³", fontsize=10)
axes[2].set_title("Estacionalidad — patrón anual repetido (alto en invierno, bajo en verano)", fontsize=10, fontweight="bold")

residuo_lim = residuo.dropna()
resid_std = float(residuo_lim.std())
outlier_mask = abs(residuo_lim) > 2 * resid_std
axes[3].plot(df["fecha"], residuo, color="#43A047", linewidth=0.6, alpha=0.7)
axes[3].axhline(0, color="black", linewidth=0.8, linestyle="--")
outlier_fechas  = df["fecha"][residuo_lim[outlier_mask].index]
outlier_valores = residuo_lim[outlier_mask]
axes[3].scatter(outlier_fechas, outlier_valores, color="red", s=15, zorder=5,
                label=f"Outliers (|residuo|>2σ): {int(outlier_mask.sum())}")
axes[3].set_ylabel("µg/m³", fontsize=10)
axes[3].set_xlabel("Fecha", fontsize=10)
axes[3].set_title("Residuo — variación no explicada por tendencia ni estacionalidad", fontsize=10, fontweight="bold")
axes[3].legend(fontsize=8)

plt.tight_layout()
plt.savefig(CARPETA + "F0_02_descomposicion_NO2.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ F0_02_descomposicion_NO2.png")

# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 3: DESCOMPOSICIÓN MANUAL DE TRÁFICO
# ═══════════════════════════════════════════════════════════════════
print("\n[3] Descomposición serie tráfico...")

traf = df["trafico_medio"].copy()
tend_traf  = traf.rolling(window=365, center=True, min_periods=180).mean()
resid_tend_traf = traf - tend_traf
estac_traf = resid_tend_traf.groupby(df["doy"]).transform("mean")
resid_traf = traf - tend_traf - estac_traf

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle("Descomposición de la serie temporal de Tráfico\n"
             "Tendencia · Estacionalidad · Residuo · Madrid 2021–2025",
             fontsize=13, fontweight="bold")

axes[0].plot(df["fecha"], traf, color="#1E88E5", linewidth=0.7, alpha=0.7)
axes[0].set_ylabel("veh/h", fontsize=10)
axes[0].set_title("Serie original", fontsize=10, fontweight="bold")

axes[1].plot(df["fecha"], tend_traf, color="#0D47A1", linewidth=2.2)
axes[1].set_ylabel("veh/h", fontsize=10)
axes[1].set_title("Tendencia — componente de largo plazo", fontsize=10, fontweight="bold")

axes[2].plot(df["fecha"], estac_traf, color="#FB8C00", linewidth=0.9)
axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[2].set_ylabel("veh/h", fontsize=10)
axes[2].set_title("Estacionalidad — patrón anual repetido", fontsize=10, fontweight="bold")

axes[3].plot(df["fecha"], resid_traf, color="#43A047", linewidth=0.6, alpha=0.7)
axes[3].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[3].set_ylabel("veh/h", fontsize=10)
axes[3].set_xlabel("Fecha", fontsize=10)
axes[3].set_title("Residuo", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(CARPETA + "F0_03_descomposicion_trafico.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ F0_03_descomposicion_trafico.png")

# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 4: AUTOCORRELACIÓN MANUAL DE NO2 (sin statsmodels)
# ACF = correlación de la serie consigo misma con distintos desfases
# ═══════════════════════════════════════════════════════════════════
print("\n[4] Autocorrelación NO₂...")

def calcular_acf(serie, max_lag):
    serie = serie.dropna()
    n = len(serie)
    media = serie.mean()
    var   = ((serie - media)**2).mean()
    acf_vals = []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            acf_vals.append(1.0)
        else:
            cov = ((serie[lag:].values - media) * (serie[:-lag].values - media)).mean()
            acf_vals.append(float(cov / var))
    return acf_vals

max_lag = 60
acf_no2 = calcular_acf(df["NO2"], max_lag)
lags = list(range(max_lag + 1))
intervalo_conf = 1.96 / np.sqrt(df["NO2"].dropna().shape[0])

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(lags, acf_no2, color=["#B71C1C" if abs(v) > intervalo_conf else "#FFCDD2" for v in acf_no2],
       width=0.7)
ax.axhline(intervalo_conf,  color="blue", linestyle="--", linewidth=1.2,
           alpha=0.7, label=f"IC 95% (±{intervalo_conf:.3f})")
ax.axhline(-intervalo_conf, color="blue", linestyle="--", linewidth=1.2, alpha=0.7)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Lag (días)", fontsize=11)
ax.set_ylabel("Autocorrelación", fontsize=11)
ax.set_title("Función de Autocorrelación (ACF) del NO₂\n"
             "Rojo oscuro = correlación significativa (fuera del intervalo de confianza al 95%)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xticks(range(0, max_lag+1, 5))

# Anotar los lags más relevantes
for lag_destacado in [1, 7, 14, 30]:
    v = acf_no2[lag_destacado]
    ax.annotate(f"lag {lag_destacado}\n{v:.2f}",
                xy=(lag_destacado, v + 0.02),
                fontsize=7.5, ha="center", color="#B71C1C",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

plt.tight_layout()
plt.savefig(CARPETA + "F0_04_ACF_NO2.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ F0_04_ACF_NO2.png")

# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 5: BOXPLOT ESTACIONAL
# ═══════════════════════════════════════════════════════════════════
print("\n[5] Boxplot estacional...")

meses_es = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
dias_es  = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
df["mes_num"]      = df["fecha"].dt.month
df["dia_sem_num"]  = df["fecha"].dt.dayofweek

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Patrones estacionales de NO₂ y Tráfico\n"
             "Por mes y por día de la semana · Madrid 2021–2025",
             fontsize=13, fontweight="bold")

def boxplot_coloreado(ax, datos_grupos, etiquetas, color, titulo, ylabel):
    bp = ax.boxplot(datos_grupos, patch_artist=True, showfliers=False,
                    medianprops={"color":"white","linewidth":2.0},
                    whiskerprops={"linewidth":1.2},
                    capprops={"linewidth":1.2},
                    boxprops={"linewidth":1.0})
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels(etiquetas, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(titulo, fontsize=10, fontweight="bold")

data_mes_no2  = [df[df["mes_num"]==m]["NO2"].dropna().values for m in range(1,13)]
data_dia_no2  = [df[df["dia_sem_num"]==d]["NO2"].dropna().values for d in range(7)]
data_mes_traf = [df[df["mes_num"]==m]["trafico_medio"].dropna().values for m in range(1,13)]
data_dia_traf = [df[df["dia_sem_num"]==d]["trafico_medio"].dropna().values for d in range(7)]

boxplot_coloreado(axes[0,0], data_mes_no2,  meses_es, "#E53935",
                  "NO₂ por mes — Patrón estacional invernal", "NO₂ (µg/m³)")
boxplot_coloreado(axes[0,1], data_dia_no2,  dias_es,  "#E53935",
                  "NO₂ por día — Huella del tráfico laboral", "NO₂ (µg/m³)")
boxplot_coloreado(axes[1,0], data_mes_traf, meses_es, "#1E88E5",
                  "Tráfico por mes — Caídas en agosto y diciembre", "veh/hora")
boxplot_coloreado(axes[1,1], data_dia_traf, dias_es,  "#1E88E5",
                  "Tráfico por día — Caída clara en fin de semana", "veh/hora")

plt.tight_layout()
plt.savefig(CARPETA + "F0_05_boxplot_estacional.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ F0_05_boxplot_estacional.png")

# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 6: MEDIAS ANUALES
# ═══════════════════════════════════════════════════════════════════
print("\n[6] Medias anuales...")

anyos = [2021, 2022, 2023, 2024]
vars_plot = ["NO2", "PM10", "O3", "trafico_medio"]
titulos   = ["NO₂ (µg/m³)", "PM10 (µg/m³)", "O₃ (µg/m³)", "Tráfico (veh/h)"]
colores   = ["#E53935", "#8E24AA", "#00897B", "#1E88E5"]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Evolución de las medias anuales · Madrid 2021–2025\n"
             "(2025 excluido por ser parcial: enero–marzo)",
             fontsize=13, fontweight="bold")
axes = axes.flatten()

for i, (var, titulo, color) in enumerate(zip(vars_plot, titulos, colores)):
    medias = [float(df[df["fecha"].dt.year == a][var].mean()) for a in anyos]
    axes[i].bar([str(a) for a in anyos], medias, color=color, alpha=0.8, edgecolor="white")
    for j, m in enumerate(medias):
        axes[i].text(j, m + max(medias)*0.02, f"{m:.1f}", ha="center",
                     fontsize=10, fontweight="bold")
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
print("  ✓ F0_06_medias_anuales.png")

# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 7: MATRIZ DE CORRELACIONES
# ═══════════════════════════════════════════════════════════════════
print("\n[7] Matriz de correlaciones...")

vars_corr = ["trafico_medio","NO2","NO","NOx","PM10","PM25","O3",
             "T2M_MAX","T2M_MIN","RH2M","WS10M","PRECTOTCORR"]
vars_corr = [v for v in vars_corr if v in df.columns]
corr = df[vars_corr].corr().round(2)

etq = {"trafico_medio":"Tráfico", "NO2":"NO₂", "NO":"NO", "NOx":"NOₓ",
       "PM10":"PM10", "PM25":"PM2.5", "O3":"O₃", "T2M_MAX":"T_max",
       "T2M_MIN":"T_min", "RH2M":"Humedad", "WS10M":"Viento",
       "PRECTOTCORR":"Lluvia"}
etiquetas = [etq.get(v, v) for v in vars_corr]

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks(range(len(vars_corr)))
ax.set_yticks(range(len(vars_corr)))
ax.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(etiquetas, fontsize=9)

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
print("  ✓ F0_07_matriz_correlaciones.png")

# ═══════════════════════════════════════════════════════════════════
# RESUMEN ESTADÍSTICO POR CONSOLA
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  RESUMEN ESTADÍSTICO PARA LA MEMORIA")
print("=" * 65)

print("\n  Medias anuales NO₂ (µg/m³):")
for a in [2021, 2022, 2023, 2024]:
    m = float(df[df["fecha"].dt.year == a]["NO2"].mean())
    print(f"    {a}: {m:.2f}")

print("\n  Medias anuales Tráfico (veh/h):")
for a in [2021, 2022, 2023, 2024]:
    m = float(df[df["fecha"].dt.year == a]["trafico_medio"].mean())
    print(f"    {a}: {m:.1f}")

print("\n  NO₂ laborable vs fin de semana:")
lab = float(df[df["dia_sem_num"] < 5]["NO2"].mean())
fds = float(df[df["dia_sem_num"] >= 5]["NO2"].mean())
print(f"    Laborable: {lab:.2f} µg/m³")
print(f"    FdS      : {fds:.2f} µg/m³")
print(f"    Diferencia: {lab-fds:.2f} µg/m³ ({(lab-fds)/lab*100:.1f}%)")

print("\n  Tráfico laborable vs fin de semana:")
lab_t = float(df[df["dia_sem_num"] < 5]["trafico_medio"].mean())
fds_t = float(df[df["dia_sem_num"] >= 5]["trafico_medio"].mean())
print(f"    Laborable: {lab_t:.1f} veh/h")
print(f"    FdS      : {fds_t:.1f} veh/h")
print(f"    Diferencia: {lab_t-fds_t:.1f} veh/h ({(lab_t-fds_t)/lab_t*100:.1f}%)")

print("\n  Tendencia lineal (pendiente por año):")
medias_no2  = [float(df[df["fecha"].dt.year==a]["NO2"].mean()) for a in anyos]
medias_traf = [float(df[df["fecha"].dt.year==a]["trafico_medio"].mean()) for a in anyos]
z_no2  = np.polyfit(range(len(anyos)), medias_no2, 1)
z_traf = np.polyfit(range(len(anyos)), medias_traf, 1)
print(f"    NO₂:     {z_no2[0]:+.3f} µg/m³/año")
print(f"    Tráfico: {z_traf[0]:+.1f} veh/h/año")

print("\n  Correlaciones con NO₂:")
for var in vars_corr:
    if var != "NO2":
        r = float(corr.loc["NO2", var])
        print(f"    {etq.get(var,var):<12}: {r:+.3f}")

print(f"\n  Gráficos en: {CARPETA}")
print("\nFIN — Pega el output en el chat.")