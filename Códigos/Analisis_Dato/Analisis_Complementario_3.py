#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:10:52 2026

@author: jaime
"""

"""
Evolución de picos de tráfico por año
Complemento al análisis horario - TFG
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

RUTA_INT = "/Users/jaime/Documents/Universidad/TFG/Dataset_Diario_Integrado.csv"
RUTA_TRAF = "/Users/jaime/Documents/Universidad/TFG/Trafico_Aforos_Final_2.csv"
CARPETA = "/Users/jaime/Documents/Universidad/TFG/Graficos_Horario/"
os.makedirs(CARPETA, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 150,
})

print("=" * 65)
print("  EVOLUCIÓN DE PICOS DE TRÁFICO Y NO₂ POR AÑO")
print("=" * 65)

# ── Desde el dataset diario integrado (más rápido) ───────────────────
df = pd.read_csv(RUTA_INT, sep=";", encoding="utf-8-sig", low_memory=False)
df["fecha"] = pd.to_datetime(df["fecha"])
df["anyo"]  = df["fecha"].dt.year

anyos = [2021, 2022, 2023, 2024]

print("\n  Estadísticas de tráfico por año (veh/hora):")
print(f"  {'Año':>5} {'Media':>8} {'Mediana':>9} {'P90':>8} {'P95':>8} {'P99':>8} {'Máx':>8}")
print(f"  {'-'*58}")

stats_traf = {}
for a in anyos:
    sub = df[df["anyo"] == a]["trafico_medio"].dropna()
    media  = float(sub.mean())
    mediana = float(sub.median())
    p90    = float(sub.quantile(0.90))
    p95    = float(sub.quantile(0.95))
    p99    = float(sub.quantile(0.99))
    maximo = float(sub.max())
    stats_traf[a] = {"media": media, "mediana": mediana,
                     "p90": p90, "p95": p95, "p99": p99, "max": maximo}
    print(f"  {a:>5} {media:>8.1f} {mediana:>9.1f} {p90:>8.1f} {p95:>8.1f} {p99:>8.1f} {maximo:>8.1f}")

print("\n  Estadísticas de NO₂ por año (µg/m³):")
print(f"  {'Año':>5} {'Media':>8} {'Mediana':>9} {'P90':>8} {'P95':>8} {'P99':>8} {'Máx':>8}")
print(f"  {'-'*58}")

stats_no2 = {}
for a in anyos:
    sub = df[df["anyo"] == a]["NO2"].dropna()
    media  = float(sub.mean())
    mediana = float(sub.median())
    p90    = float(sub.quantile(0.90))
    p95    = float(sub.quantile(0.95))
    p99    = float(sub.quantile(0.99))
    maximo = float(sub.max())
    stats_no2[a] = {"media": media, "mediana": mediana,
                    "p90": p90, "p95": p95, "p99": p99, "max": maximo}
    print(f"  {a:>5} {media:>8.1f} {mediana:>9.1f} {p90:>8.1f} {p95:>8.1f} {p99:>8.1f} {maximo:>8.1f}")

print("\n  Cambio 2024 vs 2021 (puntos percentiles):")
print(f"  {'Métrica':<12} {'Tráfico':>12} {'NO₂':>12}")
print(f"  {'-'*38}")
for key, label in [("media","Media"),("p90","P90"),("p95","P95"),("p99","P99"),("max","Máximo")]:
    dt = stats_traf[2024][key] - stats_traf[2021][key]
    dn = stats_no2[2024][key]  - stats_no2[2021][key]
    print(f"  {label:<12} {dt:>+11.1f} {dn:>+11.1f}")

# ── GRÁFICO COMPARATIVO ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Evolución de la distribución de Tráfico y NO₂ por año\n"
             "Comparativa de medias, percentiles y máximos · Madrid 2021–2024",
             fontsize=13, fontweight="bold")
axes = axes.flatten()

colores_anyo = {2021:"#1E88E5", 2022:"#43A047", 2023:"#FB8C00", 2024:"#E53935"}
metricas     = ["media", "p90", "p95", "p99"]
etiquetas    = ["Media", "Percentil 90", "Percentil 95", "Percentil 99"]

# Panel 1: Evolución de métricas de tráfico
ax = axes[0]
x = np.arange(len(anyos))
for i, (met, etq) in enumerate(zip(metricas, etiquetas)):
    vals = [stats_traf[a][met] for a in anyos]
    ax.plot(anyos, vals, marker="o", linewidth=2, markersize=7, label=etq)
ax.set_title("Tráfico — evolución de percentiles por año", fontsize=11, fontweight="bold")
ax.set_ylabel("veh/hora")
ax.set_xlabel("Año")
ax.legend(fontsize=8)
ax.set_xticks(anyos)

# Panel 2: Evolución de métricas de NO₂
ax = axes[1]
for met, etq in zip(metricas, etiquetas):
    vals = [stats_no2[a][met] for a in anyos]
    ax.plot(anyos, vals, marker="o", linewidth=2, markersize=7, label=etq)
ax.set_title("NO₂ — evolución de percentiles por año", fontsize=11, fontweight="bold")
ax.set_ylabel("µg/m³")
ax.set_xlabel("Año")
ax.legend(fontsize=8)
ax.set_xticks(anyos)

# Panel 3: Boxplot tráfico por año
ax = axes[2]
datos_box_t = [df[df["anyo"]==a]["trafico_medio"].dropna().values for a in anyos]
bp = ax.boxplot(datos_box_t, patch_artist=True, showfliers=True,
                flierprops={"marker":".", "markersize":3, "alpha":0.3},
                medianprops={"color":"white","linewidth":2})
for patch, a in zip(bp["boxes"], anyos):
    patch.set_facecolor(colores_anyo[a])
    patch.set_alpha(0.8)
ax.set_xticklabels([str(a) for a in anyos])
ax.set_title("Distribución diaria del Tráfico por año", fontsize=11, fontweight="bold")
ax.set_ylabel("Intensidad media (veh/hora)")

# Panel 4: Boxplot NO₂ por año
ax = axes[3]
datos_box_n = [df[df["anyo"]==a]["NO2"].dropna().values for a in anyos]
bp = ax.boxplot(datos_box_n, patch_artist=True, showfliers=True,
                flierprops={"marker":".", "markersize":3, "alpha":0.3},
                medianprops={"color":"white","linewidth":2})
for patch, a in zip(bp["boxes"], anyos):
    patch.set_facecolor(colores_anyo[a])
    patch.set_alpha(0.8)
ax.set_xticklabels([str(a) for a in anyos])
ax.set_title("Distribución diaria del NO₂ por año", fontsize=11, fontweight="bold")
ax.set_ylabel("NO₂ (µg/m³)")

plt.tight_layout()
ruta = CARPETA + "H06_picos_trafico_NO2_por_anyo.png"
plt.savefig(ruta, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  ✓ H06_picos_trafico_NO2_por_anyo.png")
print(f"\n  Gráfico en: {CARPETA}")
print("\nFIN — Pega el output en el chat.")