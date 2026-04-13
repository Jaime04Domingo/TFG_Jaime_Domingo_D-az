#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:45:58 2026

@author: jaime
"""

"""
ANÁLISIS COMPLEMENTARIO — Análisis horario y matriz de correlaciones ampliada
1. Correlación tráfico vs cada contaminante (nivel horario)
2. Curvas horarias por año
3. Picos extremos: ¿coinciden tráfico y NO2?
4. Matriz de correlaciones grande con variables no usadas en modelos
TFG Madrid
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

RUTA_TRAF   = "/Users/jaime/Documents/Universidad/TFG/Trafico_Aforos_Final_2.csv"
RUTA_CONT   = "/Users/jaime/Documents/Universidad/TFG/Contaminacion_Definitivo_2.csv"
RUTA_INT    = "/Users/jaime/Documents/Universidad/TFG/Dataset_Diario_Integrado.csv"
CARPETA     = "/Users/jaime/Documents/Universidad/TFG/Graficos_Horario/"
os.makedirs(CARPETA, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 150,
})

print("=" * 65)
print("  ANÁLISIS COMPLEMENTARIO — HORARIO Y CORRELACIONES AMPLIADAS")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════════
# CARGA Y PREPARACIÓN DE DATOS HORARIOS
# ═══════════════════════════════════════════════════════════════════
print("\n[CARGA] Leyendo archivos horarios...")

# ── TRÁFICO ─────────────────────────────────────────────────────────
df_t = pd.read_csv(RUTA_TRAF, sep=";", encoding="utf-8-sig", low_memory=False)
df_t["FDIA"] = pd.to_datetime(df_t["FDIA"], errors="coerce")
df_t = df_t.dropna(subset=["FDIA"])
print(f"  Tráfico: {len(df_t):,} filas | columnas: {list(df_t.columns)[:8]}...")

cols_hor = [f"HOR{i}" for i in range(1, 13)]
# Convertir HOR a numérico
for col in cols_hor:
    df_t[col] = pd.to_numeric(df_t[col], errors="coerce")

# Reconstruir horario: FSEN '-' → horas 1-12, '=' → horas 13-24
print("  Reconstruyendo series horarias de tráfico...")
registros_t = []
for _, row in df_t.iterrows():
    fecha = row["FDIA"]
    est   = row["FEST"]
    fsen  = str(row["FSEN"])
    offset = 12 if "=" in fsen else 0
    for i, col in enumerate(cols_hor, 1):
        val = row[col]
        if pd.notna(val):
            registros_t.append({
                "fecha": fecha,
                "hora":  i + offset,
                "est":   est,
                "trafico": float(val)
            })

df_th = pd.DataFrame(registros_t)
# Media de sentidos por estación, hora y día
df_th = df_th.groupby(["fecha", "hora", "est"])["trafico"].sum().reset_index()
# Media entre estaciones
df_th = df_th.groupby(["fecha", "hora"])["trafico"].mean().reset_index()
df_th.columns = ["fecha", "hora", "trafico_hora"]
print(f"  Tráfico horario: {len(df_th):,} registros")

# ── CONTAMINACIÓN ────────────────────────────────────────────────────
df_c = pd.read_csv(RUTA_CONT, sep=";", encoding="utf-8-sig", low_memory=False)
df_c["fecha"] = pd.to_datetime(df_c["fecha"], errors="coerce")
df_c = df_c.dropna(subset=["fecha"])
print(f"  Contaminación: {len(df_c):,} filas")

# Columnas H
cols_h = [f"H{i:02d}" for i in range(1, 25)]
cols_h = [c for c in cols_h if c in df_c.columns]

# Magnitudes de interés
MAGS = {8: "NO2", 7: "NO", 12: "NOx", 10: "PM10", 9: "PM25", 6: "CO", 14: "O3"}

# Convertir a largo (fecha, hora, magnitud, valor)
print("  Reconstruyendo series horarias de contaminación...")
registros_c = []
for mag_cod, mag_nombre in MAGS.items():
    sub = df_c[df_c["MAGNITUD"] == mag_cod][["fecha"] + cols_h].copy()
    if sub.empty:
        continue
    for col_idx, col in enumerate(cols_h, 1):
        vals = pd.to_numeric(sub[col], errors="coerce")
        fechas = sub["fecha"]
        for f, v in zip(fechas, vals):
            if pd.notna(v):
                registros_c.append({"fecha": f, "hora": col_idx,
                                     "magnitud": mag_nombre, "valor": float(v)})

df_ch = pd.DataFrame(registros_c)
# Media entre estaciones
df_ch = df_ch.groupby(["fecha", "hora", "magnitud"])["valor"].mean().reset_index()
# Pivot
df_ch_pivot = df_ch.pivot_table(index=["fecha","hora"], columns="magnitud",
                                 values="valor", aggfunc="mean").reset_index()
df_ch_pivot.columns.name = None
print(f"  Contaminación horaria: {len(df_ch_pivot):,} registros")

# ── MERGE ────────────────────────────────────────────────────────────
df_hora = pd.merge(df_th, df_ch_pivot, on=["fecha","hora"], how="inner")
df_hora["anyo"] = df_hora["fecha"].dt.year
df_hora["mes"]  = df_hora["fecha"].dt.month
df_hora["dow"]  = df_hora["fecha"].dt.dayofweek
df_hora["es_lab"] = (df_hora["dow"] < 5).astype(int)
print(f"  Dataset horario integrado: {len(df_hora):,} registros")

cont_cols = [c for c in MAGS.values() if c in df_hora.columns]

# ═══════════════════════════════════════════════════════════════════
# ANÁLISIS 1: CORRELACIÓN TRÁFICO VS CADA CONTAMINANTE
# ═══════════════════════════════════════════════════════════════════
print("\n[1] Correlación tráfico vs contaminantes (nivel horario)...")

print("\n  Correlaciones Pearson (nivel horario, todos los datos):")
print(f"  {'Contaminante':<12} {'r global':>10} {'r laborable':>13} {'r fin semana':>14}")
print(f"  {'-'*55}")
corr_results = {}
for cont in cont_cols:
    df_v = df_hora[["trafico_hora", cont, "es_lab"]].dropna()
    r_global = float(df_v["trafico_hora"].corr(df_v[cont]))
    r_lab    = float(df_v[df_v["es_lab"]==1]["trafico_hora"].corr(df_v[df_v["es_lab"]==1][cont]))
    r_fds    = float(df_v[df_v["es_lab"]==0]["trafico_hora"].corr(df_v[df_v["es_lab"]==0][cont]))
    corr_results[cont] = {"global": r_global, "lab": r_lab, "fds": r_fds}
    print(f"  {cont:<12} {r_global:>10.3f} {r_lab:>13.3f} {r_fds:>14.3f}")

# GRÁFICO: correlaciones por contaminante
fig, ax = plt.subplots(figsize=(11, 6))
fig.suptitle("Correlación Pearson entre Tráfico y Contaminantes\n"
             "Análisis a nivel horario · Madrid 2021–2025",
             fontsize=13, fontweight="bold")

x     = np.arange(len(cont_cols))
w     = 0.28
cols_bar = ["#1E88E5", "#E53935", "#43A047"]
labels   = ["Global", "Laborable (L–V)", "Fin de semana"]

for i, (key, color, label) in enumerate(zip(["global","lab","fds"], cols_bar, labels)):
    vals = [corr_results[c][key] for c in cont_cols]
    bars = ax.bar(x + i*w, vals, width=w, color=color, alpha=0.85,
                  edgecolor="white", label=label)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                float(bar.get_height()) + 0.005,
                f"{val:.2f}", ha="center", fontsize=7.5, fontweight="bold")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x + w)
ax.set_xticklabels(cont_cols, fontsize=10)
ax.set_ylabel("Correlación de Pearson (r)", fontsize=11)
ax.set_title("Mayor correlación horaria que diaria (r=0,21 en diario)\n"
             "Separación laborable vs fin de semana muestra el efecto del patrón semanal",
             fontsize=10)
ax.legend(fontsize=9)
ax.set_ylim(-0.1, max(max(corr_results[c]["lab"] for c in cont_cols),
                       max(corr_results[c]["global"] for c in cont_cols)) + 0.15)

plt.tight_layout()
plt.savefig(CARPETA + "H01_correlacion_trafico_contaminantes.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ H01_correlacion_trafico_contaminantes.png")

# ═══════════════════════════════════════════════════════════════════
# ANÁLISIS 2: CURVAS HORARIAS DE NO2 POR AÑO
# ═══════════════════════════════════════════════════════════════════
print("\n[2] Curvas horarias por año...")

anyos_plot = [2021, 2022, 2023, 2024]
colores_anyo = {2021:"#1E88E5", 2022:"#43A047", 2023:"#FB8C00", 2024:"#E53935"}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Perfil horario de NO₂ y Tráfico por año\n"
             "Media de días laborables (L–V) · Madrid 2021–2024",
             fontsize=13, fontweight="bold")

# Solo laborables para comparación limpia
df_lab = df_hora[df_hora["es_lab"] == 1]

for anyo in anyos_plot:
    sub = df_lab[df_lab["anyo"] == anyo]
    perfil_no2  = sub.groupby("hora")["NO2"].mean()
    perfil_traf = sub.groupby("hora")["trafico_hora"].mean()
    color = colores_anyo[anyo]

    axes[0].plot(perfil_no2.index, perfil_no2.values,
                 marker="o", markersize=4, linewidth=2,
                 color=color, label=str(anyo))
    axes[1].plot(perfil_traf.index, perfil_traf.values,
                 marker="o", markersize=4, linewidth=2,
                 color=color, label=str(anyo))

for ax, titulo, ylabel, zonas in zip(
    axes,
    ["NO₂ — Perfil horario por año (laborables)", "Tráfico — Perfil horario por año (laborables)"],
    ["NO₂ (µg/m³)", "Tráfico (veh/hora)"],
    [True, True]
):
    ax.set_xlabel("Hora del día", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(titulo, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, title="Año")
    ax.set_xticks(range(1, 25, 2))
    if zonas:
        ax.axvspan(7, 10, alpha=0.08, color="red", label="Hora punta mañana")
        ax.axvspan(17, 21, alpha=0.08, color="orange", label="Hora punta tarde")

plt.tight_layout()
plt.savefig(CARPETA + "H02_curvas_horarias_por_anyo.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ H02_curvas_horarias_por_anyo.png")

# Imprimir valores clave para la memoria
print("\n  Pico matutino NO₂ (hora 9, laborables) por año:")
for anyo in anyos_plot:
    sub = df_lab[(df_lab["anyo"] == anyo) & (df_lab["hora"] == 9)]
    m = float(sub["NO2"].mean()) if len(sub) > 0 else np.nan
    print(f"    {anyo}: {m:.2f} µg/m³")

print("\n  Media NO₂ horas punta (7-10h, laborables) por año:")
for anyo in anyos_plot:
    sub = df_lab[(df_lab["anyo"] == anyo) & (df_lab["hora"].between(7,10))]
    m = float(sub["NO2"].mean()) if len(sub) > 0 else np.nan
    print(f"    {anyo}: {m:.2f} µg/m³")

# ═══════════════════════════════════════════════════════════════════
# ANÁLISIS 3: PICOS EXTREMOS — ¿COINCIDEN TRÁFICO Y NO2?
# ═══════════════════════════════════════════════════════════════════
print("\n[3] Análisis de picos extremos...")

p95_traf = float(df_hora["trafico_hora"].quantile(0.95))
p95_no2  = float(df_hora["NO2"].quantile(0.95)) if "NO2" in df_hora.columns else None
p05_traf = float(df_hora["trafico_hora"].quantile(0.05))
p05_no2  = float(df_hora["NO2"].quantile(0.05)) if "NO2" in df_hora.columns else None

if p95_no2:
    mask_traf_alto = df_hora["trafico_hora"] >= p95_traf
    mask_no2_alto  = df_hora["NO2"] >= p95_no2
    coincidencia_alta = (mask_traf_alto & mask_no2_alto).sum()
    n_traf_alto = mask_traf_alto.sum()
    n_no2_alto  = mask_no2_alto.sum()

    mask_traf_bajo = df_hora["trafico_hora"] <= p05_traf
    mask_no2_bajo  = df_hora["NO2"] <= p05_no2
    coincidencia_baja = (mask_traf_bajo & mask_no2_bajo).sum()

    print(f"  P95 tráfico: {p95_traf:.0f} veh/h | P95 NO₂: {p95_no2:.1f} µg/m³")
    print(f"  Horas con tráfico extremo (≥P95): {int(n_traf_alto):,}")
    print(f"  Horas con NO₂ extremo (≥P95): {int(n_no2_alto):,}")
    print(f"  Coincidencia (ambos extremos altos): {int(coincidencia_alta):,} "
          f"({coincidencia_alta/n_traf_alto*100:.1f}% de horas de tráfico extremo)")
    print(f"  Coincidencia (ambos extremos bajos): {int(coincidencia_baja):,}")

    # Distribución de NO2 cuando tráfico es extremo vs normal
    no2_traf_alto = df_hora[mask_traf_alto]["NO2"].dropna()
    no2_traf_norm = df_hora[~mask_traf_alto]["NO2"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Picos extremos de Tráfico y NO₂\n"
                 "¿Cuando el tráfico es máximo, el NO₂ también lo es?",
                 fontsize=13, fontweight="bold")

    axes[0].hist(no2_traf_norm.values, bins=50, alpha=0.6, color="#1E88E5",
                 density=True, label=f"Tráfico normal (n={len(no2_traf_norm):,})")
    axes[0].hist(no2_traf_alto.values, bins=50, alpha=0.6, color="#E53935",
                 density=True, label=f"Tráfico ≥P95 ({p95_traf:.0f} veh/h) (n={len(no2_traf_alto):,})")
    axes[0].axvline(float(no2_traf_norm.mean()), color="#1E88E5", linewidth=2, linestyle="--",
                    label=f"Media normal: {float(no2_traf_norm.mean()):.1f}")
    axes[0].axvline(float(no2_traf_alto.mean()), color="#E53935", linewidth=2, linestyle="--",
                    label=f"Media alto: {float(no2_traf_alto.mean()):.1f}")
    axes[0].set_xlabel("NO₂ (µg/m³)", fontsize=10)
    axes[0].set_ylabel("Densidad", fontsize=10)
    axes[0].set_title("Distribución de NO₂ según nivel de tráfico",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8)

    # Scatter por hora del día
    media_hora = df_hora.groupby("hora").agg(
        traf_medio=("trafico_hora","mean"),
        no2_medio=("NO2","mean")
    ).reset_index()
    scatter = axes[1].scatter(media_hora["traf_medio"], media_hora["no2_medio"],
                              c=media_hora["hora"], cmap="RdYlGn_r", s=100, zorder=5)
    for _, row in media_hora.iterrows():
        axes[1].annotate(f"h{int(row['hora'])}",
                         (row["traf_medio"], row["no2_medio"]),
                         fontsize=7, ha="center", va="bottom")
    plt.colorbar(scatter, ax=axes[1], label="Hora del día")
    r_hora = float(media_hora["traf_medio"].corr(media_hora["no2_medio"]))
    axes[1].set_xlabel("Tráfico medio (veh/hora)", fontsize=10)
    axes[1].set_ylabel("NO₂ medio (µg/m³)", fontsize=10)
    axes[1].set_title(f"Media por hora del día\nr={r_hora:.3f}",
                      fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(CARPETA + "H03_picos_extremos.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ H03_picos_extremos.png")

# ═══════════════════════════════════════════════════════════════════
# ANÁLISIS 4: DIFERENCIA HORARIA DE NO2 ENTRE 2021 Y 2024
# ═══════════════════════════════════════════════════════════════════
print("\n[4] Diferencia horaria NO₂ entre 2021 y 2024...")

if "NO2" in df_hora.columns:
    perf_21 = df_lab[df_lab["anyo"]==2021].groupby("hora")["NO2"].mean()
    perf_24 = df_lab[df_lab["anyo"]==2024].groupby("hora")["NO2"].mean()
    horas_comunes = perf_21.index.intersection(perf_24.index)
    diff = perf_24[horas_comunes] - perf_21[horas_comunes]

    print("\n  Diferencia NO₂ 2024 vs 2021 por hora (laborables):")
    for hora in horas_comunes:
        print(f"    H{int(hora):02d}: {float(diff[hora]):+.2f} µg/m³")

    fig, ax = plt.subplots(figsize=(12, 6))
    colors_diff = ["#E53935" if v > 0 else "#43A047" for v in diff.values]
    bars = ax.bar(horas_comunes, diff.values, color=colors_diff, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=1)
    ax.axvspan(7, 10, alpha=0.08, color="gray", label="Hora punta mañana")
    ax.axvspan(17, 21, alpha=0.08, color="blue", label="Hora punta tarde")
    for bar, val in zip(bars, diff.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                float(val) + (0.2 if val >= 0 else -0.5),
                f"{val:+.1f}", ha="center", fontsize=7, fontweight="bold")
    ax.set_xlabel("Hora del día", fontsize=11)
    ax.set_ylabel("Diferencia NO₂ (µg/m³)\n2024 − 2021", fontsize=11)
    ax.set_title("Reducción horaria del NO₂: 2024 vs 2021 (días laborables)\n"
                 "Verde = mejora · Rojo = empeoramiento",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, 25))
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(CARPETA + "H04_diferencia_horaria_2021_2024.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ H04_diferencia_horaria_2021_2024.png")

# ═══════════════════════════════════════════════════════════════════
# ANÁLISIS 5: MATRIZ DE CORRELACIONES AMPLIADA
# (incluyendo variables no usadas en modelos)
# ═══════════════════════════════════════════════════════════════════
print("\n[5] Matriz de correlaciones ampliada...")

df_int = pd.read_csv(RUTA_INT, sep=";", encoding="utf-8-sig", low_memory=False)
df_int["fecha"] = pd.to_datetime(df_int["fecha"])
df_int["es_fin_semana"] = df_int["es_fin_semana"].astype(int)

# Todas las variables numéricas del dataset integrado
vars_todas = [
    # Variables usadas en los modelos
    "trafico_medio", "T2M_MAX", "T2M_MIN", "RH2M", "WS10M",
    "PRECTOTCORR", "mes", "es_fin_semana",
    # Variables NO usadas — a explorar
    "WD10M", "ALLSKY_SFC_SW_DWN", "PS", "T2M_RANGE", "anyo",
    # Contaminantes
    "NO2", "NO", "NOx", "PM10", "PM25", "CO", "O3",
]
vars_todas = [v for v in vars_todas if v in df_int.columns]
df_corr = df_int[vars_todas].copy()
corr_matrix = df_corr.corr().round(3)

# Nombres legibles
ETIQUETAS = {
    "trafico_medio": "Tráfico", "T2M_MAX": "T_max", "T2M_MIN": "T_min",
    "RH2M": "Humedad", "WS10M": "Viento", "PRECTOTCORR": "Lluvia",
    "mes": "Mes", "es_fin_semana": "Fin semana",
    "WD10M": "Dir.viento★", "ALLSKY_SFC_SW_DWN": "Radiación★",
    "PS": "Presión★", "T2M_RANGE": "Rango_T★", "anyo": "Año★",
    "NO2": "NO₂", "NO": "NO", "NOx": "NOₓ",
    "PM10": "PM10", "PM25": "PM2.5", "CO": "CO", "O3": "O₃",
}
etq = [ETIQUETAS.get(v, v) for v in vars_todas]

# Columnas nuevas (★) con su correlación con NO2
print("\n  Variables NO usadas en modelos — correlación con NO₂:")
vars_nuevas = ["WD10M", "ALLSKY_SFC_SW_DWN", "PS", "T2M_RANGE", "anyo"]
for v in vars_nuevas:
    if v in corr_matrix.columns and "NO2" in corr_matrix.index:
        r = float(corr_matrix.loc["NO2", v])
        print(f"  {ETIQUETAS[v]:<18}: r={r:+.3f} con NO₂")

fig, ax = plt.subplots(figsize=(15, 13))
im = ax.imshow(corr_matrix.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.7, label="Correlación de Pearson")
ax.set_xticks(range(len(vars_todas)))
ax.set_yticks(range(len(vars_todas)))
ax.set_xticklabels(etq, rotation=45, ha="right", fontsize=8.5)
ax.set_yticklabels(etq, fontsize=8.5)

for i in range(len(vars_todas)):
    for j in range(len(vars_todas)):
        val = float(corr_matrix.values[i, j])
        color_txt = "white" if abs(val) > 0.65 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=6.5, color=color_txt, fontweight="bold")

# Marcar variables nuevas con borde
n_mod = 8   # primeras 8 son las del modelo
n_nue = 5   # siguientes 5 son las nuevas
for idx in range(n_mod, n_mod + n_nue):
    ax.axvline(idx - 0.5, color="navy", linewidth=1.5, alpha=0.5)
    ax.axvline(idx + 0.5, color="navy", linewidth=1.5, alpha=0.5)

ax.set_title("Matriz de correlaciones ampliada\n"
             "★ = variables NO incluidas en los modelos · "
             "Bandas azules = zona de variables nuevas",
             fontsize=12, fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig(CARPETA + "H05_matriz_correlaciones_ampliada.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ H05_matriz_correlaciones_ampliada.png")

# ── RESUMEN ───────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  RESUMEN DEL ANÁLISIS COMPLEMENTARIO")
print("="*65)
print(f"\n  Gráficos en: {CARPETA}")
print("  H01_correlacion_trafico_contaminantes.png")
print("  H02_curvas_horarias_por_anyo.png")
print("  H03_picos_extremos.png")
print("  H04_diferencia_horaria_2021_2024.png")
print("  H05_matriz_correlaciones_ampliada.png")
print("\nFIN — Pega el output en el chat.")