#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:34:44 2026

@author: jaime
"""

### VISUALIZACIÓN CONTAMINACIÓN

"""
Visualizaciones del dataset de Contaminación - TFG
Fuente: Red de Calidad del Aire - Ayuntamiento de Madrid
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os
 
RUTA    = "/Users/jaime/Documents/Universidad/TFG/Contaminacion_Final.csv"
CARPETA = "/Users/jaime/Documents/Universidad/TFG/Graficos_Contaminacion/"
os.makedirs(CARPETA, exist_ok=True)
 
# ── DICCIONARIOS OFICIALES (PDF Ayuntamiento de Madrid) ──────────────
MAGNITUDES = {
    1:  ("SO₂",   "µg/m³"),
    6:  ("CO",    "mg/m³"),
    7:  ("NO",    "µg/m³"),
    8:  ("NO₂",   "µg/m³"),
    9:  ("PM2.5", "µg/m³"),
    10: ("PM10",  "µg/m³"),
    12: ("NOₓ",   "µg/m³"),
    14: ("O₃",    "µg/m³"),
    20: ("TOL",   "µg/m³"),
    30: ("BEN",   "µg/m³"),
    35: ("EBE",   "µg/m³"),
    42: ("TCH",   "mg/m³"),
    43: ("CH₄",   "mg/m³"),
    44: ("NMHC",  "mg/m³"),
}
 
# Estaciones de la red (Anexo I del PDF - solo las activas "Alta.-")
ESTACIONES = {
    4:  "Pza. de España",
    8:  "Escuelas Aguirre",
    11: "Av. Ramón y Cajal",
    16: "Arturo Soria",
    17: "Villaverde Alto",
    18: "C/ Farolillo",
    24: "Casa de Campo",
    27: "Barajas",
    35: "Pza. del Carmen",
    36: "Moratalaz",
    38: "Cuatro Caminos",
    39: "Barrio del Pilar",
    40: "Vallecas",
    47: "Méndez Álvaro",
    48: "Pº. Castellana",
    49: "Retiro",
    50: "Pza. Castilla",
    54: "Ensanche Vallecas",
    55: "Urb. Embajada",
    56: "Plaza Elíptica",
    57: "Sanchinarro",
    58: "El Pardo",
    59: "Pque. Juan Carlos I",
    60: "Tres Olivos",
}
 
# Coordenadas aproximadas de estaciones (lat, lon)
COORDS = {
    4:  (40.4237, -3.7124),   # Pza. de España
    8:  (40.4191, -3.6821),   # Escuelas Aguirre
    11: (40.4517, -3.6767),   # Av. Ramón y Cajal
    16: (40.4441, -3.6398),   # Arturo Soria
    17: (40.3494, -3.7151),   # Villaverde Alto
    18: (40.3983, -3.7187),   # C/ Farolillo
    24: (40.4195, -3.7442),   # Casa de Campo
    27: (40.4774, -3.5800),   # Barajas
    35: (40.4180, -3.7028),   # Pza. del Carmen
    36: (40.4068, -3.6576),   # Moratalaz
    38: (40.4454, -3.7041),   # Cuatro Caminos
    39: (40.4812, -3.7063),   # Barrio del Pilar
    40: (40.3858, -3.6613),   # Vallecas
    47: (40.3962, -3.6839),   # Méndez Álvaro
    48: (40.4589, -3.6919),   # Pº. Castellana
    49: (40.4138, -3.6836),   # Retiro
    50: (40.4658, -3.6932),   # Pza. Castilla
    54: (40.3731, -3.6401),   # Ensanche Vallecas
    55: (40.4762, -3.5793),   # Urb. Embajada
    56: (40.3877, -3.7112),   # Plaza Elíptica
    57: (40.4936, -3.6591),   # Sanchinarro
    58: (40.5127, -3.7741),   # El Pardo
    59: (40.4681, -3.6199),   # Pque. Juan Carlos I
    60: (40.5005, -3.7094),   # Tres Olivos
}
 
# Tipo de estación
TIPO_EST = {
    4: "tráfico", 8: "tráfico", 11: "tráfico", 16: "fondo urbano",
    17: "fondo urbano", 18: "tráfico", 24: "suburbana", 27: "suburbana",
    35: "tráfico", 36: "fondo urbano", 38: "tráfico", 39: "fondo urbano",
    40: "fondo urbano", 47: "tráfico", 48: "tráfico", 49: "fondo urbano",
    50: "fondo urbano", 54: "fondo urbano", 55: "suburbana", 56: "tráfico",
    57: "fondo urbano", 58: "suburbana", 59: "suburbana", 60: "suburbana",
}
 
print("Cargando datos...")
df = pd.read_csv(RUTA, sep=";", encoding="utf-8-sig", low_memory=False)
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
df["anyo"]  = df["fecha"].dt.year
df["mes"]   = df["fecha"].dt.month
df["dia_semana"] = df["fecha"].dt.dayofweek  # 0=lunes, 6=domingo
 
cols_h = sorted([c for c in df.columns if c.startswith("H") and c[1:].isdigit()],
                key=lambda x: int(x[1:]))
 
# Estilo global
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":     True,
    "grid.alpha":    0.3,
    "grid.linestyle": "--",
    "figure.dpi":    150,
})
 
COLORES = {
    "NO₂":   "#E53935",
    "PM10":  "#8E24AA",
    "O₃":    "#00897B",
    "PM2.5": "#F4511E",
    "NOₓ":   "#FB8C00",
    "NO":    "#43A047",
    "SO₂":   "#1E88E5",
    "CO":    "#6D4C41",
}
 
# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 1: MAPA DE ESTACIONES
# ═══════════════════════════════════════════════════════════════════
print("Generando Gráfico 1: Mapa de estaciones...")
 
fig, ax = plt.subplots(figsize=(11, 10))
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#EEF2F7")
 
colores_tipo = {"tráfico": "#E53935", "fondo urbano": "#1E88E5", "suburbana": "#43A047"}
 
for cod, (lat, lon) in COORDS.items():
    tipo = TIPO_EST.get(cod, "fondo urbano")
    color = colores_tipo[tipo]
    ax.scatter(lon, lat, s=180, c=color, zorder=5, edgecolors="white", linewidths=1.5)
    nombre = ESTACIONES.get(cod, str(cod))
    nombre_corto = nombre[:18]
    ax.annotate(nombre_corto, (lon, lat),
                textcoords="offset points", xytext=(6, 4),
                fontsize=6.5, color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))
 
# Leyenda
parches = [mpatches.Patch(color=c, label=t.capitalize()) for t, c in colores_tipo.items()]
ax.legend(handles=parches, title="Tipo de estación", loc="lower right",
          framealpha=0.9, fontsize=9, title_fontsize=9)
 
ax.set_xlabel("Longitud", fontsize=10)
ax.set_ylabel("Latitud", fontsize=10)
ax.set_title("Red de estaciones de calidad del aire\nAyuntamiento de Madrid (24 estaciones activas)",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlim(-3.82, -3.54)
ax.set_ylim(40.32, 40.55)
 
plt.tight_layout()
ruta1 = CARPETA + "01_mapa_estaciones.png"
plt.savefig(ruta1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Guardado: {ruta1}")
 
# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 2: EVOLUCIÓN ANUAL DE CONTAMINANTES CLAVE
# ═══════════════════════════════════════════════════════════════════
print("Generando Gráfico 2: Evolución anual...")
 
mags_clave = [8, 10, 14, 9]  # NO2, PM10, O3, PM2.5
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Evolución anual de contaminantes clave\nMedia de todas las estaciones · 2021–2025",
             fontsize=13, fontweight="bold")
axes = axes.flatten()
 
for idx, mag in enumerate(mags_clave):
    nombre, unidad = MAGNITUDES[mag]
    color = COLORES.get(nombre, "#555555")
    df_mag = df[df["MAGNITUD"] == mag].copy()
 
    # Media anual por año (promedio de todas las horas y estaciones)
    filas = []
    for anyo in sorted(df_mag["anyo"].dropna().unique()):
        vals = df_mag[df_mag["anyo"] == anyo][cols_h].values.flatten()
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            filas.append({"anyo": int(anyo), "media": np.mean(vals),
                          "p25": np.percentile(vals, 25),
                          "p75": np.percentile(vals, 75)})
 
    if not filas:
        continue
    res = pd.DataFrame(filas)
 
    ax = axes[idx]
    ax.fill_between(res["anyo"], res["p25"], res["p75"],
                    alpha=0.2, color=color, label="P25–P75")
    ax.plot(res["anyo"], res["media"], "o-", color=color,
            linewidth=2.5, markersize=7, label="Media anual")
 
    # Líneas de referencia OMS / UE
    refs = {8: (40, "Límite UE"), 10: (40, "Límite UE"),
            14: (120, "Valor objetivo UE"), 9: (25, "Límite UE")}
    if mag in refs:
        val_ref, etq = refs[mag]
        ax.axhline(val_ref, color="red", linestyle="--",
                   linewidth=1.2, alpha=0.7, label=etq)
 
    ax.set_title(f"{nombre}", fontsize=12, fontweight="bold", color=color)
    ax.set_xlabel("Año")
    ax.set_ylabel(f"Concentración ({unidad})")
    ax.set_xticks(res["anyo"])
    ax.legend(fontsize=8)
 
plt.tight_layout()
ruta2 = CARPETA + "02_evolucion_anual.png"
plt.savefig(ruta2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Guardado: {ruta2}")
 
# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 3: PERFIL HORARIO (LABORABLE VS FIN DE SEMANA)
# ═══════════════════════════════════════════════════════════════════
print("Generando Gráfico 3: Perfil horario...")
 
mags_horario = [8, 10, 14]  # NO2, PM10, O3
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Perfil horario de contaminantes\nDías laborables vs. fines de semana · Media 2021–2025",
             fontsize=13, fontweight="bold")
 
for idx, mag in enumerate(mags_horario):
    nombre, unidad = MAGNITUDES[mag]
    color = COLORES.get(nombre, "#555555")
    df_mag = df[df["MAGNITUD"] == mag].copy()
    df_lab = df_mag[df_mag["dia_semana"] < 5]   # lun–vie
    df_fds = df_mag[df_mag["dia_semana"] >= 5]  # sáb–dom
 
    horas = list(range(1, 25))
    medias_lab, medias_fds = [], []
    for i, col in enumerate(cols_h):
        medias_lab.append(df_lab[col].mean())
        medias_fds.append(df_fds[col].mean())
 
    ax = axes[idx]
    ax.plot(horas, medias_lab, "o-", color=color, linewidth=2.5,
            markersize=5, label="Laborable (L–V)")
    ax.plot(horas, medias_fds, "s--", color=color, linewidth=2,
            markersize=5, alpha=0.6, label="Fin de semana")
    ax.fill_between(horas, medias_fds, medias_lab,
                    alpha=0.1, color=color)
 
    ax.axvspan(7.5, 9.5,  alpha=0.07, color="red")
    ax.axvspan(17.5, 20,  alpha=0.07, color="orange")
    ax.set_title(f"{nombre} ({unidad})", fontsize=11, fontweight="bold", color=color)
    ax.set_xlabel("Hora del día")
    ax.set_ylabel(f"Media ({unidad})")
    ax.set_xticks(range(1, 25, 2))
    ax.legend(fontsize=8)
 
plt.tight_layout()
ruta3 = CARPETA + "03_perfil_horario.png"
plt.savefig(ruta3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Guardado: {ruta3}")
 
# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 4: BOXPLOT DE DISTRIBUCIÓN POR CONTAMINANTE
# ═══════════════════════════════════════════════════════════════════
print("Generando Gráfico 4: Boxplot distribución...")
 
mags_box = [1, 6, 7, 8, 9, 10, 12, 14, 20, 30]
fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle("Distribución de concentraciones por contaminante\n(µg/m³ salvo CO, TCH, CH₄, NMHC en mg/m³)",
             fontsize=12, fontweight="bold")
 
datos_box, etiquetas, colores_box = [], [], []
for mag in mags_box:
    if mag not in MAGNITUDES:
        continue
    nombre, unidad = MAGNITUDES[mag]
    df_mag = df[df["MAGNITUD"] == mag][cols_h].values.flatten()
    df_mag = df_mag[~np.isnan(df_mag)]
    if len(df_mag) == 0:
        continue
    datos_box.append(df_mag)
    etiquetas.append(f"{nombre}\n({unidad})")
    colores_box.append(COLORES.get(nombre, "#90CAF9"))
 
bp = ax.boxplot(datos_box, patch_artist=True, showfliers=False,
                medianprops={"color": "white", "linewidth": 2.5})
for patch, color in zip(bp["boxes"], colores_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
for whisker in bp["whiskers"]:
    whisker.set_color("#555555")
    whisker.set_linewidth(1.5)
for cap in bp["caps"]:
    cap.set_color("#555555")
    cap.set_linewidth(1.5)
 
ax.set_xticklabels(etiquetas, fontsize=9)
ax.set_ylabel("Concentración")
ax.set_title("")
plt.tight_layout()
ruta4 = CARPETA + "04_boxplot_contaminantes.png"
plt.savefig(ruta4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Guardado: {ruta4}")
 
# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 5: MAPA DE CALOR — COBERTURA DE DATOS POR ESTACIÓN Y AÑO
# ═══════════════════════════════════════════════════════════════════
print("Generando Gráfico 5: Mapa de calor cobertura...")
 
mag_ref = 8  # NO2 — presente en todas las estaciones
df_no2 = df[df["MAGNITUD"] == mag_ref].copy()
 
anyos = sorted(df_no2["anyo"].dropna().unique().astype(int))
estaciones_ord = sorted(df_no2["ESTACION"].unique())
 
matriz = np.zeros((len(estaciones_ord), len(anyos)))
for i, est in enumerate(estaciones_ord):
    for j, anyo in enumerate(anyos):
        sub = df_no2[(df_no2["ESTACION"] == est) & (df_no2["anyo"] == anyo)][cols_h]
        total = sub.size
        validos = sub.notna().sum().sum()
        matriz[i, j] = (validos / total * 100) if total > 0 else 0
 
cmap = LinearSegmentedColormap.from_list("cobertura",
       ["#FFEBEE", "#FFCDD2", "#EF9A9A", "#E57373", "#C62828"])
cmap = plt.cm.RdYlGn
 
fig, ax = plt.subplots(figsize=(10, 11))
im = ax.imshow(matriz, cmap=cmap, aspect="auto", vmin=0, vmax=100)
 
# Etiquetas
nombres_est = [ESTACIONES.get(e, str(e)) for e in estaciones_ord]
ax.set_xticks(range(len(anyos)))
ax.set_xticklabels([str(a) for a in anyos], fontsize=10)
ax.set_yticks(range(len(estaciones_ord)))
ax.set_yticklabels(nombres_est, fontsize=8)
 
# Valores en las celdas
for i in range(len(estaciones_ord)):
    for j in range(len(anyos)):
        val = matriz[i, j]
        color_txt = "white" if val < 40 else "black"
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                fontsize=8, color=color_txt, fontweight="bold")
 
cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label("% datos válidos de NO₂", fontsize=9)
 
ax.set_title("Cobertura de datos válidos de NO₂ por estación y año\n(% sobre el total de mediciones horarias esperadas)",
             fontsize=11, fontweight="bold", pad=15)
ax.set_xlabel("Año", fontsize=10)
 
plt.tight_layout()
ruta5 = CARPETA + "05_mapa_calor_cobertura.png"
plt.savefig(ruta5, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Guardado: {ruta5}")
 
print("\n" + "=" * 55)
print("  TODOS LOS GRÁFICOS GENERADOS")
print("=" * 55)
print(f"  Carpeta: {CARPETA}")
print("  01_mapa_estaciones.png")
print("  02_evolucion_anual.png")
print("  03_perfil_horario.png")
print("  04_boxplot_contaminantes.png")
print("  05_mapa_calor_cobertura.png")
print("\nFIN")
 