#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:51:15 2026

@author: jaime
"""

### LIMPIEZA GRANDE CONTAMINACIÓN CON VISUALIZACIÓN

"""
Limpieza final del dataset de Contaminación - TFG
1. Gráficos que justifican la eliminación de magnitudes 42, 43, 44
2. Sustituir valores negativos por NaN
3. Eliminar magnitudes irrelevantes (42, 43, 44)
4. Validar y guardar CSV definitivo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

RUTA_ENTRADA = "/Users/jaime/Documents/Universidad/TFG/Contaminacion_Final.csv"
RUTA_SALIDA  = "/Users/jaime/Documents/Universidad/TFG/Contaminacion_Definitivo.csv"
CARPETA_GRAF = "/Users/jaime/Documents/Universidad/TFG/Graficos_Contaminacion/"
os.makedirs(CARPETA_GRAF, exist_ok=True)

NOMBRES_MAG = {
    1: "SO₂", 6: "CO", 7: "NO", 8: "NO₂", 9: "PM2.5",
    10: "PM10", 12: "NOₓ", 14: "O₃", 20: "TOL", 30: "BEN",
    35: "EBE", 42: "TCH\n(Hidrocarburos totales)",
    43: "CH₄\n(Metano)", 44: "NMHC\n(Hidroc. no metánicos)"
}
MAGS_ELIMINAR = [42, 43, 44]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

print("=" * 65)
print("  LIMPIEZA FINAL - CONTAMINACIÓN")
print("=" * 65)

df = pd.read_csv(RUTA_ENTRADA, sep=";", encoding="utf-8-sig", low_memory=False)
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
df["anyo"]  = df["fecha"].dt.year
df["mes"]   = df["fecha"].dt.month
cols_h = sorted([c for c in df.columns if c.startswith("H") and c[1:].isdigit()],
                key=lambda x: int(x[1:]))

print(f"\nDataset: {len(df):,} filas | Magnitudes: {sorted(df['MAGNITUD'].unique())}")

# ═══════════════════════════════════════════════════════════════════
# GRÁFICOS DE JUSTIFICACIÓN (antes de eliminar nada)
# ═══════════════════════════════════════════════════════════════════
print("\nGenerando gráficos de justificación...")

magnitudes_todas = sorted(df["MAGNITUD"].unique())

# ── MÉTRICAS POR MAGNITUD ────────────────────────────────────────────
resumen = []
for mag in magnitudes_todas:
    df_mag = df[df["MAGNITUD"] == mag][cols_h]
    total  = df_mag.size
    validos = df_mag.notna().sum().sum()
    n_filas = (df["MAGNITUD"] == mag).sum()
    pct = validos / total * 100 if total > 0 else 0
    resumen.append({
        "mag": mag,
        "nombre": NOMBRES_MAG.get(mag, str(mag)),
        "filas": n_filas,
        "total_mediciones": total,
        "validas": validos,
        "pct_valido": pct,
        "eliminar": mag in MAGS_ELIMINAR
    })

res = pd.DataFrame(resumen)

# ── FIGURA: 3 PANELES ────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.suptitle("Justificación de la eliminación de magnitudes TCH, CH₄ y NMHC\n"
             "Red de Calidad del Aire — Ayuntamiento de Madrid · 2021–2025",
             fontsize=13, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

colores = ["#E53935" if e else "#1E88E5" for e in res["eliminar"]]
nombres = res["nombre"].tolist()

# ── Panel 1: % de datos válidos por magnitud ─────────────────────────
ax1 = fig.add_subplot(gs[0, :])
bars = ax1.barh(nombres, res["pct_valido"], color=colores, edgecolor="white", height=0.6)

# Etiquetas de valor
for bar, pct in zip(bars, res["pct_valido"]):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f"{pct:.1f}%", va="center", fontsize=9,
             color="black", fontweight="bold")

ax1.axvline(50, color="orange", linestyle="--", linewidth=1.5,
            alpha=0.8, label="Umbral 50%")
ax1.set_xlabel("% de mediciones horarias con dato válido", fontsize=10)
ax1.set_title("Porcentaje de datos válidos por magnitud\n"
              "(barras rojas = magnitudes eliminadas)", fontsize=11, fontweight="bold")
ax1.set_xlim(0, 108)
ax1.legend(fontsize=9)

# Leyenda manual
from matplotlib.patches import Patch
leyenda = [Patch(facecolor="#1E88E5", label="Magnitudes mantenidas"),
           Patch(facecolor="#E53935", label="Magnitudes eliminadas (42, 43, 44)")]
ax1.legend(handles=leyenda, fontsize=9, loc="lower right")

# ── Panel 2: Número de filas por magnitud ────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
bars2 = ax2.barh(nombres, res["filas"], color=colores, edgecolor="white", height=0.6)
for bar, val in zip(bars2, res["filas"]):
    ax2.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
             f"{val:,}", va="center", fontsize=8)
ax2.set_xlabel("Número de filas en el dataset", fontsize=10)
ax2.set_title("Filas por magnitud\n(una fila = una estación × un día)", fontsize=11, fontweight="bold")

# ── Panel 3: Evolución temporal de datos válidos (42, 43, 44 vs NO2) ─
ax3 = fig.add_subplot(gs[1, 1])

# NO2 como referencia
mag_ref = 8
colores_evol = {42: "#E53935", 43: "#FF7043", 44: "#D81B60", 8: "#1E88E5"}
estilos = {42: "-o", 43: "-s", 44: "-^", 8: "-D"}

for mag in [8, 42, 43, 44]:
    df_mag = df[df["MAGNITUD"] == mag]
    if df_mag.empty:
        continue
    datos_anyo = []
    for anyo in sorted(df["anyo"].dropna().unique()):
        sub = df_mag[df_mag["anyo"] == anyo][cols_h]
        total = sub.size
        validos = sub.notna().sum().sum()
        pct = validos / total * 100 if total > 0 else 0
        datos_anyo.append({"anyo": int(anyo), "pct": pct})
    da = pd.DataFrame(datos_anyo)
    nombre_corto = {42: "TCH (42)", 43: "CH₄ (43)", 44: "NMHC (44)", 8: "NO₂ (ref.)"}[mag]
    ax3.plot(da["anyo"], da["pct"], estilos[mag],
             color=colores_evol[mag], linewidth=2, markersize=7,
             label=nombre_corto)

ax3.axhline(50, color="orange", linestyle="--", linewidth=1.5, alpha=0.8, label="Umbral 50%")
ax3.set_xlabel("Año", fontsize=10)
ax3.set_ylabel("% datos válidos", fontsize=10)
ax3.set_title("Evolución temporal del % de datos válidos\n(42, 43, 44 vs NO₂ como referencia)", fontsize=11, fontweight="bold")
ax3.legend(fontsize=8)
ax3.set_xticks(sorted(df["anyo"].dropna().unique().astype(int)))
ax3.set_ylim(0, 105)

ruta_graf = CARPETA_GRAF + "06_justificacion_eliminacion_magnitudes.png"
plt.savefig(ruta_graf, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Gráfico guardado: {ruta_graf}")

# ── FIGURA 2: MAPA DE CALOR — cobertura por estación para mags 42,43,44
print("  Generando mapa de calor de cobertura...")

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 9))
fig2.suptitle("Cobertura de datos válidos por estación y año\n"
              "Magnitudes eliminadas: TCH (42), CH₄ (43), NMHC (44)",
              fontsize=12, fontweight="bold")

estaciones_ids = sorted(df["ESTACION"].unique())
NOMBRES_EST = {
    4:"Pza. España", 8:"Escuelas Aguirre", 11:"Av. Ramón y Cajal",
    16:"Arturo Soria", 17:"Villaverde Alto", 18:"C/ Farolillo",
    24:"Casa de Campo", 27:"Barajas", 35:"Pza. del Carmen",
    36:"Moratalaz", 38:"Cuatro Caminos", 39:"Barrio del Pilar",
    40:"Vallecas", 47:"Méndez Álvaro", 48:"Pº. Castellana",
    49:"Retiro", 50:"Pza. Castilla", 54:"Ensanche Vallecas",
    55:"Urb. Embajada", 56:"Plaza Elíptica", 57:"Sanchinarro",
    58:"El Pardo", 59:"Pque. Juan Carlos I", 60:"Tres Olivos"
}
nombres_est = [NOMBRES_EST.get(e, str(e)) for e in estaciones_ids]
anyos = sorted(df["anyo"].dropna().unique().astype(int))

for idx, mag in enumerate([42, 43, 44]):
    ax = axes2[idx]
    df_mag = df[df["MAGNITUD"] == mag]
    nombre_mag = {42: "TCH (42) — Hidrocarburos totales",
                  43: "CH₄ (43) — Metano",
                  44: "NMHC (44) — Hidroc. no metánicos"}[mag]

    matriz = np.zeros((len(estaciones_ids), len(anyos)))
    for i, est in enumerate(estaciones_ids):
        for j, anyo in enumerate(anyos):
            sub = df_mag[(df_mag["ESTACION"] == est) & (df_mag["anyo"] == anyo)][cols_h]
            if sub.size > 0:
                matriz[i, j] = sub.notna().sum().sum() / sub.size * 100

    im = ax.imshow(matriz, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(anyos)))
    ax.set_xticklabels([str(a) for a in anyos], fontsize=9)
    ax.set_yticks(range(len(estaciones_ids)))
    ax.set_yticklabels(nombres_est, fontsize=7)

    for i in range(len(estaciones_ids)):
        for j in range(len(anyos)):
            val = matriz[i, j]
            txt = f"{val:.0f}%" if val > 0 else "–"
            color_txt = "white" if val < 40 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=6.5, color=color_txt, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.5, label="% válidos")
    ax.set_title(nombre_mag, fontsize=9, fontweight="bold", pad=10)
    ax.set_xlabel("Año", fontsize=9)

plt.tight_layout()
ruta_graf2 = CARPETA_GRAF + "07_cobertura_mags_eliminadas.png"
plt.savefig(ruta_graf2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Gráfico guardado: {ruta_graf2}")

# ═══════════════════════════════════════════════════════════════════
# LIMPIEZA
# ═══════════════════════════════════════════════════════════════════

# ── 1. VALORES NEGATIVOS → NaN ───────────────────────────────────────
print("\n[1] Sustituyendo valores negativos por NaN...")
total_negativos = 0
for col in cols_h:
    n = (df[col] < 0).sum()
    if n > 0:
        total_negativos += n
        df.loc[df[col] < 0, col] = np.nan
print(f"  Total negativos → NaN: {total_negativos:,}")

# ── 2. ELIMINAR MAGNITUDES 42, 43, 44 ───────────────────────────────
print("\n[2] Eliminando magnitudes 42, 43, 44...")
antes = len(df)
df = df[~df["MAGNITUD"].isin(MAGS_ELIMINAR)].copy()
eliminadas = antes - len(df)
print(f"  Filas eliminadas: {eliminadas:,} | Restantes: {len(df):,}")
print(f"  Magnitudes finales: {sorted(df['MAGNITUD'].unique())}")

# ── 3. VALIDACIÓN ────────────────────────────────────────────────────
print("\n[3] Validación...")
n_neg = sum((df[col] < 0).sum() for col in cols_h)
dups  = df.duplicated().sum()
print(f"  Negativos restantes: {n_neg} {'✓' if n_neg == 0 else '✗'}")
print(f"  Duplicados         : {dups} {'✓' if dups == 0 else '✗'}")
print(f"  Mags 42/43/44      : {'no presentes ✓' if not set(MAGS_ELIMINAR) & set(df['MAGNITUD'].unique()) else '✗'}")

# ── 4. GUARDAR ───────────────────────────────────────────────────────
df.drop(columns=["anyo","mes"], inplace=True, errors="ignore")
df.to_csv(RUTA_SALIDA, sep=";", index=False, encoding="utf-8-sig")
tam = os.path.getsize(RUTA_SALIDA) / (1024**2)

print("\n" + "=" * 65)
print("  RESUMEN FINAL")
print("=" * 65)
print(f"  Filas             : {len(df):,}")
print(f"  Magnitudes        : {sorted(df['MAGNITUD'].unique())}")
print(f"  Negativos → NaN   : {total_negativos:,}")
print(f"  Filas eliminadas  : {eliminadas:,} (mags 42, 43, 44)")
print(f"  Archivo guardado  : {RUTA_SALIDA} ({tam:.1f} MB)")
print(f"\n  Gráficos:")
print(f"    06_justificacion_eliminacion_magnitudes.png")
print(f"    07_cobertura_mags_eliminadas.png")
print(f"\n  ✅ LIMPIEZA COMPLETADA")
print("\nFIN — Pega el output en el chat.")