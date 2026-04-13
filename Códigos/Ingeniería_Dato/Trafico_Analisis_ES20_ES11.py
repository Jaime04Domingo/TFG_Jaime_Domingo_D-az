#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:08:09 2026

@author: jaime
"""

"""
Análisis horario de ES20 y ES11
¿Los valores extremos aparecen en horas punta o son aleatorios?
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
 
RUTA    = "/Users/jaime/Documents/Universidad/TFG/Trafico_Aforos_Definitivo.csv"
CARPETA = "/Users/jaime/Documents/Universidad/TFG/"
 
print("=" * 65)
print("  ANÁLISIS HORARIO ES20 y ES11 - M-30")
print("=" * 65)
 
df = pd.read_csv(RUTA, sep=";", encoding="utf-8-sig", low_memory=False)
df["FDIA"] = pd.to_datetime(df["FDIA"], errors="coerce")
cols_hor = [c for c in df.columns if c.startswith("HOR")]
 
# ── RECONSTRUIR PERFIL HORARIO COMPLETO ──────────────────────────────
# FSEN con "-" → HOR1-HOR12 = horas 1 a 12
# FSEN con "=" → HOR1-HOR12 = horas 13 a 24
# Renombramos cada columna a su hora real
 
def reconstruir_horario(df_est):
    """Convierte el formato de 2 bloques a 24 columnas horarias reales."""
    bloques = []
    for fsen, offset in [("-", 0), ("=", 12)]:
        bloque = df_est[df_est["FSEN"].str.endswith(fsen)].copy()
        if bloque.empty:
            continue
        rename = {f"HOR{i}": f"H{i+offset:02d}" for i in range(1, 13)}
        bloque = bloque.rename(columns=rename)
        cols_h = [f"H{i+offset:02d}" for i in range(1, 13)]
        bloques.append(bloque[["FDIA", "FEST", "FSEN"] + cols_h])
    return bloques
 
# ── ANÁLISIS POR ESTACIÓN ────────────────────────────────────────────
estaciones = {
    "ES20": "Av. del Manzanares (M-30) — ES20",
    "ES11": "Av. del Manzanares (M-30) — ES11"
}
 
fig = plt.figure(figsize=(18, 20))
fig.suptitle("Análisis horario de estaciones M-30\n¿Los valores extremos ocurren en horas punta?",
             fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)
 
for idx, (est, titulo) in enumerate(estaciones.items()):
    df_est = df[df["FEST"] == est].copy()
    print(f"\n{'='*55}")
    print(f"  {titulo}")
    print(f"  Filas totales: {len(df_est):,}")
    print(f"  Sentidos: {sorted(df_est['FSEN'].unique())}")
 
    # Perfil horario: media por hora real del día
    # Unimos los dos bloques horarios
    datos_hora = {}
    for fsen in df_est["FSEN"].unique():
        offset = 12 if "=" in fsen else 0
        bloque = df_est[df_est["FSEN"] == fsen]
        for i, col in enumerate(cols_hor, 1):
            hora_real = i + offset
            vals = bloque[col].dropna()
            if hora_real not in datos_hora:
                datos_hora[hora_real] = []
            datos_hora[hora_real].extend(vals.tolist())
 
    horas = sorted(datos_hora.keys())
    media_hora    = [np.mean(datos_hora[h]) for h in horas]
    mediana_hora  = [np.median(datos_hora[h]) for h in horas]
    p95_hora      = [np.percentile(datos_hora[h], 95) for h in horas]
    p99_hora      = [np.percentile(datos_hora[h], 99) for h in horas]
    max_hora      = [np.max(datos_hora[h]) for h in horas]
 
    print(f"\n  Perfil horario (media de todos los sentidos y días):")
    print(f"  {'Hora':>5} {'Media':>8} {'Mediana':>8} {'P95':>8} {'P99':>8} {'Max':>8}")
    print(f"  {'-'*47}")
    for h, med, mdn, p95, p99, mx in zip(horas, media_hora, mediana_hora, p95_hora, p99_hora, max_hora):
        flag = " ← PUNTA" if h in [8, 9, 18, 19, 20] else ""
        print(f"  {h:>5}h {med:>8.0f} {mdn:>8.0f} {p95:>8.0f} {p99:>8.0f} {mx:>8.0f}{flag}")
 
    # ── Gráfico 1: Perfil horario completo ──────────────────────────
    ax1 = fig.add_subplot(gs[idx*2, :])
    ax1.fill_between(horas, mediana_hora, p95_hora, alpha=0.2, color="#2196F3", label="Mediana–P95")
    ax1.fill_between(horas, p95_hora, p99_hora, alpha=0.2, color="#FF9800", label="P95–P99")
    ax1.plot(horas, media_hora,   color="#2196F3", linewidth=2, label="Media")
    ax1.plot(horas, mediana_hora, color="#4CAF50", linewidth=2, linestyle="--", label="Mediana")
    ax1.plot(horas, max_hora,     color="red", linewidth=1, linestyle=":", label="Máximo")
    ax1.axvspan(7.5, 9.5, alpha=0.1, color="red", label="Horas punta mañana")
    ax1.axvspan(17.5, 20, alpha=0.1, color="orange", label="Horas punta tarde")
    ax1.axhline(5000, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Umbral 5.000")
    ax1.axhline(8000, color="darkred", linestyle="--", linewidth=1, alpha=0.7, label="Umbral 8.000")
    ax1.set_xlabel("Hora del día")
    ax1.set_ylabel("Vehículos/hora")
    ax1.set_title(f"{titulo}\nPerfil horario medio (todos los días y sentidos)")
    ax1.set_xticks(range(1, 25))
    ax1.legend(fontsize=7, ncol=4)
    ax1.grid(True, alpha=0.3)
 
    # ── Gráfico 2: Distribución de valores por hora (boxplot) ────────
    ax2 = fig.add_subplot(gs[idx*2+1, :])
    datos_box = [datos_hora[h] for h in horas]
    bp = ax2.boxplot(datos_box, positions=horas, widths=0.6,
                     patch_artist=True, showfliers=False,
                     medianprops={"color": "red", "linewidth": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor("#90CAF9")
        patch.set_alpha(0.7)
    ax2.axvspan(7.5, 9.5, alpha=0.1, color="red")
    ax2.axvspan(17.5, 20, alpha=0.1, color="orange")
    ax2.axhline(5000, color="red", linestyle="--", linewidth=1, alpha=0.7, label="5.000")
    ax2.axhline(8000, color="darkred", linestyle="--", linewidth=1, alpha=0.7, label="8.000")
    ax2.set_xlabel("Hora del día")
    ax2.set_ylabel("Vehículos/hora")
    ax2.set_title(f"{titulo}\nDistribución por hora (sin outliers extremos)")
    ax2.set_xticks(range(1, 25))
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
 
ruta_graf = os.path.join(CARPETA, "perfil_horario_ES20_ES11.png")
plt.savefig(ruta_graf, dpi=150, bbox_inches="tight")
print(f"\n✓ Gráfico guardado en: {ruta_graf}")
plt.show()
 
print("\nFIN — Pega el output en el chat.")
 