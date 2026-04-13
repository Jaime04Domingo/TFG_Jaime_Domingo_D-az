#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:22:49 2026

@author: jaime
"""

"""
MODELO 1 — Regresión Lineal Múltiple (versión corregida)
Cambio: dia_semana → es_fin_semana (variable binaria 0/1)
Justificación: dia_semana es ordinal y la regresión la trata linealmente,
lo que es incorrecto. es_fin_semana captura correctamente
la discontinuidad laboral/fin de semana.
TFG Madrid - Análisis del Dato
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

RUTA    = "/Users/jaime/Documents/Universidad/TFG/Dataset_Diario_Integrado.csv"
CARPETA = "/Users/jaime/Documents/Universidad/TFG/Graficos_Modelo1/"
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
print("  MODELO 1 — REGRESIÓN LINEAL MÚLTIPLE (versión corregida)")
print("  Cambio: dia_semana → es_fin_semana")
print("=" * 65)

# ── CARGA ────────────────────────────────────────────────────────────
df = pd.read_csv(RUTA, sep=";", encoding="utf-8-sig", low_memory=False)
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)

# Convertir es_fin_semana a 0/1 por si es booleano
df["es_fin_semana"] = df["es_fin_semana"].astype(int)

TARGET   = "NO2"
FEATURES = [
    "trafico_medio",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "WS10M",
    "PRECTOTCORR",
    "mes",
    "es_fin_semana",   # ← corregido (antes: dia_semana)
]
NOMBRES = {
    "trafico_medio":  "Tráfico (veh/h)",
    "T2M_MAX":        "Temp. máxima (°C)",
    "T2M_MIN":        "Temp. mínima (°C)",
    "RH2M":           "Humedad relativa (%)",
    "WS10M":          "Velocidad viento (m/s)",
    "PRECTOTCORR":    "Precipitación (mm)",
    "mes":            "Mes del año",
    "es_fin_semana":  "Fin de semana (0/1)",
}

df_model = df[[TARGET] + FEATURES].dropna().copy()
n        = len(df_model)
n_train  = int(n * 0.70)
n_test   = n - n_train

X_all = df_model[FEATURES].values.astype(float)
y_all = df_model[TARGET].values.astype(float)
X_train, X_test = X_all[:n_train], X_all[n_train:]
y_train, y_test = y_all[:n_train], y_all[n_train:]

print(f"\n  Dataset: {len(df_model):,} días")
print(f"  Partición: {n_train} entrenamiento | {n_test} test")

# ── OLS ──────────────────────────────────────────────────────────────
def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]

def estandarizar(arr):
    return (arr - arr.mean()) / arr.std()

X_mat  = np.column_stack([np.ones(n_train), X_train])
X_t_mat = np.column_stack([np.ones(n_test),  X_test])
beta   = ols(X_mat, y_train)

y_pred_train = X_mat   @ beta
y_pred_test  = X_t_mat @ beta

# Coeficientes estandarizados
X_std = np.column_stack([estandarizar(df_model[c].values) for c in FEATURES])
y_std = estandarizar(y_all)
X_std_mat = np.column_stack([np.ones(n_train), X_std[:n_train]])
beta_std  = ols(X_std_mat, y_std[:n_train])

# ── MÉTRICAS ─────────────────────────────────────────────────────────
def r2(yr, yp):
    return float(1 - np.sum((yr-yp)**2) / np.sum((yr-yr.mean())**2))
def rmse(yr, yp):
    return float(np.sqrt(np.mean((yr-yp)**2)))
def mae(yr, yp):
    return float(np.mean(np.abs(yr-yp)))

r2_train   = r2(y_train, y_pred_train)
r2_test    = r2(y_test,  y_pred_test)
rmse_train = rmse(y_train, y_pred_train)
rmse_test  = rmse(y_test,  y_pred_test)
mae_train  = mae(y_train, y_pred_train)
mae_test   = mae(y_test,  y_pred_test)

print("\n" + "-"*55)
print("  MÉTRICAS")
print("-"*55)
print(f"  {'Métrica':<20} {'Train':>10} {'Test':>10}")
print(f"  {'-'*40}")
print(f"  {'R²':<20} {r2_train:>10.4f} {r2_test:>10.4f}")
print(f"  {'RMSE (µg/m³)':<20} {rmse_train:>10.3f} {rmse_test:>10.3f}")
print(f"  {'MAE  (µg/m³)':<20} {mae_train:>10.3f} {mae_test:>10.3f}")

print("\n" + "-"*55)
print("  COEFICIENTES (unidades originales)")
print("-"*55)
print(f"  {'Variable':<25} {'Coef':>12}  Interpretación")
print(f"  {'-'*70}")
print(f"  {'Intercepto':<25} {float(beta[0]):>12.4f}")
for i, col in enumerate(FEATURES):
    c = float(beta[i+1])
    interp = ""
    if col == "trafico_medio":
        interp = f"→ +100 veh/h = {c*100:+.4f} µg/m³"
    elif col == "T2M_MIN":
        interp = f"→ +1°C = {c:+.4f} µg/m³"
    elif col == "WS10M":
        interp = f"→ +1 m/s = {c:+.4f} µg/m³"
    elif col == "es_fin_semana":
        interp = f"→ FdS reduce NO₂ en {abs(c):.4f} µg/m³"
    print(f"  {NOMBRES[col]:<25} {c:>12.4f}  {interp}")

print("\n" + "-"*55)
print("  COEFICIENTES ESTANDARIZADOS (importancia relativa)")
print("-"*55)
for i, col in enumerate(FEATURES):
    b = float(beta_std[i+1])
    barra = "█" * int(abs(b)*30)
    signo = "+" if b > 0 else "-"
    print(f"  {NOMBRES[col]:<25} {b:>10.4f}  {signo}{barra}")

# ═══════════════════════════════════════════════════════════════════
# GRÁFICOS
# ═══════════════════════════════════════════════════════════════════
fechas_test = df["fecha"].iloc[n_train:n_train+n_test].values

# G1: Predicciones vs real
print("\n[G1] Predicciones vs real...")
residuos_test = y_test - y_pred_test

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle("Modelo 1 — Regresión Lineal Múltiple (versión corregida)\n"
             "Predicciones vs valores reales de NO₂ · Conjunto de test",
             fontsize=13, fontweight="bold")

ax1.plot(fechas_test, y_test,      color="#E53935", linewidth=1.2, alpha=0.8, label="NO₂ real")
ax1.plot(fechas_test, y_pred_test, color="#1E88E5", linewidth=1.2, alpha=0.8,
         linestyle="--", label="NO₂ predicho")
ax1.set_ylabel("NO₂ (µg/m³)", fontsize=10)
ax1.set_title(f"R²={r2_test:.3f}  RMSE={rmse_test:.2f} µg/m³  MAE={mae_test:.2f} µg/m³",
              fontsize=10, fontweight="bold")
ax1.legend(fontsize=9)

ax2.bar(fechas_test, residuos_test,
        color=["#E53935" if r > 0 else "#1E88E5" for r in residuos_test],
        alpha=0.5, width=1)
ax2.axhline(0, color="black", linewidth=1)
ax2.axhline( 2*float(np.std(residuos_test)), color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax2.axhline(-2*float(np.std(residuos_test)), color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax2.set_ylabel("Residuo (µg/m³)", fontsize=10)
ax2.set_xlabel("Fecha", fontsize=10)
ax2.set_title("Residuos (NO₂ real − predicho)", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(CARPETA + "M1_01_predicciones_vs_real.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ M1_01")

# G2: Scatter
print("[G2] Scatter...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Modelo 1 — Regresión Lineal Múltiple (versión corregida)\n"
             "Dispersión: valores reales vs predichos", fontsize=13, fontweight="bold")

for ax, yr, yp, titulo, color in zip(
    axes,
    [y_train, y_test], [y_pred_train, y_pred_test],
    ["Entrenamiento", "Test"], ["#1E88E5", "#E53935"]
):
    r2v = r2(yr, yp); rmsev = rmse(yr, yp); maev = mae(yr, yp)
    ax.scatter(yr, yp, alpha=0.3, s=12, color=color)
    lim = [min(yr.min(), yp.min())-2, max(yr.max(), yp.max())+2]
    ax.plot(lim, lim, "k--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("NO₂ real (µg/m³)", fontsize=10)
    ax.set_ylabel("NO₂ predicho (µg/m³)", fontsize=10)
    ax.set_title(f"{titulo}\nR²={r2v:.3f}  RMSE={rmsev:.2f}  MAE={maev:.2f} µg/m³",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(lim); ax.set_ylim(lim)

plt.tight_layout()
plt.savefig(CARPETA + "M1_02_scatter_real_vs_predicho.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ M1_02")

# G3: Coeficientes estandarizados
print("[G3] Coeficientes estandarizados...")
betas_plot  = [float(beta_std[i+1]) for i in range(len(FEATURES))]
nombres_p   = [NOMBRES[c] for c in FEATURES]
orden       = np.argsort(np.abs(betas_plot))[::-1]
nombres_ord = [nombres_p[i] for i in orden]
betas_ord   = [betas_plot[i] for i in orden]
cols_bar    = ["#E53935" if b > 0 else "#1E88E5" for b in betas_ord]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(nombres_ord, betas_ord, color=cols_bar, alpha=0.8, edgecolor="white")
ax.axvline(0, color="black", linewidth=1)
for bar, val in zip(bars, betas_ord):
    xp = float(bar.get_width()) + (0.01 if val >= 0 else -0.01)
    ha = "left" if val >= 0 else "right"
    ax.text(xp, float(bar.get_y())+float(bar.get_height())/2,
            f"{val:+.3f}", va="center", ha=ha, fontsize=9, fontweight="bold")
ax.set_xlabel("Coeficiente estandarizado (β)", fontsize=11)
ax.set_title("Importancia relativa de variables — Coeficientes estandarizados\n"
             "Rojo = aumenta NO₂ · Azul = reduce NO₂",
             fontsize=12, fontweight="bold")
leg = [mpatches.Patch(color="#E53935", label="Efecto positivo"),
       mpatches.Patch(color="#1E88E5", label="Efecto negativo")]
ax.legend(handles=leg, fontsize=9)

plt.tight_layout()
plt.savefig(CARPETA + "M1_03_coeficientes_estandarizados.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ M1_03")

# G4: Residuos
print("[G4] Residuos...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Modelo 1 — Análisis de residuos (conjunto de test, versión corregida)",
             fontsize=13, fontweight="bold")

axes[0].hist(residuos_test, bins=40, color="#1E88E5", alpha=0.8, edgecolor="white")
axes[0].axvline(0, color="red", linewidth=2, linestyle="--")
axes[0].axvline(float(np.mean(residuos_test)), color="orange", linewidth=1.5,
                label=f"Media: {float(np.mean(residuos_test)):.2f}")
axes[0].set_xlabel("Residuo (µg/m³)", fontsize=10)
axes[0].set_ylabel("Frecuencia", fontsize=10)
axes[0].set_title("Distribución de residuos", fontsize=10, fontweight="bold")
axes[0].legend(fontsize=8)

axes[1].scatter(y_pred_test, residuos_test, alpha=0.3, s=12, color="#E53935")
axes[1].axhline(0, color="black", linewidth=1)
axes[1].axhline( 2*float(np.std(residuos_test)), color="gray", linestyle="--", linewidth=1)
axes[1].axhline(-2*float(np.std(residuos_test)), color="gray", linestyle="--", linewidth=1)
axes[1].set_xlabel("NO₂ predicho (µg/m³)", fontsize=10)
axes[1].set_ylabel("Residuo (µg/m³)", fontsize=10)
axes[1].set_title("Residuos vs predichos", fontsize=10, fontweight="bold")

res_sorted = np.sort(residuos_test)
n_res = len(res_sorted)
teoricos = np.array([float(np.percentile(np.random.randn(10000), 100*(i+0.5)/n_res))
                     for i in range(n_res)])
axes[2].scatter(teoricos, res_sorted, alpha=0.3, s=10, color="#43A047")
lq = max(abs(teoricos.min()), abs(teoricos.max()))
lr = max(abs(res_sorted.min()), abs(res_sorted.max()))
axes[2].plot([-lq, lq], [-lq*lr/lq, lq*lr/lq], "k--", linewidth=1.5, alpha=0.7)
axes[2].set_xlabel("Cuantiles teóricos", fontsize=10)
axes[2].set_ylabel("Cuantiles residuos", fontsize=10)
axes[2].set_title("Q-Q Plot", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(CARPETA + "M1_04_analisis_residuos.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ M1_04")

# G5: Efecto tráfico
print("[G5] Efecto tráfico...")
coef_traf   = float(beta[FEATURES.index("trafico_medio") + 1])
intercepto  = float(beta[0])
medias_resto = np.array([df_model[c].mean() for c in FEATURES])
traf_range  = np.linspace(float(df_model["trafico_medio"].min()),
                           float(df_model["trafico_medio"].max()), 200)
no2_pred_t  = []
for t in traf_range:
    x = medias_resto.copy(); x[0] = t
    no2_pred_t.append(intercepto + float(np.dot(x, beta[1:])))

fig, ax = plt.subplots(figsize=(11, 6))
ax.scatter(df_model["trafico_medio"], df_model["NO2"],
           alpha=0.15, s=8, color="#E53935", label="Datos reales")
ax.plot(traf_range, no2_pred_t, color="#0D47A1", linewidth=2.5,
        label=f"Efecto tráfico (modelo)\nβ = {coef_traf:+.4f} µg/m³ por veh/h")
ax.annotate(f"+100 veh/h → {coef_traf*100:+.2f} µg/m³ NO₂",
            xy=(1300, float(np.interp(1300, traf_range, no2_pred_t))),
            xytext=(1050, 50), fontsize=10, fontweight="bold", color="#0D47A1",
            arrowprops=dict(arrowstyle="->", color="#0D47A1", lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))
ax.set_xlabel("Tráfico medio diario (veh/hora)", fontsize=11)
ax.set_ylabel("NO₂ (µg/m³)", fontsize=11)
ax.set_title("Efecto del tráfico sobre el NO₂\n"
             "(manteniendo el resto de variables en su media)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(CARPETA + "M1_05_efecto_trafico.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ M1_05")

# ── RESUMEN ───────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  RESUMEN FINAL — MODELO 1 (versión corregida)")
print("="*65)
print(f"\n  R²   train : {r2_train:.4f}  |  test : {r2_test:.4f}")
print(f"  RMSE train : {rmse_train:.3f} µg/m³  |  test : {rmse_test:.3f} µg/m³")
print(f"  MAE  train : {mae_train:.3f} µg/m³  |  test : {mae_test:.3f} µg/m³")
print(f"\n  Coeficiente tráfico  : {coef_traf:+.6f} µg/m³ por veh/h")
print(f"  +100 veh/h adicionales → {coef_traf*100:+.4f} µg/m³")
coef_fds = float(beta[FEATURES.index("es_fin_semana")+1])
print(f"\n  Coeficiente FdS      : {coef_fds:+.4f} µg/m³")
print(f"  → Ser fin de semana reduce el NO₂ en {abs(coef_fds):.4f} µg/m³")
print(f"    controlando por tráfico y clima")
print(f"\n  Gráficos en: {CARPETA}")
print("\nFIN — Pega el output en el chat.")