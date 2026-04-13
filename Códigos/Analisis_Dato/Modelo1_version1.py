#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:23:07 2026

@author: jaime
"""

"""
MODELO 1 — Regresión Lineal Múltiple
Variable dependiente: NO2
Variables independientes: tráfico + clima + variables temporales
TFG Madrid - Análisis del Dato
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
print("  MODELO 1 — REGRESIÓN LINEAL MÚLTIPLE")
print("=" * 65)

# ── CARGA ────────────────────────────────────────────────────────────
df = pd.read_csv(RUTA, sep=";", encoding="utf-8-sig", low_memory=False)
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)
print(f"\n  Dataset: {len(df):,} días")

# ── VARIABLES ────────────────────────────────────────────────────────
# Variable dependiente
TARGET = "NO2"

# Variables independientes
FEATURES = [
    "trafico_medio",   # tráfico — variable de interés principal
    "T2M_MAX",         # temperatura máxima
    "T2M_MIN",         # temperatura mínima
    "RH2M",            # humedad relativa
    "WS10M",           # velocidad del viento
    "PRECTOTCORR",     # precipitación
    "mes",             # mes del año (1-12) — captura estacionalidad
    "dia_semana",      # día de la semana (0=lun, 6=dom)
]

NOMBRES = {
    "trafico_medio":  "Tráfico (veh/h)",
    "T2M_MAX":        "Temp. máxima (°C)",
    "T2M_MIN":        "Temp. mínima (°C)",
    "RH2M":           "Humedad relativa (%)",
    "WS10M":          "Velocidad viento (m/s)",
    "PRECTOTCORR":    "Precipitación (mm)",
    "mes":            "Mes del año",
    "dia_semana":     "Día de la semana",
}

# Eliminar filas con NaN en las variables del modelo
df_model = df[[TARGET] + FEATURES].dropna().copy()
print(f"  Filas para el modelo (sin NaN): {len(df_model):,}")

# ─────────────────────────────────────────────────────────────────────
# IMPLEMENTACIÓN DE REGRESIÓN LINEAL MÚLTIPLE (OLS, sin scikit-learn)
# Usando solo numpy — mínimos cuadrados ordinarios
# β = (X'X)^-1 X'y
# ─────────────────────────────────────────────────────────────────────

# ── NORMALIZACIÓN (estandarización z-score para comparar coeficientes)
def estandarizar(arr):
    return (arr - arr.mean()) / arr.std()

# Guardar medias y desviaciones para interpretar después
stats_features = {}
for col in FEATURES:
    stats_features[col] = {"mean": float(df_model[col].mean()), "std": float(df_model[col].std())}
stats_target = {"mean": float(df_model[TARGET].mean()), "std": float(df_model[TARGET].std())}

# ── PREPARAR MATRICES ─────────────────────────────────────────────────
y = df_model[TARGET].values
X_raw = df_model[FEATURES].values

# Estandarizar X e y para los coeficientes estandarizados (beta)
X_std = np.column_stack([estandarizar(df_model[col].values) for col in FEATURES])
y_std = estandarizar(y)

# Sin estandarizar (para coeficientes en unidades originales)
X = np.column_stack([np.ones(len(X_raw)), X_raw])   # añadir columna de 1s para intercepto

# ── PARTICIÓN TRAIN / TEST (70% / 30%) ───────────────────────────────
# Temporal: los primeros 70% de días para entrenar, los últimos 30% para test
n = len(df_model)
n_train = int(n * 0.70)
n_test  = n - n_train

X_train, X_test   = X[:n_train], X[n_train:]
y_train, y_test   = y[:n_train], y[n_train:]
X_std_train, X_std_test = X_std[:n_train], X_std[n_train:]
y_std_train, y_std_test = y_std[:n_train], y_std[n_train:]

print(f"  Partición: {n_train} días entrenamiento | {n_test} días test")
print(f"  Entrenamiento: {df_model['fecha'].iloc[0].date() if 'fecha' in df_model else ''} → ...")
print(f"  Test: últimos {n_test} días (≈ {n_test/365:.1f} años)")

# ── OLS EN UNIDADES ORIGINALES ────────────────────────────────────────
def ols(X, y):
    """Regresión OLS: β = (X'X)^-1 X'y"""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta

beta_orig = ols(X_train, y_train)

# Predicciones
y_pred_train = X_train @ beta_orig
y_pred_test  = X_test  @ beta_orig

# ── OLS ESTANDARIZADO (para coeficientes beta comparables) ───────────
X_std_with_intercept_train = np.column_stack([np.ones(n_train), X_std_train])
X_std_with_intercept_test  = np.column_stack([np.ones(n_test),  X_std_test])
beta_std = ols(X_std_with_intercept_train, y_std_train)

# ── MÉTRICAS ──────────────────────────────────────────────────────────
def r2(y_real, y_pred):
    ss_res = np.sum((y_real - y_pred)**2)
    ss_tot = np.sum((y_real - np.mean(y_real))**2)
    return float(1 - ss_res / ss_tot)

def rmse(y_real, y_pred):
    return float(np.sqrt(np.mean((y_real - y_pred)**2)))

def mae(y_real, y_pred):
    return float(np.mean(np.abs(y_real - y_pred)))

r2_train  = r2(y_train, y_pred_train)
r2_test   = r2(y_test,  y_pred_test)
rmse_train = rmse(y_train, y_pred_train)
rmse_test  = rmse(y_test,  y_pred_test)
mae_train  = mae(y_train, y_pred_train)
mae_test   = mae(y_test,  y_pred_test)

print("\n" + "-" * 55)
print("  MÉTRICAS DEL MODELO")
print("-" * 55)
print(f"  {'Métrica':<20} {'Train':>10} {'Test':>10}")
print(f"  {'-'*40}")
print(f"  {'R²':<20} {r2_train:>10.4f} {r2_test:>10.4f}")
print(f"  {'RMSE (µg/m³)':<20} {rmse_train:>10.3f} {rmse_test:>10.3f}")
print(f"  {'MAE  (µg/m³)':<20} {mae_train:>10.3f} {mae_test:>10.3f}")

print("\n" + "-" * 55)
print("  COEFICIENTES (unidades originales)")
print("-" * 55)
print(f"  {'Variable':<25} {'Coeficiente':>14} {'Interpretación'}")
print(f"  {'-'*65}")
print(f"  {'Intercepto':<25} {beta_orig[0]:>14.4f}")
for i, col in enumerate(FEATURES):
    coef = float(beta_orig[i+1])
    nombre = NOMBRES[col]
    if col == "trafico_medio":
        interp = f"→ +1 veh/h = {coef:+.4f} µg/m³ NO₂"
    elif col == "T2M_MAX":
        interp = f"→ +1°C = {coef:+.4f} µg/m³"
    elif col == "WS10M":
        interp = f"→ +1 m/s = {coef:+.4f} µg/m³"
    elif col == "RH2M":
        interp = f"→ +1% humedad = {coef:+.4f} µg/m³"
    elif col == "PRECTOTCORR":
        interp = f"→ +1 mm lluvia = {coef:+.4f} µg/m³"
    else:
        interp = ""
    print(f"  {nombre:<25} {coef:>14.4f}  {interp}")

print("\n" + "-" * 55)
print("  COEFICIENTES ESTANDARIZADOS (beta — importancia relativa)")
print("-" * 55)
print(f"  {'Variable':<25} {'Beta':>10} {'Importancia'}")
print(f"  {'-'*50}")
for i, col in enumerate(FEATURES):
    beta_s = float(beta_std[i+1])
    barra = "█" * int(abs(beta_s) * 30)
    signo = "+" if beta_s > 0 else "-"
    print(f"  {NOMBRES[col]:<25} {beta_s:>10.4f}  {signo}{barra}")

# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 1: PREDICCIONES vs VALORES REALES
# ═══════════════════════════════════════════════════════════════════
print("\n[G1] Predicciones vs valores reales...")

# Reconstruir fechas del test
fechas_test = df["fecha"].iloc[n_train:n_train+n_test].values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle("Modelo 1 — Regresión Lineal Múltiple\n"
             "Predicciones vs valores reales de NO₂ · Conjunto de test",
             fontsize=13, fontweight="bold")

ax1.plot(fechas_test, y_test,      color="#E53935", linewidth=1.2, alpha=0.8, label="NO₂ real")
ax1.plot(fechas_test, y_pred_test, color="#1E88E5", linewidth=1.2, alpha=0.8, linestyle="--", label="NO₂ predicho")
ax1.set_ylabel("NO₂ (µg/m³)", fontsize=10)
ax1.set_title(f"Serie temporal: real vs predicho  |  R²={r2_test:.3f}  RMSE={rmse_test:.2f} µg/m³  MAE={mae_test:.2f} µg/m³",
              fontsize=10, fontweight="bold")
ax1.legend(fontsize=9)

# Residuos
residuos_test = y_test - y_pred_test
ax2.bar(fechas_test, residuos_test, color=["#E53935" if r > 0 else "#1E88E5" for r in residuos_test],
        alpha=0.5, width=1)
ax2.axhline(0, color="black", linewidth=1)
ax2.axhline( 2*rmse_test, color="gray", linestyle="--", linewidth=1, alpha=0.7, label=f"+2σ ({2*rmse_test:.1f})")
ax2.axhline(-2*rmse_test, color="gray", linestyle="--", linewidth=1, alpha=0.7, label=f"-2σ ({-2*rmse_test:.1f})")
ax2.set_ylabel("Residuo (µg/m³)", fontsize=10)
ax2.set_xlabel("Fecha", fontsize=10)
ax2.set_title("Residuos (NO₂ real − predicho)  |  Rojo=subestimación · Azul=sobreestimación",
              fontsize=10, fontweight="bold")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig(CARPETA + "M1_01_predicciones_vs_real.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ M1_01_predicciones_vs_real.png")

# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 2: SCATTER REAL vs PREDICHO
# ═══════════════════════════════════════════════════════════════════
print("[G2] Scatter real vs predicho...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Modelo 1 — Regresión Lineal Múltiple\n"
             "Dispersión: valores reales vs predichos",
             fontsize=13, fontweight="bold")

for ax, y_r, y_p, titulo, n_pts in zip(
    axes,
    [y_train, y_test],
    [y_pred_train, y_pred_test],
    ["Entrenamiento", "Test"],
    [n_train, n_test]
):
    r2v = r2(y_r, y_p)
    rmsev = rmse(y_r, y_p)
    maev  = mae(y_r, y_p)
    color = "#1E88E5" if titulo == "Entrenamiento" else "#E53935"
    ax.scatter(y_r, y_p, alpha=0.3, s=12, color=color)
    lim = [min(y_r.min(), y_p.min()) - 2, max(y_r.max(), y_p.max()) + 2]
    ax.plot(lim, lim, "k--", linewidth=1.5, alpha=0.7, label="Predicción perfecta")
    ax.set_xlabel("NO₂ real (µg/m³)", fontsize=10)
    ax.set_ylabel("NO₂ predicho (µg/m³)", fontsize=10)
    ax.set_title(f"{titulo} (n={n_pts})\nR²={r2v:.3f}  RMSE={rmsev:.2f}  MAE={maev:.2f} µg/m³",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(CARPETA + "M1_02_scatter_real_vs_predicho.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ M1_02_scatter_real_vs_predicho.png")

# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 3: COEFICIENTES ESTANDARIZADOS (importancia variables)
# ═══════════════════════════════════════════════════════════════════
print("[G3] Coeficientes estandarizados...")

nombres_plot  = [NOMBRES[col] for col in FEATURES]
betas_plot    = [float(beta_std[i+1]) for i in range(len(FEATURES))]
orden         = np.argsort(np.abs(betas_plot))[::-1]
nombres_ord   = [nombres_plot[i] for i in orden]
betas_ord     = [betas_plot[i] for i in orden]
colores_bar   = ["#E53935" if b > 0 else "#1E88E5" for b in betas_ord]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(nombres_ord, betas_ord, color=colores_bar, alpha=0.8, edgecolor="white")
ax.axvline(0, color="black", linewidth=1)

for bar, val in zip(bars, betas_ord):
    x_pos = float(bar.get_width()) + (0.01 if val >= 0 else -0.01)
    ha = "left" if val >= 0 else "right"
    ax.text(x_pos, float(bar.get_y()) + float(bar.get_height())/2,
            f"{val:+.3f}", va="center", ha=ha, fontsize=9, fontweight="bold")

ax.set_xlabel("Coeficiente estandarizado (β)", fontsize=11)
ax.set_title("Importancia relativa de variables — Coeficientes estandarizados\n"
             "Rojo = aumenta NO₂ · Azul = reduce NO₂",
             fontsize=12, fontweight="bold")

import matplotlib.patches as mpatches
leg = [mpatches.Patch(color="#E53935", label="Efecto positivo sobre NO₂"),
       mpatches.Patch(color="#1E88E5", label="Efecto negativo sobre NO₂")]
ax.legend(handles=leg, fontsize=9)

plt.tight_layout()
plt.savefig(CARPETA + "M1_03_coeficientes_estandarizados.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ M1_03_coeficientes_estandarizados.png")

# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 4: DISTRIBUCIÓN DE RESIDUOS
# ═══════════════════════════════════════════════════════════════════
print("[G4] Distribución de residuos...")

residuos_train = y_train - y_pred_train
residuos_test  = y_test  - y_pred_test

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Modelo 1 — Análisis de residuos (conjunto de test)",
             fontsize=13, fontweight="bold")

# Histograma de residuos
axes[0].hist(residuos_test, bins=40, color="#1E88E5", alpha=0.8, edgecolor="white")
axes[0].axvline(0, color="red", linewidth=2, linestyle="--")
axes[0].axvline(float(np.mean(residuos_test)), color="orange", linewidth=1.5,
                label=f"Media residuos: {float(np.mean(residuos_test)):.2f}")
axes[0].set_xlabel("Residuo (µg/m³)", fontsize=10)
axes[0].set_ylabel("Frecuencia", fontsize=10)
axes[0].set_title("Distribución de residuos\n(idealmente centrada en 0)",
                  fontsize=10, fontweight="bold")
axes[0].legend(fontsize=8)

# Residuos vs predichos (heterocedasticidad)
axes[1].scatter(y_pred_test, residuos_test, alpha=0.3, s=12, color="#E53935")
axes[1].axhline(0, color="black", linewidth=1)
axes[1].axhline( 2*float(np.std(residuos_test)), color="gray", linestyle="--", linewidth=1)
axes[1].axhline(-2*float(np.std(residuos_test)), color="gray", linestyle="--", linewidth=1)
axes[1].set_xlabel("NO₂ predicho (µg/m³)", fontsize=10)
axes[1].set_ylabel("Residuo (µg/m³)", fontsize=10)
axes[1].set_title("Residuos vs predichos\n(idealmente sin patrón)",
                  fontsize=10, fontweight="bold")

# QQ-plot manual (cuantiles teóricos vs empíricos)
residuos_sorted = np.sort(residuos_test)
n_res = len(residuos_sorted)
teoricos = np.array([float(np.percentile(np.random.randn(10000),
                    100*(i+0.5)/n_res)) for i in range(n_res)])
axes[2].scatter(teoricos, residuos_sorted, alpha=0.3, s=10, color="#43A047")
lim_qq = max(abs(teoricos.min()), abs(teoricos.max()))
lim_rr = max(abs(residuos_sorted.min()), abs(residuos_sorted.max()))
axes[2].plot([-lim_qq, lim_qq],
             [-lim_qq * lim_rr/lim_qq, lim_qq * lim_rr/lim_qq],
             "k--", linewidth=1.5, alpha=0.7, label="Referencia normal")
axes[2].set_xlabel("Cuantiles teóricos (distribución normal)", fontsize=10)
axes[2].set_ylabel("Cuantiles de residuos", fontsize=10)
axes[2].set_title("Q-Q Plot\n(lineal = residuos normalmente distribuidos)",
                  fontsize=10, fontweight="bold")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(CARPETA + "M1_04_analisis_residuos.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ M1_04_analisis_residuos.png")

# ═══════════════════════════════════════════════════════════════════
# GRÁFICO 5: EFECTO DEL TRÁFICO SOBRE NO2 (variable de interés)
# ═══════════════════════════════════════════════════════════════════
print("[G5] Efecto del tráfico sobre NO₂...")

coef_trafico = float(beta_orig[FEATURES.index("trafico_medio") + 1])
intercepto   = float(beta_orig[0])

# Calcular el efecto del tráfico manteniendo el resto de variables en su media
medias_resto = np.array([df_model[col].mean() for col in FEATURES])
trafico_range = np.linspace(float(df_model["trafico_medio"].min()),
                             float(df_model["trafico_medio"].max()), 200)

no2_pred_trafico = []
for t in trafico_range:
    x_punto = medias_resto.copy()
    x_punto[0] = t  # sustituir solo el tráfico
    pred = intercepto + float(np.dot(x_punto, beta_orig[1:]))
    no2_pred_trafico.append(pred)

fig, ax = plt.subplots(figsize=(11, 6))

# Datos reales
ax.scatter(df_model["trafico_medio"], df_model["NO2"],
           alpha=0.15, s=8, color="#E53935", label="Datos reales (días)")
# Línea del modelo
ax.plot(trafico_range, no2_pred_trafico, color="#0D47A1", linewidth=2.5,
        label=f"Efecto tráfico (modelo)\nβ = {coef_trafico:+.4f} µg/m³ por veh/h")

# Anotar el impacto de 100 veh/h adicionales
delta_100 = coef_trafico * 100
ax.annotate(
    f"+100 veh/h → {delta_100:+.2f} µg/m³ NO₂",
    xy=(1300, float(np.interp(1300, trafico_range, no2_pred_trafico))),
    xytext=(1100, 50),
    fontsize=10, fontweight="bold", color="#0D47A1",
    arrowprops=dict(arrowstyle="->", color="#0D47A1", lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9)
)

ax.set_xlabel("Tráfico medio diario (veh/hora)", fontsize=11)
ax.set_ylabel("NO₂ (µg/m³)", fontsize=11)
ax.set_title("Efecto del tráfico sobre el NO₂\n"
             "(manteniendo el resto de variables en su media)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(CARPETA + "M1_05_efecto_trafico.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ M1_05_efecto_trafico.png")

# ═══════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  RESUMEN FINAL — MODELO 1")
print("=" * 65)
print(f"\n  Variable dependiente : NO₂ (µg/m³)")
print(f"  Variables independ.  : {len(FEATURES)}")
print(f"  N train              : {n_train} días")
print(f"  N test               : {n_test} días")
print(f"\n  MÉTRICAS:")
print(f"    R²   train : {r2_train:.4f}  |  test : {r2_test:.4f}")
print(f"    RMSE train : {rmse_train:.3f} µg/m³  |  test : {rmse_test:.3f} µg/m³")
print(f"    MAE  train : {mae_train:.3f} µg/m³  |  test : {mae_test:.3f} µg/m³")
print(f"\n  COEFICIENTE TRÁFICO:")
print(f"    β tráfico = {coef_trafico:+.6f} µg/m³ por veh/h")
print(f"    Impacto de 100 veh/h adicionales: {coef_trafico*100:+.4f} µg/m³")
print(f"    Impacto de 332 veh/h (FdS vs lab): {coef_trafico*332:+.4f} µg/m³")
print(f"\n  VARIABLE MÁS IMPORTANTE (β estandarizado):")
abs_betas = [abs(float(beta_std[i+1])) for i in range(len(FEATURES))]
idx_max = int(np.argmax(abs_betas))
print(f"    {NOMBRES[FEATURES[idx_max]]} : β={float(beta_std[idx_max+1]):+.4f}")
print(f"\n  Gráficos en: {CARPETA}")
print("\nFIN — Pega el output en el chat.")