#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 14:13:46 2026

@author: jaime
"""
### Código para hacer visualizaciones de residuos. Chatgpt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from statistics import NormalDist

# =========================================================
# RUTAS
# =========================================================
RUTA = "/Users/jaime/Documents/Universidad/TFG/Dataset_Diario_Integrado.csv"
SALIDA = "/Users/jaime/Documents/Universidad/TFG/Graficos_Analisis_Final/"
os.makedirs(SALIDA, exist_ok=True)

# =========================================================
# CONFIGURACIÓN VISUAL
# =========================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

# =========================================================
# CARGA Y PREPARACIÓN
# =========================================================
df = pd.read_csv(RUTA, sep=";", encoding="utf-8-sig", low_memory=False)
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)
df["es_fin_semana"] = df["es_fin_semana"].astype(int)

# Temperatura media
df["T2M_MEDIA"] = (df["T2M_MAX"] + df["T2M_MIN"]) / 2

# Dummies de mes (referencia = diciembre)
MESES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre"
}
for m in MESES_ES:
    df[f"mes_{m:02d}"] = (df["mes"] == m).astype(int)

TARGET = "NO2"

FEATURES_BASE = [
    "trafico_medio",
    "T2M_MEDIA",
    "RH2M",
    "WS10M",
    "PRECTOTCORR",
    "PS",
    "ALLSKY_SFC_SW_DWN",
    "es_fin_semana",
    "anyo",
]

DUMMIES_MES = [f"mes_{m:02d}" for m in sorted(MESES_ES.keys())]
FEATURES = FEATURES_BASE + DUMMIES_MES

df_model = df[["fecha", TARGET] + FEATURES].dropna().copy()

# =========================================================
# PARTICIÓN TEMPORAL
# =========================================================
n = len(df_model)
n_train = int(n * 0.70)
n_test = n - n_train

X_all = df_model[FEATURES].values.astype(float)
y_all = df_model[TARGET].values.astype(float)

X_train, X_test = X_all[:n_train], X_all[n_train:]
y_train, y_test = y_all[:n_train], y_all[n_train:]

# =========================================================
# FUNCIONES
# =========================================================
def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]

def r2(yr, yp):
    ss_res = np.sum((yr - yp) ** 2)
    ss_tot = np.sum((yr - yr.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

def rmse(yr, yp):
    return float(np.sqrt(np.mean((yr - yp) ** 2)))

def mae(yr, yp):
    return float(np.mean(np.abs(yr - yp)))

# Cuantiles teóricos normales para Q-Q plot sin scipy
def qq_theoretical_quantiles(n_points):
    probs = (np.arange(1, n_points + 1) - 0.5) / n_points
    nd = NormalDist()
    return np.array([nd.inv_cdf(p) for p in probs])

# =========================================================
# MODELO OLS
# =========================================================
X_mat_train = np.column_stack([np.ones(n_train), X_train])
X_mat_test = np.column_stack([np.ones(n_test), X_test])

beta = ols(X_mat_train, y_train)

y_pred_train = X_mat_train @ beta
y_pred_test = X_mat_test @ beta

r2_train = r2(y_train, y_pred_train)
r2_test = r2(y_test, y_pred_test)
rmse_train = rmse(y_train, y_pred_train)
rmse_test = rmse(y_test, y_pred_test)
mae_train = mae(y_train, y_pred_train)
mae_test = mae(y_test, y_pred_test)

# Residuos en test
res_test = y_test - y_pred_test
sesgo_test = float(np.mean(res_test))
std_res_test = float(np.std(res_test))

# =========================================================
# FIGURA 1 — REALES VS PREDICHOS
# =========================================================
lim_min = min(y_train.min(), y_test.min(), y_pred_train.min(), y_pred_test.min())
lim_max = max(y_train.max(), y_test.max(), y_pred_train.max(), y_pred_test.max())

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig.suptitle(
    "Modelo 1 — Regresión Lineal Múltiple (dummies de mes + temperatura media)\n"
    "Dispersión: valores reales vs predichos",
    fontsize=15,
    fontweight="bold"
)

# Entrenamiento
axes[0].scatter(y_train, y_pred_train, alpha=0.35, s=14)
axes[0].plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="black", alpha=0.7)
axes[0].set_title(
    f"Entrenamiento\nR²={r2_train:.3f}   RMSE={rmse_train:.2f}   MAE={mae_train:.2f} µg/m³",
    fontsize=11,
    fontweight="bold"
)
axes[0].set_xlabel("NO₂ real (µg/m³)")
axes[0].set_ylabel("NO₂ predicho (µg/m³)")
axes[0].set_xlim(lim_min, lim_max)
axes[0].set_ylim(lim_min, lim_max)

# Test
axes[1].scatter(y_test, y_pred_test, alpha=0.35, s=14, color="indianred")
axes[1].plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="black", alpha=0.7)
axes[1].set_title(
    f"Test\nR²={r2_test:.3f}   RMSE={rmse_test:.2f}   MAE={mae_test:.2f} µg/m³",
    fontsize=11,
    fontweight="bold"
)
axes[1].set_xlabel("NO₂ real (µg/m³)")
axes[1].set_ylabel("NO₂ predicho (µg/m³)")
axes[1].set_xlim(lim_min, lim_max)
axes[1].set_ylim(lim_min, lim_max)

plt.tight_layout()
plt.savefig(os.path.join(SALIDA, "M1_02_scatter_real_vs_predicho_final.png"), dpi=150, bbox_inches="tight")
plt.close()

# =========================================================
# FIGURA 2 — ANÁLISIS DE RESIDUOS (solo test)
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    "Modelo 1 — Análisis de residuos (conjunto de test)\n"
    "Versión final: dummies de mes + temperatura media",
    fontsize=15,
    fontweight="bold"
)

# 1) Histograma
axes[0].hist(res_test, bins=30, alpha=0.85, edgecolor="white")
axes[0].axvline(0, color="red", linestyle="--", linewidth=2, label="Cero")
axes[0].axvline(sesgo_test, color="orange", linestyle="-", linewidth=2, label=f"Media: {sesgo_test:.2f}")
axes[0].set_title("Distribución de residuos", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Residuo (µg/m³)")
axes[0].set_ylabel("Frecuencia")
axes[0].legend()

# 2) Residuos vs predichos
axes[1].scatter(y_pred_test, res_test, alpha=0.35, s=18, color="indianred")
axes[1].axhline(0, color="black", linewidth=1)
axes[1].axhline(2 * std_res_test, color="gray", linestyle="--", linewidth=1)
axes[1].axhline(-2 * std_res_test, color="gray", linestyle="--", linewidth=1)
axes[1].set_title("Residuos vs predichos", fontsize=11, fontweight="bold")
axes[1].set_xlabel("NO₂ predicho (µg/m³)")
axes[1].set_ylabel("Residuo (µg/m³)")

# 3) Q-Q Plot
res_sorted = np.sort(res_test)
res_std = (res_sorted - np.mean(res_sorted)) / (np.std(res_sorted) + 1e-10)
theoretical = qq_theoretical_quantiles(len(res_std))

axes[2].scatter(theoretical, res_std, alpha=0.5, s=16, color="#4CAF50")
min_q = min(theoretical.min(), res_std.min())
max_q = max(theoretical.max(), res_std.max())
axes[2].plot([min_q, max_q], [min_q, max_q], linestyle="--", color="black", alpha=0.7)
axes[2].set_title("Q-Q Plot", fontsize=11, fontweight="bold")
axes[2].set_xlabel("Cuantiles teóricos")
axes[2].set_ylabel("Cuantiles estandarizados de residuos")

plt.tight_layout()
plt.savefig(os.path.join(SALIDA, "M1_04_analisis_residuos_final.png"), dpi=150, bbox_inches="tight")
plt.close()

# =========================================================
# RESUMEN
# =========================================================
print("=" * 72)
print("MODELO 1 FINAL — VISUALIZACIONES GENERADAS")
print("=" * 72)
print(f"Train -> R²={r2_train:.4f} | RMSE={rmse_train:.3f} | MAE={mae_train:.3f}")
print(f"Test  -> R²={r2_test:.4f} | RMSE={rmse_test:.3f} | MAE={mae_test:.3f}")
print(f"Sesgo test: {sesgo_test:+.3f} µg/m³")
print(f"Desviación típica residuos test: {std_res_test:.3f} µg/m³")
print()
print("Archivos guardados:")
print(" - M1_02_scatter_real_vs_predicho_final.png")
print(" - M1_04_analisis_residuos_final.png")
print("=" * 72)