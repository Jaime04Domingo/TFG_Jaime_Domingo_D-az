#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 12:53:16 2026

@author: jaime
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# =========================================================
# RUTAS
# =========================================================
RUTA = "/Users/jaime/Documents/Universidad/TFG/Dataset_Diario_Integrado.csv"
CARPETA_SALIDA = "/Users/jaime/Documents/Universidad/TFG/Graficos_Analisis_Final/"
os.makedirs(CARPETA_SALIDA, exist_ok=True)

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

# Variable binaria fin de semana
df["es_fin_semana"] = df["es_fin_semana"].astype(int)

# Temperatura media
df["T2M_MEDIA"] = (df["T2M_MAX"] + df["T2M_MIN"]) / 2

# Dummies de mes (referencia = diciembre)
MESES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre"
}
for mes_num in MESES_ES:
    df[f"mes_{mes_num:02d}"] = (df["mes"] == mes_num).astype(int)

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

NOMBRES_BASE = {
    "trafico_medio": "Tráfico (veh/h)",
    "T2M_MEDIA": "Temp. media (°C)",
    "RH2M": "Humedad relativa (%)",
    "WS10M": "Velocidad viento (m/s)",
    "PRECTOTCORR": "Precipitación (mm)",
    "PS": "Presión atmosférica (kPa)",
    "ALLSKY_SFC_SW_DWN": "Radiación solar (MJ/m²)",
    "es_fin_semana": "Fin de semana (0/1)",
    "anyo": "Año",
}

# Dataset final
df_model = df[["fecha", TARGET] + FEATURES].dropna().copy()

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

def estandarizar(arr):
    return (arr - arr.mean()) / (arr.std() + 1e-10)

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

# =========================================================
# COEFICIENTES ESTANDARIZADOS
# Solo variables base (las dummies ya van en otro gráfico)
# =========================================================
X_std = np.column_stack([estandarizar(df_model[c].values) for c in FEATURES])
y_std = estandarizar(y_all)

beta_std = ols(
    np.column_stack([np.ones(n_train), X_std[:n_train]]),
    y_std[:n_train]
)

coef_std_base = []
for col in FEATURES_BASE:
    idx = FEATURES.index(col)
    coef_std_base.append(float(beta_std[idx + 1]))

coef_std_df = pd.DataFrame({
    "variable": FEATURES_BASE,
    "nombre": [NOMBRES_BASE[c] for c in FEATURES_BASE],
    "beta_std": coef_std_base
}).sort_values("beta_std")

# =========================================================
# GRÁFICO 1 — DISPERSIÓN REAL VS PREDICHO
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

# Train
axes[0].scatter(y_train, y_pred_train, alpha=0.35, s=12)
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
axes[1].scatter(y_test, y_pred_test, alpha=0.35, s=12, color="indianred")
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
plt.savefig(
    os.path.join(CARPETA_SALIDA, "M1_02_scatter_real_vs_predicho_actualizado.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# =========================================================
# GRÁFICO 2 — COEFICIENTES ESTANDARIZADOS
# =========================================================
fig, ax = plt.subplots(figsize=(12, 7))

colores = ["#e74c3c" if v > 0 else "#3498db" for v in coef_std_df["beta_std"]]
bars = ax.barh(coef_std_df["nombre"], coef_std_df["beta_std"], color=colores, alpha=0.85)

ax.axvline(0, color="black", linewidth=1)
ax.set_title(
    "Importancia relativa de variables — Coeficientes estandarizados\n"
    "Rojo = aumenta NO₂ · Azul = reduce NO₂",
    fontsize=15,
    fontweight="bold"
)
ax.set_xlabel("Coeficiente estandarizado (β)")
ax.set_ylabel("")

for bar, val in zip(bars, coef_std_df["beta_std"]):
    x = float(bar.get_width())
    y = float(bar.get_y()) + float(bar.get_height()) / 2
    if val >= 0:
        ax.text(x + 0.01, y, f"{val:+.3f}", va="center", ha="left", fontsize=10, fontweight="bold")
    else:
        ax.text(x - 0.01, y, f"{val:+.3f}", va="center", ha="right", fontsize=10, fontweight="bold")

# Nota abajo
fig.text(
    0.5, 0.01,
    "Nota: las dummies de mes no se incluyen en este gráfico porque se representan por separado en la figura específica de efectos mensuales.",
    ha="center", fontsize=9
)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(
    os.path.join(CARPETA_SALIDA, "M1_03_coeficientes_estandarizados_actualizado.png"),
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# =========================================================
# SALIDA RESUMEN
# =========================================================
print("=" * 70)
print("MODELO 1 ACTUALIZADO — GRÁFICOS GENERADOS")
print("=" * 70)
print(f"Train -> R²={r2_train:.4f} | RMSE={rmse_train:.3f} | MAE={mae_train:.3f}")
print(f"Test  -> R²={r2_test:.4f} | RMSE={rmse_test:.3f} | MAE={mae_test:.3f}")
print()
print("Archivos guardados:")
print(" - M1_02_scatter_real_vs_predicho_actualizado.png")
print(" - M1_03_coeficientes_estandarizados_actualizado.png")
print("=" * 70)