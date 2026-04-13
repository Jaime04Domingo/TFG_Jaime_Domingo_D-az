#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 14:34:59 2026

@author: jaime
"""
### Código para crear visualizacion VIF
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
SALIDA = "/Users/jaime/Documents/Universidad/TFG/Graficos_Analisis_Final/"
os.makedirs(SALIDA, exist_ok=True)

# =========================================================
# CONFIGURACIÓN VISUAL MINIMALISTA
# =========================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "figure.dpi": 160,
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

NOMBRES = {
    "trafico_medio": "Tráfico",
    "T2M_MEDIA": "Temp. media",
    "RH2M": "Humedad",
    "WS10M": "Viento",
    "PRECTOTCORR": "Precipitación",
    "PS": "Presión",
    "ALLSKY_SFC_SW_DWN": "Radiación",
    "es_fin_semana": "Fin de semana",
    "anyo": "Año",
    **{f"mes_{m:02d}": MESES_ES[m] for m in MESES_ES}
}

df_model = df[[TARGET] + FEATURES].dropna().copy()

# Partición temporal 70/30 (VIF sobre train)
n = len(df_model)
n_train = int(n * 0.70)
X_train = df_model[FEATURES].iloc[:n_train].values.astype(float)

# =========================================================
# FUNCIONES
# =========================================================
def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]

def r2(y_real, y_pred):
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

def calcular_vif(X, nombres):
    vifs = []
    for j in range(X.shape[1]):
        y_j = X[:, j]
        X_j = np.delete(X, j, axis=1)
        X_j_mat = np.column_stack([np.ones(len(y_j)), X_j])
        beta_j = ols(X_j_mat, y_j)
        y_pred_j = X_j_mat @ beta_j
        r2_j = r2(y_j, y_pred_j)
        vif_j = 1 / (1 - r2_j) if r2_j < 1 else np.inf
        vifs.append(vif_j)
    return pd.DataFrame({
        "variable": nombres,
        "vif": vifs
    })

# =========================================================
# CÁLCULO VIF
# =========================================================
vif_df = calcular_vif(X_train, FEATURES)
vif_df["nombre"] = vif_df["variable"].map(NOMBRES)
vif_df = vif_df.sort_values("vif", ascending=True)

# =========================================================
# GRÁFICO VIF
# =========================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Colores según umbral
colores = []
for v in vif_df["vif"]:
    if v < 5:
        colores.append("#5B8FF9")   # azul suave
    elif v < 10:
        colores.append("#F6BD16")   # ámbar
    else:
        colores.append("#E8684A")   # rojo suave

bars = ax.barh(vif_df["nombre"], vif_df["vif"], color=colores, alpha=0.9)

# Líneas de referencia
ax.axvline(5, color="#F6BD16", linestyle="--", linewidth=1.4, alpha=0.9)
ax.axvline(10, color="#E8684A", linestyle="--", linewidth=1.4, alpha=0.9)

# Etiquetas al final de cada barra
for bar, val in zip(bars, vif_df["vif"]):
    ax.text(
        bar.get_width() + 0.12,
        bar.get_y() + bar.get_height()/2,
        f"{val:.2f}",
        va="center",
        ha="left",
        fontsize=10
    )

ax.set_title("Factor de Inflación de la Varianza (VIF)")
ax.set_xlabel("VIF")
ax.set_ylabel("")
ax.set_xlim(0, max(vif_df["vif"].max() + 1.2, 11))

# Nota discreta
fig.text(
    0.5, 0.01,
    "Azul: VIF < 5 · Ámbar: 5 ≤ VIF < 10 · Rojo: VIF ≥ 10",
    ha="center",
    fontsize=10,
    color="dimgray"
)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(os.path.join(SALIDA, "M1_VIF_minimalista.png"), bbox_inches="tight")
plt.close()

print(vif_df[["nombre", "vif"]].to_string(index=False))
print("\nGráfico guardado en:", os.path.join(SALIDA, "M1_VIF_minimalista.png"))