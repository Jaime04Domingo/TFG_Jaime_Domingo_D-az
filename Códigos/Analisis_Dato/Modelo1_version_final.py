#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:14:20 2026

@author: jaime
"""
### Hecho por chatgpt
"""
MODELO 1 ALTERNATIVO — Regresión Lineal con Dummies de Mes + Temperatura Media
Incluye verificación de multicolinealidad (VIF)
TFG Madrid
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

RUTA    = "/Users/jaime/Documents/Universidad/TFG/Dataset_Diario_Integrado.csv"
CARPETA = "/Users/jaime/Documents/Universidad/TFG/Graficos_Analisis_Final/"
os.makedirs(CARPETA, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 150,
})

print("=" * 72)
print("  MODELO 1 ALTERNATIVO — REGRESIÓN CON DUMMIES + TEMPERATURA MEDIA")
print("=" * 72)

# ── CARGA ─────────────────────────────────────────────────────────────
df = pd.read_csv(RUTA, sep=";", encoding="utf-8-sig", low_memory=False)
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)
df["es_fin_semana"] = df["es_fin_semana"].astype(int)

TARGET = "NO2"

# Crear temperatura media
df["T2M_MEDIA"] = (df["T2M_MAX"] + df["T2M_MIN"]) / 2

# Variables base (sin T_max ni T_min, y sin mes numérico)
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

NOMBRES_BASE = {
    "trafico_medio":      "Tráfico (veh/h)",
    "T2M_MEDIA":          "Temp. media (°C)",
    "RH2M":               "Humedad relativa (%)",
    "WS10M":              "Velocidad viento (m/s)",
    "PRECTOTCORR":        "Precipitación (mm)",
    "PS":                 "Presión atmosférica (kPa)",
    "ALLSKY_SFC_SW_DWN":  "Radiación solar (MJ/m²)",
    "es_fin_semana":      "Fin de semana (0/1)",
    "anyo":               "Año",
}

MESES_ES = {
    1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril",
    5:"Mayo", 6:"Junio", 7:"Julio", 8:"Agosto",
    9:"Septiembre", 10:"Octubre", 11:"Noviembre"
    # Diciembre = referencia
}

# Crear dummies de mes (referencia = diciembre)
for mes_num, mes_nom in MESES_ES.items():
    df[f"mes_{mes_num:02d}"] = (df["mes"] == mes_num).astype(int)

DUMMIES_MES = [f"mes_{m:02d}" for m in sorted(MESES_ES.keys())]
NOMBRES_DUMMIES = {f"mes_{m:02d}": f"{MESES_ES[m]} (vs Dic)" for m in MESES_ES}

FEATURES = FEATURES_BASE + DUMMIES_MES
NOMBRES  = {**NOMBRES_BASE, **NOMBRES_DUMMIES}

# Verificar disponibilidad
FEATURES = [f for f in FEATURES if f in df.columns]
df_model = df[[TARGET] + FEATURES + ["fecha"]].dropna().copy()

n       = len(df_model)
n_train = int(n * 0.70)
n_test  = n - n_train

X_all = df_model[FEATURES].values.astype(float)
y_all = df_model[TARGET].values.astype(float)

X_train, X_test = X_all[:n_train], X_all[n_train:]
y_train, y_test = y_all[:n_train], y_all[n_train:]

print(f"\n  Features totales: {len(FEATURES)}")
print(f"  - Variables base: {len(FEATURES_BASE)}")
print(f"  - Dummies mes   : {len(DUMMIES_MES)} (ref=Diciembre)")
print(f"  Partición: {n_train} entrenamiento | {n_test} test")

# ═══════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════
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

def calcular_vif(X_data, nombres):
    """
    Calcula el VIF para cada variable:
    VIF_j = 1 / (1 - R²_j)
    donde R²_j es el R² de la regresión de la variable j contra todas las demás.
    """
    n_vars = X_data.shape[1]
    vifs = []
    for j in range(n_vars):
        y_j = X_data[:, j]
        X_j = np.delete(X_data, j, axis=1)
        X_j_mat = np.column_stack([np.ones(len(y_j)), X_j])
        beta_j = ols(X_j_mat, y_j)
        y_pred_j = X_j_mat @ beta_j
        r2_j = r2(y_j, y_pred_j)
        vif_j = 1 / (1 - r2_j) if r2_j < 1.0 else np.inf
        vifs.append(float(vif_j))
    return vifs

# ═══════════════════════════════════════════════════════════════════════
# VERIFICACIÓN DE MULTICOLINEALIDAD — VIF
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("  VERIFICACIÓN DE MULTICOLINEALIDAD (VIF)")
print("="*72)
print("  Referencia: VIF<5 = sin problema | 5-10 = moderada | >10 = alta")
print("-"*72)

vifs = calcular_vif(X_train, FEATURES)

print(f"\n  {'Variable':<35} {'VIF':>8}  Estado")
print(f"  {'-'*62}")
problemas_vif = []

for feat, vif in zip(FEATURES, vifs):
    nombre = NOMBRES.get(feat, feat)
    if vif == np.inf or vif > 100:
        estado = "⛔ INFINITO — multicolinealidad perfecta"
        problemas_vif.append(feat)
    elif vif > 10:
        estado = "⚠  ALTO"
        problemas_vif.append(feat)
    elif vif > 5:
        estado = "△  MODERADO"
    else:
        estado = "✓  OK"
    vif_str = f"{vif:.2f}" if vif < 1000 else "∞"
    print(f"  {nombre:<35} {vif_str:>8}  {estado}")

if not problemas_vif:
    print(f"\n  ✅ Sin problemas graves de multicolinealidad. Todos los VIF < 10.")
else:
    print(f"\n  ⚠  Variables con VIF > 10: {problemas_vif}")

# ═══════════════════════════════════════════════════════════════════════
# REGRESIÓN CON DUMMIES + TEMPERATURA MEDIA
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("  MODELO CON DUMMIES DE MES + TEMPERATURA MEDIA")
print("="*72)

X_mat   = np.column_stack([np.ones(n_train), X_train])
X_t_mat = np.column_stack([np.ones(n_test),  X_test])
beta    = ols(X_mat, y_train)

y_pred_train = X_mat   @ beta
y_pred_test  = X_t_mat @ beta

r2_train   = r2(y_train, y_pred_train)
r2_test    = r2(y_test,  y_pred_test)
rmse_train = rmse(y_train, y_pred_train)
rmse_test  = rmse(y_test,  y_pred_test)
mae_train  = mae(y_train, y_pred_train)
mae_test   = mae(y_test,  y_pred_test)

print(f"\n  {'Métrica':<22} {'Train':>10} {'Test':>10}")
print(f"  {'-'*44}")
print(f"  {'R²':<22} {r2_train:>10.4f} {r2_test:>10.4f}")
print(f"  {'RMSE (µg/m³)':<22} {rmse_train:>10.3f} {rmse_test:>10.3f}")
print(f"  {'MAE  (µg/m³)':<22} {mae_train:>10.3f} {mae_test:>10.3f}")

# Coeficientes estandarizados
X_std    = np.column_stack([estandarizar(df_model[c].values) for c in FEATURES])
y_std    = estandarizar(y_all)
beta_std = ols(np.column_stack([np.ones(n_train), X_std[:n_train]]), y_std[:n_train])

print(f"\n  COEFICIENTES DE VARIABLES BASE (unidades originales):")
print(f"  {'Variable':<35} {'Coef':>12}  {'Beta_std':>10}  Interpretación")
print(f"  {'-'*82}")
print(f"  {'Intercepto':<35} {float(beta[0]):>12.4f}")

for i, col in enumerate(FEATURES_BASE):
    if col not in FEATURES:
        continue
    fi  = FEATURES.index(col)
    c   = float(beta[fi + 1])
    b_s = float(beta_std[fi + 1])
    interp = ""

    if col == "trafico_medio":
        interp = f"→ +100 veh/h = {c*100:+.3f} µg/m³"
    elif col == "T2M_MEDIA":
        interp = f"→ +1°C = {c:+.3f} µg/m³"
    elif col == "WS10M":
        interp = f"→ +1 m/s = {c:+.3f} µg/m³"
    elif col == "PS":
        interp = f"→ +1 kPa = {c:+.3f} µg/m³"
    elif col == "ALLSKY_SFC_SW_DWN":
        interp = f"→ +1 MJ/m² = {c:+.3f} µg/m³"
    elif col == "es_fin_semana":
        interp = f"→ FdS cambia {c:+.3f} µg/m³"
    elif col == "anyo":
        interp = f"→ por año = {c:+.3f} µg/m³"

    print(f"  {NOMBRES[col]:<35} {c:>12.4f}  {b_s:>10.4f}  {interp}")

print(f"\n  COEFICIENTES DE DUMMIES DE MES (referencia = Diciembre):")
print(f"  {'Mes':<20} {'Coef':>10}  Interp. (vs Diciembre)")
print(f"  {'-'*57}")
dummy_vals = []
for col in DUMMIES_MES:
    fi = FEATURES.index(col)
    c  = float(beta[fi + 1])
    dummy_vals.append((NOMBRES_DUMMIES[col].split(" ")[0], c))
    print(f"  {NOMBRES_DUMMIES[col]:<20} {c:>10.3f}  → {c:+.3f} µg/m³ vs Diciembre")

coef_traf = float(beta[FEATURES.index("trafico_medio") + 1])
coef_temp = float(beta[FEATURES.index("T2M_MEDIA") + 1])
coef_fds  = float(beta[FEATURES.index("es_fin_semana") + 1])
coef_anyo = float(beta[FEATURES.index("anyo") + 1])

print(f"\n  HALLAZGOS CLAVE:")
print(f"    Coef. tráfico      : +100 veh/h = {coef_traf*100:+.3f} µg/m³")
print(f"    Coef. temp. media  : +1°C = {coef_temp:+.3f} µg/m³")
print(f"    Coef. FdS          : {coef_fds:+.3f} µg/m³")
print(f"    Coef. año          : {coef_anyo:+.3f} µg/m³/año")

# ═══════════════════════════════════════════════════════════════════════
# COMPARATIVAS
# ═══════════════════════════════════════════════════════════════════════
print(f"\n  COMPARATIVA CON VERSIÓN ANTERIOR (mes numérico):")
print(f"  R² test anterior             : 0.6849")
print(f"  R² test temp. media + dummies: {r2_test:.4f}  ({r2_test-0.6849:+.4f})")
print(f"  RMSE test anterior           : 7.454 µg/m³")
print(f"  RMSE test temp. media        : {rmse_test:.3f} µg/m³  ({rmse_test-7.454:+.3f})")
print(f"  MAE test anterior            : 5.984 µg/m³")
print(f"  MAE test temp. media         : {mae_test:.3f} µg/m³  ({mae_test-5.984:+.3f})")

print(f"\n  COMPARATIVA CON VERSIÓN DUMMIES + T_MAX/T_MIN:")
print(f"  R² test dummies anterior     : [PEGAR]")
print(f"  RMSE test dummies anterior   : [PEGAR]")
print(f"  MAE test dummies anterior    : [PEGAR]")
print(f"  Sesgo medio anterior         : [PEGAR]")

# ═══════════════════════════════════════════════════════════════════════
# GRÁFICOS
# ═══════════════════════════════════════════════════════════════════════

# G1: Coeficientes de los dummies de mes
print("\n[G1] Coeficientes dummies de mes...")
meses_nom  = [d[0] for d in dummy_vals] + ["Diciembre\n(ref.)"]
meses_coef = [d[1] for d in dummy_vals] + [0.0]

fig, ax = plt.subplots(figsize=(13, 6))
colores = ["#E53935" if v > 0 else "#43A047" for v in meses_coef]
colores[-1] = "#90A4AE"

bars = ax.bar(meses_nom, meses_coef, color=colores, alpha=0.85, edgecolor="white")
ax.axhline(0, color="black", linewidth=1)

for bar, val in zip(bars, meses_coef):
    ypos = float(bar.get_height()) + (0.3 if val >= 0 else -1.2)
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f"{val:+.1f}" if val != 0 else "ref.",
            ha="center", fontsize=9, fontweight="bold")

ax.set_ylabel("Efecto sobre NO₂ (µg/m³) vs Diciembre", fontsize=11)
ax.set_title("Efecto de cada mes sobre el NO₂\n"
             "Modelo con temperatura media · Referencia = Diciembre",
             fontsize=12, fontweight="bold")

leg = [
    mpatches.Patch(color="#E53935", label="Más NO₂ que diciembre"),
    mpatches.Patch(color="#43A047", label="Menos NO₂ que diciembre"),
    mpatches.Patch(color="#90A4AE", label="Referencia (Diciembre)")
]
ax.legend(handles=leg, fontsize=9)

plt.tight_layout()
plt.savefig(CARPETA + "A12_coeficientes_dummies_mes_temp_media.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ A12_coeficientes_dummies_mes_temp_media.png")

# G2: VIF por variable
print("[G2] VIF por variable...")
nombres_vif = [NOMBRES.get(f, f) for f in FEATURES]
vifs_plot   = [min(v, 50) for v in vifs]
cols_vif    = ["#E53935" if v > 10 else "#FB8C00" if v > 5 else "#43A047" for v in vifs]

fig, ax = plt.subplots(figsize=(13, 7))
bars = ax.barh(nombres_vif, vifs_plot, color=cols_vif, alpha=0.85, edgecolor="white")
ax.axvline(5,  color="orange", linestyle="--", linewidth=1.5, alpha=0.8, label="VIF=5 (moderado)")
ax.axvline(10, color="red",    linestyle="--", linewidth=1.5, alpha=0.8, label="VIF=10 (alto)")

for bar, val in zip(bars, vifs):
    ax.text(float(bar.get_width()) + 0.2,
            float(bar.get_y()) + float(bar.get_height())/2,
            f"{val:.1f}",
            va="center", fontsize=7.5, fontweight="bold")

ax.set_xlabel("Factor de Inflación de la Varianza (VIF)", fontsize=11)
ax.set_title("Multicolinealidad — modelo con temperatura media\n"
             "Verde <5 · Naranja 5-10 · Rojo >10",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(CARPETA + "A13_VIF_temp_media.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ A13_VIF_temp_media.png")

# G3: Predicciones vs real
print("[G3] Predicciones vs real...")
fechas_test = df_model["fecha"].iloc[n_train:n_train+n_test].values
res_test    = y_test - y_pred_test
sesgo_test  = float(np.mean(res_test))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle("Modelo 1 con dummies de mes + temperatura media — Predicciones vs reales\n"
             f"R²={r2_test:.3f}  RMSE={rmse_test:.2f} µg/m³  MAE={mae_test:.2f} µg/m³",
             fontsize=13, fontweight="bold")

ax1.plot(fechas_test, y_test,      color="#E53935", linewidth=1.0, alpha=0.7, label="NO₂ real")
ax1.plot(fechas_test, y_pred_test, color="#1E88E5", linewidth=1.2, linestyle="--",
         alpha=0.8, label="NO₂ predicho")
ax1.set_ylabel("NO₂ (µg/m³)", fontsize=10)
ax1.legend(fontsize=9)

ax2.bar(fechas_test, res_test,
        color=["#E53935" if r > 0 else "#1E88E5" for r in res_test],
        alpha=0.5, width=1)
ax2.axhline(0, color="black", linewidth=1)
ax2.axhline( 2*float(np.std(res_test)), color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax2.axhline(-2*float(np.std(res_test)), color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax2.set_ylabel("Residuo (µg/m³)", fontsize=10)
ax2.set_xlabel("Fecha", fontsize=10)
ax2.set_title(f"Residuos | Media={sesgo_test:.2f} µg/m³",
              fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(CARPETA + "A14_predicciones_temp_media.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ A14_predicciones_temp_media.png")

# ── RESUMEN ───────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  RESUMEN FINAL")
print("="*72)
print(f"\n  Modelo con dummies + temperatura media:")
print(f"  R²   test: {r2_test:.4f}")
print(f"  RMSE test: {rmse_test:.3f} µg/m³")
print(f"  MAE  test: {mae_test:.3f} µg/m³")
print(f"  Sesgo test: {sesgo_test:+.3f} µg/m³")
print(f"\n  Multicolinealidad: {'✅ Sin problemas graves' if not problemas_vif else f'⚠ {problemas_vif}'}")
print(f"\n  Coeficiente tráfico     : {coef_traf*100:+.3f} µg/m³ por 100 veh/h")
print(f"  Coeficiente temp. media : {coef_temp:+.3f} µg/m³ por 1°C")
print(f"  Coeficiente FdS         : {coef_fds:+.3f} µg/m³")
print(f"  Coeficiente año         : {coef_anyo:+.3f} µg/m³/año")
print(f"\n  Gráficos:")
print(f"    A12_coeficientes_dummies_mes_temp_media.png")
print(f"    A13_VIF_temp_media.png")
print(f"    A14_predicciones_temp_media.png")
print("\nFIN — pega el output en el chat.")