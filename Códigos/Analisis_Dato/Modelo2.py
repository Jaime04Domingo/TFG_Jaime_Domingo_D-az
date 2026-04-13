#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:47:50 2026

@author: jaime
"""

"""
MODELO 2 — Random Forest
Implementado con numpy y pandas (sin scikit-learn)
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
CARPETA = "/Users/jaime/Documents/Universidad/TFG/Graficos_Modelo2/"
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
print("  MODELO 2 — RANDOM FOREST")
print("=" * 65)
 
# ── CARGA ─────────────────────────────────────────────────────────────
df = pd.read_csv(RUTA, sep=";", encoding="utf-8-sig", low_memory=False)
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)
 
TARGET   = "NO2"
FEATURES = ["trafico_medio", "T2M_MAX", "T2M_MIN", "RH2M",
            "WS10M", "PRECTOTCORR", "mes", "dia_semana"]
NOMBRES  = {
    "trafico_medio":  "Tráfico (veh/h)",
    "T2M_MAX":        "Temp. máxima (°C)",
    "T2M_MIN":        "Temp. mínima (°C)",
    "RH2M":           "Humedad relativa (%)",
    "WS10M":          "Velocidad viento (m/s)",
    "PRECTOTCORR":    "Precipitación (mm)",
    "mes":            "Mes del año",
    "dia_semana":     "Día de la semana",
}
 
df_model = df[[TARGET] + FEATURES].dropna().copy()
n = len(df_model)
n_train = int(n * 0.70)
n_test  = n - n_train
 
X_all = df_model[FEATURES].values.astype(float)
y_all = df_model[TARGET].values.astype(float)
X_train, X_test = X_all[:n_train], X_all[n_train:]
y_train, y_test = y_all[:n_train], y_all[n_train:]
 
print(f"\n  Dataset: {len(df_model):,} días")
print(f"  Partición: {n_train} entrenamiento | {n_test} test")
 
# ═══════════════════════════════════════════════════════════════════════
# IMPLEMENTACIÓN DE ÁRBOL DE DECISIÓN Y RANDOM FOREST CON NUMPY
# ═══════════════════════════════════════════════════════════════════════
 
def varianza_ponderada(y_izq, y_der):
    """Criterio de división: reducción de varianza"""
    n_total = len(y_izq) + len(y_der)
    if n_total == 0:
        return 0.0
    v_izq = float(np.var(y_izq)) * len(y_izq) if len(y_izq) > 0 else 0.0
    v_der = float(np.var(y_der)) * len(y_der) if len(y_der) > 0 else 0.0
    return -(v_izq + v_der) / n_total
 
def mejor_split(X, y, n_features_split, rng):
    """Encuentra la mejor división para un nodo"""
    n_samples, n_feats = X.shape
    if n_samples < 2:
        return None, None, None, -np.inf
 
    # Selección aleatoria de features (clave del Random Forest)
    feat_indices = rng.choice(n_feats, size=min(n_features_split, n_feats), replace=False)
 
    mejor_feat  = None
    mejor_umbral = None
    mejor_score  = -np.inf
    mejor_mascara = None
 
    for feat in feat_indices:
        valores = X[:, feat]
        umbrales = np.unique(valores)
        # Muestrear umbrales para eficiencia
        if len(umbrales) > 20:
            umbrales = rng.choice(umbrales, size=20, replace=False)
 
        for umbral in umbrales:
            mascara_izq = valores <= umbral
            mascara_der = ~mascara_izq
            if mascara_izq.sum() < 2 or mascara_der.sum() < 2:
                continue
            score = varianza_ponderada(y[mascara_izq], y[mascara_der])
            if score > mejor_score:
                mejor_score  = score
                mejor_feat   = feat
                mejor_umbral = float(umbral)
                mejor_mascara = mascara_izq
 
    return mejor_feat, mejor_umbral, mejor_mascara, mejor_score
 
def construir_arbol(X, y, profundidad_max, min_muestras, n_features_split, rng, profundidad=0):
    """Construye un árbol de decisión recursivamente"""
    # Condiciones de parada
    if profundidad >= profundidad_max or len(y) <= min_muestras or np.var(y) < 1e-6:
        return {"hoja": True, "valor": float(np.mean(y))}
 
    feat, umbral, mascara_izq, score = mejor_split(X, y, n_features_split, rng)
 
    if feat is None or mascara_izq is None:
        return {"hoja": True, "valor": float(np.mean(y))}
 
    mascara_der = ~mascara_izq
    if mascara_izq.sum() < 2 or mascara_der.sum() < 2:
        return {"hoja": True, "valor": float(np.mean(y))}
 
    return {
        "hoja":       False,
        "feature":    int(feat),
        "umbral":     float(umbral),
        "izquierda":  construir_arbol(X[mascara_izq], y[mascara_izq], profundidad_max, min_muestras, n_features_split, rng, profundidad+1),
        "derecha":    construir_arbol(X[mascara_der], y[mascara_der], profundidad_max, min_muestras, n_features_split, rng, profundidad+1),
        "n_muestras": len(y),
        "varianza_antes": float(np.var(y)),
        "varianza_red": abs(score),
    }
 
def predecir_arbol(nodo, x):
    """Predicción de un árbol para una muestra"""
    if nodo["hoja"]:
        return nodo["valor"]
    if x[nodo["feature"]] <= nodo["umbral"]:
        return predecir_arbol(nodo["izquierda"], x)
    return predecir_arbol(nodo["derecha"], x)
 
def predecir_batch(arbol, X):
    return np.array([predecir_arbol(arbol, X[i]) for i in range(len(X))])
 
def importancia_variables(arbol, n_features, importancias=None):
    """Calcula importancia de variables (reducción de varianza ponderada)"""
    if importancias is None:
        importancias = np.zeros(n_features)
    if arbol["hoja"]:
        return importancias
    feat = arbol["feature"]
    importancias[feat] += arbol["n_muestras"] * arbol["varianza_red"]
    importancia_variables(arbol["izquierda"], n_features, importancias)
    importancia_variables(arbol["derecha"], n_features, importancias)
    return importancias
 
# ── HIPERPARÁMETROS ────────────────────────────────────────────────────
N_ARBOLES        = 80    # número de árboles
PROFUNDIDAD_MAX  = 7     # profundidad máxima de cada árbol
MIN_MUESTRAS     = 10    # mínimo de muestras para dividir un nodo
N_FEATURES_SPLIT = 4     # nº de features a considerar en cada split (≈ sqrt de 8)
SEED             = 42
 
print(f"\n  Hiperparámetros:")
print(f"    Árboles         : {N_ARBOLES}")
print(f"    Profundidad máx : {PROFUNDIDAD_MAX}")
print(f"    Min. muestras   : {MIN_MUESTRAS}")
print(f"    Features/split  : {N_FEATURES_SPLIT}")
print(f"\n  Entrenando {N_ARBOLES} árboles...")
 
rng = np.random.default_rng(SEED)
bosque    = []
oob_preds = np.full(n_train, np.nan)
oob_count = np.zeros(n_train)
importancias_total = np.zeros(len(FEATURES))
 
for i in range(N_ARBOLES):
    if (i + 1) % 20 == 0:
        print(f"    Árbol {i+1}/{N_ARBOLES}...")
 
    # Bootstrap: muestreo con reemplazo
    indices_boot = rng.integers(0, n_train, size=n_train)
    indices_oob  = np.setdiff1d(np.arange(n_train), np.unique(indices_boot))
 
    X_boot = X_train[indices_boot]
    y_boot = y_train[indices_boot]
 
    arbol = construir_arbol(X_boot, y_boot, PROFUNDIDAD_MAX, MIN_MUESTRAS, N_FEATURES_SPLIT, rng)
    bosque.append(arbol)
 
    # OOB predictions
    if len(indices_oob) > 0:
        preds_oob = predecir_batch(arbol, X_train[indices_oob])
        oob_preds[indices_oob] = np.where(
            np.isnan(oob_preds[indices_oob]),
            preds_oob,
            (oob_preds[indices_oob] * oob_count[indices_oob] + preds_oob) / (oob_count[indices_oob] + 1)
        )
        oob_count[indices_oob] += 1
 
    # Importancia de variables
    imp = importancia_variables(arbol, len(FEATURES))
    importancias_total += imp
 
print("  ✓ Entrenamiento completado")
 
# ── PREDICCIONES FINALES (media de todos los árboles) ─────────────────
print("\n  Generando predicciones...")
preds_train_list = np.array([predecir_batch(a, X_train) for a in bosque])
preds_test_list  = np.array([predecir_batch(a, X_test)  for a in bosque])
 
y_pred_train = preds_train_list.mean(axis=0)
y_pred_test  = preds_test_list.mean(axis=0)
 
# ── MÉTRICAS ────────────────────────────────────────────────────────────
def r2(y_r, y_p):
    ss_res = np.sum((y_r - y_p)**2)
    ss_tot = np.sum((y_r - y_r.mean())**2)
    return float(1 - ss_res/ss_tot)
 
def rmse(y_r, y_p):
    return float(np.sqrt(np.mean((y_r - y_p)**2)))
 
def mae(y_r, y_p):
    return float(np.mean(np.abs(y_r - y_p)))
 
r2_train   = r2(y_train, y_pred_train)
r2_test    = r2(y_test,  y_pred_test)
rmse_train = rmse(y_train, y_pred_train)
rmse_test  = rmse(y_test,  y_pred_test)
mae_train  = mae(y_train, y_pred_train)
mae_test   = mae(y_test,  y_pred_test)
 
# OOB R²
mascara_oob_valida = ~np.isnan(oob_preds)
r2_oob  = r2(y_train[mascara_oob_valida], oob_preds[mascara_oob_valida])
rmse_oob = rmse(y_train[mascara_oob_valida], oob_preds[mascara_oob_valida])
mae_oob  = mae(y_train[mascara_oob_valida], oob_preds[mascara_oob_valida])
 
print("\n" + "-" * 60)
print("  MÉTRICAS DEL MODELO")
print("-" * 60)
print(f"  {'Métrica':<22} {'Train':>10} {'OOB':>10} {'Test':>10}")
print(f"  {'-'*52}")
print(f"  {'R²':<22} {r2_train:>10.4f} {r2_oob:>10.4f} {r2_test:>10.4f}")
print(f"  {'RMSE (µg/m³)':<22} {rmse_train:>10.3f} {rmse_oob:>10.3f} {rmse_test:>10.3f}")
print(f"  {'MAE  (µg/m³)':<22} {mae_train:>10.3f} {mae_oob:>10.3f} {mae_test:>10.3f}")
 
# ── IMPORTANCIA DE VARIABLES ───────────────────────────────────────────
imp_normalizada = importancias_total / importancias_total.sum()
nombres_imp = [NOMBRES[f] for f in FEATURES]
orden = np.argsort(imp_normalizada)[::-1]
 
print("\n" + "-" * 60)
print("  IMPORTANCIA DE VARIABLES (reducción de varianza normalizada)")
print("-" * 60)
for i in orden:
    barra = "█" * int(imp_normalizada[i] * 50)
    print(f"  {nombres_imp[i]:<25} {imp_normalizada[i]:.4f}  {barra}")
 
# ═══════════════════════════════════════════════════════════════════
# GRÁFICOS
# ═══════════════════════════════════════════════════════════════════
 
fechas_test = df["fecha"].iloc[n_train:n_train+n_test].values
 
# ── G1: Predicciones vs real ────────────────────────────────────────
print("\n[G1] Predicciones vs valores reales...")
 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle("Modelo 2 — Random Forest\n"
             "Predicciones vs valores reales de NO₂ · Conjunto de test",
             fontsize=13, fontweight="bold")
 
ax1.plot(fechas_test, y_test,      color="#E53935", linewidth=1.2, alpha=0.8, label="NO₂ real")
ax1.plot(fechas_test, y_pred_test, color="#43A047", linewidth=1.2, alpha=0.8,
         linestyle="--", label="NO₂ predicho (RF)")
ax1.set_ylabel("NO₂ (µg/m³)", fontsize=10)
ax1.set_title(f"Serie temporal: real vs predicho  |  R²={r2_test:.3f}  RMSE={rmse_test:.2f} µg/m³  MAE={mae_test:.2f} µg/m³",
              fontsize=10, fontweight="bold")
ax1.legend(fontsize=9)
 
residuos_test = y_test - y_pred_test
ax2.bar(fechas_test, residuos_test,
        color=["#E53935" if r > 0 else "#43A047" for r in residuos_test],
        alpha=0.5, width=1)
ax2.axhline(0, color="black", linewidth=1)
ax2.axhline( 2*float(np.std(residuos_test)), color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax2.axhline(-2*float(np.std(residuos_test)), color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax2.set_ylabel("Residuo (µg/m³)", fontsize=10)
ax2.set_xlabel("Fecha", fontsize=10)
ax2.set_title("Residuos (NO₂ real − predicho)", fontsize=10, fontweight="bold")
 
plt.tight_layout()
plt.savefig(CARPETA + "M2_01_predicciones_vs_real.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ M2_01_predicciones_vs_real.png")
 
# ── G2: Scatter real vs predicho ────────────────────────────────────
print("[G2] Scatter real vs predicho...")
 
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Modelo 2 — Random Forest\n"
             "Dispersión: valores reales vs predichos",
             fontsize=13, fontweight="bold")
 
for ax, y_r, y_p, titulo, color in zip(
    axes,
    [y_train, y_test],
    [y_pred_train, y_pred_test],
    ["Entrenamiento", "Test"],
    ["#43A047", "#E53935"]
):
    r2v    = r2(y_r, y_p)
    rmsev  = rmse(y_r, y_p)
    maev   = mae(y_r, y_p)
    ax.scatter(y_r, y_p, alpha=0.3, s=12, color=color)
    lim = [min(y_r.min(), y_p.min()) - 2, max(y_r.max(), y_p.max()) + 2]
    ax.plot(lim, lim, "k--", linewidth=1.5, alpha=0.7, label="Predicción perfecta")
    ax.set_xlabel("NO₂ real (µg/m³)", fontsize=10)
    ax.set_ylabel("NO₂ predicho (µg/m³)", fontsize=10)
    ax.set_title(f"{titulo}\nR²={r2v:.3f}  RMSE={rmsev:.2f}  MAE={maev:.2f} µg/m³",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.legend(fontsize=8)
 
plt.tight_layout()
plt.savefig(CARPETA + "M2_02_scatter_real_vs_predicho.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ M2_02_scatter_real_vs_predicho.png")
 
# ── G3: Importancia de variables ─────────────────────────────────────
print("[G3] Importancia de variables...")
 
nombres_ord = [nombres_imp[i] for i in orden]
imp_ord     = [float(imp_normalizada[i]) for i in orden]
COLORES_IMP = {"Tráfico (veh/h)": "#E53935",
               "Temp. máxima (°C)": "#1E88E5",
               "Temp. mínima (°C)": "#1565C0",
               "Humedad relativa (%)": "#00897B",
               "Velocidad viento (m/s)": "#0288D1",
               "Precipitación (mm)": "#26A69A",
               "Mes del año": "#FB8C00",
               "Día de la semana": "#F4511E"}
colores_ord = [COLORES_IMP.get(n, "#888888") for n in nombres_ord]
 
fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(nombres_ord, imp_ord, color=colores_ord, alpha=0.85, edgecolor="white")
for bar, val in zip(bars, imp_ord):
    ax.text(float(bar.get_width()) + 0.003,
            float(bar.get_y()) + float(bar.get_height())/2,
            f"{val:.3f} ({val*100:.1f}%)",
            va="center", fontsize=9, fontweight="bold")
ax.set_xlabel("Importancia (reducción de varianza normalizada)", fontsize=11)
ax.set_title("Importancia de variables — Random Forest\n"
             "Contribución de cada variable a la reducción de varianza del NO₂",
             fontsize=12, fontweight="bold")
 
# Destacar el tráfico
idx_trafico = nombres_ord.index("Tráfico (veh/h)") if "Tráfico (veh/h)" in nombres_ord else -1
if idx_trafico >= 0:
    ax.get_children()[idx_trafico].set_edgecolor("darkred")
    ax.get_children()[idx_trafico].set_linewidth(2)
 
plt.tight_layout()
plt.savefig(CARPETA + "M2_03_importancia_variables.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ M2_03_importancia_variables.png")
 
# ── G4: Análisis de residuos ─────────────────────────────────────────
print("[G4] Análisis de residuos...")
 
residuos_train_rf = y_train - y_pred_train
 
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Modelo 2 — Random Forest — Análisis de residuos (conjunto de test)",
             fontsize=13, fontweight="bold")
 
axes[0].hist(residuos_test, bins=40, color="#43A047", alpha=0.8, edgecolor="white")
axes[0].axvline(0, color="red", linewidth=2, linestyle="--")
axes[0].axvline(float(np.mean(residuos_test)), color="orange", linewidth=1.5,
                label=f"Media residuos: {float(np.mean(residuos_test)):.2f}")
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
 
residuos_sorted = np.sort(residuos_test)
n_res = len(residuos_sorted)
teoricos = np.array([float(np.percentile(np.random.randn(10000),
                    100*(i+0.5)/n_res)) for i in range(n_res)])
axes[2].scatter(teoricos, residuos_sorted, alpha=0.3, s=10, color="#43A047")
lim_qq = max(abs(teoricos.min()), abs(teoricos.max()))
lim_rr = max(abs(residuos_sorted.min()), abs(residuos_sorted.max()))
axes[2].plot([-lim_qq, lim_qq],
             [-lim_qq*lim_rr/lim_qq, lim_qq*lim_rr/lim_qq],
             "k--", linewidth=1.5, alpha=0.7)
axes[2].set_xlabel("Cuantiles teóricos", fontsize=10)
axes[2].set_ylabel("Cuantiles de residuos", fontsize=10)
axes[2].set_title("Q-Q Plot", fontsize=10, fontweight="bold")
 
plt.tight_layout()
plt.savefig(CARPETA + "M2_04_analisis_residuos.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ M2_04_analisis_residuos.png")
 
# ── G5: Comparativa Modelo 1 vs Modelo 2 ────────────────────────────
print("[G5] Comparativa M1 vs M2...")
 
# Cargar predicciones del Modelo 1 (OLS) para comparar
# Recalculamos OLS rápido
def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]
 
X_ols = np.column_stack([np.ones(n_train), X_train])
beta  = ols(X_ols, y_train)
y_pred_ols_test = np.column_stack([np.ones(n_test), X_test]) @ beta
 
metricas = {
    "Regresión Lineal": {
        "R²":   r2(y_test, y_pred_ols_test),
        "RMSE": rmse(y_test, y_pred_ols_test),
        "MAE":  mae(y_test, y_pred_ols_test),
        "color": "#1E88E5"
    },
    "Random Forest": {
        "R²":   r2_test,
        "RMSE": rmse_test,
        "MAE":  mae_test,
        "color": "#43A047"
    }
}
 
fig, axes = plt.subplots(1, 3, figsize=(14, 6))
fig.suptitle("Comparativa de modelos — Conjunto de test\n"
             "Regresión Lineal Múltiple vs Random Forest",
             fontsize=13, fontweight="bold")
 
for ax, metrica in zip(axes, ["R²", "RMSE", "MAE"]):
    modelos  = list(metricas.keys())
    valores  = [metricas[m][metrica] for m in modelos]
    colores  = [metricas[m]["color"] for m in modelos]
    bars = ax.bar(modelos, valores, color=colores, alpha=0.85, edgecolor="white", width=0.5)
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width()/2,
                float(bar.get_height()) + max(valores)*0.02,
                f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")
    unidad = "" if metrica == "R²" else " (µg/m³)"
    ax.set_title(f"{metrica}{unidad}", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(valores) * 1.25)
    ax.tick_params(axis="x", labelsize=10)
 
plt.tight_layout()
plt.savefig(CARPETA + "M2_05_comparativa_modelos.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ M2_05_comparativa_modelos.png")
 
# ─────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESUMEN FINAL — MODELO 2")
print("=" * 65)
print(f"\n  Árboles entrenados   : {N_ARBOLES}")
print(f"  Profundidad máxima   : {PROFUNDIDAD_MAX}")
print(f"\n  MÉTRICAS (conjunto de test):")
print(f"    R²   : {r2_test:.4f}")
print(f"    RMSE : {rmse_test:.3f} µg/m³")
print(f"    MAE  : {mae_test:.3f} µg/m³")
print(f"    R² OOB (estimación interna): {r2_oob:.4f}")
print(f"\n  IMPORTANCIA DE VARIABLES (top 3):")
for i in orden[:3]:
    print(f"    {nombres_imp[i]:<25} : {imp_normalizada[i]:.3f} ({imp_normalizada[i]*100:.1f}%)")
print(f"\n  IMPORTANCIA DEL TRÁFICO:")
idx_traf = list(FEATURES).index("trafico_medio")
print(f"    {imp_normalizada[idx_traf]:.4f} ({imp_normalizada[idx_traf]*100:.1f}% de la varianza explicada)")
print(f"\n  COMPARATIVA CON MODELO 1 (test):")
print(f"    {'Métrica':<10} {'Regresión':>12} {'Random Forest':>15} {'Mejora':>10}")
print(f"    {'-'*50}")
for met in ["R²","RMSE","MAE"]:
    v1 = metricas["Regresión Lineal"][met]
    v2 = metricas["Random Forest"][met]
    if met == "R²":
        mejora = f"+{v2-v1:.4f}"
    else:
        mejora = f"{v2-v1:+.3f} µg/m³"
    print(f"    {met:<10} {v1:>12.4f} {v2:>15.4f} {mejora:>10}")
print(f"\n  Gráficos en: {CARPETA}")
print("\nFIN — Pega el output en el chat.")