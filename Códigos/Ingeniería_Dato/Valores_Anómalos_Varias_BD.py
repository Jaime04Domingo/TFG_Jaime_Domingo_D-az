#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:36:33 2026

@author: jaime
"""

#### Hecho por CHATGPT ; Ver si hay valores "malos"

# -*- coding: utf-8 -*-
"""
04_diagnostico_datos_malos.py

Objetivo:
- Analizar "datos malos" o problemáticos en las bases del TFG
- Generar tablas y gráficos útiles para la memoria

Se entiende por "datos malos" aquí:
- valores nulos/NaN
- valores negativos físicamente imposibles (cuando aplique)
- cobertura incompleta de mediciones esperadas

NO modifica los archivos originales.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. RUTAS
# =========================================================

BASE_DIR = Path("/Users/jaime/Documents/Universidad/TFG")

TRAFICO_PATH = BASE_DIR / "Trafico_Aforos_Final_2.csv"
CLIMA_PATH = BASE_DIR / "Clima_Final.csv"
CONTAM_PATH = BASE_DIR / "Contaminacion_Definitivo_2.csv"

OUT_DIR = BASE_DIR / "Diagnostico_Datos_Malos"
OUT_DIR.mkdir(exist_ok=True)

FIG_DIR = OUT_DIR / "graficos"
FIG_DIR.mkdir(exist_ok=True)

EXCEL_PATH = OUT_DIR / "diagnostico_datos_malos.xlsx"
TXT_PATH = OUT_DIR / "resumen_datos_malos.txt"

# =========================================================
# 2. CONFIG
# =========================================================

plt.rcParams["figure.figsize"] = (11, 6)

MAGNITUDS_CLAVE = ["NO2", "PM10", "O3", "PM2.5", "NO", "NOx"]

# =========================================================
# 3. UTILIDADES
# =========================================================

def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()

def log(msg, f=None):
    print(msg)
    if f:
        f.write(msg + "\n")

def pct(a, b):
    return round((a / b) * 100, 2) if b != 0 else np.nan

# =========================================================
# 4. CARGA
# =========================================================

def load_trafico():
    df = pd.read_csv(TRAFICO_PATH, sep=";", encoding="utf-8")
    df = clean_columns(df)
    df["FDIA"] = pd.to_datetime(df["FDIA"], dayfirst=True, errors="coerce")
    df["anio"] = df["FDIA"].dt.year

    hor_cols = [f"HOR{i}" for i in range(1, 13)]
    for c in hor_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def load_clima():
    df = pd.read_csv(CLIMA_PATH, sep=";", encoding="utf-8")
    df = clean_columns(df)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["anio"] = df["fecha"].dt.year

    for c in df.columns:
        if c not in ["fecha"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def load_contam():
    df = pd.read_csv(CONTAM_PATH, sep=";", encoding="utf-8")
    df = clean_columns(df)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["anio"] = df["fecha"].dt.year
    df["ESTACION"] = pd.to_numeric(df["ESTACION"], errors="coerce")
    df["MAGNITUD"] = pd.to_numeric(df["MAGNITUD"], errors="coerce")

    hcols = [f"H{str(i).zfill(2)}" for i in range(1, 25)]
    for c in hcols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# =========================================================
# 5. TRÁFICO
# =========================================================

def analizar_trafico(df, writer, txt):
    log("\n" + "="*90, txt)
    log("TRÁFICO — DIAGNÓSTICO DE DATOS MALOS", txt)
    log("="*90, txt)

    hor_cols = [f"HOR{i}" for i in range(1, 13)]

    total_celdas = df[hor_cols].size
    n_nulos = int(df[hor_cols].isna().sum().sum())
    n_negativos = int((df[hor_cols] < 0).sum().sum())
    n_ceros = int((df[hor_cols] == 0).sum().sum())

    resumen = pd.DataFrame([{
        "dataset": "trafico",
        "filas": len(df),
        "columnas": df.shape[1],
        "celdas_horarias": total_celdas,
        "nulos_horarios": n_nulos,
        "pct_nulos_horarios": pct(n_nulos, total_celdas),
        "negativos_horarios": n_negativos,
        "pct_negativos_horarios": pct(n_negativos, total_celdas),
        "ceros_horarios": n_ceros,
        "pct_ceros_horarios": pct(n_ceros, total_celdas),
    }])

    log(resumen.to_string(index=False), txt)
    resumen.to_excel(writer, sheet_name="trafico_resumen", index=False)

    # Cobertura por estación y año
    traf_cov = (
        df.groupby(["nombre_estacion", "anio"])[hor_cols]
        .apply(lambda x: x.notna().sum().sum())
        .reset_index(name="medidas_validas")
    )

    traf_exp = (
        df.groupby(["nombre_estacion", "anio"])[hor_cols]
        .size()
        .reset_index(name="n_filas")
    )
    traf_exp["medidas_esperadas"] = traf_exp["n_filas"] * len(hor_cols)

    traf_cov = traf_cov.merge(traf_exp[["nombre_estacion", "anio", "medidas_esperadas"]],
                              on=["nombre_estacion", "anio"], how="left")
    traf_cov["pct_validas"] = (traf_cov["medidas_validas"] / traf_cov["medidas_esperadas"] * 100).round(2)
    traf_cov["pct_malas"] = (100 - traf_cov["pct_validas"]).round(2)

    traf_cov.to_excel(writer, sheet_name="trafico_cobertura", index=False)

    heat = traf_cov.pivot(index="nombre_estacion", columns="anio", values="pct_malas").sort_index()

    plt.figure(figsize=(8, 10))
    plt.imshow(heat.fillna(0), aspect="auto")
    plt.xticks(range(len(heat.columns)), heat.columns)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.colorbar(label="% de datos malos (faltantes)")
    plt.title("Tráfico — % de datos malos por estación y año")
    savefig("01_trafico_heatmap_datos_malos.png")

    # Ranking estaciones peores
    ranking = (
        traf_cov.groupby("nombre_estacion", as_index=False)["pct_malas"]
        .mean()
        .sort_values("pct_malas", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(ranking["nombre_estacion"], ranking["pct_malas"])
    plt.gca().invert_yaxis()
    plt.title("Tráfico — estaciones con mayor % medio de datos malos")
    plt.xlabel("% medio de datos malos")
    savefig("02_trafico_ranking_peores_estaciones.png")

    return resumen, traf_cov

# =========================================================
# 6. CLIMA
# =========================================================

def analizar_clima(df, writer, txt):
    log("\n" + "="*90, txt)
    log("CLIMA — DIAGNÓSTICO DE DATOS MALOS", txt)
    log("="*90, txt)

    var_cols = [c for c in df.columns if c not in ["fecha", "anio"]]

    rows = []
    for c in var_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append({
            "variable": c,
            "nulos": int(s.isna().sum()),
            "pct_nulos": pct(int(s.isna().sum()), len(s)),
            "negativos": int((s < 0).sum()),
            "pct_negativos": pct(int((s < 0).sum()), len(s)),
            "min": s.min(),
            "max": s.max(),
        })

    resumen = pd.DataFrame(rows).sort_values("pct_nulos", ascending=False)
    log(resumen.to_string(index=False), txt)
    resumen.to_excel(writer, sheet_name="clima_resumen", index=False)

    # Solo nulos por variable
    plt.figure(figsize=(10, 5))
    plt.bar(resumen["variable"], resumen["pct_nulos"])
    plt.xticks(rotation=45)
    plt.title("Clima — % de valores nulos por variable")
    plt.ylabel("% nulos")
    savefig("03_clima_pct_nulos_variables.png")

    # Matriz variable-año
    clima_anio = []
    for anio in sorted(df["anio"].dropna().unique()):
        sub = df[df["anio"] == anio]
        for c in var_cols:
            nulos = sub[c].isna().sum()
            clima_anio.append({
                "anio": anio,
                "variable": c,
                "pct_malos": pct(nulos, len(sub))
            })

    clima_anio = pd.DataFrame(clima_anio)
    clima_anio.to_excel(writer, sheet_name="clima_por_anio", index=False)

    heat = clima_anio.pivot(index="variable", columns="anio", values="pct_malos")

    plt.figure(figsize=(8, 5))
    plt.imshow(heat.fillna(0), aspect="auto")
    plt.xticks(range(len(heat.columns)), heat.columns)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.colorbar(label="% de datos malos")
    plt.title("Clima — % de datos malos por variable y año")
    savefig("04_clima_heatmap_datos_malos.png")

    return resumen, clima_anio

# =========================================================
# 7. CONTAMINACIÓN
# =========================================================

def analizar_contaminacion(df, writer, txt):
    log("\n" + "="*90, txt)
    log("CONTAMINACIÓN — DIAGNÓSTICO DE DATOS MALOS", txt)
    log("="*90, txt)

    hcols = [f"H{str(i).zfill(2)}" for i in range(1, 25)]

    total_celdas = df[hcols].size
    n_nulos = int(df[hcols].isna().sum().sum())
    n_negativos = int((df[hcols] < 0).sum().sum())

    resumen = pd.DataFrame([{
        "dataset": "contaminacion",
        "filas": len(df),
        "columnas": df.shape[1],
        "celdas_horarias": total_celdas,
        "nulos_horarios": n_nulos,
        "pct_nulos_horarios": pct(n_nulos, total_celdas),
        "negativos_horarios": n_negativos,
        "pct_negativos_horarios": pct(n_negativos, total_celdas),
    }])

    log(resumen.to_string(index=False), txt)
    resumen.to_excel(writer, sheet_name="contam_resumen", index=False)

    # Cobertura por contaminante
    por_cont = (
        df.groupby("abreviatura_contaminante")[hcols]
        .apply(lambda x: x.notna().sum().sum())
        .reset_index(name="validas")
    )

    exp_cont = (
        df.groupby("abreviatura_contaminante")[hcols]
        .size()
        .reset_index(name="n_filas")
    )
    exp_cont["esperadas"] = exp_cont["n_filas"] * len(hcols)

    por_cont = por_cont.merge(exp_cont[["abreviatura_contaminante", "esperadas"]], on="abreviatura_contaminante")
    por_cont["pct_validas"] = (por_cont["validas"] / por_cont["esperadas"] * 100).round(2)
    por_cont["pct_malas"] = (100 - por_cont["pct_validas"]).round(2)
    por_cont = por_cont.sort_values("pct_malas", ascending=False)

    por_cont.to_excel(writer, sheet_name="contam_por_contaminante", index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(por_cont["abreviatura_contaminante"], por_cont["pct_malas"])
    plt.gca().invert_yaxis()
    plt.title("Contaminación — % de datos malos por contaminante")
    plt.xlabel("% de datos malos")
    savefig("05_contam_pct_malos_por_contaminante.png")

    # Heatmap por estación y año para NO2 (útil para memoria)
    sub_no2 = df[df["abreviatura_contaminante"] == "NO2"].copy()

    no2_cov = (
        sub_no2.groupby(["ESTACION", "anio"])[hcols]
        .apply(lambda x: x.notna().sum().sum())
        .reset_index(name="validas")
    )

    no2_exp = (
        sub_no2.groupby(["ESTACION", "anio"])[hcols]
        .size()
        .reset_index(name="n_filas")
    )
    no2_exp["esperadas"] = no2_exp["n_filas"] * len(hcols)

    no2_cov = no2_cov.merge(no2_exp[["ESTACION", "anio", "esperadas"]], on=["ESTACION", "anio"], how="left")
    no2_cov["pct_validas"] = (no2_cov["validas"] / no2_cov["esperadas"] * 100).round(2)
    no2_cov["pct_malas"] = (100 - no2_cov["pct_validas"]).round(2)

    # Si tienes nombre de estación en contaminación, lo usa; si no, usa código
    if "nombre_estacion" in sub_no2.columns:
        names = sub_no2[["ESTACION", "nombre_estacion"]].drop_duplicates()
        no2_cov = no2_cov.merge(names, on="ESTACION", how="left")
        idx_col = "nombre_estacion"
    else:
        idx_col = "ESTACION"

    no2_cov.to_excel(writer, sheet_name="contam_NO2_est_anio", index=False)

    heat = no2_cov.pivot(index=idx_col, columns="anio", values="pct_malas")

    plt.figure(figsize=(8, 10))
    plt.imshow(heat.fillna(0), aspect="auto")
    plt.xticks(range(len(heat.columns)), heat.columns)
    plt.yticks(range(len(heat.index)), heat.index)
    plt.colorbar(label="% de datos malos de NO2")
    plt.title("NO2 — % de datos malos por estación y año")
    savefig("06_no2_heatmap_datos_malos.png")

    # Ranking estaciones peores para NO2
    ranking = (
        no2_cov.groupby(idx_col, as_index=False)["pct_malas"]
        .mean()
        .sort_values("pct_malas", ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(ranking[idx_col].astype(str), ranking["pct_malas"])
    plt.gca().invert_yaxis()
    plt.title("NO2 — estaciones con mayor % medio de datos malos")
    plt.xlabel("% medio de datos malos")
    savefig("07_no2_ranking_peores_estaciones.png")

    return resumen, por_cont, no2_cov

# =========================================================
# 8. RESUMEN GLOBAL
# =========================================================

def resumen_global(traf_cov, clima_res, cont_por_cont, writer):
    # Tabla compacta útil para memoria
    filas = []

    filas.append({
        "base": "Tráfico",
        "criterio": "Celdas horarias faltantes",
        "valor_medio_pct": round(traf_cov["pct_malas"].mean(), 2),
        "valor_max_pct": round(traf_cov["pct_malas"].max(), 2),
    })

    filas.append({
        "base": "Clima",
        "criterio": "% nulos por variable",
        "valor_medio_pct": round(clima_res["pct_nulos"].mean(), 2),
        "valor_max_pct": round(clima_res["pct_nulos"].max(), 2),
    })

    filas.append({
        "base": "Contaminación",
        "criterio": "% datos malos por contaminante",
        "valor_medio_pct": round(cont_por_cont["pct_malas"].mean(), 2),
        "valor_max_pct": round(cont_por_cont["pct_malas"].max(), 2),
    })

    resumen = pd.DataFrame(filas)
    resumen.to_excel(writer, sheet_name="resumen_global", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(resumen["base"], resumen["valor_medio_pct"])
    plt.title("Resumen comparado del % medio de datos malos por base")
    plt.ylabel("% medio")
    savefig("08_resumen_global_pct_malos.png")

# =========================================================
# 9. MAIN
# =========================================================

def main():
    traf = load_trafico()
    clima = load_clima()
    contam = load_contam()

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        with open(TXT_PATH, "w", encoding="utf-8") as txt:
            traf_res, traf_cov = analizar_trafico(traf, writer, txt)
            clima_res, clima_anio = analizar_clima(clima, writer, txt)
            cont_res, cont_por_cont, no2_cov = analizar_contaminacion(contam, writer, txt)
            resumen_global(traf_cov, clima_res, cont_por_cont, writer)

            log("\n" + "="*90, txt)
            log("FIN DEL DIAGNÓSTICO", txt)
            log("="*90, txt)
            log(f"Excel: {EXCEL_PATH}", txt)
            log(f"TXT: {TXT_PATH}", txt)
            log(f"Gráficos: {FIG_DIR}", txt)

    print("\nProceso terminado.")
    print(f"Excel guardado en: {EXCEL_PATH}")
    print(f"TXT guardado en: {TXT_PATH}")
    print(f"Gráficos guardados en: {FIG_DIR}")

if __name__ == "__main__":
    main()
    
    
    ### Verificación
    
# -*- coding: utf-8 -*-
"""
05_verificar_diagnostico_datos_malos.py

Verifica que el script 04_diagnostico_datos_malos.py
ha generado correctamente los entregables esperados.
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path("/Users/jaime/Documents/Universidad/TFG")
OUT_DIR = BASE_DIR / "Diagnostico_Datos_Malos"
FIG_DIR = OUT_DIR / "graficos"

EXCEL_PATH = OUT_DIR / "diagnostico_datos_malos.xlsx"
TXT_PATH = OUT_DIR / "resumen_datos_malos.txt"

EXPECTED_FIGS = [
    "01_trafico_heatmap_datos_malos.png",
    "02_trafico_ranking_peores_estaciones.png",
    "03_clima_pct_nulos_variables.png",
    "04_clima_heatmap_datos_malos.png",
    "05_contam_pct_malos_por_contaminante.png",
    "06_no2_heatmap_datos_malos.png",
    "07_no2_ranking_peores_estaciones.png",
    "08_resumen_global_pct_malos.png",
]

EXPECTED_SHEETS = [
    "trafico_resumen",
    "trafico_cobertura",
    "clima_resumen",
    "clima_por_anio",
    "contam_resumen",
    "contam_por_contaminante",
    "contam_NO2_est_anio",
    "resumen_global",
]

def ok(cond, msg):
    if cond:
        print(f"[OK] {msg}")
    else:
        print(f"[ERROR] {msg}")

def main():
    print("="*80)
    print("VERIFICACIÓN DEL DIAGNÓSTICO DE DATOS MALOS")
    print("="*80)

    ok(OUT_DIR.exists(), "Existe la carpeta de salida")
    ok(FIG_DIR.exists(), "Existe la carpeta de gráficos")
    ok(EXCEL_PATH.exists(), "Existe el Excel de diagnóstico")
    ok(TXT_PATH.exists(), "Existe el TXT resumen")

    if EXCEL_PATH.exists():
        xls = pd.ExcelFile(EXCEL_PATH)
        hojas = xls.sheet_names
        print("\nHojas detectadas en el Excel:")
        for h in hojas:
            print("-", h)

        for s in EXPECTED_SHEETS:
            ok(s in hojas, f"Existe la hoja '{s}'")

    print("\nGráficos detectados:")
    for fig in EXPECTED_FIGS:
        path = FIG_DIR / fig
        ok(path.exists(), f"Existe {fig}")

    if TXT_PATH.exists():
        with open(TXT_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        ok(len(lines) > 10, "El TXT contiene contenido suficiente")
        print(f"\nNúmero de líneas en TXT: {len(lines)}")

    print("\n" + "="*80)
    print("FIN DE VERIFICACIÓN")
    print("="*80)

if __name__ == "__main__":
    main()  
    
### Creación de gráficos sobre valores anómalos en todas las bases de datos. 

# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path("/Users/jaime/Documents/Universidad/TFG")
OUT = BASE / "Graficos_Calidad_Dato_Academicos"
OUT.mkdir(exist_ok=True)

traf = pd.read_csv(BASE / "Trafico_Aforos_Final_2.csv", sep=";")
clima = pd.read_csv(BASE / "Clima_Final.csv", sep=";")
cont = pd.read_csv(BASE / "Contaminacion_Definitivo_2.csv", sep=";")

traf.columns = [str(c).strip().replace("\ufeff", "") for c in traf.columns]
clima.columns = [str(c).strip().replace("\ufeff", "") for c in clima.columns]
cont.columns = [str(c).strip().replace("\ufeff", "") for c in cont.columns]

traf["FDIA"] = pd.to_datetime(traf["FDIA"], dayfirst=True, errors="coerce")
traf["anio"] = traf["FDIA"].dt.year

clima["fecha"] = pd.to_datetime(clima["fecha"], errors="coerce")
clima["anio"] = clima["fecha"].dt.year

cont["fecha"] = pd.to_datetime(cont["fecha"], errors="coerce")
cont["anio"] = cont["fecha"].dt.year

hor_traf = [f"HOR{i}" for i in range(1, 13)]
hor_cont = [f"H{str(i).zfill(2)}" for i in range(1, 25)]

for c in hor_traf:
    traf[c] = pd.to_numeric(traf[c], errors="coerce")
for c in hor_cont:
    cont[c] = pd.to_numeric(cont[c], errors="coerce")

# -------------------------
# 1. TRÁFICO: % faltantes por estación y año
# -------------------------
t1 = (
    traf.groupby(["nombre_estacion", "anio"])[hor_traf]
    .apply(lambda x: x.isna().sum().sum() / x.size * 100)
    .reset_index(name="pct_faltantes")
)
heat1 = t1.pivot(index="nombre_estacion", columns="anio", values="pct_faltantes").fillna(0)

plt.figure(figsize=(8, 10))
plt.imshow(heat1, aspect="auto")
plt.xticks(range(len(heat1.columns)), heat1.columns)
plt.yticks(range(len(heat1.index)), heat1.index)
plt.colorbar(label="% de valores faltantes")
plt.title("Tráfico: porcentaje de valores faltantes por estación y año")
plt.tight_layout()
plt.savefig(OUT / "01_trafico_faltantes_estacion_anio.png", dpi=180)
plt.close()

# -------------------------
# 2. TRÁFICO: % faltantes por tramo horario
# -------------------------
t2 = pd.DataFrame({
    "tramo_horario": hor_traf,
    "pct_faltantes": [traf[c].isna().mean() * 100 for c in hor_traf]
})

plt.figure(figsize=(8, 4.5))
plt.bar(t2["tramo_horario"], t2["pct_faltantes"])
plt.title("Tráfico: porcentaje de valores faltantes por tramo horario")
plt.ylabel("% de valores faltantes")
plt.tight_layout()
plt.savefig(OUT / "02_trafico_faltantes_tramo_horario.png", dpi=180)
plt.close()

# -------------------------
# 3. CLIMA: % faltantes por variable
# -------------------------
vars_clima = [c for c in clima.columns if c not in ["fecha", "anio"]]
c1 = pd.DataFrame({
    "variable": vars_clima,
    "pct_faltantes": [clima[c].isna().mean() * 100 for c in vars_clima]
}).sort_values("pct_faltantes", ascending=False)

plt.figure(figsize=(9, 4.5))
plt.bar(c1["variable"], c1["pct_faltantes"])
plt.xticks(rotation=45)
plt.title("Clima: porcentaje de valores faltantes por variable")
plt.ylabel("% de valores faltantes")
plt.tight_layout()
plt.savefig(OUT / "03_clima_faltantes_variable.png", dpi=180)
plt.close()

# -------------------------
# 4. CLIMA: % faltantes por variable y año
# -------------------------
rows = []
for a in sorted(clima["anio"].dropna().unique()):
    sub = clima[clima["anio"] == a]
    for c in vars_clima:
        rows.append({
            "anio": a,
            "variable": c,
            "pct_faltantes": sub[c].isna().mean() * 100
        })
c2 = pd.DataFrame(rows)
heat2 = c2.pivot(index="variable", columns="anio", values="pct_faltantes").fillna(0)

plt.figure(figsize=(8, 5))
plt.imshow(heat2, aspect="auto")
plt.xticks(range(len(heat2.columns)), heat2.columns)
plt.yticks(range(len(heat2.index)), heat2.index)
plt.colorbar(label="% de valores faltantes")
plt.title("Clima: porcentaje de valores faltantes por variable y año")
plt.tight_layout()
plt.savefig(OUT / "04_clima_faltantes_variable_anio.png", dpi=180)
plt.close()

# -------------------------
# 5. CONTAMINACIÓN: % faltantes por contaminante
# -------------------------
k = "abreviatura_contaminante" if "abreviatura_contaminante" in cont.columns else "MAGNITUD"

p1 = (
    cont.groupby(k)[hor_cont]
    .apply(lambda x: x.isna().sum().sum() / x.size * 100)
    .reset_index(name="pct_faltantes")
    .sort_values("pct_faltantes", ascending=False)
)

plt.figure(figsize=(8, 5))
plt.barh(p1[k].astype(str), p1["pct_faltantes"])
plt.gca().invert_yaxis()
plt.title("Contaminación: porcentaje de valores faltantes por contaminante")
plt.xlabel("% de valores faltantes")
plt.tight_layout()
plt.savefig(OUT / "05_contaminacion_faltantes_contaminante.png", dpi=180)
plt.close()

# -------------------------
# 6. CONTAMINACIÓN: cobertura de NO2 por estación y año
# -------------------------
if "abreviatura_contaminante" in cont.columns:
    no2 = cont[cont["abreviatura_contaminante"] == "NO2"].copy()
else:
    no2 = cont[cont["MAGNITUD"] == 8].copy()

p2 = (
    no2.groupby(["ESTACION", "anio"])[hor_cont]
    .apply(lambda x: x.notna().sum().sum() / x.size * 100)
    .reset_index(name="pct_validos")
)

heat3 = p2.pivot(index="ESTACION", columns="anio", values="pct_validos").fillna(0)

plt.figure(figsize=(7, 8))
plt.imshow(heat3, aspect="auto", vmin=0, vmax=100)
plt.xticks(range(len(heat3.columns)), heat3.columns)
plt.yticks(range(len(heat3.index)), heat3.index)
plt.colorbar(label="% de valores válidos")
plt.title("Contaminación (NO₂): cobertura de valores válidos por estación y año")
plt.tight_layout()
plt.savefig(OUT / "06_contaminacion_no2_cobertura_estacion_anio.png", dpi=180)
plt.close()

print("Gráficos guardados en:", OUT)
for f in sorted(OUT.glob("*.png")):
    print("-", f.name)