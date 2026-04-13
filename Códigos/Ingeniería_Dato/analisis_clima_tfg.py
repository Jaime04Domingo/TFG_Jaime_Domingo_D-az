"""
Análisis de la base de datos de Clima - TFG
Comunidad de Madrid
"""

import pandas as pd
import numpy as np

RUTA = "/Users/jaime/Documents/Universidad/TFG/Clima.csv"

print("=" * 65)
print("  ANÁLISIS BASE DE DATOS CLIMA - TFG")
print("=" * 65)

# ── 1. CARGA ─────────────────────────────────────────────────────────
print("\n[1] Cargando archivo...")
df = None
for enc in ["utf-8-sig", "utf-8", "latin-1"]:
    for sep in [";", ","]:
        try:
            tmp = pd.read_csv(RUTA, sep=sep, encoding=enc, low_memory=False)
            if tmp.shape[1] > 1:
                df = tmp
                print(f"  ✓ Encoding: {enc} | Separador: '{sep}'")
                break
        except Exception:
            continue
    if df is not None:
        break

if df is None:
    print("  ⚠ No se pudo cargar.")
    exit()

print(f"  Filas: {len(df):,} | Columnas: {df.shape[1]}")

# ── 2. COLUMNAS Y NULOS ──────────────────────────────────────────────
print("\n[2] COLUMNAS, TIPOS Y NULOS")
print("-" * 55)
for col in df.columns:
    n = df[col].isnull().sum()
    pct = n / len(df) * 100
    print(f"  {col:<30} | {str(df[col].dtype):<12} | nulos: {n:,} ({pct:.1f}%)")

# ── 3. PRIMERAS FILAS ────────────────────────────────────────────────
print("\n[3] PRIMERAS 5 FILAS")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 20)
print(df.head(5).to_string())

# ── 4. FECHAS ────────────────────────────────────────────────────────
print("\n[4] ANÁLISIS DE FECHAS")
print("-" * 55)
cols_fecha = [c for c in df.columns if any(k in c.lower() for k in
              ["fecha", "date", "año", "anyo", "year", "dia", "mes", "time"])]
print(f"  Columnas candidatas: {cols_fecha}")

col_fecha_ok = None
for cf in cols_fecha:
    parsed = pd.to_datetime(df[cf], errors="coerce", dayfirst=True)
    pct_ok = parsed.notna().sum() / len(df) * 100
    muestra = df[cf].dropna().unique()[:4]
    print(f"\n  '{cf}' → {pct_ok:.1f}% parseado")
    print(f"    Muestra valores: {list(muestra)}")
    if pct_ok > 80 and col_fecha_ok is None:
        col_fecha_ok = cf
        df["_fecha"] = parsed

if col_fecha_ok:
    print(f"\n  ✓ Fecha principal: '{col_fecha_ok}'")
    print(f"  Rango: {df['_fecha'].min()} → {df['_fecha'].max()}")
    print(f"  Años: {sorted(df['_fecha'].dt.year.dropna().unique().astype(int))}")
else:
    # Intentar construir fecha desde columnas separadas
    print("\n  Intentando construir fecha desde columnas separadas...")
    col_anyo = next((c for c in df.columns if c.upper() in ["ANO","AÑO","YEAR","ANYO"]), None)
    col_mes  = next((c for c in df.columns if c.upper() in ["MES","MONTH"]), None)
    col_dia  = next((c for c in df.columns if c.upper() in ["DIA","DÍA","DAY"]), None)
    print(f"  Año: '{col_anyo}' | Mes: '{col_mes}' | Día: '{col_dia}'")
    if col_anyo and col_mes and col_dia:
        df["_fecha"] = pd.to_datetime(
            df[[col_anyo, col_mes, col_dia]].rename(
                columns={col_anyo:"year", col_mes:"month", col_dia:"day"}),
            errors="coerce")
        print(f"  Rango: {df['_fecha'].min()} → {df['_fecha'].max()}")
        print(f"  Años: {sorted(df['_fecha'].dt.year.dropna().unique().astype(int))}")

# ── 5. COBERTURA TEMPORAL ────────────────────────────────────────────
print("\n[5] COBERTURA POR AÑO Y MES")
print("-" * 55)
if "_fecha" in df.columns:
    df["_anyo"] = df["_fecha"].dt.year
    df["_mes"]  = df["_fecha"].dt.month
    cob = df.groupby(["_anyo","_mes"]).size().unstack(fill_value=0)
    print(cob.to_string())
    print("\n  Meses con menos de 10 registros:")
    hay_huecos = False
    for anyo in cob.index:
        for mes in cob.columns:
            if cob.loc[anyo, mes] < 10:
                print(f"    ⚠ {int(anyo)}-{int(mes):02d}: {cob.loc[anyo, mes]} registros")
                hay_huecos = True
    if not hay_huecos:
        print("    ✓ Sin huecos detectados")

# ── 6. DUPLICADOS ────────────────────────────────────────────────────
print("\n[6] DUPLICADOS")
dups = df.duplicated().sum()
print(f"  Filas 100% duplicadas: {dups:,}")

# ── 7. ESTADÍSTICAS NUMÉRICAS ────────────────────────────────────────
print("\n[7] ESTADÍSTICAS DE COLUMNAS NUMÉRICAS")
print("-" * 55)
cols_aux = [c for c in df.columns if c.startswith("_")]
num = df.drop(columns=cols_aux).select_dtypes(include="number")
if not num.empty:
    print(num.describe().round(3).to_string())
else:
    print("  No hay columnas numéricas.")

# ── 8. COLUMNAS CATEGÓRICAS ──────────────────────────────────────────
print("\n[8] COLUMNAS CATEGÓRICAS")
print("-" * 55)
for col in df.drop(columns=cols_aux).select_dtypes(include="object").columns:
    unicos = df[col].nunique()
    muestra = df[col].dropna().unique()[:6]
    print(f"\n  '{col}' → {unicos} valores únicos")
    print(f"    {list(muestra)}")

# ── 9. VARIABLES CLIMÁTICAS DETECTADAS ──────────────────────────────
print("\n[9] POSIBLES VARIABLES CLIMÁTICAS")
print("-" * 55)
keywords = ["temp", "prec", "lluv", "hum", "vien", "wind", "rain",
            "pres", "rad", "sol", "niev", "visib", "rocio", "tmax",
            "tmin", "tmed", "vel", "dir", "rafag", "evap"]
cols_clima = [c for c in df.columns if any(k in c.lower() for k in keywords)]
if cols_clima:
    print(f"  Detectadas: {cols_clima}")
    for col in cols_clima:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(df[cols_clima].describe().round(3).to_string())
else:
    print("  No se detectaron nombres estándar de variables climáticas.")
    print("  → Revisa el bloque [8] para identificarlas manualmente.")

# ── 10. VALORES ANÓMALOS ─────────────────────────────────────────────
print("\n[10] VALORES ANÓMALOS EN VARIABLES CLIMÁTICAS")
print("-" * 55)
if cols_clima:
    for col in cols_clima:
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        neg = (vals < 0).sum()
        ext = (vals > vals.quantile(0.999)).sum() if len(vals) > 100 else 0
        print(f"  '{col}': min={vals.min():.2f} max={vals.max():.2f} | "
              f"negativos={neg:,} | extremos(>p99.9)={ext:,}")

# ── 11. ESTACIONES ───────────────────────────────────────────────────
print("\n[11] ESTACIONES DE MEDICIÓN")
print("-" * 55)
cols_est = [c for c in df.columns if any(k in c.lower() for k in
            ["estaci", "station", "punto", "indicat", "cod", "id"])]
for col in cols_est[:4]:
    print(f"\n  '{col}' → {df[col].nunique()} únicos: {df[col].dropna().unique()[:8]}")

print("\n" + "=" * 65)
print("  FIN — Pega el output en el chat.")
print("=" * 65)
