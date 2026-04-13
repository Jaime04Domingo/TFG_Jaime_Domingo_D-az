"""
Análisis del dataset de Clima (formato NASA POWER) - TFG
"""
 
import pandas as pd
import numpy as np
 
RUTA = "/Users/jaime/Documents/Universidad/TFG/Clima.csv"
 
print("=" * 65)
print("  ANÁLISIS CLIMA - NASA POWER")
print("=" * 65)
 
# ── 1. LEER CABECERA PARA ENTENDER LA ESTRUCTURA ─────────────────────
print("\n[1] CABECERA DEL ARCHIVO")
print("-" * 55)
cabecera = []
linea_inicio_datos = 0
with open(RUTA, "r", encoding="utf-8") as f:
    for i, linea in enumerate(f):
        cabecera.append(linea.strip())
        print(f"  {i:>3}: {linea.rstrip()}")
        if "-END HEADER-" in linea:
            linea_inicio_datos = i + 1
            break
 
print(f"\n  → Datos empiezan en la línea: {linea_inicio_datos}")
 
# ── 2. CARGAR LOS DATOS SALTANDO LA CABECERA ─────────────────────────
print("\n[2] CARGANDO DATOS")
print("-" * 55)
df = pd.read_csv(RUTA, skiprows=linea_inicio_datos, encoding="utf-8")
# NASA POWER usa espacios múltiples como separador en algunos formatos
if df.shape[1] == 1:
    df = pd.read_csv(RUTA, skiprows=linea_inicio_datos,
                     sep=r"\s+", encoding="utf-8")
 
print(f"  Filas: {len(df):,} | Columnas: {df.shape[1]}")
print(f"  Columnas: {list(df.columns)}")
 
# ── 3. PRIMERAS Y ÚLTIMAS FILAS ──────────────────────────────────────
print("\n[3] PRIMERAS 5 FILAS")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(df.head(5).to_string())
 
print("\n  ÚLTIMAS 3 FILAS")
print(df.tail(3).to_string())
 
# ── 4. CONSTRUIR FECHA ───────────────────────────────────────────────
print("\n[4] CONSTRUYENDO FECHA")
print("-" * 55)
# NASA POWER suele tener columnas YEAR, MO (mes), DY (día)
col_year = next((c for c in df.columns if c.upper() in ["YEAR","AÑO","ANO"]), None)
col_mo   = next((c for c in df.columns if c.upper() in ["MO","MES","MONTH","MM"]), None)
col_dy   = next((c for c in df.columns if c.upper() in ["DY","DIA","DAY","DD"]), None)
print(f"  Año: '{col_year}' | Mes: '{col_mo}' | Día: '{col_dy}'")
 
if col_year and col_mo and col_dy:
    df["fecha"] = pd.to_datetime(
        df.rename(columns={col_year:"year", col_mo:"month", col_dy:"day"})[["year","month","day"]],
        errors="coerce")
    print(f"  Rango: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    print(f"  Total días: {df['fecha'].notna().sum():,}")
    nulos_fecha = df["fecha"].isnull().sum()
    if nulos_fecha:
        print(f"  ⚠ Fechas no parseadas: {nulos_fecha}")
 
# ── 5. COBERTURA TEMPORAL ────────────────────────────────────────────
print("\n[5] COBERTURA POR AÑO")
print("-" * 55)
if "fecha" in df.columns:
    df["anyo"] = df["fecha"].dt.year
    df["mes"]  = df["fecha"].dt.month
    cob = df.groupby(["anyo","mes"]).size().unstack(fill_value=0)
    print(cob.to_string())
 
# ── 6. VARIABLES DISPONIBLES ─────────────────────────────────────────
print("\n[6] VARIABLES CLIMÁTICAS DISPONIBLES")
print("-" * 55)
cols_excluir = [col_year, col_mo, col_dy, "fecha", "anyo", "mes"]
cols_vars = [c for c in df.columns if c not in cols_excluir and c is not None]
 
# Reemplazar -999 (valor missing de NASA POWER) por NaN
for col in cols_vars:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    n_missing = (df[col] == -999).sum()
    if n_missing > 0:
        print(f"  ⚠ '{col}': {n_missing} valores -999 → sustituidos por NaN")
        df.loc[df[col] == -999, col] = np.nan
 
print(f"\n  Variables detectadas: {cols_vars}")
 
# ── 7. ESTADÍSTICAS DE VARIABLES ─────────────────────────────────────
print("\n[7] ESTADÍSTICAS")
print("-" * 55)
if cols_vars:
    print(df[cols_vars].describe().round(3).to_string())
 
# ── 8. VALORES ANÓMALOS ──────────────────────────────────────────────
print("\n[8] REVISIÓN DE VALORES")
print("-" * 55)
for col in cols_vars:
    vals = df[col].dropna()
    if len(vals) == 0:
        continue
    print(f"  {col:<25} | min={vals.min():>10.3f} | max={vals.max():>10.3f} | "
          f"media={vals.mean():>8.3f} | nulos={df[col].isnull().sum()}")
 
# ── 9. DUPLICADOS ────────────────────────────────────────────────────
print("\n[9] DUPLICADOS")
dups = df.duplicated().sum()
print(f"  Filas duplicadas: {dups}")
 
# ── 10. COMPATIBILIDAD CON CONTAMINACIÓN ─────────────────────────────
print("\n[10] COMPATIBILIDAD CON DATASET DE CONTAMINACIÓN (2021-2025)")
print("-" * 55)
if "fecha" in df.columns:
    años_clima = sorted(df["anyo"].dropna().unique().astype(int))
    años_contam = [2021, 2022, 2023, 2024, 2025]
    coinciden = [a for a in años_clima if a in años_contam]
    faltan = [a for a in años_contam if a not in años_clima]
    print(f"  Años en clima     : {años_clima}")
    print(f"  Años en contaminación: {años_contam}")
    print(f"  Años en común     : {coinciden}")
    if faltan:
        print(f"  ⚠ Años que faltan : {faltan}")
    else:
        print(f"  ✓ Cobertura completa — todos los años de contaminación cubiertos")
 
print("\n" + "=" * 65)
print("  FIN — Pega el output en el chat.")
print("=" * 65)
