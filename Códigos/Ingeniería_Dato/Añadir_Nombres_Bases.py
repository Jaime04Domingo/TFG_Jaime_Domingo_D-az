#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:05:16 2026

@author: jaime
"""

### Código hecho por chatgpt, se añaden las columnas sobre el nombre de los contaminantes y de las estaciones. 
# -*- coding: utf-8 -*-
"""
02_anadir_nombres_bases.py

Añade:
1) A la base de tráfico: nombre de estación
2) A la base de contaminación: nombre y abreviatura del contaminante

NO sobrescribe los originales.
Genera nuevas versiones con sufijo numérico progresivo.
"""

from pathlib import Path
import pandas as pd
import re

# =========================================================
# 1. RUTAS
# =========================================================

BASE_DIR = Path("/Users/jaime/Documents/Universidad/TFG")

TRAFICO_PATH = BASE_DIR / "Trafico_Aforos_Final.csv"
CONTAM_PATH = BASE_DIR / "Contaminacion_Definitivo.csv"
META_TRAFICO_PATH = BASE_DIR / "aforos_trafico_permanentes" / "300233-5-aforo-trafico-permanentes.csv"

# =========================================================
# 2. DICCIONARIO MAGNITUDES
# =========================================================

MAGNITUD_INFO = {
    1:   {"contaminante_nombre": "Dióxido de Azufre", "abreviatura": "SO2"},
    6:   {"contaminante_nombre": "Monóxido de Carbono", "abreviatura": "CO"},
    7:   {"contaminante_nombre": "Monóxido de Nitrógeno", "abreviatura": "NO"},
    8:   {"contaminante_nombre": "Dióxido de Nitrógeno", "abreviatura": "NO2"},
    9:   {"contaminante_nombre": "Partículas < 2.5 µm", "abreviatura": "PM2.5"},
    10:  {"contaminante_nombre": "Partículas < 10 µm", "abreviatura": "PM10"},
    12:  {"contaminante_nombre": "Óxidos de Nitrógeno", "abreviatura": "NOx"},
    14:  {"contaminante_nombre": "Ozono", "abreviatura": "O3"},
    20:  {"contaminante_nombre": "Tolueno", "abreviatura": "TOL"},
    30:  {"contaminante_nombre": "Benceno", "abreviatura": "BEN"},
    35:  {"contaminante_nombre": "Etilbenceno", "abreviatura": "EBE"},
    37:  {"contaminante_nombre": "Metaxileno", "abreviatura": "MXY"},
    38:  {"contaminante_nombre": "Paraxileno", "abreviatura": "PXY"},
    39:  {"contaminante_nombre": "Ortoxileno", "abreviatura": "OXY"},
    42:  {"contaminante_nombre": "Hidrocarburos totales (hexano)", "abreviatura": "TCH"},
    43:  {"contaminante_nombre": "Metano", "abreviatura": "CH4"},
    44:  {"contaminante_nombre": "Hidrocarburos no metánicos (hexano)", "abreviatura": "NMHC"},
    431: {"contaminante_nombre": "Metaparaxileno", "abreviatura": "MPX"},
}

# =========================================================
# 3. FUNCIONES AUXILIARES
# =========================================================

def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df

def get_next_version_path(original_path: Path) -> Path:
    """
    Si el archivo es Trafico_Aforos_Final.csv
    genera Trafico_Aforos_Final_1.csv, _2.csv, etc.
    """
    stem = original_path.stem
    suffix = original_path.suffix

    pattern = re.compile(rf"^{re.escape(stem)}_(\d+){re.escape(suffix)}$")

    existing_numbers = []
    for f in original_path.parent.iterdir():
        if f.is_file():
            m = pattern.match(f.name)
            if m:
                existing_numbers.append(int(m.group(1)))

    next_n = 1 if not existing_numbers else max(existing_numbers) + 1
    return original_path.parent / f"{stem}_{next_n}{suffix}"

# =========================================================
# 4. TRÁFICO: AÑADIR NOMBRE DE ESTACIÓN
# =========================================================

def process_traffic():
    traf = pd.read_csv(TRAFICO_PATH, sep=";", encoding="utf-8")
    traf = clean_columns(traf)

    meta = pd.read_csv(META_TRAFICO_PATH, sep=";", encoding="latin1")
    meta = clean_columns(meta)

    # Preparar claves
    traf["num_estacion"] = pd.to_numeric(
        traf["FEST"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce"
    )

    meta["Nº"] = pd.to_numeric(meta["Nº"], errors="coerce")

    # Join solo con el nombre
    meta_small = meta[["Nº", "ESTACION"]].drop_duplicates().rename(
        columns={"Nº": "num_estacion", "ESTACION": "nombre_estacion"}
    )

    traf_new = traf.merge(meta_small, on="num_estacion", how="left")

    # Opcional: colocar nombre_estacion al lado de FEST
    cols = list(traf_new.columns)
    if "FEST" in cols and "nombre_estacion" in cols:
        cols.remove("nombre_estacion")
        idx = cols.index("FEST") + 1
        cols.insert(idx, "nombre_estacion")
        traf_new = traf_new[cols]

    out_path = get_next_version_path(TRAFICO_PATH)
    traf_new.to_csv(out_path, sep=";", index=False, encoding="utf-8")

    print(f"Tráfico guardado en: {out_path}")
    print(f"Filas: {len(traf_new)} | Columnas: {traf_new.shape[1]}")
    print(f"Cobertura nombre_estacion: {traf_new['nombre_estacion'].notna().mean()*100:.2f}%")

# =========================================================
# 5. CONTAMINACIÓN: AÑADIR NOMBRE Y ABREVIATURA
# =========================================================

def process_contamination():
    cont = pd.read_csv(CONTAM_PATH, sep=";", encoding="utf-8")
    cont = clean_columns(cont)

    cont["MAGNITUD"] = pd.to_numeric(cont["MAGNITUD"], errors="coerce")

    cont["contaminante_nombre"] = cont["MAGNITUD"].map(
        lambda x: MAGNITUD_INFO.get(x, {}).get("contaminante_nombre")
    )
    cont["abreviatura_contaminante"] = cont["MAGNITUD"].map(
        lambda x: MAGNITUD_INFO.get(x, {}).get("abreviatura")
    )

    # Colocar columnas nuevas junto a MAGNITUD
    cols = list(cont.columns)
    if "MAGNITUD" in cols:
        cols.remove("contaminante_nombre")
        cols.remove("abreviatura_contaminante")
        idx = cols.index("MAGNITUD") + 1
        cols.insert(idx, "contaminante_nombre")
        cols.insert(idx + 1, "abreviatura_contaminante")
        cont = cont[cols]

    out_path = get_next_version_path(CONTAM_PATH)
    cont.to_csv(out_path, sep=";", index=False, encoding="utf-8")

    print(f"Contaminación guardada en: {out_path}")
    print(f"Filas: {len(cont)} | Columnas: {cont.shape[1]}")
    print(f"Cobertura contaminante_nombre: {cont['contaminante_nombre'].notna().mean()*100:.2f}%")

# =========================================================
# 6. MAIN
# =========================================================

def main():
    print("Procesando tráfico...")
    process_traffic()

    print("\nProcesando contaminación...")
    process_contamination()

    print("\nProceso terminado. No se han modificado los archivos originales.")

if __name__ == "__main__":
    main()
    
    
    
    ### Comprobación
    
# -*- coding: utf-8 -*-
"""
03_verificar_anadidos_bases.py

Verifica que:
1) En tráfico se ha añadido correctamente nombre_estacion
2) En contaminación se han añadido correctamente contaminante_nombre y abreviatura_contaminante

NO modifica nada.
Solo comprueba y muestra resultados.
"""

from pathlib import Path
import pandas as pd
import re

# =========================================================
# 1. RUTAS
# =========================================================

BASE_DIR = Path("/Users/jaime/Documents/Universidad/TFG")

TRAFICO_ORIGINAL = BASE_DIR / "Trafico_Aforos_Final.csv"
CONTAM_ORIGINAL = BASE_DIR / "Contaminacion_Definitivo.csv"

# =========================================================
# 2. DICCIONARIO ESPERADO
# =========================================================

MAGNITUD_INFO = {
    1:   {"contaminante_nombre": "Dióxido de Azufre", "abreviatura": "SO2"},
    6:   {"contaminante_nombre": "Monóxido de Carbono", "abreviatura": "CO"},
    7:   {"contaminante_nombre": "Monóxido de Nitrógeno", "abreviatura": "NO"},
    8:   {"contaminante_nombre": "Dióxido de Nitrógeno", "abreviatura": "NO2"},
    9:   {"contaminante_nombre": "Partículas < 2.5 µm", "abreviatura": "PM2.5"},
    10:  {"contaminante_nombre": "Partículas < 10 µm", "abreviatura": "PM10"},
    12:  {"contaminante_nombre": "Óxidos de Nitrógeno", "abreviatura": "NOx"},
    14:  {"contaminante_nombre": "Ozono", "abreviatura": "O3"},
    20:  {"contaminante_nombre": "Tolueno", "abreviatura": "TOL"},
    30:  {"contaminante_nombre": "Benceno", "abreviatura": "BEN"},
    35:  {"contaminante_nombre": "Etilbenceno", "abreviatura": "EBE"},
    37:  {"contaminante_nombre": "Metaxileno", "abreviatura": "MXY"},
    38:  {"contaminante_nombre": "Paraxileno", "abreviatura": "PXY"},
    39:  {"contaminante_nombre": "Ortoxileno", "abreviatura": "OXY"},
    42:  {"contaminante_nombre": "Hidrocarburos totales (hexano)", "abreviatura": "TCH"},
    43:  {"contaminante_nombre": "Metano", "abreviatura": "CH4"},
    44:  {"contaminante_nombre": "Hidrocarburos no metánicos (hexano)", "abreviatura": "NMHC"},
    431: {"contaminante_nombre": "Metaparaxileno", "abreviatura": "MPX"},
}

# =========================================================
# 3. FUNCIONES AUXILIARES
# =========================================================

def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df

def get_latest_version_path(original_path: Path):
    """
    Busca el archivo numerado más reciente:
    Trafico_Aforos_Final_1.csv, _2.csv, etc.
    """
    stem = original_path.stem
    suffix = original_path.suffix
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+){re.escape(suffix)}$")

    candidates = []
    for f in original_path.parent.iterdir():
        if f.is_file():
            m = pattern.match(f.name)
            if m:
                candidates.append((int(m.group(1)), f))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def ok(cond, text):
    if cond:
        print(f"[OK] {text}")
    else:
        print(f"[ERROR] {text}")

# =========================================================
# 4. VERIFICACIÓN TRÁFICO
# =========================================================

def verify_traffic():
    print("\n" + "="*80)
    print("VERIFICACIÓN TRÁFICO")
    print("="*80)

    latest = get_latest_version_path(TRAFICO_ORIGINAL)
    ok(latest is not None, "Existe una versión numerada de tráfico")

    if latest is None:
        return

    print(f"Archivo verificado: {latest.name}")

    orig = pd.read_csv(TRAFICO_ORIGINAL, sep=";", encoding="utf-8")
    new = pd.read_csv(latest, sep=";", encoding="utf-8")

    orig = clean_columns(orig)
    new = clean_columns(new)

    ok(len(orig) == len(new), f"Mismo número de filas que el original ({len(orig)})")
    ok("nombre_estacion" in new.columns, "Existe la columna nombre_estacion")
    ok("num_estacion" in new.columns, "Existe la columna num_estacion")
    ok("FEST" in new.columns, "Sigue existiendo la columna FEST")

    if "nombre_estacion" in new.columns:
        cobertura = new["nombre_estacion"].notna().mean() * 100
        print(f"Cobertura nombre_estacion: {cobertura:.2f}%")
        ok(cobertura > 95, "La cobertura de nombre_estacion es alta (>95%)")

        ejemplos = new[["FEST", "nombre_estacion"]].drop_duplicates().head(10)
        print("\nEjemplos FEST -> nombre_estacion:")
        print(ejemplos.to_string(index=False))

    if "num_estacion" in new.columns and "FEST" in new.columns:
        extraido = pd.to_numeric(new["FEST"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        iguales = (extraido.fillna(-9999) == pd.to_numeric(new["num_estacion"], errors="coerce").fillna(-9999)).mean() * 100
        print(f"\nCoincidencia FEST -> num_estacion: {iguales:.2f}%")
        ok(iguales == 100, "num_estacion coincide al 100% con el número extraído de FEST")

# =========================================================
# 5. VERIFICACIÓN CONTAMINACIÓN
# =========================================================

def verify_contamination():
    print("\n" + "="*80)
    print("VERIFICACIÓN CONTAMINACIÓN")
    print("="*80)

    latest = get_latest_version_path(CONTAM_ORIGINAL)
    ok(latest is not None, "Existe una versión numerada de contaminación")

    if latest is None:
        return

    print(f"Archivo verificado: {latest.name}")

    orig = pd.read_csv(CONTAM_ORIGINAL, sep=";", encoding="utf-8")
    new = pd.read_csv(latest, sep=";", encoding="utf-8")

    orig = clean_columns(orig)
    new = clean_columns(new)

    ok(len(orig) == len(new), f"Mismo número de filas que el original ({len(orig)})")
    ok("contaminante_nombre" in new.columns, "Existe la columna contaminante_nombre")
    ok("abreviatura_contaminante" in new.columns, "Existe la columna abreviatura_contaminante")
    ok("MAGNITUD" in new.columns, "Sigue existiendo la columna MAGNITUD")

    if "MAGNITUD" in new.columns:
        new["MAGNITUD"] = pd.to_numeric(new["MAGNITUD"], errors="coerce")

    if "contaminante_nombre" in new.columns:
        cobertura_nom = new["contaminante_nombre"].notna().mean() * 100
        print(f"Cobertura contaminante_nombre: {cobertura_nom:.2f}%")

    if "abreviatura_contaminante" in new.columns:
        cobertura_abr = new["abreviatura_contaminante"].notna().mean() * 100
        print(f"Cobertura abreviatura_contaminante: {cobertura_abr:.2f}%")

    # Verificación exacta de correspondencia
    errores = []

    for mag, info in MAGNITUD_INFO.items():
        sub = new[new["MAGNITUD"] == mag]
        if len(sub) == 0:
            continue

        nombre_ok = (sub["contaminante_nombre"] == info["contaminante_nombre"]).all()
        abr_ok = (sub["abreviatura_contaminante"] == info["abreviatura"]).all()

        if not nombre_ok or not abr_ok:
            errores.append((mag, info["contaminante_nombre"], info["abreviatura"]))

    ok(len(errores) == 0, "La traducción MAGNITUD -> nombre/abreviatura es correcta en todos los códigos presentes")

    if len(errores) > 0:
        print("\nErrores detectados en magnitudes:")
        for e in errores:
            print(e)

    ejemplos = (
        new[["MAGNITUD", "contaminante_nombre", "abreviatura_contaminante"]]
        .drop_duplicates()
        .sort_values("MAGNITUD")
        .head(20)
    )
    print("\nEjemplos MAGNITUD -> contaminante:")
    print(ejemplos.to_string(index=False))

# =========================================================
# 6. MAIN
# =========================================================

def main():
    verify_traffic()
    verify_contamination()

    print("\n" + "="*80)
    print("FIN DE VERIFICACIÓN")
    print("="*80)
    print("Si todo sale con [OK], el proceso ha funcionado correctamente.")

if __name__ == "__main__":
    main()    