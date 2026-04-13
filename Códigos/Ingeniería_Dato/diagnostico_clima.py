"""
Diagnóstico del archivo Clima.csv
"""

import pandas as pd
import sys

RUTA = "/Users/jaime/Documents/Universidad/TFG/Clima.csv"

print("=" * 65)
print("  DIAGNÓSTICO CLIMA.CSV")
print("=" * 65)

# ── 1. LEER LAS PRIMERAS LÍNEAS EN CRUDO ────────────────────────────
print("\n[1] PRIMERAS 10 LÍNEAS EN CRUDO (bytes reales)")
print("-" * 55)
for enc in ["utf-8", "latin-1", "utf-8-sig", "cp1252"]:
    try:
        with open(RUTA, "r", encoding=enc) as f:
            lineas = [f.readline() for _ in range(10)]
        print(f"\n  Encoding '{enc}' funciona:")
        for i, l in enumerate(lineas):
            print(f"    {i}: {repr(l)}")
        break
    except Exception as e:
        print(f"  Encoding '{enc}': error → {e}")

# ── 2. TAMAÑO DEL ARCHIVO ────────────────────────────────────────────
import os
tam = os.path.getsize(RUTA)
print(f"\n[2] Tamaño del archivo: {tam:,} bytes ({tam/1024/1024:.2f} MB)")

# ── 3. INTENTAR CARGA CON DISTINTOS PARÁMETROS ──────────────────────
print("\n[3] INTENTOS DE CARGA")
print("-" * 55)
intentos = [
    {"sep": ";",  "enc": "utf-8"},
    {"sep": ",",  "enc": "utf-8"},
    {"sep": ";",  "enc": "latin-1"},
    {"sep": ",",  "enc": "latin-1"},
    {"sep": "\t", "enc": "utf-8"},
    {"sep": "\t", "enc": "latin-1"},
    {"sep": ";",  "enc": "cp1252"},
    {"sep": ",",  "enc": "cp1252"},
]
for p in intentos:
    try:
        tmp = pd.read_csv(RUTA, sep=p["sep"], encoding=p["enc"],
                          nrows=3, low_memory=False)
        print(f"\n  ✓ sep='{p['sep']}' enc='{p['enc']}' → {tmp.shape[1]} columnas")
        print(f"    Columnas: {list(tmp.columns)}")
        print(tmp.to_string())
    except Exception as e:
        print(f"  ✗ sep='{p['sep']}' enc='{p['enc']}' → {e}")

print("\n" + "=" * 65)
print("  FIN — Pega el output en el chat.")
print("=" * 65)
