"""
Reprocesa los JSON de HARDWARE REAL al formato con ORDEN DE BANDA (diabático), SIN volver
a ejecutar en hardware: reordena los valores ya medidos y añade el campo 'bandas', usando
autovalores_diabaticos() y asignar_bandas() de version_1.py.

Se conserva una copia RAW en el scratchpad. Modifica en el sitio:
  hardware_agnr_2q.json  (armchair, ibm_fez)
  hardware_zgnr_2q.json  (zigzag,   ibm_kingston)
"""
import os
import sys
import json
import numpy as np

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RAIZ)
from version_1 import autovalores_diabaticos, asignar_bandas

for fn, tipo in [("hardware_agnr_2q.json", 'a'), ("hardware_zgnr_2q.json", 'z')]:
    path = os.path.join(RAIZ, fn)
    d = json.load(open(path, encoding='utf-8'))
    nq = d["parametros"].get("num_qubits", 2)
    t = d["parametros"].get("t", -1.0)
    res = d["resultados"]
    kpts = np.array([r["k"] for r in res])
    E_dia = autovalores_diabaticos(nq, tipo, t, kpts)   # bandas exactas diabáticas

    nuevos = []
    for n, r in enumerate(res):
        calc = r["valores_calculados"]      # ideal (statevector de los params óptimos)
        med = r["valores_medidos"]          # medida en hardware
        itr = r.get("iteraciones", [0] * len(calc))
        orden, bandas = asignar_bandas(calc, E_dia[n])   # asigna cada estado a su banda física
        nuevos.append({
            "k": float(r["k"]),
            "bandas": [int(bj) for bj in bandas],
            "valores_exactos": [float(E_dia[n][bj]) for bj in bandas],
            "valores_calculados": [float(calc[i]) for i in orden],
            "valores_medidos": [float(med[i]) for i in orden],
            "iteraciones": [int(itr[i]) for i in orden],
        })
    d["resultados"] = nuevos
    d["parametros"]["orden"] = "diabatico"
    json.dump(d, open(path, "w", encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f"reprocesado: {fn}  ({len(nuevos)} puntos, orden diabático + campo 'bandas')")
