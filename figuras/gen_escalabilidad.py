"""
Estudio de ESCALABILIDAD y ANÁLISIS DE ERROR: para n_q = 2..6 (anchura N=2^(n_q-1)),
calcula las 4 bandas inferiores por VQD (statevector, SLSQP, reps=2) y mide el error
frente a la diagonalización exacta y los recursos (parámetros del ansatz, nº de términos
de Pauli, evaluaciones del optimizador, tiempo).

Guarda: ../figuras/escalabilidad.json
"""
import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RAIZ)
import version_1 as v
from qiskit.circuit.library import efficient_su2 as EfficientSU2
from qiskit.quantum_info import SparsePauliOp

NQS = [2, 3, 4, 5, 6]
REPS = 2
N_TICKS = 7
tabla = []

for tipo, nombre in [('a', 'armchair'), ('z', 'zigzag')]:
    for nq in NQS:
        fn = os.path.join(RAIZ, "figuras", f"_scal_{nombre}_nq{nq}.json")
        t0 = time.perf_counter()
        v.calculos(num_qubits=nq, tipo=tipo, b=3, modo='statevector', grafico='n',
                   n_ticks=N_TICKS, reps=REPS, optimizador='SLSQP', fichero_salida=fn)
        dt = time.perf_counter() - t0
        d = json.load(open(fn, encoding='utf-8'))
        ex = np.array([r["valores_exactos"] for r in d["resultados"]])
        ca = np.array([r["valores_calculados"] for r in d["resultados"]])
        err = np.abs(ca - ex)
        evals = int(sum(sum(r["iteraciones"]) for r in d["resultados"]))
        nparams = EfficientSU2(num_qubits=nq, reps=REPS).num_parameters
        npauli = len(SparsePauliOp.from_operator(v.build_matrix(nq, tipo, 0.7)))
        fila = {"tipo": nombre, "nq": nq, "N": 2 ** (nq - 1), "nparams": nparams,
                "npauli": npauli, "err_medio": float(np.mean(err)), "err_max": float(np.max(err)),
                "evals": evals, "tiempo_s": round(dt, 1)}
        tabla.append(fila)
        print(f"{nombre:8s} nq={nq} N={2**(nq-1):3d} params={nparams:3d} pauli={npauli:5d} "
              f"err_medio={fila['err_medio']:.4f} err_max={fila['err_max']:.4f} "
              f"evals={evals:6d} t={dt:.1f}s", flush=True)

json.dump(tabla, open(os.path.join(RAIZ, "figuras", "escalabilidad.json"), "w"),
          indent=2, ensure_ascii=False)
print("\nHECHO", flush=True)
