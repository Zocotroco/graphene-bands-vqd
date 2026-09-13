"""
Barrido de PROFUNDIDAD DEL ANSATZ (repeticiones de EfficientSU2) para documentar con datos
propios la tensión del régimen NISQ: más repeticiones recuperan precisión en simulación ideal,
pero por debajo del ruido del dispositivo la mejora se agota y acaba revirtiéndose, porque el
circuito más profundo acumula más error de puertas.

Para cada valor de reps se calculan las 4 bandas inferiores por VQD de la nanocinta zigzag
(n_q=4, N=8) sobre 7 puntos k en [0, pi], en dos regímenes:

  - statevector : SLSQP, sin muestreo ni ruido  -> error puramente del ansatz/optimizador.
  - aer_noise   : SPSA, backend genérico de n_q cubits, 1024 shots
                  -> se guardan por separado el error de la energía MEDIDA (lo que daría un
                     dispositivo) y el de la energía IDEAL de los parámetros óptimos (cuánto
                     ha estropeado el ruido la OPTIMIZACIÓN, sin el error de medida).

Elegimos zigzag con n_q=4 porque es donde el barrido de escalabilidad (Tabla II) ya muestra
el régimen limitado por el ansatz: err_medio/t ~ 1.8e-2 con reps=2.

CUIDADO: el brazo aer_noise es caro (estimación ~3 h en total; crece con reps). El script
guarda el JSON después de CADA valor de reps y, si lo relanzas, se salta lo ya calculado.

Uso:   cd figuras && python3 gen_reps.py
Salida: ../figuras/reps_zGNR_4q.json
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

FIGDIR = os.path.join(RAIZ, "figuras")
SALIDA = os.path.join(FIGDIR, "reps_zGNR_4q.json")

TIPO, NOMBRE = 'z', 'zigzag'
NQ = 4
B = 3                       # 4 bandas (fundamental + 3 excitados)
N_TICKS = 7
REPS_LIST = [1, 2, 3, 4]
SHOTS = 1024


def errores(fichero, clave):
    """Error medio y máximo (en unidades de |t|) de una clave de energías frente a la exacta."""
    d = json.load(open(fichero, encoding='utf-8'))
    ex = np.array([r["valores_exactos"] for r in d["resultados"]])
    ca = np.array([r[clave] for r in d["resultados"]])
    err = np.abs(ca - ex)
    evals = int(sum(sum(r["iteraciones"]) for r in d["resultados"]))
    return float(np.mean(err)), float(np.max(err)), evals


def coste_circuito(reps):
    """Parámetros y número de puertas de dos cubits del ansatz (lo que dispara el ruido)."""
    qc = EfficientSU2(num_qubits=NQ, reps=reps)
    dec = qc.decompose()
    cx = sum(n for inst, n in dec.count_ops().items() if inst in ('cx', 'cz', 'ecr'))
    return qc.num_parameters, dec.depth(), int(cx)


tabla = []
if os.path.exists(SALIDA):
    tabla = json.load(open(SALIDA, encoding='utf-8'))
    print(f"Reanudando: ya hay {len(tabla)} valores de reps calculados", flush=True)
hechos = {f["reps"] for f in tabla}

for reps in REPS_LIST:
    if reps in hechos:
        print(f"reps={reps}: ya estaba, se salta", flush=True)
        continue
    nparams, depth, cx = coste_circuito(reps)
    fila = {"tipo": NOMBRE, "nq": NQ, "N": 2 ** (NQ - 1), "reps": reps,
            "nparams": nparams, "depth": depth, "cx": cx}

    # --- 1. simulación ideal: SLSQP, sin ruido ---
    print(f"\n===== reps={reps} | statevector (SLSQP) | params={nparams} cx={cx} =====", flush=True)
    fn = os.path.join(FIGDIR, f"_reps_{NOMBRE}_nq{NQ}_r{reps}_sv.json")
    t0 = time.perf_counter()
    v.calculos(num_qubits=NQ, tipo=TIPO, b=B, modo='statevector', grafico='n',
               n_ticks=N_TICKS, reps=reps, optimizador='SLSQP', fichero_salida=fn)
    fila["t_sv_s"] = round(time.perf_counter() - t0, 1)
    fila["err_sv"], fila["err_sv_max"], fila["evals_sv"] = errores(fn, "valores_calculados")
    print(f"  err_sv = {fila['err_sv']:.3e}   ({fila['t_sv_s']} s)", flush=True)

    # --- 2. con ruido de dispositivo: SPSA, 1024 shots ---
    print(f"===== reps={reps} | aer_noise (SPSA, {SHOTS} shots) =====", flush=True)
    fn = os.path.join(FIGDIR, f"_reps_{NOMBRE}_nq{NQ}_r{reps}_noise.json")
    t0 = time.perf_counter()
    v.calculos(num_qubits=NQ, tipo=TIPO, b=B, modo='aer_noise', noise_model='generic',
               optimizador='SPSA', grafico='n', n_ticks=N_TICKS, reps=reps,
               shots=SHOTS, fichero_salida=fn)
    fila["t_noise_s"] = round(time.perf_counter() - t0, 1)
    fila["err_medido"], fila["err_medido_max"], fila["evals_noise"] = errores(fn, "valores_medidos")
    fila["err_ideal_opt"], _, _ = errores(fn, "valores_calculados")
    print(f"  err_medido = {fila['err_medido']:.3e}   "
          f"err_ideal_de_los_optimos = {fila['err_ideal_opt']:.3e}   "
          f"({fila['t_noise_s']} s)", flush=True)

    tabla.append(fila)
    tabla.sort(key=lambda f: f["reps"])
    json.dump(tabla, open(SALIDA, "w", encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"  guardado parcial en {os.path.basename(SALIDA)}", flush=True)

print("\n===== RESUMEN =====", flush=True)
print(f"{'reps':>4} {'params':>7} {'cx':>4} {'err_sv':>11} {'err_medido':>11} {'err_ideal':>11}")
for f in tabla:
    print(f"{f['reps']:>4} {f['nparams']:>7} {f['cx']:>4} "
          f"{f['err_sv']:>11.3e} {f['err_medido']:>11.3e} {f['err_ideal_opt']:>11.3e}")
print("\nHECHO", flush=True)
