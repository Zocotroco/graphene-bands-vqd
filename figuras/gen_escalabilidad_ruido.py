"""
Columnas de RUIDO para la tabla de escalabilidad: repite el barrido en n_q de
`gen_escalabilidad.py` pero en modo `aer_noise` (dispositivo genérico de n_q cubits, 1024
shots, SPSA), para poder contrastar en la misma tabla el suelo intrínseco del ansatz
(statevector, SLSQP) con lo que se mediría en un dispositivo ruidoso.

Guarda por cada (cinta, n_q) el error medio de la energía MEDIDA con ruido y el de la energía
IDEAL de los parámetros óptimos, que separa el ruido de medida del efecto del ruido sobre la
optimización.

CUIDADO: caro. Medido: zigzag n_q=4 tarda ~80 min. El coste por evaluación crece con el número
de términos de Pauli (que se duplica con cada cubit) y con el tamaño del circuito simulado, de
modo que cada cubit adicional multiplica el tiempo por ~4. Estimación acumulada por cinta:
n_q=2 ~5 min, n_q=3 ~20 min, n_q=4 ~80 min, n_q=5 ~5 h, n_q=6 ~20 h.

El script guarda el JSON después de CADA celda y, al relanzarlo, se salta las ya calculadas:
puede pararse y reanudarse tantas veces como haga falta.

Uso:   cd figuras && python3 gen_escalabilidad_ruido.py
       NQS=2,3,4,5 python3 gen_escalabilidad_ruido.py    (para limitar el alcance)
Salida: ../figuras/escalabilidad_ruido.json
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

FIGDIR = os.path.join(RAIZ, "figuras")
SALIDA = os.path.join(FIGDIR, "escalabilidad_ruido.json")

NQS = [int(x) for x in os.environ.get("NQS", "2,3,4,5,6").split(",")]
REPS = 2          # misma profundidad que la tabla de escalabilidad
B = 3             # 4 bandas
N_TICKS = 7       # misma malla de k que la tabla
SHOTS = 1024


def errores(fichero, clave):
    d = json.load(open(fichero, encoding='utf-8'))
    ex = np.array([r["valores_exactos"] for r in d["resultados"]])
    ca = np.array([r[clave] for r in d["resultados"]])
    err = np.abs(ca - ex)
    evals = int(sum(sum(r["iteraciones"]) for r in d["resultados"]))
    return float(np.mean(err)), float(np.max(err)), evals


tabla = []
if os.path.exists(SALIDA):
    tabla = json.load(open(SALIDA, encoding='utf-8'))
    print(f"Reanudando: ya hay {len(tabla)} celdas calculadas", flush=True)
hechas = {(f["tipo"], f["nq"]) for f in tabla}

for tipo, nombre in [('a', 'armchair'), ('z', 'zigzag')]:
    for nq in NQS:
        if (nombre, nq) in hechas:
            print(f"{nombre} nq={nq}: ya estaba, se salta", flush=True)
            continue
        print(f"\n===== {nombre}  nq={nq}  (aer_noise, SPSA, {SHOTS} shots, reps={REPS}) =====",
              flush=True)
        fn = os.path.join(FIGDIR, f"_escrui_{nombre}_nq{nq}.json")
        t0 = time.perf_counter()
        v.calculos(num_qubits=nq, tipo=tipo, b=B, modo='aer_noise', noise_model='generic',
                   optimizador='SPSA', grafico='n', n_ticks=N_TICKS, reps=REPS,
                   shots=SHOTS, fichero_salida=fn)
        dt = time.perf_counter() - t0
        med, med_max, evals = errores(fn, "valores_medidos")
        ide, _, _ = errores(fn, "valores_calculados")
        fila = {"tipo": nombre, "nq": nq, "N": 2 ** (nq - 1),
                "err_medido": med, "err_medido_max": med_max, "err_ideal_opt": ide,
                "evals": evals, "tiempo_s": round(dt, 1)}
        tabla.append(fila)
        tabla.sort(key=lambda f: (f["tipo"], f["nq"]))
        json.dump(tabla, open(SALIDA, "w", encoding='utf-8'), indent=2, ensure_ascii=False)
        print(f"  err_medido={med:.3e}  err_ideal_de_los_optimos={ide:.3e}  "
              f"({dt/60:.1f} min)  [guardado]", flush=True)

print("\n===== RESUMEN =====", flush=True)
print(f"{'cinta':10} {'nq':>3} {'err_medido':>12} {'err_ideal':>12} {'min':>8}")
for f in tabla:
    print(f"{f['tipo']:10} {f['nq']:>3} {f['err_medido']:>12.3e} "
          f"{f['err_ideal_opt']:>12.3e} {f['tiempo_s']/60:>8.1f}")
print("\nHECHO", flush=True)
