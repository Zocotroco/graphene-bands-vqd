"""
EJERCICIO HW-3 (v2, BARATO): contención de una NANOCINTA en hardware real (sistema 1D).

Método (medición en la base propia): se optimizan los parámetros en simulación (converge bien,
gratis) y en el dispositivo se mide la contención del estado realmente preparado, proyectando en
la base propia exacta de H(k). Evita la optimización completa en HW (que para 2-3 cubits no
converge dentro del presupuesto de la cuenta) y da un resultado limpio de la calidad de
preparación en la máquina.

COSTE: 2·(b+1)·n_ticks = 2·3·5 = ~30 jobs por cinta (~5-6 min a 11 s/job). UNA cinta por cuenta.
Empezar por zigzag (multiplete de borde).

Uso:
  1. TIPO = 'z' (zigzag, recomendado primero) o 'a' (armchair). Una cinta por cuenta.
  2. Poner TOKEN (o save_account) y opcional BACKEND_NAME.
  3. python3 ejercicio_hw3_containment_nanocinta.py
Salida: ../figuras/hardware_containment_{zGNR,aGNR}_2q.json
"""
import os
import sys
import numpy as np
np.random.seed(7)

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RAIZ)
import version_1 as v

FIGDIR = os.path.join(RAIZ, "figuras")

TIPO = 'a'              # 'z' zigzag (recomendado primero) o 'a' armchair
TOKEN = None            # <-- token de IBM Quantum (o None si usas save_account)
BACKEND_NAME = None     # <-- None = least_busy; o p. ej. "ibm_marrakesh"

nombre = 'zGNR' if TIPO == 'z' else 'aGNR'
v.calculos(
    num_qubits=2, tipo=TIPO, b=2,              # N=2 (4 estados): fundamental + 2 excitados
    modo='hardware', optimizar_en='sim',       # optimiza en sim (gratis, converge)
    optimizador='SLSQP', reps=2, maxiter=1000,
    n_ticks=5, grafico='n',                    # 5 puntos k en [0, π]
    shots=1024, resilience_level=1,
    token=TOKEN, channel='ibm_quantum_platform', backend_name=BACKEND_NAME,
    usar_session=False,                        # plan Open -> job mode
    containment=True, containment_hw=True,     # <-- contención MEDIDA en el dispositivo
    fichero_salida=os.path.join(FIGDIR, f"hardware_containment_{nombre}_2q.json"))
print(f"\nHECHO HW-3 ({nombre})")
