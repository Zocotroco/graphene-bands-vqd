"""
EJERCICIO HW-2 (v2, BARATO): contención del grafeno MONOCAPA en hardware real.

>>> IMPORTANTE: sustituye a la versión anterior (que optimizaba en HW y necesitaba ~600 jobs,
    muy por encima de los ~54 jobs que caben en los 10 min de una cuenta). <<<

Método (medición en la base propia): se optimizan los parámetros en simulación (converge bien,
gratis) y en el dispositivo se prepara el circuito óptimo U(θ), se aplica el cambio a la base
propia exacta V† de H(k) y se mide: la distribución de salidas da la contención del estado
REALMENTE preparado en la máquina (incluye ruido de preparación y de medida). 1 job por (k, estado)
para la contención + 1 por (k, estado) para la energía.

COSTE: 2·(b+1)·n_puntos = 2·2·10 = ~40 jobs (~7-8 min a 11 s/job). Cabe en una cuenta.

Uso: poner TOKEN (o save_account) y opcional BACKEND_NAME. python3 ejercicio_hw2_containment_monocapa.py
Salida: ../figuras/hardware_monocapa_containment.json
"""
import os
import sys
import numpy as np
np.random.seed(7)

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RAIZ)
import version_1 as v

FIGDIR = os.path.join(RAIZ, "figuras")

TOKEN = None            # <-- token de IBM Quantum (o None si usas save_account)
BACKEND_NAME = None     # <-- None = least_busy; o p. ej. "ibm_marrakesh"

v.calculos_monocapa(
    num_points=1,               # 7 puntos k a lo largo de Γ-K-M-Γ (~28 jobs, cabe en ~5 min)
    modo='hardware', optimizar_en='sim',      # optimiza en sim (gratis, converge)
    optimizador='SLSQP', reps=2, vecinos=2, maxiter=1000,
    shots=1024, resilience_level=1,
    token=TOKEN, channel='ibm_quantum_platform', backend_name=BACKEND_NAME,
    usar_session=False,         # plan Open -> job mode
    containment=True, containment_hw=True,    # <-- contención MEDIDA en el dispositivo
    fichero_salida=os.path.join(FIGDIR, "hardware_monocapa_containment.json"))
print("\nHECHO HW-2")
