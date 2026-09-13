"""
EJERCICIO HW-1: bandas del grafeno MONOCAPA en hardware real de IBM Quantum.

Estrategia 'sim': los parámetros del ansatz se optimizan en simulación (gratis) y SOLO se mide
la energía del circuito óptimo en el dispositivo real → 1 job por (k, banda) (~20 jobs).
Barato; ideal para verificar la conexión y obtener las bandas 2D "reales".

Uso:
  1. Poner el TOKEN de IBM Quantum abajo (o usar credenciales guardadas con save_account).
  2. (Opcional) fijar BACKEND_NAME a un backend concreto; None -> el menos ocupado.
  3. python3 ejercicio_hw1_bandas_monocapa.py
Salida: ../figuras/hardware_monocapa_bandas.json
"""
import os
import sys
import numpy as np
np.random.seed(7)

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RAIZ)
import version_1 as v

FIGDIR = os.path.join(RAIZ, "figuras")

TOKEN = None            # <-- pon aquí tu token de IBM Quantum (o None si usas save_account)
BACKEND_NAME = None     # <-- None = least_busy; o p. ej. "ibm_fez"

v.calculos_monocapa(
    num_points=2,               # ~10 puntos k a lo largo de Γ-K-M-Γ
    modo='hardware', optimizar_en='sim',
    optimizador='SLSQP', reps=2, vecinos=2, maxiter=1000,
    shots=1024, resilience_level=1,
    token=TOKEN, channel='ibm_quantum_platform', backend_name=BACKEND_NAME,
    usar_session=False,         # plan Open -> job mode
    containment=False,
    fichero_salida=os.path.join(FIGDIR, "hardware_monocapa_bandas.json"))
print("\nHECHO HW-1")
