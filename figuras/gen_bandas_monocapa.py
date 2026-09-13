"""
Bandas del grafeno MONOCAPA en simulación IDEAL (statevector), VQD de valencia y conducción,
para validar frente a la diagonalización exacta. Se generan dos casos: solo primeros vecinos
(vecinos=1) y primeros+segundos vecinos (vecinos=2).

Guarda: ../figuras/bandas_monocapa_v{1,2}.json
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
np.random.seed(7)

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RAIZ)
import version_1 as v

FIGDIR = os.path.join(RAIZ, "figuras")

for vec in (1, 2):
    print(f"\n===== MONOCAPA vecinos={vec} (statevector, SLSQP) =====", flush=True)
    v.calculos_monocapa(num_points=20, modo='statevector', optimizador='SLSQP',
                        reps=2, vecinos=vec, containment=False,
                        fichero_salida=os.path.join(FIGDIR, f"bandas_monocapa_v{vec}.json"))
print("\nHECHO", flush=True)
