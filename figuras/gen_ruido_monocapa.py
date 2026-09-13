"""
Datos del grafeno MONOCAPA con RUIDO (aer_noise), para la sección de efecto del ruido y la
comparativa de optimizadores (COBYLA vs SPSA), a lo largo de Γ→K→M→Γ. VQD de valencia y
conducción (1 cubit), primeros+segundos vecinos.

Guarda: ../figuras/ruido_monocapa_{COBYLA,SPSA}.json
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import numpy as np

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RAIZ)
import version_1 as v

FIGDIR = os.path.join(RAIZ, "figuras")

for opt in ("COBYLA", "SPSA"):
    np.random.seed(7)   # misma inicialización para comparar en igualdad de condiciones
    print(f"\n===== MONOCAPA aer_noise, optimizador={opt} =====", flush=True)
    v.calculos_monocapa(num_points=8, modo='aer_noise', noise_model='generic',
                        optimizador=opt, reps=2, maxiter=300, vecinos=2, shots=1024,
                        containment=False,
                        fichero_salida=os.path.join(FIGDIR, f"ruido_monocapa_{opt}.json"))
print("\nHECHO", flush=True)
