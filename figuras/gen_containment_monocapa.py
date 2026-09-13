"""
Genera los datos del grafeno MONOCAPA (2D) para el paper reorientado: bandas a lo largo del
camino Γ→K→M→Γ y análisis de contención en subespacios (subspace containment), optimizando
CON RUIDO (aer_noise), igual que para las nanocintas. Sistema de 1 cubit (2 bandas: valencia
y conducción) mediante VQD.

Guarda: ../figuras/containment_monocapa.json
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

print("===== MONOCAPA (1 cubit, aer_noise generic, SPSA, containment) =====", flush=True)
v.calculos_monocapa(num_points=12, modo='aer_noise', noise_model='generic',
                    optimizador='SPSA', reps=2, maxiter=1000, vecinos=2,
                    containment=True, shots=1024,
                    fichero_salida=os.path.join(FIGDIR, "containment_monocapa.json"))
print("\nHECHO", flush=True)
