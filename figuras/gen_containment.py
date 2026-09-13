"""
Genera los datos del análisis de CONTENCIÓN EN SUBESPACIOS (subspace containment),
siguiendo la metodología de Expósito, Aseginolaza, Guerrero-Avilés et al.
(arXiv:2607.11380, Tablas II y IV).

Idea (sugerencia del director): en lugar de calcular TODO el espectro, se estudia un
sistema mayor (3 cubits -> 8 estados) y se reconstruyen solo el estado fundamental y un
par de estados excitados (b=2 -> 3 estados) mediante VQD. Para cada punto k y cada estado
variacional se evalúa

    C_alpha(θ) = <psi(θ)| P_alpha |psi(θ)> = sum_i |<phi_{alpha,i}^ED | psi(θ)>|^2

es decir, qué fracción de la función de onda variacional vive en cada autoespacio exacto
de baja energía. Esto distingue una reconstrucción genuina de la función de onda de un
mero acuerdo en energía. Los parámetros se optimizan CON RUIDO (aer_noise) y la contención
se evalúa con el statevector ideal de esos parámetros, aislando la calidad de la
optimización ruidosa del ruido de muestreo.

Guarda: ../figuras/containment_{zGNR,aGNR}_3q.json
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import numpy as np
np.random.seed(7)  # reproducibilidad (Math.random no disponible en scripts de workflow)

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RAIZ)
import version_1 as v

FIGDIR = os.path.join(RAIZ, "figuras")

# 3 cubits (8 estados), VQD del fundamental + 2 excitados, optimización con ruido (SPSA).
# n_restarts=3: con una sola semilla, SPSA se quedaba en un mínimo local en algunos k y un
# estado se saltaba un nivel (armchair k=pi/6: C=0.04) o dos estados quedaban rotados dentro
# del subespacio que generan (zigzag k=5pi/6: C=0.55). Subir maxiter NO lo corrige (empeora);
# reintentar desde otras semillas, sí: la contención mínima pasa de 0.04 a >0.98.
comun = dict(num_qubits=3, b=2, modo='aer_noise', noise_model='generic',
             optimizador='SPSA', ansatz_str='EfficientSU2', grafico='n',
             n_ticks=7, shots=1024, reps=3, n_restarts=3,
             containment=True, n_espacios=5)

for tipo, nombre in [('a', 'aGNR'), ('z', 'zGNR')]:
    print(f"\n===== {nombre} (3 cubits, aer_noise generic, SPSA, b=2, containment) =====",
          flush=True)
    v.calculos(tipo=tipo,
               fichero_salida=os.path.join(FIGDIR, f"containment_{nombre}_3q.json"),
               **comun)
print("\nHECHO", flush=True)
