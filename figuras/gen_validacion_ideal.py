"""
Genera las figuras de validación en simulación ideal (statevector):
estructura de bandas VQD frente a la diagonalización exacta, para las nanocintas
zigzag y armchair, con 3 cubits y k de -pi a pi (todas las bandas).

Salida: ../latex/imagenes/fig_val_{zGNR,aGNR}_3q.png
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")  # headless
RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RAIZ)          # para poder importar version_1
import version_1 as v

OUT = os.path.join(RAIZ, "latex", "imagenes")
FIGDIR = os.path.join(RAIZ, "figuras")

# b=3 -> 4 bandas inferiores (las que mostramos); reps=2 -> ansatz más expresivo
# SLSQP (gradiente) converge mejor/más rápido que COBYLA en statevector
# grafico='f' -> k en [-pi, pi] (aunque no usemos la gráfica de version_1)
comun = dict(num_qubits=3, modo='statevector', b=3, grafico='f', optimizador='SLSQP',
             n_ticks=13, reps=2)

for tipo, nombre in [('z', 'zGNR'), ('a', 'aGNR')]:
    print(f"\n===== {nombre} (3 cubits, statevector, -pi..pi, reps=3) =====")
    v.calculos(tipo=tipo,
               fichero_salida=os.path.join(FIGDIR, f"val_{nombre}_3q.json"),
               **comun)
print("\nHECHO")
