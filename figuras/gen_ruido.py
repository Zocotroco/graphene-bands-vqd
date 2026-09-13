"""
Genera los datos de la sección "Efecto del ruido": VQD en modo aer_noise (simulación
con ruido de un dispositivo, backend pequeño y rápido), optimizando con SPSA (robusto
al ruido), para las nanocintas zigzag y armchair (2 cubits, 4 bandas, k en [0, pi]).

Guarda: ../figuras/ruido_{zGNR,aGNR}_2q.json
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")
RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RAIZ)
import version_1 as v

FIGDIR = os.path.join(RAIZ, "figuras")

comun = dict(num_qubits=2, b=3, modo='aer_noise', noise_model='generic',
             optimizador='SPSA', ansatz_str='EfficientSU2', grafico='n',
             n_ticks=12, shots=1024, reps=1)

for tipo, nombre in [('a', 'aGNR'), ('z', 'zGNR')]:
    print(f"\n===== {nombre} (2 cubits, aer_noise generic, SPSA, b=3) =====", flush=True)
    v.calculos(tipo=tipo, fichero_salida=os.path.join(FIGDIR, f"ruido_{nombre}_2q.json"), **comun)
print("\nHECHO", flush=True)
