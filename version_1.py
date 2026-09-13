import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import json

# --- IMPORTACIONES ---
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import efficient_su2 as EfficientSU2
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_algorithms.optimizers import SPSA, COBYLA, SLSQP
from qiskit_aer.primitives import EstimatorV2
from qiskit_aer.noise import NoiseModel
# Backends "fake" grandes (127-133 cúbits) -> simulación de ruido MUY lenta
from qiskit_ibm_runtime.fake_provider import FakeKyiv, FakeTorino, FakeSherbrooke
# Backends "fake" pequeños (5-7 cúbits) -> simulación de ruido rápida (recomendados)
from qiskit_ibm_runtime.fake_provider import (
    FakeManilaV2, FakeLimaV2, FakeBelemV2, FakeQuitoV2,   # 5 cúbits
    FakeLagosV2, FakeNairobiV2, FakePerth,                # 7 cúbits
)
# Máquina fake genérica con el nº de cúbits que se quiera (ruido sintético reproducible)
from qiskit.providers.fake_provider import GenericBackendV2


# --- Diccionario de backends "fake" para simulación con ruido ---
# Nota del director: el tiempo de la simulación con ruido crece con el nº de cúbits
# de la máquina emulada. FakeTorino/Kyiv/Sherbrooke tienen 127-133 cúbits y son
# ~20-500x más lentos que un backend de 5 cúbits o una máquina genérica pequeña.
FAKE_BACKENDS = {
    'torino': FakeTorino,          # 133 cúbits (lento)
    'kyiv': FakeKyiv,              # 127 cúbits (lento)
    'sherbrooke': FakeSherbrooke,  # 127 cúbits (lento)
    'manila': FakeManilaV2,        # 5 cúbits (rápido)
    'lima': FakeLimaV2,            # 5 cúbits (rápido)
    'belem': FakeBelemV2,          # 5 cúbits (rápido)
    'quito': FakeQuitoV2,          # 5 cúbits (rápido)
    'lagos': FakeLagosV2,          # 7 cúbits
    'nairobi': FakeNairobiV2,      # 7 cúbits
    'perth': FakePerth,            # 7 cúbits
    # 'generic' -> GenericBackendV2(num_qubits) se construye en crear_estimator()
}


# --- Función de coste (energía) a optimizar en VQE/VQD (modos de simulación) ---
def cost_func(beta, params, ansatz, operator, estimator, wavefuncs, eidx, evaluaciones, modo):
    # 1. Asignamos los parámetros al circuito
    circuito = ansatz.assign_parameters(params)
    # 2. Calculamos el valor de expectación (Energía)
    sv = Statevector.from_instruction(circuito)
    if modo == "statevector":
        energy = sv.expectation_value(operator).real
    else:
        job = estimator.run([(circuito, operator)])
        energy = float(job.result()[0].data.evs)
    # 3. Calculamos la penalización por solapamiento (Fidelidad) con statevector
    penalty = 0.0
    for prev_sv in wavefuncs[:eidx]:
        penalty += beta * np.abs(sv.inner(prev_sv)) ** 2
    evaluaciones[eidx] += 1

    return energy + penalty



# --- Optimización con REINICIOS MÚLTIPLES (multi-start) ---
def optimizar_con_reinicios(optimizer, cost, ansatz, operator, wavefuncs, beta,
                            semilla_warm, n_restarts=1):
    """Lanza la optimización desde n_restarts semillas y devuelve la MEJOR.

    La primera semilla es siempre la del warm-starting (parámetros óptimos del punto k
    anterior); las demás son aleatorias. El criterio de selección es el COSTE IDEAL
    ---energía + penalización de deflación evaluadas sobre el statevector--- y no el valor
    medido, de modo que la elección no depende del ruido de muestreo.

    Motivo: con SPSA sobre una función de coste ruidosa el optimizador puede quedarse en un
    mínimo local y saltarse un nivel (el estado converge a un autoespacio que no es el suyo).
    Aumentar maxiter no lo corrige; volver a intentarlo desde otra semilla, sí.

    Devuelve (params, statevector, energia_ideal).
    """
    mejor = None
    for r in range(max(1, int(n_restarts))):
        x0 = semilla_warm if r == 0 else np.pi * np.random.random(ansatz.num_parameters)
        resultado = optimizer.minimize(cost, x0)
        sv = Statevector.from_instruction(ansatz.assign_parameters(resultado.x))
        e_ideal = sv.expectation_value(operator).real
        penal = sum(beta * np.abs(sv.inner(prev)) ** 2 for prev in wavefuncs)
        if mejor is None or (e_ideal + penal) < mejor[0]:
            mejor = (e_ideal + penal, resultado.x, sv, e_ideal)
    return mejor[1], mejor[2], mejor[3]


# --- Plantilla (transpilada UNA sola vez) para la fidelidad por muestreo ---
# En hardware no hay acceso al statevector: la fidelidad |<psi(pa)|psi(pb)>|^2 se mide como
# la probabilidad de obtener |0...0> al ejecutar U(pb)^dag U(pa). El circuito se construye y
# transpila una única vez con dos vectores de parámetros independientes (theta, phi); en cada
# evaluación solo se re-vinculan los valores, evitando miles de transpilaciones/llamadas.
def construir_overlap_template(ansatz, pm):
    from qiskit.circuit import ParameterVector
    theta = ParameterVector('θov', ansatz.num_parameters)
    phi = ParameterVector('φov', ansatz.num_parameters)
    ans_a = ansatz.assign_parameters(dict(zip(list(ansatz.parameters), list(theta))))
    ans_b = ansatz.assign_parameters(dict(zip(list(ansatz.parameters), list(phi))))
    ov = QuantumCircuit(ansatz.num_qubits)
    ov.compose(ans_a, inplace=True)
    ov.compose(ans_b.inverse(), inplace=True)
    ov.measure_all()
    ov_isa = pm.run(ov)
    return (ov_isa, list(theta), list(phi))


def _overlap_por_muestreo(sampler, overlap_template, nq, pa, pb):
    ov_isa, theta, phi = overlap_template
    binding = {}
    binding.update(dict(zip(theta, pa)))
    binding.update(dict(zip(phi, pb)))
    bound = ov_isa.assign_parameters(binding)
    res = sampler.run([bound]).result()[0]
    counts = res.data.meas.get_counts()
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return counts.get('0' * nq, 0) / total


# --- Contención medida en HARDWARE por proyección en la base propia exacta ---
# C_alpha = |<phi_alpha|psi(theta)>|^2 es la probabilidad de medir |psi(theta)> en la base propia
# de H. Basta preparar U(theta), aplicar el cambio de base V^dagger (V diagonaliza H) y medir en la
# base computacional: la distribución de salidas da todas las contenciones a la vez -> 1 job por
# (k, estado), mucho más barato que la tomografía completa. Mide la calidad del estado REALMENTE
# preparado en el dispositivo (incluye ruido de preparación y de medida).
def containment_hw_por_muestreo(sampler, pm, ansatz, params, V_ex, grupos, nq):
    from qiskit.circuit.library import UnitaryGate
    circ = QuantumCircuit(nq)
    circ.compose(ansatz.assign_parameters(params), inplace=True)
    circ.append(UnitaryGate(V_ex.conj().T), list(range(nq)))
    circ.measure_all()
    circ_isa = pm.run(circ)
    res = sampler.run([circ_isa]).result()[0]
    counts = res.data.meas.get_counts()
    total = sum(counts.values()) or 1
    probs = np.zeros(2 ** nq)
    for bitstr, c in counts.items():
        probs[int(bitstr.replace(' ', ''), 2)] += c / total
    return [float(sum(probs[i] for i in g)) for g in grupos]


# --- Función de coste para el modo hardware real (sin statevector) ---
def cost_func_hardware(beta, params, nq, ansatz_isa, operator_isa, estimator, sampler,
                       overlap_template, prev_params, eidx, evaluaciones):
    # Energía mediante el estimador del runtime sobre el circuito transpilado (ISA)
    job = estimator.run([(ansatz_isa, operator_isa, params)])
    energy = float(job.result()[0].data.evs)
    # Penalización por solapamiento con los estados anteriores (fidelidad por muestreo)
    penalty = 0.0
    for pb in prev_params[:eidx]:
        penalty += beta * _overlap_por_muestreo(sampler, overlap_template, nq, params, pb)
    evaluaciones[eidx] += 1
    return energy + penalty


# --- Creación del estimador para los modos de simulación ---
# IMPORTANTE: el EstimatorV2 de qiskit-aer controla el nº de muestras mediante la PRECISIÓN
# (precision = 1/sqrt(shots)), NO mediante run_options={"shots":...} (que ignora). Si no se
# fija la precisión, calcula el valor esperado EXACTO por statevector, sin shots NI ruido
# (aunque se le pase un noise_model). Por eso se fija default_precision = 1/sqrt(shots).
def crear_estimator(modo="statevector", fake_backend_name="torino", shots=1024, num_qubits=2):
    if modo == "statevector":
        return None
    precision = 1.0 / np.sqrt(shots)
    if modo == "aer":
        return EstimatorV2(options={"default_precision": precision})
    elif modo == "aer_noise":
        nombre = fake_backend_name.lower()
        if nombre == 'generic':
            # Máquina fake pequeña, ajustada al nº de cúbits del circuito (muy rápida)
            backend = GenericBackendV2(num_qubits=num_qubits, seed=1234)
        elif nombre in FAKE_BACKENDS:
            backend = FAKE_BACKENDS[nombre]()
        else:
            raise ValueError(f"Fake backend no soportado: {fake_backend_name}")
        noise_model = NoiseModel.from_backend(backend)
        return EstimatorV2(options={"backend_options": {"noise_model": noise_model},
                                    "default_precision": precision})
    else:
        raise ValueError("Modo no reconocido")


# --- Preparación del backend de hardware real (o backend de prueba local) ---
def preparar_hardware(num_qubits, ansatz, token=None, channel="ibm_quantum_platform",
                      instance=None, backend_name=None, optimization_level=1, backend_obj=None):
    """
    Devuelve (service, backend, pm, ansatz_isa).
    - Si backend_obj no es None se usa directamente (útil para PRUEBAS LOCALES pasando
      un fake backend, p. ej. FakeManilaV2(), sin necesidad de token). service = None.
    - En otro caso se conecta a IBM Quantum con QiskitRuntimeService y se elige el backend
      indicado (backend_name) o el menos ocupado (least_busy).
    """
    if backend_obj is not None:
        service = None
        backend = backend_obj
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(channel=channel, token=token, instance=instance)
        if backend_name:
            backend = service.backend(backend_name)
        else:
            backend = service.least_busy(operational=True, simulator=False, min_num_qubits=num_qubits)
    pm = generate_preset_pass_manager(target=backend.target, optimization_level=optimization_level)
    ansatz_isa = pm.run(ansatz)
    return service, backend, pm, ansatz_isa


# --- Hamiltoniano de Bloch de la nanocinta ---
def build_matrix(num_qubits, tipo, k, t=-2.7):
    n = 2 ** (num_qubits - 1)
    M = np.zeros((n, n), dtype=complex)
    if tipo == 'z':
        phase_factor = t * (1 + np.exp(1j * k))
        for i in range(n):
            M[i, i] = phase_factor
            if i > 0: M[i, i-1] = t
    elif tipo == 'a':
        phase_factor = t * np.exp(1j * k * 0.5)
        for i in range(n):
            M[i,i] = phase_factor
            if i > 0: M[i, i-1] = t
            if i < n-1: M[i, i+1] = t
    H = np.block([[np.zeros((n, n)), M], [np.conj(M).T, np.zeros((n, n))]])
    return H


# --- Hamiltoniano de Bloch del GRAFENO MONOCAPA (2D, 2 bandas, 1 cubit) ---
# Mismo modelo tight-binding que el TFM (BandasEnergiaGrafeno): primeros vecinos (t) y,
# opcionalmente, segundos vecinos (t_prima). k es un vector 2D (kx, ky). El fichero del TFM
# NO se modifica; aquí se replica el modelo para unificar containment + hardware moderno.
def _delta_monocapa(a):
    return np.array([[a/2, np.sqrt(3)*a/2], [a/2, -np.sqrt(3)*a/2], [-a, 0]])


def _delta2_monocapa(a):
    return np.array([[a*3/2, np.sqrt(3)*a/2], [-a*3/2, -np.sqrt(3)*a/2],
                     [a*3/2, -np.sqrt(3)*a/2], [-a*3/2, np.sqrt(3)*a/2],
                     [0, -a*np.sqrt(3)], [0, a*np.sqrt(3)]])


def hamiltoniano_monocapa(k_vec, t=1.0, t_prima=1.0/28, a=1.42, vecinos=2):
    """Hamiltoniano de Bloch 2x2 del grafeno monocapa. k_vec: (kx, ky).
    vecinos=1 (solo primeros vecinos) o 2 (incluye segundos vecinos, término diagonal c)."""
    d = _delta_monocapa(a)
    aa = sum(np.cos(np.dot(k_vec, x)) for x in d)
    bb = sum(np.sin(np.dot(k_vec, x)) for x in d)
    c = 0.0
    if vecinos == 2:
        c = 2 * t_prima * sum(np.cos(np.dot(k_vec, x)) for x in _delta2_monocapa(a))
    return np.array([[c, -t*(aa + 1j*bb)], [-t*(aa - 1j*bb), c]], dtype=complex)


def camino_monocapa(num_points=12, a=1.42):
    """Camino Γ→K→M→Γ en la zona de Brillouin (mismo esquema que el TFM).
    Devuelve (k_vals[Nx2], coord[N] distancia acumulada para el eje x, pos[4] índices de Γ,K,M,Γ)."""
    G = np.array([0.0, 0.0])
    K = (2*np.pi/(3*a)) * np.array([1.0, 1/np.sqrt(3)])
    M = (2*np.pi/(3*a)) * np.array([1.0, 0.0])
    puntos = [G, K, M, G]
    k_vals = [G]
    pos = [0]
    for i in range(len(puntos) - 1):
        for alpha in np.linspace(0, 1, num_points + 2):
            if alpha > 0:
                k_vals.append((1 - alpha)*puntos[i] + alpha*puntos[i + 1])
        pos.append(len(k_vals) - 1)
    k_vals = np.array(k_vals)
    coord = np.zeros(len(k_vals))
    for i in range(1, len(k_vals)):
        coord[i] = coord[i-1] + np.linalg.norm(k_vals[i] - k_vals[i-1])
    return k_vals, coord, pos


# --- Autovalores exactos ordenados por CONTINUIDAD de banda (no por energía) ---
# np.linalg.eigvalsh devuelve los autovalores ordenados por energía, lo que hace que en
# los cruces el orden se intercambie. Aquí seguimos cada banda física por el solapamiento
# de autovectores entre puntos k consecutivos, de modo que la columna 0 sea siempre la
# banda 0, la 1 la banda 1, etc. (bandas continuas que se cruzan suavemente).
def autovalores_diabaticos(num_qubits, tipo, t, k_points):
    nb = 2 ** num_qubits
    E = np.zeros((len(k_points), nb))
    prev_V = None
    for idx, k in enumerate(k_points):
        w, V = np.linalg.eigh(build_matrix(num_qubits, tipo, k, t))  # w ascendente + autovectores
        if prev_V is not None:
            solape = np.abs(prev_V.conj().T @ V)     # |<v_prev_i | v_new_j>|
            perm = -np.ones(nb, dtype=int)
            usados, asig = set(), set()
            for _, i, j in sorted(((solape[i, j], i, j)
                                   for i in range(nb) for j in range(nb)), reverse=True):
                if i in asig or j in usados:
                    continue
                perm[i] = j; asig.add(i); usados.add(j)
            w, V = w[perm], V[:, perm]
        E[idx] = w
        prev_V = V
    return E   # (Nk, 2^num_qubits): columna b = banda b (continua)


# --- Asignación de los estados calculados a su banda física (diabática) ---
# Empareja cada valor calculado (orden de energía/deflación) con la banda diabática más
# cercana (uno a uno) y devuelve los índices de banda y el orden para reordenar.
def asignar_bandas(valores, E_dia_k):
    nb = len(E_dia_k)
    pares = sorted((abs(valores[i] - E_dia_k[j]), i, j)
                   for i in range(len(valores)) for j in range(nb))
    asig, usados = {}, set()
    for _, i, j in pares:
        if i in asig or j in usados:
            continue
        asig[i] = j; usados.add(j)
    orden = sorted(range(len(valores)), key=lambda i: asig[i])   # índices ordenados por banda
    bandas = [asig[i] for i in orden]
    return orden, bandas


# --- Análisis de contención en subespacios (subspace containment) ---
# Para un estado variacional |psi(θ)> y un autoespacio exacto H_alpha con proyector P_alpha,
# la contención se define como
#     C_alpha(θ) = <psi(θ)| P_alpha |psi(θ)> = sum_i |<phi_{alpha,i}^ED | psi(θ)>|^2
# (ecs. 24-25 de Expósito, Aseginolaza, Guerrero-Avilés et al., arXiv:2607.11380). Mide qué
# fracción de la función de onda variacional vive REALMENTE en el autoespacio exacto, de modo
# que distingue una reconstrucción genuina de la función de onda frente a un mero acuerdo en
# energía. Para un autoestado no degenerado se reduce al solapamiento al cuadrado; para un
# subespacio (casi) degenerado -p. ej. los estados de borde del zigzag en E~0- la proyección
# sobre TODO el autoespacio es la magnitud correcta e independiente de la base. Se evalúa con el
# statevector IDEAL de los parámetros optimizados (con o sin ruido), lo que aísla la calidad de
# los parámetros del ruido de muestreo/hardware.

def espacios_propios(w, tol=1e-6):
    """Agrupa autovalores ASCENDENTES en autoespacios: índices consecutivos cuya separación
    sea <= tol se consideran (casi) degenerados. Devuelve una lista de listas de índices."""
    grupos = [[0]]
    for i in range(1, len(w)):
        if abs(w[i] - w[i - 1]) <= tol:
            grupos[-1].append(i)
        else:
            grupos.append([i])
    return grupos


def subspace_containment(psi, V, grupos):
    """C_alpha para cada autoespacio: suma de |<V_i|psi>|^2 sobre los índices del grupo.
    psi: amplitudes del estado variacional (statevector.data); V: autovectores por columnas
    (np.linalg.eigh); grupos: salida de espacios_propios."""
    return [float(sum(abs(np.vdot(V[:, i], psi)) ** 2 for i in g)) for g in grupos]


# --- CONFIGURACIÓN Y BUCLE VQE/VQD ---
def calculos(num_qubits=2, tipo='z', modo='statevector', optimizador='COBYLA',
             ansatz_str='EfficientSU2', grafico='n', b=0, noise_model='torino',
             shots=1024, n_ticks=12, t=-1.0, fichero_salida=None, fichero_figura=None,
             plot_energia='ideal',
             token=None, channel='ibm_quantum_platform', instance=None,
             backend_name=None, optimization_level=1, resilience_level=1, backend_obj=None,
             usar_session=False, optimizar_en='sim', reps=1,
             containment=False, n_espacios=None, tol_deg=1e-6, maxiter=1000,
             containment_hw=False, n_restarts=1):
    # num_qubits: >1, defecto=2
    # tipo: 'a'->armchair, 'z'-> zigzag , defecto='z'
    # modo: 'statevector','aer','aer_noise','hardware', defecto='statevector'
    # optimizador: 'COBYLA','SPSA','SLSQP' (SLSQP=gradiente, ideal para statevector). Defecto='COBYLA'
    # ansatz_str: 'EfficientSU2', defecto='EfficientSU2'
    # reps: nº de repeticiones del ansatz EfficientSU2 (más reps = más expresivo). Defecto=1
    # grafico: 'f'-> [-pi,pi],  'm'-> [0,pi],   'n'-> sin gráfico,  defecto='n'
    # b (Número de estados excitados en VQD): >=0, defecto=0
    # shots (modos 'aer','aer_noise','hardware'): defecto = 1024
    # noise_model (modo 'aer_noise'): 'torino','kyiv','sherbrooke' (127-133 cúbits, LENTOS);
    #             'manila','lima','belem','quito' (5 cúbits, rápidos); 'lagos','nairobi','perth'
    #             (7 cúbits); 'generic' (máquina fake de num_qubits cúbits, la más rápida). Defecto='torino'
    # n_ticks (número de puntos a calcular entre 0 y pi): >2, defecto=12
    # t (hopping): defecto=-1.0
    # fichero_salida: defecto=None
    # fichero_figura: defecto=None
    # plot_energia: 'ideal' (statevector) o 'medida' (con ruido/hardware), defecto='ideal'
    # --- Parámetros para modo='hardware' ---
    # token: token de IBM Quantum (o usar credenciales guardadas). Defecto=None
    # channel: 'ibm_quantum_platform','ibm_cloud','local'. Defecto='ibm_quantum_platform'
    # instance: instancia/CRN de IBM Quantum. Defecto=None
    # backend_name: nombre del backend real; si None se usa el menos ocupado. Defecto=None
    # optimization_level: nivel de transpilación (0-3). Defecto=1
    # resilience_level: mitigación de errores del estimador (0-2). Defecto=1
    # backend_obj: backend ya construido (para pruebas locales con un fake backend). Defecto=None
    # usar_session: True usa modo Session (solo planes de pago); False usa "job mode"
    #               (obligatorio en el plan Open gratuito). Defecto=False
    # optimizar_en (solo modo='hardware'): 'sim' optimiza los parámetros en simulación
    #               (statevector, gratis) y SOLO mide la energía del circuito óptimo en el
    #               hardware real (1 job por k y banda; recomendado para el plan Open).
    #               'hardware' hace toda la optimización en el hardware (cientos de jobs por k,
    #               inviable en Open). Defecto='sim'
    # --- Parámetros del análisis de contención en subespacios (subspace containment) ---
    # containment: si True, para cada punto k y cada estado variacional calculado se evalúa
    #              su contención C_alpha en los autoespacios exactos de baja energía (statevector
    #              de los parámetros optimizados). Se guarda en el JSON. Defecto=False
    # n_espacios: nº de autoespacios exactos de menor energía sobre los que se reporta la
    #             contención. Si None -> (b+2) para ver también la fuga al siguiente. Defecto=None
    # tol_deg: tolerancia para agrupar autovalores casi degenerados en un mismo autoespacio.
    #          Defecto=1e-6
    # n_restarts: nº de semillas desde las que se optimiza cada estado en cada punto k. La
    #             primera es siempre la del warm-starting; las demás, aleatorias. Se queda con
    #             la de menor coste IDEAL (energía + penalización sobre el statevector), lo que
    #             evita que SPSA se quede en un mínimo local y un estado se salte un nivel.
    #             n_restarts=1 (defecto) reproduce exactamente el comportamiento anterior.
    #             OJO: con modo='hardware' y optimizar_en='hardware' multiplica los jobs por
    #             n_restarts; con optimizar_en='sim' los reinicios son gratis (statevector).

    # Comprobación de los parámetros
    if num_qubits <= 1: raise Exception("El número de cubits debe ser mayor que 1")
    if tipo.lower() not in ['a','z']: raise Exception("El tipo debe ser 'a' o 'z'")
    if modo.lower() not in ['statevector','aer','aer_noise','hardware']:
        raise Exception("Modo no reconocido. Opciones:'statevector','aer','aer_noise','hardware'")
    if optimizador.upper() not in ['COBYLA','SPSA','SLSQP']: raise Exception("Optimizador no reconocido. Opciones:'COBYLA','SPSA','SLSQP'")
    if ansatz_str.lower() != 'efficientsu2': raise Exception("Ansatz no reconocido. Opciones:'EfficientSU2'")
    if grafico.lower() not in ['f','m','n']: raise Exception("El valor grafico debe ser 'f' o 'm' o 'n'")
    if shots <= 0: raise Exception("El número de shots debe ser mayor que 0")
    if b < 0: raise Exception("El valor b debe ser mayor o igual que 0")
    if (b >= 2**num_qubits): b = 2**num_qubits -1
    if noise_model.lower() not in FAKE_BACKENDS and noise_model.lower() != 'generic':
        raise Exception("noise_model no reconocido. Opciones: " + ", ".join(list(FAKE_BACKENDS) + ['generic']))
    if n_ticks <= 2: raise Exception("El número de puntos debe ser mayor que 2")
    if plot_energia.lower() not in ['ideal','medida']: raise Exception("plot_energia debe ser 'ideal' o 'medida'")
    if optimizar_en.lower() not in ['sim','hardware']: raise Exception("optimizar_en debe ser 'sim' o 'hardware'")
    optimizar_en = optimizar_en.lower()

    modo = modo.lower()
    tipo = tipo.lower()

    inicio_reloj = time.perf_counter()
    json_data = {}
    if fichero_salida is not None:
        import sys
        import qiskit
        import qiskit_aer
        import qiskit_algorithms
        import qiskit_ibm_runtime
        json_data = {
            "versiones": {
                "python": sys.version,
                "qiskit": qiskit.__version__,
                "qiskit-aer": qiskit_aer.__version__,
                "qiskit-algorithms": qiskit_algorithms.__version__,
                "qiskit-ibm-runtime": qiskit_ibm_runtime.__version__
            },
            "parametros": {
                "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_qubits": num_qubits,
                "tipo": tipo,
                "modo": modo,
                "optimizador": optimizador,
                "ansatz": ansatz_str,
                "reps": reps,
                "shots": shots,
                "noise_model": noise_model,
                "backend_name": backend_name,
                "resilience_level": resilience_level,
                "optimizar_en": optimizar_en,
                "b": b,
                "n_ticks": n_ticks,
                "t": t,
                "n_restarts": n_restarts
            },
            "tiempo-proceso": 0,
            "resultados": []
        }

    # k-points
    if grafico == 'f': k_points = np.linspace(-np.pi, np.pi, 2*n_ticks)
    else: k_points = np.linspace(0, np.pi, n_ticks)

    # optimizer
    optimizer = COBYLA(maxiter=maxiter)
    optimizador = optimizador.upper()
    if optimizador == 'SPSA': optimizer = SPSA(maxiter=maxiter)
    elif optimizador == 'SLSQP': optimizer = SLSQP(maxiter=maxiter)  # gradiente: rápido y preciso en statevector

    # ansatz
    ansatz = EfficientSU2(num_qubits=num_qubits, reps=reps)
    ansatz_str = ansatz_str.lower()

    # --- Preparación según el modo ---
    estimator = None
    sampler = None
    ansatz_isa = None
    pm = None
    session = None
    overlap_template = None
    if modo == 'hardware':
        from qiskit_ibm_runtime import Session
        from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimator, SamplerV2 as RuntimeSampler
        service, backend, pm, ansatz_isa = preparar_hardware(
            num_qubits, ansatz, token=token, channel=channel, instance=instance,
            backend_name=backend_name, optimization_level=optimization_level, backend_obj=backend_obj)
        # IMPORTANTE: el plan Open (gratuito) de IBM NO permite el modo Session.
        # Por defecto usamos "job mode" (mode=backend), válido en todos los planes.
        # usar_session=True solo si el plan lo permite (planes de pago).
        if usar_session and service is not None:
            mode = Session(backend=backend)
        else:
            mode = backend
        estimator = RuntimeEstimator(mode=mode, options={"default_shots": shots, "resilience_level": resilience_level})
        sampler = RuntimeSampler(mode=mode, options={"default_shots": shots})
        # Plantilla de solapamiento: solo si se optimiza EN hardware y hay estados excitados.
        # (Con optimizar_en='sim' la fidelidad se calcula por statevector, gratis.)
        if b > 0 and optimizar_en == 'hardware':
            overlap_template = construir_overlap_template(ansatz, pm)
        # Trazabilidad: registrar el backend REAL resuelto (no solo el parámetro backend_name)
        if fichero_salida is not None:
            json_data["parametros"]["backend_name"] = getattr(backend, "name", str(backend))
        print("Backend hardware:", getattr(backend, "name", backend), "| optimizar_en:", optimizar_en)
    else:
        estimator = crear_estimator(modo, noise_model, shots, num_qubits)

    print("Modo:", modo)
    print("Qubits ansatz:", ansatz.num_qubits)
    print("Parámetros:", ansatz.num_parameters)
    if modo == 'aer_noise': print("Noise Model:", noise_model)
    print(ansatz)

    beta = 10
    energias_ideal = []    # energía IDEAL (statevector) por punto k -> retrocompatibilidad
    energias_medidas = []  # energía MEDIDA (con ruido/hardware) por punto k
    energias_exactas = []
    containment_por_k = []  # (subspace containment) contención por k, estado y autoespacio

    # nº de autoespacios de baja energía a reportar en el análisis de contención
    if n_espacios is None:
        n_espacios = b + 2

    c = 'VQE'
    if b > 0: c = 'VQD'
    print(f"Iniciando cálculo {c} para {num_qubits} cubits...")

    # --- MATRIZ DE HISTORIAL PARA WARM-STARTING ---
    historial_params = [np.pi * np.random.random(ansatz.num_parameters) for _ in range(b + 1)]

    # --- Autovalores exactos en ORDEN DE BANDA (diabático), coherentes con el gráfico ---
    # (espectro completo 2^num_qubits; se calcula una sola vez sobre todos los k)
    E_dia = autovalores_diabaticos(num_qubits, tipo, t, k_points)

    try:
        # --- CÁLCULOS ---
        for n, k in enumerate(k_points):
            # -- Referencia exacta
            H = build_matrix(num_qubits, tipo, k, t)
            resultados_exactos = np.linalg.eigvalsh(H)

            # -- VQE y VQD
            operator = SparsePauliOp.from_operator(H)
            try:
                evaluaciones = [0] * (b + 1)
                resultados_ideal = []
                resultados_medida = []

                if modo == 'hardware':
                    # --- Ruta hardware real ---
                    operator_isa = operator.apply_layout(ansatz_isa.layout)
                    if optimizar_en == 'sim':
                        # ESTRATEGIA RECOMENDADA (plan Open): optimizar en simulación
                        # (statevector, gratis) y medir el circuito óptimo en hardware
                        # -> solo 1 job por (k, banda).
                        wavefuncs = []
                        for eidx in range(b + 1):
                            cost = lambda par: cost_func(beta, par, ansatz, operator, None,
                                                         wavefuncs, eidx, evaluaciones, 'statevector')
                            params_opt, opt_sv, e_ideal = optimizar_con_reinicios(
                                optimizer, cost, ansatz, operator, wavefuncs, beta,
                                historial_params[eidx], n_restarts)
                            historial_params[eidx] = params_opt
                            wavefuncs.append(opt_sv)
                            # Única llamada al hardware: energía del circuito óptimo
                            job = estimator.run([(ansatz_isa, operator_isa, params_opt)])
                            e_medida = float(job.result()[0].data.evs)
                            resultados_ideal.append(e_ideal)
                            resultados_medida.append(e_medida)
                    else:
                        # Optimización COMPLETA en hardware (cientos de jobs por k; inviable en Open)
                        prev_params = []
                        wavefuncs = []
                        for eidx in range(b + 1):
                            inicial_guess = historial_params[eidx]
                            cost = lambda par: cost_func_hardware(
                                beta, par, num_qubits, ansatz_isa, operator_isa, estimator, sampler,
                                overlap_template, prev_params, eidx, evaluaciones)
                            resultado = optimizer.minimize(cost, inicial_guess)
                            historial_params[eidx] = resultado.x
                            prev_params.append(resultado.x)
                            # Energía medida (hardware) sobre los parámetros óptimos
                            job = estimator.run([(ansatz_isa, operator_isa, resultado.x)])
                            e_medida = float(job.result()[0].data.evs)
                            # Energía ideal de referencia (statevector, barato para pocos cúbits)
                            opt_sv = Statevector.from_instruction(ansatz.assign_parameters(resultado.x))
                            wavefuncs.append(opt_sv)
                            e_ideal = opt_sv.expectation_value(operator).real
                            resultados_ideal.append(e_ideal)
                            resultados_medida.append(e_medida)
                else:
                    # --- Rutas de simulación (statevector / aer / aer_noise) ---
                    wavefuncs = []
                    for eidx in range(b + 1):
                        cost = lambda par: cost_func(beta, par, ansatz, operator, estimator,
                                                     wavefuncs, eidx, evaluaciones, modo)
                        params_opt, opt_sv, e_ideal = optimizar_con_reinicios(
                            optimizer, cost, ansatz, operator, wavefuncs, beta,
                            historial_params[eidx], n_restarts)
                        historial_params[eidx] = params_opt
                        wavefuncs.append(opt_sv)
                        if modo == 'statevector':
                            e_medida = e_ideal
                        else:
                            job = estimator.run([(ansatz.assign_parameters(params_opt), operator)])
                            e_medida = float(job.result()[0].data.evs)
                        resultados_ideal.append(e_ideal)
                        resultados_medida.append(e_medida)

                energias_exactas.append(resultados_exactos)
                energias_ideal.append(resultados_ideal)
                energias_medidas.append(resultados_medida)

                # --- Análisis de contención en subespacios (subspace containment) ---
                cont_k = None
                if containment:
                    w_ex, V_ex = np.linalg.eigh(H)                 # autovalores + autovectores exactos
                    grupos = espacios_propios(w_ex, tol_deg)       # autoespacios (agrupa degenerados)
                    grupos_low = grupos[:n_espacios]               # los de menor energía
                    energias_esp = [float(np.mean([w_ex[i] for i in g])) for g in grupos_low]
                    degeneracion = [len(g) for g in grupos_low]
                    if containment_hw and modo == 'hardware':
                        # Contención MEDIDA en hardware (proyección en la base propia, 1 job/estado);
                        # mide el estado realmente preparado en el dispositivo.
                        matriz, total = [], []
                        for e in range(b + 1):
                            c_all = containment_hw_por_muestreo(sampler, pm, ansatz,
                                                                historial_params[e], V_ex, grupos, num_qubits)
                            matriz.append(c_all[:n_espacios])
                            total.append(float(sum(c_all)))
                    else:
                        # matriz (estado calculado x autoespacio) de contención (statevector ideal)
                        matriz = [subspace_containment(sv.data, V_ex, grupos_low) for sv in wavefuncs]
                        # verificación de completitud: contención sobre TODOS los autoespacios ~ 1
                        total = [float(sum(subspace_containment(sv.data, V_ex, grupos))) for sv in wavefuncs]
                    cont_k = {
                        "k": float(k),
                        "energias_autoespacios": energias_esp,
                        "degeneracion": degeneracion,
                        "matriz": matriz,               # matriz[estado][autoespacio] = C_alpha
                        "suma_total": total             # ~1.0 si la base es consistente
                    }
                    containment_por_k.append(cont_k)
                    diag = [matriz[e][e] if e < len(grupos_low) else float('nan')
                            for e in range(len(wavefuncs))]
                    print(f"      contención (diag C_e,e) = {np.round(diag,3)} | Σ={np.round(total,3)}")

                print(f"k = {k:.2f} | E_ideal = {np.round(resultados_ideal,4)} | "
                      f"E_medida = {np.round(resultados_medida,4)} | "
                      f"E_exacto = {np.round(energias_exactas[n][:b+1],4)} | iter= {evaluaciones}")
                if fichero_salida is not None:
                    # Reordenar todo por BANDA física (diabática) para que datos y gráfico
                    # sean coherentes (índice j = banda j en las cuatro listas).
                    orden, bandas = asignar_bandas(resultados_ideal, E_dia[n])
                    resultado_k = {
                        "k": float(k),
                        "bandas": [int(bj) for bj in bandas],
                        "valores_exactos": [float(E_dia[n][bj]) for bj in bandas],
                        "valores_calculados": [float(resultados_ideal[i]) for i in orden],
                        "valores_medidos": [float(resultados_medida[i]) for i in orden],
                        "iteraciones": [int(evaluaciones[i]) for i in orden]
                    }
                    if cont_k is not None:
                        resultado_k["containment"] = cont_k
                    json_data["resultados"].append(resultado_k)
            except Exception as e:
                print(f"Error en k={k:.2f}: {e}")
                energias_exactas.append([np.nan] * (b + 1))
                energias_ideal.append([np.nan] * (b + 1))
                energias_medidas.append([np.nan] * (b + 1))
    finally:
        if session is not None:
            session.close()

    fin_reloj = time.perf_counter()
    print(f"* TIEMPO TOTAL DEL PROCESO: {fin_reloj - inicio_reloj:.4f} segundos")
    # --- Almacenar los parámetros en el fichero de salida ---
    if fichero_salida is not None:
        json_data["tiempo-proceso"] = fin_reloj - inicio_reloj
        if os.path.exists(fichero_salida):
            os.remove(fichero_salida)
        with open(fichero_salida, "w", encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

    # --- GRÁFICA ---
    if grafico != 'n':
        datos_plot = energias_medidas if plot_energia.lower() == 'medida' else energias_ideal
        datos_plot = np.array(datos_plot)
        energias_exactas = np.array(energias_exactas)
        plt.figure(figsize=(10, 6))

        if grafico == 'f': plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[ '-π', '-π/2', '0', 'π/2', 'π'])
        else: plt.xticks([0, np.pi/2, np.pi],['0', 'π/2', 'π'])

        # Líneas exactas en ORDEN DE BANDA (diabático, ya calculado): cruzan suavemente
        for i in range(2**num_qubits):
            plt.plot(k_points, E_dia[:, i], color='black', alpha=0.4)

        if b == 0:
            valid_indices = [i for i, v in enumerate(datos_plot) if v is not None]
            plt.scatter([k_points[i] for i in valid_indices], [datos_plot[i] for i in valid_indices],
                            color='red', label=f'{c} Banda de Valencia')
            plt.legend()
        else:
            # Cada punto se colorea según la banda física (diabática) más cercana en ese k,
            # de modo que el color no "salte" de una banda a otra en los cruces.
            colores = plt.rcParams['axes.prop_cycle'].by_key()['color']
            puntos = {bd: ([], []) for bd in range(2**num_qubits)}
            for ik in range(len(k_points)):
                fila = datos_plot[ik]
                if fila is None:
                    continue
                for val in np.atleast_1d(fila):
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        continue
                    bd = int(np.argmin(np.abs(E_dia[ik] - val)))
                    puntos[bd][0].append(k_points[ik]); puntos[bd][1].append(val)
            for bd in range(2**num_qubits):
                xs, ys = puntos[bd]
                if xs:
                    plt.scatter(xs, ys, color=colores[bd % len(colores)], label=f'{c} Banda {bd}')
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles[::-1], labels[::-1])

        titulo = f"Estructura de Bandas {'zGNR' if tipo=='z' else 'aGNR'}"
        etiqueta_energia = 'ideal' if plot_energia.lower() == 'ideal' else 'medida'
        texto = f"{c} con {num_qubits} cubits\nOptimizador: {optimizador}\nAnsatz: {ansatz_str}\nModo: {modo} (E {etiqueta_energia})"
        if modo != 'statevector':
            texto += f" ({shots} shots)"
            if modo == 'aer_noise':
                texto += f" - Noise Model: {noise_model}"
            elif modo == 'hardware':
                texto += f" - Backend: {backend_name or 'least_busy'}"
        plt.suptitle(titulo, y=0.97, fontsize=16, style='italic')
        plt.xlabel("Vector de onda k")
        plt.ylabel("Energía (eV)")
        plt.axhline(0, color='red', linestyle=':', alpha=0.2)
        plt.title(texto)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if fichero_figura is not None:
            plt.savefig(fichero_figura, dpi=200, bbox_inches='tight')
            print("Figura guardada en:", fichero_figura)
        plt.show()

    return energias_ideal, energias_exactas


# ============================================================================
#   GRAFENO MONOCAPA (2D): VQD de las 2 bandas + contención, camino Γ-K-M-Γ
#   Reutiliza cost_func, crear_estimator, espacios_propios y subspace_containment.
# ============================================================================
def calculos_monocapa(num_points=12, modo='statevector', optimizador='COBYLA',
                      t=1.0, t_prima=1.0/28, a=1.42, vecinos=2, reps=1,
                      shots=1024, noise_model='generic', maxiter=1000, beta=10,
                      containment=False, n_espacios=None, tol_deg=1e-6,
                      fichero_salida=None,
                      token=None, channel='ibm_quantum_platform', instance=None,
                      backend_name=None, optimization_level=1, resilience_level=1,
                      backend_obj=None, usar_session=False, optimizar_en='sim',
                      containment_hw=False):
    # num_points: nº de puntos por segmento del camino Γ→K→M→Γ. Defecto=12
    # containment_hw: si True (modo hardware), la contención se MIDE en el dispositivo por
    #                 proyección en la base propia (1 job/estado), en vez de por statevector.
    # modo: 'statevector','aer','aer_noise','hardware'. Defecto='statevector'
    # optimizador: 'COBYLA','SPSA','SLSQP'. t/t_prima/a/vecinos: parámetros del modelo (como el TFM)
    # reps: repeticiones del EfficientSU2 (1 cubit). containment/n_espacios/tol_deg: análisis de contención
    # --- Parámetros de modo='hardware' (idénticos a calculos) ---
    # token/channel/instance/backend_name/optimization_level/resilience_level/backend_obj/usar_session
    # optimizar_en: 'sim' (optimiza en simulación y solo MIDE la energía del circuito óptimo en HW,
    #               1 job por (k,banda); recomendado, plan Open) o 'hardware' (optimización COMPLETA
    #               en HW; necesario para que la CONTENCIÓN refleje el ruido real, pero costoso).
    modo = modo.lower()
    if modo not in ['statevector', 'aer', 'aer_noise', 'hardware']:
        raise Exception("modo debe ser 'statevector','aer','aer_noise' o 'hardware'")
    if optimizador.upper() not in ['COBYLA', 'SPSA', 'SLSQP']:
        raise Exception("Optimizador no reconocido")
    if vecinos not in (1, 2):
        raise Exception("vecinos debe ser 1 o 2")
    optimizar_en = optimizar_en.lower()
    if optimizar_en not in ['sim', 'hardware']:
        raise Exception("optimizar_en debe ser 'sim' o 'hardware'")

    b = 1          # 2 bandas: fundamental (valencia) + 1 excitado (conducción)
    nb = 2
    if n_espacios is None:
        n_espacios = nb

    optimizer = COBYLA(maxiter=maxiter)
    ou = optimizador.upper()
    if ou == 'SPSA': optimizer = SPSA(maxiter=maxiter)
    elif ou == 'SLSQP': optimizer = SLSQP(maxiter=maxiter)

    ansatz = EfficientSU2(num_qubits=1, reps=reps)
    k_vals, coord, pos = camino_monocapa(num_points, a)

    # --- Preparación según el modo ---
    estimator = None
    sampler = None
    ansatz_isa = None
    overlap_template = None
    backend_resuelto = backend_name
    if modo == 'hardware':
        from qiskit_ibm_runtime import Session
        from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimator, SamplerV2 as RuntimeSampler
        service, backend, pm, ansatz_isa = preparar_hardware(
            1, ansatz, token=token, channel=channel, instance=instance,
            backend_name=backend_name, optimization_level=optimization_level, backend_obj=backend_obj)
        mode = Session(backend=backend) if (usar_session and service is not None) else backend
        estimator = RuntimeEstimator(mode=mode, options={"default_shots": shots, "resilience_level": resilience_level})
        sampler = RuntimeSampler(mode=mode, options={"default_shots": shots})
        if b > 0 and optimizar_en == 'hardware':
            overlap_template = construir_overlap_template(ansatz, pm)
        backend_resuelto = getattr(backend, "name", str(backend))
        print("Backend hardware:", backend_resuelto, "| optimizar_en:", optimizar_en)
    else:
        # El circuito es de 1 cubit, pero un backend genérico de 1 cubit no admite la puerta cx;
        # se construye el modelo de ruido con 2 cubits (solo actúa el ruido de puertas de 1 cubit).
        estimator = crear_estimator(modo, noise_model, shots, 2)

    print("=== GRAFENO MONOCAPA (2D) ===  modo:", modo, "| vecinos:", vecinos,
          "| puntos:", len(k_vals))

    json_data = {}
    if fichero_salida is not None:
        import sys, qiskit, qiskit_aer, qiskit_algorithms, qiskit_ibm_runtime
        json_data = {
            "versiones": {"python": sys.version, "qiskit": qiskit.__version__,
                          "qiskit-aer": qiskit_aer.__version__,
                          "qiskit-algorithms": qiskit_algorithms.__version__,
                          "qiskit-ibm-runtime": qiskit_ibm_runtime.__version__},
            "parametros": {"fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "sistema": "monocapa", "num_points": num_points, "modo": modo,
                           "optimizador": ou, "reps": reps, "shots": shots,
                           "noise_model": noise_model, "maxiter": maxiter,
                           "backend_name": backend_resuelto, "resilience_level": resilience_level,
                           "optimizar_en": optimizar_en,
                           "t": t, "t_prima": t_prima, "a": a, "vecinos": vecinos},
            "camino": {"coord": [float(x) for x in coord], "pos": pos,
                       "etiquetas": ["Γ", "K", "M", "Γ"]},
            "tiempo-proceso": 0, "resultados": []}

    historial = [np.pi * np.random.random(ansatz.num_parameters) for _ in range(b + 1)]
    inicio = time.perf_counter()
    energias_ideal, energias_medidas, energias_exactas = [], [], []

    for n, k in enumerate(k_vals):
        H = hamiltoniano_monocapa(k, t, t_prima, a, vecinos)
        operator = SparsePauliOp.from_operator(H)
        w_ex, V_ex = np.linalg.eigh(H)
        evals = [0] * (b + 1)
        wavefuncs, res_ideal, res_med = [], [], []

        if modo == 'hardware':
            operator_isa = operator.apply_layout(ansatz_isa.layout)
            prev_params = []
            for eidx in range(b + 1):
                if optimizar_en == 'sim':
                    cost = lambda p: cost_func(beta, p, ansatz, operator, None,
                                               wavefuncs, eidx, evals, 'statevector')
                else:
                    cost = lambda p: cost_func_hardware(beta, p, 1, ansatz_isa, operator_isa,
                                                        estimator, sampler, overlap_template,
                                                        prev_params, eidx, evals)
                r = optimizer.minimize(cost, historial[eidx])
                historial[eidx] = r.x
                prev_params.append(r.x)
                opt_sv = Statevector.from_instruction(ansatz.assign_parameters(r.x))
                wavefuncs.append(opt_sv)
                e_ideal = opt_sv.expectation_value(operator).real
                # Única medida en HW: energía del circuito óptimo
                job = estimator.run([(ansatz_isa, operator_isa, r.x)])
                e_med = float(job.result()[0].data.evs)
                res_ideal.append(e_ideal); res_med.append(e_med)
        else:
            for eidx in range(b + 1):
                cost = lambda p: cost_func(beta, p, ansatz, operator, estimator,
                                           wavefuncs, eidx, evals, modo)
                r = optimizer.minimize(cost, historial[eidx])
                historial[eidx] = r.x
                sv = Statevector.from_instruction(ansatz.assign_parameters(r.x))
                wavefuncs.append(sv)
                e_ideal = sv.expectation_value(operator).real
                if modo == 'statevector':
                    e_med = e_ideal
                else:
                    job = estimator.run([(ansatz.assign_parameters(r.x), operator)])
                    e_med = float(job.result()[0].data.evs)
                res_ideal.append(e_ideal); res_med.append(e_med)
        energias_ideal.append(res_ideal); energias_medidas.append(res_med)
        energias_exactas.append(w_ex)

        # --- Contención en subespacios ---
        cont_k = None
        if containment:
            grupos = espacios_propios(w_ex, tol_deg)
            grupos_low = grupos[:n_espacios]
            if containment_hw and modo == 'hardware':
                matriz, total = [], []
                for e in range(b + 1):
                    c_all = containment_hw_por_muestreo(sampler, pm, ansatz,
                                                        historial[e], V_ex, grupos, 1)
                    matriz.append(c_all[:n_espacios])
                    total.append(float(sum(c_all)))
            else:
                matriz = [subspace_containment(sv.data, V_ex, grupos_low) for sv in wavefuncs]
                total = [float(sum(subspace_containment(sv.data, V_ex, grupos))) for sv in wavefuncs]
            cont_k = {"energias_autoespacios": [float(np.mean([w_ex[i] for i in g])) for g in grupos_low],
                      "degeneracion": [len(g) for g in grupos_low],
                      "matriz": matriz, "suma_total": total}
            diag = [matriz[e][e] if e < len(grupos_low) else float('nan') for e in range(b + 1)]
            print(f"  k[{n:2d}] E={np.round(np.sort(w_ex),4)} | C={np.round(diag,3)} | Σ={np.round(total,3)}")
        else:
            print(f"  k[{n:2d}] E_calc={np.round(sorted(res_ideal),4)} | E_exact={np.round(np.sort(w_ex),4)}")

        if fichero_salida is not None:
            orden = list(np.argsort(res_ideal))    # por energía ascendente (valencia, conducción)
            rk = {"n": n, "k": [float(k[0]), float(k[1])], "coord": float(coord[n]),
                  "valores_exactos": [float(x) for x in np.sort(w_ex)],
                  "valores_calculados": [float(res_ideal[i]) for i in orden],
                  "valores_medidos": [float(res_med[i]) for i in orden]}
            if cont_k is not None:
                rk["containment"] = cont_k
            json_data["resultados"].append(rk)

    fin = time.perf_counter()
    print(f"* TIEMPO TOTAL: {fin - inicio:.2f} s")
    if fichero_salida is not None:
        json_data["tiempo-proceso"] = fin - inicio
        if os.path.exists(fichero_salida):
            os.remove(fichero_salida)
        with open(fichero_salida, "w", encoding='utf-8') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

    return energias_ideal, energias_exactas


# ***********************************************
#                Ejemplos de uso
# ***********************************************

if __name__ == "__main__":

    # Ejemplo 1: simulación ideal (statevector), zigzag, 2 cúbits, 3 bandas excitadas
    calculos(num_qubits=2, tipo='z', b=3, optimizador='SLSQP', grafico='f', modo='statevector')

    # Ejemplo 2: simulación con RUIDO RÁPIDA usando una máquina fake pequeña.
    # 'generic' crea una máquina de num_qubits cubits (la más rápida). También sirven
    # 'manila'/'lima'/'belem'/'quito' (5 cúbits) o 'lagos'/'nairobi'/'perth' (7 cúbits).
    # calculos(num_qubits=2, tipo='a', b=3, optimizador='COBYLA', grafico='m',
    #         modo='aer_noise', noise_model='generic', plot_energia='medida')

    # Ejemplo 3: HARDWARE REAL de IBM Quantum (requiere token y créditos).
    # Se recomienda b pequeño y pocos n_ticks por el coste.
    #calculos(num_qubits=2, tipo='a', b=0, n_ticks=5, grafico='m', modo='hardware',
    #          token="<TU_TOKEN_IBM>", channel='ibm_quantum_platform',
    #          backend_name=None,  # None -> backend menos ocupado
    #          fichero_salida='hardware_agnr.json')
