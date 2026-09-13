"""
conectar_ibm.py

Utilidad para conectar con IBM Quantum Platform (IBM Cloud) y verificar que todo
funciona ANTES de gastar minutos de cómputo real.

Uso recomendado (en este orden):
    1) guardar_cuenta(...)      -> se ejecuta UNA sola vez para guardar credenciales.
    2) probar_conexion()        -> NO gasta cuota: lista backends y el menos ocupado.
    3) prueba_minima_hardware() -> SÍ gasta cuota: 1 cálculo VQE minúsculo en hardware real.

Canal: 'ibm_quantum_platform' (el antiguo 'ibm_quantum' está retirado).

SEGURIDAD: no dejes tu API key escrita en este fichero si lo vas a compartir.
save_account la guarda en ~/.qiskit/qiskit-ibm.json, fuera del código.
"""

from qiskit_ibm_runtime import QiskitRuntimeService


CHANNEL = "ibm_quantum_platform"


# ---------------------------------------------------------------------------
# PASO 1 — Guardar la cuenta (ejecutar UNA sola vez)
# ---------------------------------------------------------------------------
def guardar_cuenta(token, instance, set_as_default=True):
    """
    Guarda las credenciales en disco (~/.qiskit/qiskit-ibm.json) para no tener
    que repetir el token en cada ejecución.

    token   : tu API key de IBM Quantum Platform.
    instance: el CRN de tu instancia (crn:v1:bluemix:public:quantum-computing:...).
    """
    QiskitRuntimeService.save_account(
        channel=CHANNEL,
        token=token,
        instance=instance,
        set_as_default=set_as_default,
        overwrite=True,
    )
    print("Credenciales guardadas correctamente en ~/.qiskit/qiskit-ibm.json")


# ---------------------------------------------------------------------------
# PASO 2 — Probar la conexión (NO gasta cuota)
# ---------------------------------------------------------------------------
def probar_conexion(token=None, instance=None):
    """
    Se conecta al servicio y lista los backends reales disponibles y el menos
    ocupado. No ejecuta ningún circuito, así que NO consume minutos.

    Si ya has ejecutado guardar_cuenta(), llama sin argumentos: probar_conexion().
    En caso contrario, pásale token e instance.
    """
    if token is None:
        # Usa la cuenta guardada por save_account()
        service = QiskitRuntimeService()
    else:
        service = QiskitRuntimeService(channel=CHANNEL, token=token, instance=instance)

    print("Conexión establecida.\n")
    backends = service.backends(operational=True, simulator=False)
    print(f"Backends reales operativos ({len(backends)}):")
    for b in backends:
        try:
            cola = b.status().pending_jobs
        except Exception:
            cola = "?"
        print(f"  - {b.name:20s} | {b.num_qubits:4d} cúbits | cola: {cola}")

    menos_ocupado = service.least_busy(operational=True, simulator=False)
    print(f"\nBackend menos ocupado ahora mismo: {menos_ocupado.name} "
          f"({menos_ocupado.num_qubits} cúbits)")
    return service


# ---------------------------------------------------------------------------
# PASO 3 — Prueba mínima en hardware real (SÍ gasta cuota)
# ---------------------------------------------------------------------------
def prueba_minima_hardware(token=None, instance=None, backend_name=None):
    """
    Lanza el cálculo MÁS PEQUEÑO posible en hardware real para confirmar que la
    cadena completa funciona: VQE (b=0), 2 cúbits, solo 3 puntos k.

    ATENCIÓN: esto SÍ consume minutos de cómputo. Ejecútalo solo cuando el PASO 2
    haya funcionado. Si no indicas backend_name, se usa el menos ocupado.
    """
    from version_1 import calculos
    print("Lanzando prueba mínima en hardware real (VQE, 2 cúbits, 3 puntos k)...")
    print("Esto puede tardar (cola + ejecución) y consume cuota.\n")
    energias, exactas = calculos(
        num_qubits=2, tipo='a', b=0, n_ticks=3,
        modo='hardware', grafico='n',
        token=token, instance=instance, channel=CHANNEL,
        backend_name=backend_name,
        shots=1024, optimization_level=1, resilience_level=1,
        fichero_salida='prueba_hardware.json',
    )
    print("\nPrueba completada. Resultados guardados en 'prueba_hardware.json'.")
    return energias, exactas


# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # ----- PASO 1: descomenta y rellena SOLO la primera vez -----
    #guardar_cuenta(token="TU_API_KEY", instance="TU_CRN")
    
    # ----- PASO 2: probar conexión (no gasta cuota) -----
    # Si ya guardaste la cuenta:
    #probar_conexion()
    # Si NO la guardaste, usa:
    # probar_conexion(token="TU_API_KEY", instance="TU_CRN")

    # ----- PASO 3: prueba mínima en hardware real (gasta cuota) -----
    # Descomenta cuando el paso 2 funcione:
    # prueba_minima_hardware()

    #from version_1 import calculos
    #calculos(num_qubits=2, tipo='z', b=3, n_ticks=9, modo='hardware', grafico='m',fichero_salida='hardware_agnr_03.json')

