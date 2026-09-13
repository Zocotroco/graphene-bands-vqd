# Plan de ejercicios en hardware real (IBM Quantum, plan Open)

Objetivo: completar la columna "reales" (hardware) para presentar, junto a "sin ruido"
(statevector) y "con ruido" (aer_noise), el conjunto completo pedido por los directores:
bandas del grafeno **monocapa** y análisis de **contención** (propuesta de Juan) en máquina real.

## Estado de partida
- Nanocintas — **bandas en HW ya hechas** (ibm_fez / ibm_kingston): `hardware_{agnr,zgnr}_2q.json` → Fig. 8.
- Monocapa — bandas y containment en HW: **pendientes** (estos ejercicios).
- Nanocintas — containment en HW: **costoso** (3 cubits, optimización completa en HW); ver HW-3.

## Consideraciones de coste (plan Open)
- **Presupuesto medido (dato del usuario):** cada job consume ~**11 s** de tiempo de proceso;
  una cuenta tiene ~10 min → **~54 jobs por cuenta**. (HW-1 usó 20 jobs ≈ 3 min 40 s.)
- El plan Open usa "job mode" (sin sesión). El **cómputo cuántico** por trabajo es de segundos;
  el factor limitante en la práctica es la **latencia de la cola** del dispositivo compartido.
- **Contención en HW por MEDICIÓN EN LA BASE PROPIA** (método adoptado): se optimiza θ en
  simulación (converge, gratis) y en el dispositivo se prepara U(θ), se aplica el cambio a la
  base propia exacta V† de H(k) y se mide → la distribución da la contención del estado
  realmente preparado. **1 job por (k, estado)**, en vez de la optimización completa en HW
  (~cientos de jobs, inviable) o la tomografía completa (9 bases/estado). Param `containment_hw=True`.
- Estrategia `optimizar_en='sim'`: optimiza los parámetros en simulación (gratis) y **solo mide**
  la energía del circuito óptimo en HW → **1 job por (k, banda)**. Barato. Sirve para BANDAS.
- Estrategia `optimizar_en='hardware'`: optimización VQD **completa** en HW → muchos jobs.
  Necesaria para que la CONTENCIÓN refleje el ruido real de la optimización (como en el paper
  de Juan). Solo viable con 1 cubit (monocapa) y pocos puntos k.

## Ejercicios

### HW-1 — Bandas del monocapa en hardware real  (barato, ~20 jobs)
`ejercicio_hw1_bandas_monocapa.py`
- `calculos_monocapa(modo='hardware', optimizar_en='sim', num_points=2)` → ~10 puntos k, 2 bandas.
- Jobs ≈ 10 × 2 = **20** (1 por (k, banda)). Cómputo QPU ~segundos cada uno.
- Salida: `hardware_monocapa_bandas.json` → figura análoga a la de bandas del monocapa, con
  energías MEDIDAS en HW frente a la exacta. (Columna "reales" de las bandas 2D.)

### HW-2 — Contención del monocapa en hardware real  (~40 jobs, cabe en 1 cuenta)
`ejercicio_hw2_containment_monocapa.py`  (v2: medición en la base propia, `containment_hw=True`)
- `calculos_monocapa(modo='hardware', optimizar_en='sim', containment_hw=True, num_points=2)`.
- Optimiza θ en simulación y mide en HW: energía (bandas) + contención (proyección en base propia).
- Jobs ≈ 2·(b+1)·n_puntos = 2·2·10 = **~40** (~7-8 min). Salida: `hardware_monocapa_containment.json`.
- ⚠️ La versión previa (optimización completa en HW, ~600 jobs) NO cabe en el presupuesto; usar esta v2.

### HW-3 — Contención de nanocintas en hardware  (~30 jobs/cinta, 1 cinta por cuenta)
`ejercicio_hw3_containment_nanocinta.py`  (v2: medición en la base propia, `containment_hw=True`)
- 2 cubits (N=2, 4 estados), fundamental + 2 excitados, `optimizar_en='sim'` + `containment_hw=True`,
  n_ticks=5 (k en [0,π]). Jobs ≈ 2·(b+1)·n_ticks = 2·3·5 = **~30** por cinta.
- UNA cinta por cuenta: `TIPO='z'` en una, `TIPO='a'` en otra. Empezar por zigzag (multiplete de borde).
- Salida: `hardware_containment_{zGNR,aGNR}_2q.json`. Completa la contención "real" del sistema 1D.
- ⚠️ La optimización completa en HW (2-3 cubits) NO converge dentro del presupuesto → usar esta v2.

## Estado (2026-07-26)
- HW-1 (bandas monocapa): **HECHO** en `ibm_marrakesh` (~3m40s, ε̄/t=0.021) → en el artículo.
- HW-2 (containment monocapa): la v1 lanzada (optim. en HW) SE SALE del presupuesto (~600 jobs);
  **parar y relanzar la v2** (~40 jobs).
- HW-3 (containment nanocintas): script v2 listo y validado en local; para las 2 cuentas extra.

## Antes de ejecutar (usuario)
1. Tener el token de IBM Quantum (plan Open) guardado o a mano. Ver `conectar_ibm.py`.
2. En cada script, poner el token (o usar credenciales guardadas) y, si se quiere, fijar
   `backend_name` (p. ej. un backend Heron poco ocupado); si se deja `None`, se usa el menos ocupado.
3. Ejecutar primero HW-1 (barato) para verificar la conexión y el flujo end-to-end.
4. Luego HW-2 con parámetros mínimos; vigilar la cola.

Nota: los scripts están validados localmente con un fake backend (sin token). En HW real la
autenticación y el `least_busy` no se pueden probar aquí.
