# Band structure of graphene and graphene nanoribbons with VQE/VQD

Code and data accompanying the paper *Band structure of two-dimensional graphene and of
one-dimensional nanoribbons via variational quantum algorithms: from energies to wavefunction
quality*, by R. Medrano Millán, R. Gil-Merino y Rubio, U. Aseguinolaza Aguirreche and J. Borge.

Everything reported in the paper can be reproduced from this repository: the tight-binding
Bloch Hamiltonians, their mapping to Pauli operators, the VQE/VQD solver, the four execution
modes (ideal simulation, sampling, device noise model and real IBM Quantum hardware), the
subspace-containment diagnostic, and the scripts that draw every figure.

> The source comments and identifiers are in Spanish; this README and the paper are in English.

## Layout

    version_1.py            Core library: Hamiltonians, Pauli mapping, VQE/VQD, execution modes
    conectar_ibm.py         Saving and checking IBM Quantum credentials
    figuras/                One generator per figure, the plotting routines, and the raw results
    figuras/plot_articulo.py    Draws every figure of the paper from the JSON files
    figuras/gen_*.py        Recomputes the data behind each figure
    figuras/*.json          Raw results (simulation and hardware), with run parameters
    hardware_runs/          Scripts actually submitted to IBM Quantum, and the run plan
    hardware_*.json         Raw nanoribbon band data measured on IBM Quantum processors

## Installation

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

Python 3.11. Nothing beyond `requirements.txt` is needed for the simulation modes; running on
real hardware additionally requires an IBM Quantum account (see below).

## Reproducing the figures

All figures are drawn from the stored JSON files, so this needs no quantum computation:

    cd figuras && FIGOUT=../figuras_out python3 plot_articulo.py

Set `FIGLANG=en` for the English labels used in the paper (the default is Spanish).

To recompute the underlying data, run the corresponding generator; each writes the JSON that
`plot_articulo.py` reads:

| Figure | Generator | Mode | Approximate cost |
|---|---|---|---|
| Monolayer bands | `gen_bandas_monocapa.py` | `statevector` | minutes |
| Monolayer under noise | `gen_ruido_monocapa.py` | `aer_noise` | ~1 h |
| Nanoribbon validation | `gen_validacion_ideal.py` | `statevector` | minutes |
| Nanoribbons under noise | `gen_ruido.py` | `aer_noise` | ~1 h |
| Scalability | `gen_escalabilidad.py` | `statevector` | ~1 h |
| Containment, 3 qubits | `gen_containment.py` | `aer_noise` | ~25 min per ribbon |
| Ansatz-depth sweep | `gen_reps.py` | `statevector` + `aer_noise` | ~1.5 h |
| Scalability under noise | `gen_escalabilidad_ruido.py` | `aer_noise` | ~10 h |

## Execution modes

`calculos()` in `version_1.py` takes a `modo` argument:

- `statevector` — exact expectation values, no sampling and no noise.
- `aer` — finite number of shots, no noise.
- `aer_noise` — gate-level noise model of a device, finite shots. By default the emulated
  device has exactly the number of qubits of the circuit, which is what makes this mode
  affordable (emulating a 133-qubit backend to optimize a 2-qubit Hamiltonian is ~500x slower).
- `hardware` — execution on a real IBM Quantum processor.

Results are persisted to JSON together with the parameters of the run and the versions of the
libraries used.

## Running on real hardware

Save your credentials once with `conectar_ibm.py` (never hard-code a token in a script that
you commit). The scripts in `hardware_runs/` follow the hybrid strategy described in the paper:
the ansatz parameters are optimized in simulation and only the optimal circuit of each $k$
point and band is measured on the device, which is one job per point instead of the hundreds
that a full on-device optimization loop would need. That is what makes the computation fit in
the free-access plan; `hardware_runs/PLAN_HARDWARE.md` gives the job budget of each run.

## License

MIT — see `LICENSE`.
