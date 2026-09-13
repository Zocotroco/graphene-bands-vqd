"""
Rutinas de dibujo LIMPIO (estilo unificado, sin título incrustado; el título va en
el pie de figura de LaTeX) para las figuras del artículo.

Genera:
  - fig_validacion_3q.png : 1x2 (zigzag / armchair), VQD de las 4 bandas inferiores
                            frente a la diagonalización exacta (statevector, 3 cubits, -pi..pi).
  - fig_anchura.png       : 2x2 (zigzag/armchair x N=4/N=64), estructura de bandas exacta.
  - fig_hardware.png      : 1x2 (armchair/zigzag), energía medida en hardware real vs exacta.

Las bandas se colorean por su IDENTIDAD FÍSICA (seguimiento "diabático" por solapamiento
de autovectores), de modo que cada color sigue una banda continua a través de los cruces
(y no salta de una banda a otra por el orden de energía).
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LANG = os.environ.get('FIGLANG', 'es')
def _L(es, en):
    return en if LANG == 'en' else es

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, RAIZ)
from version_1 import build_matrix, hamiltoniano_monocapa, camino_monocapa

# Destino de las figuras: por defecto la carpeta del articulo; FIGOUT lo redirige
# (util fuera del repositorio del manuscrito).
OUT = os.environ.get("FIGOUT") or os.path.join(RAIZ, "latex", "imagenes_en" if LANG == 'en' else "imagenes")
os.makedirs(OUT, exist_ok=True)
FIGDIR = os.path.join(RAIZ, "figuras")

GRIS = '0.45'
# Paleta por banda (colores distinguibles y estables)
PALETA = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']
XT = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
XTL = ['-π', '-π/2', '0', 'π/2', 'π']


def track_diabatico(nq, tipo, t, kd):
    """Bandas exactas ordenadas por CONTINUIDAD (diabáticas), siguiendo cada banda
    física por el solapamiento de autovectores entre puntos k consecutivos."""
    nb = 2 ** nq
    E = np.zeros((len(kd), nb))
    prev_v = None
    for idx, k in enumerate(kd):
        w, v = np.linalg.eigh(build_matrix(nq, tipo, k, t))   # w ascendente
        if prev_v is None:
            E[idx] = w
            prev_v = v
            continue
        ov = np.abs(prev_v.conj().T @ v)          # solape |<prev_i | v_j>|
        perm = -np.ones(nb, dtype=int)
        usados, asig = set(), set()
        for _, i, j in sorted(((ov[i, j], i, j) for i in range(nb) for j in range(nb)),
                              reverse=True):
            if i in asig or j in usados:
                continue
            perm[i] = j
            asig.add(i); usados.add(j)
        E[idx] = w[perm]
        prev_v = v[:, perm]
    return E   # (Nk, nb) diabática


def panel_bandas(ax, nq, tipo, t, k_pts, vals, m, etiqueta, xr='full'):
    """Dibuja m bandas (las de menor energía media) con seguimiento diabático:
    líneas exactas de color por banda + puntos (vals) coloreados según la banda
    física más cercana en cada k."""
    kd = np.linspace(k_pts.min(), k_pts.max(), 400)
    E = track_diabatico(nq, tipo, t, kd)
    # m bandas de menor energía media, ORDENADAS de menor a mayor energía media: así la banda
    # más baja recibe el color 0 (azul), la siguiente el 1 (naranja), etc. (Banda 0 = la más baja).
    sel = sorted(np.argsort(E.mean(axis=0))[:m], key=lambda b: float(E[:, b].mean()))
    # líneas exactas (referencia) en GRIS; el color queda para los puntos por banda
    for b in sel:
        ax.plot(kd, E[:, b], color=GRIS, lw=1.3, zorder=1)
    Ei = np.vstack([np.interp(k_pts, kd, E[:, b]) for b in sel])   # (m, Npts)
    for n in range(len(k_pts)):
        # Puntos válidos en este k, asignados UNO A UNO a las m bandas (bijección por cercanía),
        # de modo que en cada k cada banda recibe exactamente un punto de su color (sin colores
        # duplicados ni bandas sin punto, ni siquiera en las degeneraciones del borde de zona).
        fila = [v for v in np.atleast_1d(vals[n])
                if v is not None and not (isinstance(v, float) and np.isnan(v))]
        asig = _asignar_a_bandas(fila, Ei[:, n])
        for i, val in enumerate(fila):
            c = asig[i]
            ax.scatter(k_pts[n], val, s=30, color=PALETA[c % len(PALETA)],
                       edgecolors='white', linewidths=0.4, zorder=3)
    ax.axhline(0, color='red', ls=':', alpha=0.25)
    if xr == 'full':
        ax.set_xticks(XT); ax.set_xticklabels(XTL)
    else:
        ax.set_xticks([0, np.pi/2, np.pi]); ax.set_xticklabels(['0', 'π/2', 'π'])
    ax.set_xlabel(_L("Vector de onda k", "Wave vector k"))
    ax.set_title(etiqueta, fontsize=12)
    ax.grid(True, alpha=0.25)


def _leyenda(ax, m, loc):
    # raya gris "Exacta" (arriba) + un punto de cada color con su banda,
    # en orden inverso (Banda m-1 ... Banda 0) para coincidir con el orden vertical del gráfico
    handles = [Line2D([0], [0], color=GRIS, lw=1.4, label=_L('Exacta', 'Exact'))]
    handles += [Line2D([0], [0], marker='o', ls='', color=PALETA[c % len(PALETA)],
                       label=_L(f'Banda {c}', f'Band {c}')) for c in reversed(range(m))]
    ax.legend(handles=handles, loc=loc, fontsize=8, framealpha=0.9)


# ---------------- Validación (statevector, 4 bandas inferiores) ----------------
def figura_validacion():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, fn, tipo, etiq in [(ax1, "val_zGNR_3q.json", 'z', "(a) zigzag"),
                               (ax2, "val_aGNR_3q.json", 'a', "(b) armchair")]:
        d = json.load(open(os.path.join(FIGDIR, fn), encoding='utf-8'))
        nq = d["parametros"].get("num_qubits", 3)
        t = d["parametros"].get("t", -1.0)
        k = np.array([r["k"] for r in d["resultados"]])
        vals = [r["valores_calculados"][:4] for r in d["resultados"]]  # 4 bandas inferiores
        panel_bandas(ax, nq, tipo, t, k, vals, m=4, etiqueta=etiq, xr='full')
    ax1.set_ylabel(_L("Energía (eV)", "Energy (eV)"))
    _leyenda(ax1, 4, 'lower right')   # zigzag
    _leyenda(ax2, 4, 'lower right')   # armchair
    plt.tight_layout()
    out = os.path.join(OUT, "fig_validacion_3q.png")
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print("guardada:", out)


# ---------------- Anchura (bandas exactas, 2x2, manifold en gris) ----------------
def panel_anchura(ax, tipo, nq, etiqueta, t=-1.0, npts=200):
    N = 2 ** (nq - 1)
    ks = np.linspace(-np.pi, np.pi, npts)
    bandas = np.array([np.linalg.eigvalsh(build_matrix(nq, tipo, k, t)) for k in ks])
    for i in range(2 ** nq):
        ax.plot(ks, bandas[:, i], color=GRIS, lw=0.7)
    ax.axhline(0, color='red', ls=':', alpha=0.25)
    ax.set_xticks(XT); ax.set_xticklabels(XTL)
    ax.set_title(f"{etiqueta}  (N={N})", fontsize=12)
    ax.grid(True, alpha=0.25)


def figura_anchura():
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    panel_anchura(axs[0, 0], 'z', 3, "(a) zigzag")
    panel_anchura(axs[0, 1], 'z', 7, "(b) zigzag")
    panel_anchura(axs[1, 0], 'a', 3, "(c) armchair")
    panel_anchura(axs[1, 1], 'a', 7, "(d) armchair")
    for ax in axs[:, 0]:
        ax.set_ylabel(_L("Energía (eV)", "Energy (eV)"))
    for ax in axs[1, :]:
        ax.set_xlabel(_L("Vector de onda k", "Wave vector k"))
    plt.tight_layout()
    out = os.path.join(OUT, "fig_anchura.png")
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print("guardada:", out)


# ---------------- Hardware real (energía medida, todas las bandas) ----------------
def figura_hardware():
    casos = [("hardware_agnr_2q.json", 'a', "(a) armchair — ibm_fez", 'center right'),
             ("hardware_zgnr_2q.json", 'z', "(b) zigzag — ibm_kingston", 'center left')]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (fn, tipo, etiq, loc) in zip(axs, casos):
        d = json.load(open(os.path.join(RAIZ, fn), encoding='utf-8'))
        nq = d["parametros"].get("num_qubits", 2)
        t = d["parametros"].get("t", -1.0)
        k = np.array([r["k"] for r in d["resultados"]])
        vals = [r["valores_medidos"] for r in d["resultados"]]
        panel_bandas(ax, nq, tipo, t, k, vals, m=2 ** nq, etiqueta=etiq, xr='half')
        _leyenda(ax, 2 ** nq, loc)
    axs[0].set_ylabel(_L("Energía (eV)", "Energy (eV)"))
    plt.tight_layout()
    out = os.path.join(OUT, "fig_hardware.png")
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print("guardada:", out)


# ---------------- Efecto del ruido (aer_noise, energía medida) ----------------
def figura_ruido():
    casos = [("ruido_aGNR_2q.json", 'a', "(a) armchair", 'center right'),
             ("ruido_zGNR_2q.json", 'z', "(b) zigzag", 'center left')]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (fn, tipo, etiq, loc) in zip(axs, casos):
        d = json.load(open(os.path.join(FIGDIR, fn), encoding='utf-8'))
        nq = d["parametros"].get("num_qubits", 2)
        t = d["parametros"].get("t", -1.0)
        k = np.array([r["k"] for r in d["resultados"]])
        vals = [r["valores_medidos"] for r in d["resultados"]]     # energía MEDIDA con ruido
        panel_bandas(ax, nq, tipo, t, k, vals, m=2 ** nq, etiqueta=etiq, xr='half')
        _leyenda(ax, 2 ** nq, loc)
    axs[0].set_ylabel(_L("Energía (eV)", "Energy (eV)"))
    plt.tight_layout()
    out = os.path.join(OUT, "fig_ruido.png")
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print("guardada:", out)


# ---------------- Escalabilidad y análisis de error ----------------
def figura_escalabilidad():
    tabla = json.load(open(os.path.join(FIGDIR, "escalabilidad.json"), encoding='utf-8'))
    tipos = {}
    for f in tabla:
        tipos.setdefault(f["tipo"], []).append(f)
    col = {"armchair": '#1f77b4', "zigzag": '#d62728'}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    # (a) error medio vs n_q (log-y)
    for tp, filas in tipos.items():
        filas = sorted(filas, key=lambda f: f["nq"])
        nq = [f["nq"] for f in filas]
        ax1.semilogy(nq, [f["err_medio"] for f in filas], 'o-', color=col[tp], label=tp)
    ax1.set_xlabel(_L("Número de cúbits $n_q$", "Number of qubits $n_q$")); ax1.set_ylabel(_L(r"Error medio $|E_{\rm VQD}-E_{\rm exacta}|/t$", r"Mean error $|E_{\rm VQD}-E_{\rm exact}|/t$"))
    ax1.set_title(_L("(a) Error frente a la anchura", "(a) Error vs. width")); ax1.grid(True, which='both', alpha=0.25); ax1.legend()
    # (b) recursos: nº términos de Pauli y nº parámetros vs n_q
    filas = sorted(tipos["armchair"], key=lambda f: f["nq"])
    nq = [f["nq"] for f in filas]
    ax2.plot(nq, [f["npauli"] for f in filas], 's-', color='#2ca02c', label=_L('Términos de Pauli (armchair)', 'Pauli terms (armchair)'))
    ax2.plot(nq, [f["npauli"] for f in sorted(tipos["zigzag"], key=lambda f: f["nq"])], 's--',
             color='#ff7f0e', label=_L('Términos de Pauli (zigzag)', 'Pauli terms (zigzag)'))
    ax2.plot(nq, [f["nparams"] for f in filas], '^-', color='#9467bd', label=_L('Parámetros del ansatz', 'Ansatz parameters'))
    ax2.set_xlabel(_L("Número de cúbits $n_q$", "Number of qubits $n_q$")); ax2.set_ylabel(_L("Recuento", "Count"))
    ax2.set_title(_L("(b) Recursos frente a $n_q$", "(b) Resources vs. $n_q$")); ax2.grid(True, alpha=0.25); ax2.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(OUT, "fig_escalabilidad.png")
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print("guardada:", out)


# ---------------- Contención en subespacios (subspace containment) ----------------
# Etiquetas de los estados VQD reconstruidos (fundamental + excitados)
_ETIQ_ESTADO = [_L('Fundamental', 'Ground'), _L('1er excitado', '1st excited'),
                _L('2º excitado', '2nd excited'), _L('3er excitado', '3rd excited'),
                _L('4º excitado', '4th excited')]


def _containment_objetivo(r):
    """Para un resultado por k, devuelve C_objetivo[e]: contención del estado variacional e
    en el AUTOESPACIO exacto al que pertenece el e-ésimo nivel de energía. Con degeneración,
    varios estados comparten autoespacio (p. ej. el manifold de borde del zigzag en E~0)."""
    c = r["containment"]
    matriz = c["matriz"]                 # matriz[estado][autoespacio]
    deg = c["degeneracion"]              # tamaño de cada autoespacio (niveles agrupados)
    nivel_a_grupo = []
    for gi, d in enumerate(deg):
        nivel_a_grupo += [gi] * d        # nivel_a_grupo[e] = índice de autoespacio del nivel e
    out = []
    for e in range(len(matriz)):
        g = nivel_a_grupo[e] if e < len(nivel_a_grupo) else len(deg) - 1
        g = min(g, len(matriz[e]) - 1)
        out.append(matriz[e][g])
    return out


def _asignar_a_bandas(niveles, banda_E):
    """Empareja (uno a uno, por cercanía) cada nivel de energía con la banda DIABÁTICA
    más próxima. niveles: energías (orden de energía); banda_E: energías de las bandas
    diabáticas en ese k. Devuelve asig[e] = índice de banda diabática del nivel e."""
    pares = sorted((abs(niveles[i] - banda_E[j]), i, j)
                   for i in range(len(niveles)) for j in range(len(banda_E)))
    asig, usados = {}, set()
    for _, i, j in pares:
        if i in asig or j in usados:
            continue
        asig[i] = j
        usados.add(j)
    return asig


def figura_containment():
    """2x2: (fila superior) estructura de bandas exacta en [0,pi] con los niveles
    reconstruidos resaltados; (fila inferior) contención C de cada estado VQD en su
    autoespacio exacto frente a k. Columnas: zigzag / armchair.

    Tanto los puntos (arriba) como las curvas de contención (abajo) se colorean por la
    IDENTIDAD DIABÁTICA de la banda (seguimiento por continuidad), de modo que cada color
    sigue una banda física a través de los cruces y no salta por el orden de energía."""
    casos = [('z', "containment_zGNR_3q.json", "zigzag"),
             ('a', "containment_aGNR_3q.json", "armchair")]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.6), sharex='col')
    for col, (tipo, fn, nombre) in enumerate(casos):
        d = json.load(open(os.path.join(FIGDIR, fn), encoding='utf-8'))
        nq = d["parametros"].get("num_qubits", 3)
        t = d["parametros"].get("t", -1.0)
        b = d["parametros"].get("b", 2)
        k = np.array([r["k"] for r in d["resultados"]])
        n_est = b + 1

        # Bandas diabáticas (densas) y energías de las n_est bandas de menor energía
        # media, en orden de color estable (por energía en k=0 -> banda 0 = fundamental).
        kd = np.linspace(k.min(), k.max(), 400)
        E = track_diabatico(nq, tipo, t, kd)
        sel = sorted(np.argsort(E.mean(axis=0))[:n_est], key=lambda bnd: E[0, bnd])
        Ei = np.vstack([np.interp(k, kd, E[:, bnd]) for bnd in sel])   # (n_est, Nk)

        # --- (fila 0) bandas exactas (gris) + niveles reconstruidos coloreados por banda ---
        # Todas las líneas exactas en GRIS; el color queda solo para los PUNTOS, coloreados
        # según su identidad diabática (seguimiento por continuidad), como en las demás figuras.
        axb = axs[0, col]
        for i in range(2 ** nq):
            axb.plot(kd, E[:, i], color=GRIS, lw=0.7, zorder=1)
        Cband = np.full((len(k), n_est), np.nan)   # contención reordenada por banda diabática
        for n, r in enumerate(d["resultados"]):
            niveles = sorted(r["valores_exactos"])[:n_est]
            asig = _asignar_a_bandas(niveles, Ei[:, n])       # nivel (energía) -> banda diabática
            Cobj = _containment_objetivo(r)                   # contención por orden de energía
            for e in range(n_est):
                bnd = asig[e]
                axb.scatter(k[n], niveles[e], s=22, color=PALETA[bnd % len(PALETA)],
                            edgecolors='white', linewidths=0.4, zorder=3)
                Cband[n, bnd] = Cobj[e]
        axb.axhline(0, color='red', ls=':', alpha=0.25)
        axb.set_xticks([0, np.pi/2, np.pi]); axb.set_xticklabels(['0', 'π/2', 'π'])
        axb.set_title(f"({'ab'[col]}) {nombre}", fontsize=12)
        axb.grid(True, alpha=0.25)

        # --- (fila 1) contención por banda diabática vs k ---
        axc = axs[1, col]
        for bnd in range(n_est):
            axc.plot(k, Cband[:, bnd], 'o-', color=PALETA[bnd % len(PALETA)],
                     label=_ETIQ_ESTADO[bnd], lw=1.4, ms=5)
        axc.axhline(1.0, color='0.7', ls='--', lw=0.8)
        axc.set_ylim(0.78, 1.01)   # datos en 0.809-1.000
        axc.set_xticks([0, np.pi/2, np.pi]); axc.set_xticklabels(['0', 'π/2', 'π'])
        axc.set_xlabel(_L("Vector de onda k", "Wave vector k"))
        axc.grid(True, alpha=0.25)
        if col == 0:
            axc.legend(loc='lower left', fontsize=8, framealpha=0.9)
    # misma escala vertical en los dos paneles de bandas (fila superior)
    y0 = min(axs[0, 0].get_ylim()[0], axs[0, 1].get_ylim()[0])
    y1 = max(axs[0, 0].get_ylim()[1], axs[0, 1].get_ylim()[1])
    axs[0, 0].set_ylim(y0, y1); axs[0, 1].set_ylim(y0, y1)
    axs[0, 0].set_ylabel(_L("Energía (eV)", "Energy (eV)"))
    axs[1, 0].set_ylabel(_L(r"Contención $C_\alpha$", r"Containment $C_\alpha$"))
    plt.tight_layout()
    out = os.path.join(OUT, "fig_containment.png")
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print("guardada:", out)


def figura_containment_monocapa():
    """Grafeno monocapa (2D): (arriba) bandas de valencia y conducción a lo largo de Γ-K-M-Γ
    (líneas exactas en gris + puntos VQD por banda); (abajo) contención C de cada banda vs k.
    2 bandas -> banda 0 = valencia (azul), banda 1 = conducción (naranja)."""
    d = json.load(open(os.path.join(FIGDIR, "containment_monocapa.json"), encoding='utf-8'))
    par = d["parametros"]
    t, tp, a, vec = par.get("t", 1.0), par.get("t_prima", 1.0/28), par.get("a", 1.42), par.get("vecinos", 2)
    coord = np.array(d["camino"]["coord"])
    pos = d["camino"]["pos"]
    etiquetas = d["camino"]["etiquetas"]
    xticks = [coord[p] for p in pos]

    # Bandas exactas densas (energía ascendente: valencia, conducción) a lo largo del mismo camino
    kd, coord_d, _ = camino_monocapa(num_points=200, a=a)
    Ed = np.array([np.linalg.eigvalsh(hamiltoniano_monocapa(k, t, tp, a, vec)) for k in kd])

    fig, (axb, axc) = plt.subplots(2, 1, figsize=(7.5, 7.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1.4, 1]})
    # (arriba) bandas: líneas grises + puntos VQD coloreados por banda (0=valencia, 1=conducción)
    for i in range(Ed.shape[1]):
        axb.plot(coord_d, Ed[:, i], color=GRIS, lw=1.0, zorder=1)
    for r in d["resultados"]:
        for bnd, val in enumerate(r["valores_calculados"]):   # ya ordenados por energía
            axb.scatter(r["coord"], val, s=26, color=PALETA[bnd % len(PALETA)],
                        edgecolors='white', linewidths=0.4, zorder=3)
    axb.axhline(0, color='red', ls=':', alpha=0.25)
    axb.set_ylabel(_L("Energía / t", "Energy / t"))
    axb.grid(True, alpha=0.25)
    leyenda = [Line2D([0], [0], color=GRIS, lw=1.2, label=_L('Exacta', 'Exact')),
               Line2D([0], [0], marker='o', ls='', color=PALETA[0], label=_L('Valencia (VQD)', 'Valence (VQD)')),
               Line2D([0], [0], marker='o', ls='', color=PALETA[1], label=_L('Conducción (VQD)', 'Conduction (VQD)'))]
    axb.legend(handles=leyenda, loc='upper right', fontsize=8, framealpha=0.9)

    # (abajo) contención de cada banda vs k
    C = np.array([_containment_objetivo(r) for r in d["resultados"]])   # (Nk, 2)
    for bnd, etq in [(0, _L('Valencia', 'Valence')), (1, _L('Conducción', 'Conduction'))]:
        axc.plot(coord, C[:, bnd], 'o-', color=PALETA[bnd % len(PALETA)],
                 label=etq, lw=1.4, ms=4)
    axc.axhline(1.0, color='0.7', ls='--', lw=0.8)
    axc.set_ylim(-0.03, 1.05)
    axc.set_ylabel(_L(r"Contención $C_\alpha$", r"Containment $C_\alpha$"))
    axc.grid(True, alpha=0.25)
    axc.legend(loc='lower left', fontsize=8, framealpha=0.9)

    for ax in (axb, axc):
        for xt in xticks[1:-1]:
            ax.axvline(xt, color='0.8', lw=0.8, zorder=0)
    axc.set_xticks(xticks)
    axc.set_xticklabels(etiquetas)
    axc.set_xlabel(_L("Vector de onda k", "Wave vector k"))
    plt.tight_layout()
    out = os.path.join(OUT, "fig_containment_monocapa.png")
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print("guardada:", out)


def figura_ruido_monocapa():
    """Efecto del ruido en el monocapa y comparaci\'on de optimizadores: energía MEDIDA bajo
    ruido (puntos) frente a la exacta (gris) a lo largo de Γ-K-M-Γ, para (a) COBYLA y (b) SPSA."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, opt, etq in [(axs[0], "COBYLA", "(a) COBYLA"), (axs[1], "SPSA", "(b) SPSA")]:
        d = json.load(open(os.path.join(FIGDIR, f"ruido_monocapa_{opt}.json"), encoding='utf-8'))
        par = d["parametros"]
        t, tp, a, vec = par.get("t", 1.0), par.get("t_prima", 1.0/28), par.get("a", 1.42), par.get("vecinos", 2)
        coord = np.array(d["camino"]["coord"]); pos = d["camino"]["pos"]; etiquetas = d["camino"]["etiquetas"]
        xticks = [coord[p] for p in pos]
        kd, coord_d, _ = camino_monocapa(num_points=200, a=a)
        Ed = np.array([np.linalg.eigvalsh(hamiltoniano_monocapa(k, t, tp, a, vec)) for k in kd])
        for i in range(Ed.shape[1]):
            ax.plot(coord_d, Ed[:, i], color=GRIS, lw=1.0, zorder=1)
        errs = []
        for r in d["resultados"]:
            ex = np.array(sorted(r["valores_exactos"]))
            for bnd, val in enumerate(r["valores_medidos"]):
                ax.scatter(r["coord"], val, s=24, color=PALETA[bnd % len(PALETA)],
                           edgecolors='white', linewidths=0.4, zorder=3)
            errs.append(np.abs(np.array(r["valores_medidos"]) - ex))
        emean = float(np.concatenate(errs).mean())
        ax.axhline(0, color='red', ls=':', alpha=0.25)
        for xt in xticks[1:-1]:
            ax.axvline(xt, color='0.85', lw=0.8, zorder=0)
        ax.set_xticks(xticks); ax.set_xticklabels(etiquetas)
        ax.set_xlabel(_L("Vector de onda k", "Wave vector k"))
        ax.set_title(f"{etq}  ($\\bar\\epsilon/t={emean:.3f}$)", fontsize=12)
        ax.grid(True, axis='y', alpha=0.25)
    axs[0].set_ylabel(_L("Energía / t", "Energy / t"))
    leyenda = [Line2D([0], [0], color=GRIS, lw=1.2, label=_L('Exacta', 'Exact')),
               Line2D([0], [0], marker='o', ls='', color=PALETA[0], label=_L('Valencia (medida)', 'Valence (measured)')),
               Line2D([0], [0], marker='o', ls='', color=PALETA[1], label=_L('Conducción (medida)', 'Conduction (measured)'))]
    axs[1].legend(handles=leyenda, loc='upper right', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    out = os.path.join(OUT, "fig_ruido_monocapa.png")
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print("guardada:", out)


def figura_containment_nanocinta_3reg():
    """Contención de las nanocintas (2 cubits, N=2) en los tres regímenes (sin ruido / con ruido
    / hardware real), 1x2 = (a) zigzag, (b) armchair. Color = estado (fund/1er/2º exc, orden de
    energía); estilo = régimen (línea sólida sin ruido, discontinua con ruido, marcadores hardware).
    Omite el hardware de la cinta cuyo JSON aún no exista."""
    from matplotlib.lines import Line2D
    casos = [('z', 'zGNR', "(a) zigzag"), ('a', 'aGNR', "(b) armchair")]
    regs = [("2q_sv", dict(ls='-', lw=1.4)), ("2q_noise", dict(ls='--', lw=1.4)),
            ("hardware", dict(ls='', marker='o', ms=7))]
    etiq_estado = [_L('Fundamental', 'Ground'), _L('1er excitado', '1st excited'), _L('2º excitado', '2nd excited')]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (tipo, nom, tit) in zip(axs, casos):
        backend_hw = None
        for tag, st in regs:
            fn = (f"hardware_containment_{nom}_2q.json" if tag == "hardware"
                  else f"containment_{nom}_{tag}.json")
            p = os.path.join(FIGDIR, fn)
            if not os.path.exists(p):
                continue
            d = json.load(open(p, encoding='utf-8'))
            if tag == "hardware":
                backend_hw = d["parametros"].get("backend_name", "HW")
            k = np.array([r["k"] for r in d["resultados"]])
            C = np.array([_containment_objetivo(r) for r in d["resultados"]])   # (Nk, 3)
            for e in range(C.shape[1]):
                ax.plot(k, C[:, e], color=PALETA[e % len(PALETA)], **st)
        ax.axhline(1.0, color='0.7', ls=':', lw=0.8)
        ax.set_ylim(0.80, 1.01)    # datos en 0.834-1.000
        ax.set_xticks([0, np.pi/2, np.pi]); ax.set_xticklabels(['0', 'π/2', 'π'])
        ax.set_xlabel(_L("Vector de onda k", "Wave vector k"))
        ax.set_title(tit + (f" — HW: {backend_hw}" if backend_hw else ""), fontsize=12)
        ax.grid(True, alpha=0.25)
    axs[0].set_ylabel(_L(r"Contención $C_\alpha$", r"Containment $C_\alpha$"))
    # leyenda: color = estado; estilo = régimen
    h_estado = [Line2D([0], [0], color=PALETA[e], lw=3, label=etiq_estado[e]) for e in range(3)]
    h_reg = [Line2D([0], [0], color='0.3', ls='-', lw=1.4, label=_L('sin ruido', 'noiseless')),
             Line2D([0], [0], color='0.3', ls='--', lw=1.4, label=_L('con ruido', 'with noise')),
             Line2D([0], [0], color='0.3', ls='', marker='o', ms=7, label=_L('hardware', 'hardware'))]
    axs[0].legend(handles=h_estado, loc='lower left', fontsize=8, framealpha=0.9)
    axs[1].legend(handles=h_reg, loc='lower left', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    out = os.path.join(OUT, "fig_containment_nanocinta_3reg.png")
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print("guardada:", out)


def figura_containment_monocapa_3reg():
    """Contención del monocapa en los TRES regímenes (sin ruido / con ruido / hardware real),
    1x2 = (a) valencia, (b) conducción, frente a k a lo largo de Γ-K-M-Γ. Las simulaciones se
    dibujan como líneas; el hardware, como marcadores. Si aún no existe el JSON de hardware,
    se omite esa serie."""
    regimenes = [("containment_monocapa_sv.json", "sin ruido", dict(ls='-', lw=1.6)),
                 ("containment_monocapa.json", "con ruido", dict(ls='--', lw=1.6)),
                 ("hardware_monocapa_containment.json", "hardware", dict(ls='', marker='o', ms=9))]
    # camino de referencia (para los xticks Γ,K,M,Γ) del primer JSON disponible
    ref = None
    for fn, _, _ in regimenes:
        p = os.path.join(FIGDIR, fn)
        if os.path.exists(p):
            ref = json.load(open(p, encoding='utf-8')); break
    if ref is None:
        print("figura_containment_monocapa_3reg: no hay JSONs"); return
    xticks = [ref["camino"]["coord"][i] for i in ref["camino"]["pos"]]
    etiquetas = ref["camino"]["etiquetas"]

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    col = {"sin ruido": '#2ca02c', "con ruido": '#ff7f0e', "hardware": '#d62728'}
    for fn, etq, st in regimenes:
        p = os.path.join(FIGDIR, fn)
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding='utf-8'))
        coord = np.array([r["coord"] for r in d["resultados"]])
        C = np.array([_containment_objetivo(r) for r in d["resultados"]])   # (Nk, 2)
        lab = _L(etq, {"sin ruido": "noiseless", "con ruido": "with noise", "hardware": "hardware"}.get(etq, etq))
        if etq == "hardware":
            lab = "hardware (%s)" % d["parametros"].get("backend_name", "HW")
        for bnd, ax in enumerate(axs):
            ax.plot(coord, C[:, bnd], color=col[etq], label=lab, **st)
    for ax, tit in zip(axs, [_L("(a) valencia", "(a) valence"), _L("(b) conducción", "(b) conduction")]):
        ax.axhline(1.0, color='0.7', ls=':', lw=0.8)
        ax.set_ylim(0.97, 1.004)   # datos en 0.979-1.000
        for xt in xticks[1:-1]:
            ax.axvline(xt, color='0.9', lw=0.8, zorder=0)
        ax.set_xticks(xticks); ax.set_xticklabels(etiquetas)
        ax.set_xlabel(_L("Vector de onda k", "Wave vector k")); ax.set_title(tit, fontsize=12)
        ax.grid(True, axis='y', alpha=0.25)
    axs[0].set_ylabel(_L(r"Contención $C_\alpha$", r"Containment $C_\alpha$"))
    axs[0].legend(loc='lower left', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    out = os.path.join(OUT, "fig_containment_monocapa_3reg.png")
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print("guardada:", out)


def figura_hardware_monocapa():
    """Bandas del grafeno monocapa medidas en HARDWARE REAL: energía medida en el dispositivo
    (puntos) frente a la exacta (gris) a lo largo de Γ-K-M-Γ (estrategia híbrida: optimización
    en simulación + medida del circuito óptimo en el hardware)."""
    d = json.load(open(os.path.join(FIGDIR, "hardware_monocapa_bandas.json"), encoding='utf-8'))
    par = d["parametros"]
    t, tp, a, vec = par.get("t", 1.0), par.get("t_prima", 1.0/28), par.get("a", 1.42), par.get("vecinos", 2)
    backend = par.get("backend_name", "hardware")
    coord = np.array(d["camino"]["coord"]); pos = d["camino"]["pos"]; etiquetas = d["camino"]["etiquetas"]
    xticks = [coord[p] for p in pos]
    kd, coord_d, _ = camino_monocapa(num_points=200, a=a)
    Ed = np.array([np.linalg.eigvalsh(hamiltoniano_monocapa(k, t, tp, a, vec)) for k in kd])
    errs = []
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for i in range(Ed.shape[1]):
        ax.plot(coord_d, Ed[:, i], color=GRIS, lw=1.0, zorder=1)
    for r in d["resultados"]:
        ex = np.array(sorted(r["valores_exactos"]))
        for bnd, val in enumerate(r["valores_medidos"]):
            ax.scatter(r["coord"], val, s=30, color=PALETA[bnd % len(PALETA)],
                       edgecolors='white', linewidths=0.4, zorder=3)
        errs.append(np.abs(np.array(r["valores_medidos"]) - ex))
    emean = float(np.concatenate(errs).mean())
    ax.axhline(0, color='red', ls=':', alpha=0.25)
    for xt in xticks[1:-1]:
        ax.axvline(xt, color='0.85', lw=0.8, zorder=0)
    ax.set_xticks(xticks); ax.set_xticklabels(etiquetas)
    ax.set_xlabel(_L("Vector de onda k", "Wave vector k")); ax.set_ylabel(_L("Energía / t", "Energy / t"))
    ax.set_title(f"{backend}  ($\\bar\\epsilon/t={emean:.3f}$)", fontsize=12)
    ax.grid(True, axis='y', alpha=0.25)
    leyenda = [Line2D([0], [0], color=GRIS, lw=1.2, label=_L('Exacta', 'Exact')),
               Line2D([0], [0], marker='o', ls='', color=PALETA[0], label=_L('Valencia (HW)', 'Valence (HW)')),
               Line2D([0], [0], marker='o', ls='', color=PALETA[1], label=_L('Conducción (HW)', 'Conduction (HW)'))]
    ax.legend(handles=leyenda, loc='upper right', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    out = os.path.join(OUT, "fig_hardware_monocapa.png")
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print("guardada:", out)


def figura_bandas_monocapa():
    """Validación de las bandas del monocapa en simulación ideal: exacta (gris) + VQD (puntos),
    1x2 = (a) primeros vecinos, (b) primeros+segundos vecinos, a lo largo de Γ-K-M-Γ."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, vec, etq in [(axs[0], 1, _L("(a) primeros vecinos", "(a) first neighbors")),
                         (axs[1], 2, _L("(b) primeros y segundos vecinos", "(b) first and second neighbors"))]:
        d = json.load(open(os.path.join(FIGDIR, f"bandas_monocapa_v{vec}.json"), encoding='utf-8'))
        par = d["parametros"]
        t, tp, a = par.get("t", 1.0), par.get("t_prima", 1.0/28), par.get("a", 1.42)
        coord = np.array(d["camino"]["coord"]); pos = d["camino"]["pos"]; etiquetas = d["camino"]["etiquetas"]
        xticks = [coord[p] for p in pos]
        kd, coord_d, _ = camino_monocapa(num_points=200, a=a)
        Ed = np.array([np.linalg.eigvalsh(hamiltoniano_monocapa(k, t, tp, a, vec)) for k in kd])
        for i in range(Ed.shape[1]):
            ax.plot(coord_d, Ed[:, i], color=GRIS, lw=1.0, zorder=1)
        for r in d["resultados"]:
            for bnd, val in enumerate(r["valores_calculados"]):
                ax.scatter(r["coord"], val, s=24, color=PALETA[bnd % len(PALETA)],
                           edgecolors='white', linewidths=0.4, zorder=3)
        ax.axhline(0, color='red', ls=':', alpha=0.25)
        for xt in xticks[1:-1]:
            ax.axvline(xt, color='0.85', lw=0.8, zorder=0)
        ax.set_xticks(xticks); ax.set_xticklabels(etiquetas)
        ax.set_xlabel(_L("Vector de onda k", "Wave vector k")); ax.set_title(etq, fontsize=12)
        ax.grid(True, axis='y', alpha=0.25)
    axs[0].set_ylabel(_L("Energía / t", "Energy / t"))
    leyenda = [Line2D([0], [0], color=GRIS, lw=1.2, label=_L('Exacta', 'Exact')),
               Line2D([0], [0], marker='o', ls='', color=PALETA[0], label=_L('Valencia (VQD)', 'Valence (VQD)')),
               Line2D([0], [0], marker='o', ls='', color=PALETA[1], label=_L('Conducción (VQD)', 'Conduction (VQD)'))]
    axs[1].legend(handles=leyenda, loc='upper right', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    out = os.path.join(OUT, "fig_bandas_monocapa.png")
    plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
    print("guardada:", out)


if __name__ == "__main__":
    figura_validacion()
    figura_anchura()
    figura_hardware()
    figura_ruido()
    figura_escalabilidad()
    figura_containment()
    figura_bandas_monocapa()
    figura_ruido_monocapa()
    figura_hardware_monocapa()
    figura_containment_monocapa_3reg()
    figura_containment_nanocinta_3reg()
    # figura_containment_monocapa()  # (vista bandas+contención de un solo régimen; no usada en el artículo)
