"""
Genera la Figura 1 del artículo: estructura de las nanocintas de grafeno
armchair (a) y zigzag (b), con la celda unidad marcada (rectángulo discontinuo).
Figura propia (evita problemas de derechos de la figura del paper original).

Salida: ../latex/imagenes/fig_estructura_cintas.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

LANG = os.environ.get('FIGLANG', 'es')
def _L(es, en):
    return en if LANG == 'en' else es

OUT = os.environ.get("FIGOUT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "latex",
                                   "imagenes_en" if LANG == 'en' else "imagenes"))
os.makedirs(OUT, exist_ok=True)

acc = 1.0  # longitud de enlace C-C (unidades arbitrarias)

# Red de panal (honeycomb), orientación base: enlaces verticales, zigzag horizontal
a1 = np.array([np.sqrt(3) * acc, 0.0])
a2 = np.array([np.sqrt(3) / 2 * acc, 1.5 * acc])
basisA = np.array([0.0, 0.0])
basisB = np.array([0.0, acc])


def generar(n1r, n2r, rot90=False):
    pts, sub = [], []
    for n1 in n1r:
        for n2 in n2r:
            R = n1 * a1 + n2 * a2
            pts.append(R + basisA); sub.append(0)
            pts.append(R + basisB); sub.append(1)
    pts = np.array(pts); sub = np.array(sub)
    if rot90:  # rotación 90° para orientar los bordes armchair en horizontal
        M = np.array([[0.0, -1.0], [1.0, 0.0]])
        pts = pts @ M.T
    return pts, sub


def clip(pts, sub, box):
    x0, x1, y0, y1 = box
    m = (pts[:, 0] >= x0 - 1e-6) & (pts[:, 0] <= x1 + 1e-6) & \
        (pts[:, 1] >= y0 - 1e-6) & (pts[:, 1] <= y1 + 1e-6)
    return pts[m], sub[m]


def grado(pts):
    n = len(pts)
    deg = np.zeros(n, dtype=int)
    for i in range(n):
        d = np.linalg.norm(pts - pts[i], axis=1)
        deg[i] = np.sum(np.abs(d - acc) < 0.1 * acc)
    return deg


def podar(pts, sub, iters=3):
    """Elimina átomos con menos de 2 enlaces (bordes colgantes)."""
    for _ in range(iters):
        deg = grado(pts)
        keep = deg >= 2
        if keep.all():
            break
        pts, sub = pts[keep], sub[keep]
    return pts, sub


def puntos_medios(pts, tipo):
    """Devuelve las x de los puntos medios de los enlaces horizontales o diagonales."""
    mids = []
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[j] - pts[i]
            if abs(np.linalg.norm(d) - acc) < 0.1 * acc:
                dx, dy = abs(d[0]), abs(d[1])
                if tipo == 'horizontal' and dy < 0.1 * acc:
                    mids.append((pts[i, 0] + pts[j, 0]) / 2)
                elif tipo == 'diagonal' and dx > 0.1 * acc and dy > 0.1 * acc:
                    mids.append((pts[i, 0] + pts[j, 0]) / 2)
    return np.array(sorted(set(np.round(mids, 4))))


def dibujar(ax, pts, sub, periodo, celda_x0, titulo, margen_celda=0.45):
    # Enlaces
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(np.linalg.norm(pts[i] - pts[j]) - acc) < 0.1 * acc:
                ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                        color='0.35', lw=1.7, zorder=1)
    # Átomos: subred A (relleno) y B (hueco)
    A, B = pts[sub == 0], pts[sub == 1]
    ax.scatter(A[:, 0], A[:, 1], s=95, c='#222222', zorder=3)
    ax.scatter(B[:, 0], B[:, 1], s=95, facecolors='white', edgecolors='#222222',
               linewidths=1.7, zorder=3)
    # Celda unidad: rectángulo discontinuo de anchura = periodo; los bordes verticales
    # se sitúan en celda_x0 (punto medio de un enlace) y celda_x0 + periodo.
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    ax.add_patch(Rectangle((celda_x0, ymin - margen_celda), periodo, (ymax - ymin) + 2 * margen_celda,
                           fill=False, ls='--', lw=1.9, edgecolor='#c0392b', zorder=4))
    ax.set_title(titulo, fontsize=13)
    ax.set_aspect('equal')
    ax.axis('off')


fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

# --- (a) ARMCHAIR: red rotada 90°, cinta horizontal, periodo = 3*acc ---
pts, sub = generar(range(-14, 14), range(-14, 14), rot90=True)
pts, sub = clip(pts, sub, (0.0, 12.0 * acc, 0.0, 5.0 * acc))
pts, sub = podar(pts, sub)
# bordes verticales bisecando los enlaces HORIZONTALES
mids_h = puntos_medios(pts, 'horizontal')
celda_x0_a = mids_h[1]  # el segundo punto medio por la izquierda
dibujar(ax_a, pts, sub, periodo=3.0 * acc, celda_x0=celda_x0_a, titulo="(a) armchair")

# --- (b) ZIGZAG: orientación base, cinta horizontal, periodo = sqrt(3)*acc ---
pts, sub = generar(range(-14, 14), range(-14, 14))
pts, sub = clip(pts, sub, (0.0, 12.0 * acc, 0.0, 4.2 * acc))
pts, sub = podar(pts, sub)
# bordes verticales bisecando los enlaces DIAGONALES
mids_d = puntos_medios(pts, 'diagonal')
celda_x0_b = mids_d[1]  # el segundo punto medio por la izquierda
dibujar(ax_b, pts, sub, periodo=np.sqrt(3) * acc, celda_x0=celda_x0_b, titulo="(b) zigzag")

plt.tight_layout()
salida = os.path.join(OUT, "fig_estructura_cintas.png")
plt.savefig(salida, dpi=200, bbox_inches='tight')
print("Figura guardada en:", salida)
