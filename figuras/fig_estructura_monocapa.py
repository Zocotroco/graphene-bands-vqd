"""
Figura de estructura del grafeno MONOCAPA (elaboración propia), sección 2.1:
  (a) red en panal de abeja en espacio real, con subredes A y B, vectores de red a1,a2 y
      vectores a primeros vecinos.
  (b) primera zona de Brillouin (hexágono) con los puntos de alta simetría Γ, K, M y el
      camino Γ→K→M→Γ.

Guarda: ../latex/imagenes/fig_estructura_monocapa.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LANG = os.environ.get('FIGLANG', 'es')
def _L(es, en):
    return en if LANG == 'en' else es

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.environ.get("FIGOUT") or os.path.join(RAIZ, "latex", "imagenes_en" if LANG == 'en' else "imagenes")
os.makedirs(OUT, exist_ok=True)

a = 1.0  # distancia C–C (unidades arbitrarias para el dibujo)
# Vectores de red (celda del grafeno)
a1 = np.array([3*a/2,  np.sqrt(3)*a/2])
a2 = np.array([3*a/2, -np.sqrt(3)*a/2])
# Vectores a primeros vecinos (subred A -> B)
delta = np.array([[a/2, np.sqrt(3)*a/2], [a/2, -np.sqrt(3)*a/2], [-a, 0]])

CA, CB = '#1f77b4', '#d62728'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# ----------------- (a) Red en espacio real -----------------
A_pts, B_pts = [], []
for n1 in range(-2, 3):
    for n2 in range(-2, 3):
        base = n1*a1 + n2*a2
        A_pts.append(base)
        B_pts.append(base + delta[0])   # átomo B asociado
A_pts, B_pts = np.array(A_pts), np.array(B_pts)

# enlaces a primeros vecinos
for A in A_pts:
    for d in delta:
        B = A + d
        if np.hypot(*(B)) < 4.2:
            ax1.plot([A[0], B[0]], [A[1], B[1]], color='0.6', lw=1.2, zorder=1)
m = (np.hypot(A_pts[:, 0], A_pts[:, 1]) < 4.2)
ax1.scatter(A_pts[m, 0], A_pts[m, 1], s=90, color=CA, edgecolors='k',
            linewidths=0.6, zorder=3, label=_L('Subred A', 'Sublattice A'))
m = (np.hypot(B_pts[:, 0], B_pts[:, 1]) < 4.2)
ax1.scatter(B_pts[m, 0], B_pts[m, 1], s=90, color=CB, edgecolors='k',
            linewidths=0.6, zorder=3, label=_L('Subred B', 'Sublattice B'))
# vectores de red desde el origen
for v, nom in [(a1, r'$\vec{a}_1$'), (a2, r'$\vec{a}_2$')]:
    ax1.annotate('', xy=v, xytext=(0, 0),
                 arrowprops=dict(arrowstyle='-|>', color='black', lw=1.8))
    ax1.text(v[0]*0.55, v[1]*0.55 + 0.25*np.sign(v[1] or 1), nom, fontsize=13)
ax1.set_title(_L("(a) Red en panal (espacio real)", "(a) Honeycomb lattice (real space)"), fontsize=12)
ax1.set_aspect('equal'); ax1.set_xlim(-3.2, 3.2); ax1.set_ylim(-3.2, 3.2)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax1.set_xticks([]); ax1.set_yticks([])

# ----------------- (b) Zona de Brillouin -----------------
# Vectores recíprocos
b1 = (2*np.pi/(3*a)) * np.array([1,  np.sqrt(3)])
b2 = (2*np.pi/(3*a)) * np.array([1, -np.sqrt(3)])
# Puntos de alta simetría
G = np.array([0.0, 0.0])
K = (2*np.pi/(3*a)) * np.array([1, 1/np.sqrt(3)])
M = (2*np.pi/(3*a)) * np.array([1, 0])
# Hexágono de la 1ª zona de Brillouin: 6 puntos K rotados 60º
Kmag = np.hypot(*K)
hexpts = np.array([[Kmag*np.cos(np.pi/6 + i*np.pi/3), Kmag*np.sin(np.pi/6 + i*np.pi/3)]
                   for i in range(6)])
ax2.add_patch(plt.Polygon(hexpts, closed=True, fill=True, facecolor='#e8eef7',
                          edgecolor='0.4', lw=1.5, zorder=1))
# camino Γ→K→M→Γ
camino = np.array([G, K, M, G])
ax2.plot(camino[:, 0], camino[:, 1], '-', color='#2ca02c', lw=2.2, zorder=2)
for p, nom, dx, dy in [(G, r'$\Gamma$', -0.35, -0.45), (K, 'K', 0.15, 0.15),
                       (M, 'M', 0.1, -0.5)]:
    ax2.scatter(*p, s=60, color='black', zorder=3)
    ax2.text(p[0]+dx, p[1]+dy, nom, fontsize=14)
ax2.set_title(_L("(b) Zona de Brillouin y camino Γ–K–M–Γ", "(b) Brillouin zone and Γ–K–M–Γ path"), fontsize=12)
ax2.set_aspect('equal')
lim = Kmag*1.5
ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim)
ax2.set_xticks([]); ax2.set_yticks([])

plt.tight_layout()
out = os.path.join(OUT, "fig_estructura_monocapa.png")
plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
print("guardada:", out)
