"""
Transition monotone -> non-monotone de modele_tau(wp2_i) sur plusieurs couches.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib.pyplot as plt


c = 3e8
e = 1.602176634e-19
me = 9.1093837e-31
eps0 = 8.854187817e-12
a, d, n0 = 0.1, 0.2, 1e18

LAYERS = [5, 10, 15, 20, 25, 30, 35, 40, 45]      # couches à comparer

donnees = np.loadtxt("donnees_inversion_6.txt", skiprows=1)
alpha_deg, omega, tau = donnees[:, 0], donnees[:, 1], donnees[:, 2]
p   = np.abs((d / 2) * np.sin(np.radians(alpha_deg)) - a * np.cos(np.radians(alpha_deg)))
tau = 2 * tau

# intégrateur commun
def _integre(p_i, wi, r_re, wp2_val, wp2_der):
    def zeta2(r):
        return r**2 * (1.0 - wp2_val(r) / wi**2)
    u_max = np.sqrt(max(a**2 - p_i**2, 0.0))
    def integ(u):
        z2 = p_i**2 + u**2
        try:
            r = brentq(lambda r: zeta2(r) - z2, r_re, a)
        except ValueError:
            r = r_re
        wp2, wp2p = wp2_val(r), wp2_der(r)
        dz2dr = 2 * r - (2 * r * wp2 + r**2 * wp2p) / wi**2
        if dz2dr <= 0:
            dz2dr = 2 * r
        return (4 / c) * r / dz2dr
    val, _ = quad(integ, 0.0, u_max, limit=50)
    return val

# # ── modele_tau réel (interpolation linéaire sur les couches reconstruites) ─────
# def tau_lin(wp2_i, couches_r, couches_wp2, p_i, wi):
#     r_re = p_i / np.sqrt(max(1.0 - wp2_i / wi**2, 1e-12))
#     r_re = min(r_re, a - 1e-6)
#     r_all, wp2_all = couches_r + [r_re], couches_wp2 + [wp2_i]
#     o = np.argsort(r_all)
#     r_pts, wp2_pts = [r_all[j] for j in o], [wp2_all[j] for j in o]
#     def wp2_val(r):
#         return float(np.interp(r, r_pts, wp2_pts, left=wp2_pts[0], right=0.0))
#     def wp2_der(r):
#         dr = 1e-7 * a
#         return (wp2_val(r + dr) - wp2_val(r - dr)) / (2 * dr)
#     return _integre(p_i, wi, r_re, wp2_val, wp2_der)

# ── modele_tau lisse (profil parabolique proxy, sans noeuds -> sans pics) ──────
def tau_smooth(wp2_i, p_i, wi):
    r_re = p_i / np.sqrt(max(1.0 - wp2_i / wi**2, 1e-12))
    r_re = min(r_re, a - 1e-6)
    def wp2_val(r):
        if r <= r_re: return wp2_i
        if r >= a:    return 0.0
        return wp2_i * (a**2 - r**2) / (a**2 - r_re**2)
    def wp2_der(r):
        if r <= r_re or r >= a: return 0.0
        return wp2_i * (-2.0 * r) / (a**2 - r_re**2)
    return _integre(p_i, wi, r_re, wp2_val, wp2_der)

# ── snapshots aux couches voulues ─────────────────────────────────────────────
ordre = sorted(range(len(p)), key=lambda i: -p[i])
p_tri, w_tri, t_tri = ([x[i] for i in ordre] for x in (p, omega, tau))
snaps = {}

LAYERS = [L for L in LAYERS if L <= len(p_tri)]
for L in LAYERS:
    k = L - 1
    p_i, wi, tau_obs = p_tri[k], w_tri[k], t_tri[k]
    hi = wi**2 * (1.0 - (p_i / a)**2) * 0.999
    snaps[L] = (p_i, wi, tau_obs, hi)

# # ── onion-peeling complet (nécessite tau_lin) ─────────────────────────────
# couches_r, couches_wp2 = [a], [0.0]
# for k in range(max(LAYERS, default=0)):
#     p_i, wi, tau_obs = p_tri[k], w_tri[k], t_tri[k]
#     hi = wi**2 * (1.0 - (p_i / a)**2) * 0.999
#     if (k + 1) in LAYERS:
#         snaps[k + 1] = (list(couches_r), list(couches_wp2), p_i, wi, tau_obs, hi)
#     wp2_prev = couches_wp2[-1]
#     xs = np.linspace(0.0, hi, 80)
#     fs = np.array([tau_lin(x, couches_r, couches_wp2, p_i, wi) - tau_obs for x in xs])
#     rac = [brentq(lambda x: tau_lin(x, couches_r, couches_wp2, p_i, wi) - tau_obs,
#                   xs[j], xs[j + 1]) for j in range(len(xs) - 1) if fs[j] * fs[j + 1] < 0]
#     cand = [r for r in rac if r >= wp2_prev - 1e-6]
#     wp2_i = min(cand) if cand else (max(rac) if rac else 0.0)
#     r_re = min(p_i / np.sqrt(max(1.0 - wp2_i / wi**2, 1e-12)), a - 1e-6)
#     couches_r.append(r_re)
#     couches_wp2.append(wp2_i)

colors = plt.cm.tab10(np.linspace(0, 0.9, len(LAYERS)))

fig, ax = plt.subplots(figsize=(10, 6))
for (L, col) in zip(LAYERS, colors):
    p_i, wi, tau_obs, hi = snaps[L]
    xs = np.linspace(0.0, hi, 200)
    ys_s = np.array([tau_smooth(x, p_i, wi) for x in xs])
    label = f"couche {L} (p={p_i:.4f}, w={wi:.2e})"
    ax.plot(xs, (ys_s - tau_obs) * 1e9, '-', lw=2, color=col, label=label)
    ax.axhline(0, ls='--', color=col, alpha=0.7)

ax.set_xlabel(r"$w_p^2$ candidat")
ax.set_ylabel(r"$\tau_{\rm smooth} - \tau_{\rm obs}$ (ns)")
ax.legend(fontsize=14)
ax.grid(True)

plt.tight_layout()
plt.savefig("transition_couches.png", dpi=150)
plt.show()
