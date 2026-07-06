import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import matplotlib.colors as mcolors
from scipy.optimize import brentq
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid
import hashlib, json, os, sys, datetime


# ── constantes physiques ──────────────────────────────────────────────────────
c = 3e8
e = 1.6e-19
me = 9.11e-31
eps0 = 8.85e-12

a = 0.2    # rayon plasma (m)
n0 = 1e18

omega_p0 = np.sqrt((n0 * e**2) / (me * eps0))
W_antenne = 0.015   # demi-largeur de l'ouverture (m)

# ── géométrie ─────────────────────────────────────────────────────────────────
# profil gigogne : N couches paraboliques de rayon uniforme entre a et a_min
# dn de chaque couche uniforme tel que la densité centrale totale vaut n0
N_gigogne = 8
a_min_gigogne = 0.02  # rayon de la couche la plus interne (m)
radii_gigogne = np.linspace(a, a_min_gigogne, N_gigogne)
profil_gigogne = [{'dn': n0 / N_gigogne, 'a': float(ai)} for ai in radii_gigogne]

D_PRESETS = [0.40, 0.60, 0.80]   # valeurs de d disponibles (m)
_d_idx    = [0]                   # index du preset actif

def _geom_for_d(d_val):
    R  = np.sqrt(a**2 + (d_val / 2)**2)
    ac = np.degrees(np.arctan(2 * a / d_val))
    ph = np.arctan(d_val / (2 * a))
    ps = np.linspace(R * np.cos(np.radians(2) + ph),
                     R * np.cos(np.radians(ac - 0.5) + ph), 90)
    return R, np.arccos(ps / R) - ph

_R_geom_presets = [_geom_for_d(d_)[0] for d_ in D_PRESETS]
_alphas_presets = [_geom_for_d(d_)[1] for d_ in D_PRESETS]
_R_geom         = [_R_geom_presets[0]]   # mutable : suit le preset actif

# ── profil parabolique ────────────────────────────────────────────────────────
def derivees_rk4_parab(Y, omega_val):
    x, y, px, py = Y
    r2 = x**2 + (y - a)**2
    if r2 >= a**2:
        return np.array([px, py, 0.0, 0.0])
    X_val = omega_p0**2 / omega_val**2
    return np.array([px, py, (X_val / a**2) * x, (X_val / a**2) * (y - a)])

def derivees_rk4_gigogne(Y, omega_val):
    x, y, px, py = Y
    r2 = x**2 + (y - a)**2

    sum_x = 0.0
    sum_y = 0.0
    for couche in profil_gigogne:
        ai = couche['a']
        if r2 < ai**2:
            Xi = couche['dn'] * e**2 / (me * eps0 * omega_val**2)
            sum_x += Xi / ai**2
            sum_y += Xi / ai**2

    return np.array([px, py, x * sum_x, (y - a) * sum_y])

def derivees_rk4_gauss(Y, omega_val, w=0.07):
    x, y, px, py = Y
    X_val = omega_p0**2 / omega_val**2
    r2 = x**2 + (y - a)**2

    # gradient de n² = 1 - X*exp(-r²/w²)
    exp_term = X_val * np.exp(-r2 / w**2)
    dpx = (exp_term / w**2) * x
    dpy = (exp_term / w**2) * (y - a)

    return np.array([px, py, dpx, dpy])


# ── profil actif (modifiable en live via le menu) ─────────────────────────────
w_gauss = 0.07
PROFILES = {
    'parabolique': (derivees_rk4_parab,   lambda om: 1.0),
    'gaussien':    (derivees_rk4_gauss,   lambda om: np.sqrt(max(1.0 - omega_p0**2 * np.exp(-a**2/w_gauss**2) / om**2, 1e-12))),
    'gigogne':     (derivees_rk4_gigogne, lambda om: 1.0),
}
current_derivees    = [derivees_rk4_parab]
current_n_boundary  = [lambda om: 1.0]
current_profile_name = ['parabolique']

def rk4_step(Y, dsigma, omega_val):
    f = current_derivees[0]
    k1 = f(Y, omega_val)
    k2 = f(Y + 0.5 * dsigma * k1, omega_val)
    k3 = f(Y + 0.5 * dsigma * k2, omega_val)
    k4 = f(Y + dsigma * k3, omega_val)
    return Y + (dsigma / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ── ray tracing ───────────────────────────────────────────────────────────────
def calc_point_entree_from(x0, y0, alpha):
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    B = 2 * (x0 * cos_a + (y0 - a) * sin_a)
    C = x0**2 + (y0 - a)**2 - a**2
    disc = B**2 - 4 * C
    if disc < 0:
        raise ValueError("pas d'intersection")
    t = (-B - np.sqrt(disc)) / 2
    if t <= 0:
        t = (-B + np.sqrt(disc)) / 2
    if t <= 0:
        raise ValueError("pas d'intersection vers l'avant")
    return x0 + t * cos_a, y0 + t * sin_a

def simuler_rayon(alpha, omega_val, x0=-D_PRESETS[0]/2, y0=0.0, y_target=0.0, dsigma=5e-4, force_draw=False):
    try:
        x_e, y_e = calc_point_entree_from(x0, y0, alpha)
    except ValueError:
        return None, None, None, None
    # phase dans le vide : source → entrée plasma  (n=1)
    d_in = np.sqrt((x_e - x0)**2 + (y_e - y0)**2)
    phi  = (omega_val / c) * d_in
    n_a = current_n_boundary[0](omega_val)
    Y = np.array([x_e, y_e, n_a * np.cos(alpha), n_a * np.sin(alpha)])
    Y[:2] += 1e-6 * np.array([np.cos(alpha), np.sin(alpha)])
    trajectoire = [Y.copy()]
    while True:
        Y_prev = Y
        Y = rk4_step(Y, dsigma, omega_val)
        trajectoire.append(Y.copy())
        # φ += k · Δr = (ω/c) · (px·Δx + py·Δy)
        phi += (omega_val / c) * (Y_prev[2] * (Y[0] - Y_prev[0])
                                 + Y_prev[3] * (Y[1] - Y_prev[1]))
        if np.sqrt(Y[0]**2 + (Y[1] - a)**2) >= a:
            break
        if len(trajectoire) > 5000:
            return None, None, None, None
    if Y[3] >= 0:
        # rayon réfléchi — on retourne la trajectoire pour affichage si demandé
        if force_draw:
            return None, np.array(trajectoire), None, None
        return None, None, None, None
    # projection balistique vers y = y_target (position du récepteur)
    x_final = Y[0] + (y_target - Y[1]) * (Y[2] / Y[3])
    # temps de vol total : vide_in + plasma (groupe) + vide_out
    d_out   = np.sqrt((x_final - Y[0])**2 + (y_target - Y[1])**2)
    tau_rt  = (d_in + (len(trajectoire) - 1) * dsigma + d_out) / c
    # phase dans le vide : sortie plasma → récepteur  (n≈1)
    phi  += (omega_val / c) * d_out
    return x_final, np.array(trajectoire), tau_rt, phi

def trouver_omega(alpha):
    """Version originale : E à (-d/2,0), R à (d/2,0)."""
    def x_err(om):
        x_fin = simuler_rayon(alpha, om)[0]
        return None if x_fin is None else x_fin - (d / 2)
    omegas = np.linspace(0.3 * omega_p0, 1.5 * omega_p0, 200)
    errs = [x_err(om) for om in omegas]
    for j in range(len(omegas) - 1):
        e0, e1 = errs[j], errs[j + 1]
        if e0 is not None and e1 is not None and e0 * e1 < 0:
            try:
                return brentq(lambda om: simuler_rayon(alpha, om)[0] - d/2,
                              omegas[j], omegas[j + 1])
            except (ValueError, TypeError):
                continue
    return np.nanyons

def trouver_omega_general(alpha, x_E, y_E, x_R, y_R):
    """Version générale : E et R à des positions quelconques."""
    def x_err(om):
        x_fin = simuler_rayon(alpha, om, x0=x_E, y0=y_E, y_target=y_R)[0]
        return None if x_fin is None else x_fin - x_R
    omegas = np.linspace(0.3 * omega_p0, 1.5 * omega_p0, 200)
    errs = [x_err(om) for om in omegas]
    for j in range(len(omegas) - 1):
        e0, e1 = errs[j], errs[j + 1]
        if e0 is not None and e1 is not None and e0 * e1 < 0:
            try:
                return brentq(
                    lambda om: simuler_rayon(alpha, om, x0=x_E, y0=y_E, y_target=y_R)[0] - x_R,
                    omegas[j], omegas[j + 1])
            except (ValueError, TypeError):
                continue
    return np.nan

# ── grille theta pour la table 2D ────────────────────────────────────────────
_theta_grid = np.radians(np.arange(5, 176, 5))   # 5° → 175° par pas de 5°

# ── cache : un fichier par preset de d, clés (prof_name, d_idx) ──────────────
omega_2d_interps = {}   # (prof, d_idx) → RegularGridInterpolator
omega_interps    = {}   # (prof, d_idx) → 1D fallback
_omega_2d_raw    = {}   # (prof, d_idx) → table brute
_alpha_pts_raw   = {}
_theta_pts_raw   = {}

def _build_2d_interp(alpha_pts, theta_pts, omega_2d):
    return RegularGridInterpolator(
        (alpha_pts, theta_pts), omega_2d,
        method='linear', bounds_error=False, fill_value=np.nan)

_CACHE_ONLY = '--cache-only' in sys.argv   # python cone_ui.py --cache-only
_CACHE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cone_cache')
os.makedirs(_CACHE_DIR, exist_ok=True)

for _di, _d_preset in enumerate(D_PRESETS):
    _R_i      = _R_geom_presets[_di]
    _alphas_i = _alphas_presets[_di]
    _d_cm     = int(round(_d_preset * 100))

    _cp = dict(a=a, d=_d_preset, n0=n0, w_gauss=w_gauss,
               N_gigogne=N_gigogne, a_min_gigogne=a_min_gigogne,
               n_alphas=len(_alphas_i),
               alpha_min=float(_alphas_i[0]), alpha_max=float(_alphas_i[-1]),
               theta_step_deg=5, theta_min_deg=5, theta_max_deg=175)
    _ck   = hashlib.md5(json.dumps(_cp, sort_keys=True).encode()).hexdigest()[:10]
    _cf   = os.path.join(_CACHE_DIR, f"cone_ui_cache_{_d_cm}cm_{_ck}.npz")

    _cf_legacy = f"cone_ui_cache_{_ck}.npz"   # ancien emplacement (racine du projet)
    if not os.path.exists(_cf) and os.path.exists(_cf_legacy):
        _cf = _cf_legacy   # utilise l'ancien fichier tel quel

    if os.path.exists(_cf):
        print(f"Cache d={_d_cm}cm trouvé ({os.path.basename(_cf)}), chargement...")
        _cache = np.load(_cf)
        for prof_name in PROFILES:
            a_pts  = _cache[f'{prof_name}_alpha_2d']
            th_pts = _cache[f'{prof_name}_theta_2d']
            om_2d  = _cache[f'{prof_name}_omega_2d']
            key = (prof_name, _di)
            omega_2d_interps[key] = _build_2d_interp(a_pts, th_pts, om_2d)
            _omega_2d_raw[key]    = om_2d
            _alpha_pts_raw[key]   = a_pts
            _theta_pts_raw[key]   = th_pts
            j0 = np.argmin(np.abs(th_pts - np.arcsin((_d_preset / 2) / _R_i)))
            valid = ~np.isnan(om_2d[:, j0])
            omega_interps[key] = interp1d(
                a_pts[valid], om_2d[valid, j0], kind='linear',
                bounds_error=False, fill_value=(om_2d[valid, j0][0], om_2d[valid, j0][-1]))
        print(f"  d={_d_cm}cm — {len(PROFILES)} profils × {len(a_pts)}α × {len(th_pts)}θ.")
    elif _CACHE_ONLY:
        print(f"Cache d={_d_cm}cm absent — démarrage sans données (mode --cache-only)")
        n_alpha = len(_alphas_i)
        n_theta = len(_theta_grid)
        for prof_name in PROFILES:
            key = (prof_name, _di)
            _om_nan = np.full((n_alpha, n_theta), np.nan)
            omega_2d_interps[key] = _build_2d_interp(_alphas_i, _theta_grid, _om_nan)
            _omega_2d_raw[key]    = _om_nan
            _alpha_pts_raw[key]   = _alphas_i
            _theta_pts_raw[key]   = _theta_grid
            omega_interps[key]    = interp1d(
                [_alphas_i[0], _alphas_i[-1]], [omega_p0, omega_p0],
                kind='linear', bounds_error=False, fill_value=omega_p0)
    else:
        # ── caches existants pour ce preset avec des paramètres différents ────
        _old_caches = sorted([
            os.path.join(_CACHE_DIR, f)
            for f in os.listdir(_CACHE_DIR)
            if f.startswith(f'cone_ui_cache_{_d_cm}cm_') and f.endswith('.npz')
            and f != os.path.basename(_cf)
        ], key=os.path.getmtime, reverse=True)

        if _old_caches:
            print(f"\nParamètres modifiés pour d={_d_cm}cm (hash actuel : {_ck}).")
            print("Caches existants pour ce preset :")
            for _i, _op in enumerate(_old_caches):
                _mt = datetime.datetime.fromtimestamp(
                    os.path.getmtime(_op)).strftime('%Y-%m-%d %H:%M')
                print(f"  [{_i+1}] {os.path.basename(_op)}  ({_mt})")
            print(f"  [n] Calculer un nouveau cache (les anciens sont conservés)")
            print(f"  [q] Quitter")
            try:
                _choice = input("Choix : ").strip().lower()
            except EOFError:
                _choice = 'n'
            if _choice == 'q':
                sys.exit(0)
            elif _choice.isdigit() and 1 <= int(_choice) <= len(_old_caches):
                _old_path = _old_caches[int(_choice) - 1]
                print(f"  Chargement de {os.path.basename(_old_path)}...")
                _cache = np.load(_old_path)
                for prof_name in PROFILES:
                    a_pts  = _cache[f'{prof_name}_alpha_2d']
                    th_pts = _cache[f'{prof_name}_theta_2d']
                    om_2d  = _cache[f'{prof_name}_omega_2d']
                    key = (prof_name, _di)
                    omega_2d_interps[key] = _build_2d_interp(a_pts, th_pts, om_2d)
                    _omega_2d_raw[key]    = om_2d
                    _alpha_pts_raw[key]   = a_pts
                    _theta_pts_raw[key]   = th_pts
                    j0 = np.argmin(np.abs(th_pts - np.arcsin((_d_preset / 2) / _R_i)))
                    valid = ~np.isnan(om_2d[:, j0])
                    omega_interps[key] = interp1d(
                        a_pts[valid], om_2d[valid, j0], kind='linear',
                        bounds_error=False,
                        fill_value=(om_2d[valid, j0][0], om_2d[valid, j0][-1]))
                print(f"  d={_d_cm}cm — cache existant chargé ({len(PROFILES)} profils).")
                continue   # passe au d_preset suivant sans recalculer

        _cf_partial = os.path.join(_CACHE_DIR, f"cone_ui_partial_{_d_cm}cm_{_ck}.npz")
        _partial = {}
        if os.path.exists(_cf_partial):
            _pd_file = np.load(_cf_partial)
            for _pn in PROFILES:
                if f'{_pn}_omega_2d' in _pd_file:
                    _partial[_pn] = {
                        'omega_2d':  _pd_file[f'{_pn}_omega_2d'],
                        'cols_done': int(_pd_file[f'{_pn}_cols_done']),
                    }
            _resume_info = ', '.join(f"{p}: {_partial[p]['cols_done']}θ" for p in _partial)
            print(f"  Reprise partielle détectée pour d={_d_cm}cm ({_resume_info})")

        _save_dict = {}
        n_alpha = len(_alphas_i)
        n_theta = len(_theta_grid)
        print(f"\nd={_d_cm}cm — calcul cache {n_alpha}α × {n_theta}θ × {len(PROFILES)} profils...")
        for prof_name, (derivees_func, n_bnd_func) in PROFILES.items():
            current_derivees[0]   = derivees_func
            current_n_boundary[0] = n_bnd_func

            if prof_name in _partial and _partial[prof_name]['cols_done'] >= n_theta:
                omega_2d = _partial[prof_name]['omega_2d']
                print(f"\n  ── [{prof_name}] récupéré du cache partiel (complet)")
            else:
                if prof_name in _partial:
                    omega_2d = _partial[prof_name]['omega_2d']
                    j_start  = _partial[prof_name]['cols_done']
                    print(f"\n  ── [{prof_name}] reprise depuis θ colonne {j_start}/{n_theta}")
                else:
                    omega_2d = np.full((n_alpha, n_theta), np.nan)
                    j_start  = 0
                    print(f"\n  ── [{prof_name}] ── {n_alpha*n_theta} calculs")

                done = j_start * n_alpha
                for j, theta in enumerate(_theta_grid):
                    if j < j_start:
                        continue
                    x_E_j = -_R_i * np.sin(theta)
                    y_E_j =  a - _R_i * np.cos(theta)
                    x_R_j =  _R_i * np.sin(theta)
                    y_R_j =  y_E_j
                    for i_a, alpha in enumerate(_alphas_i):
                        om = trouver_omega_general(alpha, x_E_j, y_E_j, x_R_j, y_R_j)
                        omega_2d[i_a, j] = om
                        done += 1
                        pct = 100 * done / (n_alpha * n_theta)
                        print(f"  θ={np.degrees(theta):.0f}° α={np.degrees(alpha):.1f}°  "
                              f"f={om/(2e9*np.pi):.2f} GHz  [{pct:.0f}%]", end='\r')

                    # sauvegarde partielle après chaque colonne θ
                    _partial[prof_name] = {'omega_2d': omega_2d, 'cols_done': j + 1}
                    _part_save = {}
                    for _pn, _pv in _partial.items():
                        _part_save[f'{_pn}_omega_2d']  = _pv['omega_2d']
                        _part_save[f'{_pn}_cols_done'] = np.array(_pv['cols_done'])
                    np.savez(_cf_partial, **_part_save)

            key = (prof_name, _di)
            _save_dict[f'{prof_name}_alpha_2d'] = _alphas_i
            _save_dict[f'{prof_name}_theta_2d'] = _theta_grid
            _save_dict[f'{prof_name}_omega_2d'] = omega_2d
            omega_2d_interps[key] = _build_2d_interp(_alphas_i, _theta_grid, omega_2d)
            _omega_2d_raw[key]    = omega_2d
            _alpha_pts_raw[key]   = _alphas_i
            _theta_pts_raw[key]   = _theta_grid
            j0 = np.argmin(np.abs(_theta_grid - np.arcsin((_d_preset / 2) / _R_i)))
            valid = ~np.isnan(omega_2d[:, j0])
            omega_interps[key] = interp1d(
                _alphas_i[valid], omega_2d[valid, j0], kind='linear',
                bounds_error=False, fill_value=(omega_2d[valid, j0][0], omega_2d[valid, j0][-1]))
            print(f"\n    [{prof_name}] terminé.")
        np.savez(_cf, **_save_dict)
        if os.path.exists(_cf_partial):
            os.remove(_cf_partial)
        print(f"\n  Cache d={_d_cm}cm sauvegardé → {os.path.basename(_cf)}")

# profil initial : parabolique, preset d=0 (20 cm)
current_derivees[0]   = PROFILES['parabolique'][0]
current_n_boundary[0] = PROFILES['parabolique'][1]
omega_interp = [omega_interps[('parabolique', 0)]]

# ── figures : fenêtre principale + profil d'intensité séparé ─────────────────
fig  = plt.figure(figsize=(14, 7), num='Ray tracing — contrôles')
#fig2 = plt.figure(figsize=(6, 4.5), num="Profil d'intensité")
fig3 = plt.figure(figsize=(6, 4.5), num="Diagramme de rayonnement")
ax_rad = fig3.add_axes([0.13, 0.13, 0.83, 0.75])

fig4 = plt.figure(figsize=(6, 4.5), num="Distribution angulaire des rayons")
ax_distrib = fig4.add_axes([0.13, 0.13, 0.83, 0.75])

ax_ray   = fig.add_axes([0.29, 0.25, 0.40, 0.68])           # tracé rayons (centre)
# ax_int   = fig2.add_axes([0.12, 0.14, 0.83, 0.76])           # profil I(x) (fig2)
ax_amp   = fig.add_axes([0.74, 0.06, 0.24, 0.26])            # puissance vs α (droite, bas)
ax_tau   = fig.add_axes([0.74, 0.37, 0.24, 0.26])            # temps de vol   (droite, milieu)
ax_phase = fig.add_axes([0.74, 0.68, 0.24, 0.26])            # front de phase (droite, haut)

theta_circ = np.linspace(0, 2 * np.pi, 300)
_cmap_plasma = mcolors.LinearSegmentedColormap.from_list('turquoise', ['white', 'turquoise'])

def _ne_normalized(r_arr):
    name = current_profile_name[0]
    if name == 'gaussien':
        return np.exp(-r_arr**2 / w_gauss**2)
    elif name == 'parabolique':
        return np.maximum(0.0, 1.0 - r_arr**2 / a**2)
    else:  # gigogne
        ne = np.zeros_like(r_arr, dtype=float)
        for couche in profil_gigogne:
            ne += (r_arr < couche['a']).astype(float)
        return ne / N_gigogne

def draw_static(ax):
    _x = np.linspace(-a, a, 300)
    _y = np.linspace(-a, a, 300)
    _X, _Y = np.meshgrid(_x, _y)
    _R = np.sqrt(_X**2 + _Y**2)
    _ne = _ne_normalized(_R)
    rgba = _cmap_plasma(np.nan_to_num(_ne))
    rgba[..., 3] = np.where(_R > a, 0.0, 0.2 + np.nan_to_num(_ne) * 0.6)
    ax.imshow(rgba, extent=[-a, a, 0, 2*a], origin='lower',
              interpolation='bilinear', zorder=0)
    ax.plot(a * np.cos(theta_circ), a + a * np.sin(theta_circ), 'k-', alpha=0.5)
    # cercle de référence sur lequel se déplacent E et R
    ax.plot(_R_geom[0] * np.cos(theta_circ), a + _R_geom[0] * np.sin(theta_circ),
            ':', color='gray', alpha=0.3, linewidth=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle=':')
    ax.set_aspect('equal')

def draw_cone(ax, alpha, x_E, y_E, x_R, y_R):
    L = 0.03
    perp_x, perp_y = -np.sin(alpha), np.cos(alpha)

    # émetteur : aperture + bords du cône dans la direction α
    x0p = x_E + W_antenne*perp_x;  y0p = y_E + W_antenne*perp_y
    x0m = x_E - W_antenne*perp_x;  y0m = y_E - W_antenne*perp_y
    ax.plot([x0m, x0p], [y0m, y0p], 'k-', lw=3, solid_capstyle='round', zorder=5)
    #for x0, y0 in [(x0p, y0p), (x0m, y0m)]:
        #ax.plot([x0, x0 + L*np.cos(alpha)], [y0, y0 + L*np.sin(alpha)], 'k-', lw=1, alpha=0.5)

    # récepteur : direction d'arrivée (-cos α, sin α), aperture perpendiculaire = (sin α, cos α)
    rx0p = x_R - W_antenne*perp_x;  ry0p = y_R + W_antenne*perp_y
    rx0m = x_R + W_antenne*perp_x;  ry0m = y_R - W_antenne*perp_y
    ax.plot([rx0m, rx0p], [ry0m, ry0p], 'k-', lw=3, solid_capstyle='round', zorder=5)
    #for x0, y0 in [(rx0p, ry0p), (rx0m, ry0m)]:
        #ax.plot([x0, x0 - L*np.cos(alpha)], [y0, y0 + L*np.sin(alpha)], 'k-', lw=1, alpha=0.5)

def tracer_traj(ax, traj, x0_src, y0_src, x_R, y_R, color, lw, alpha_plot):
    xe0, ye0 = traj[0, 0], traj[0, 1]
    x_out, y_out, px_out, py_out = traj[-1]
    # prolongement rectiligne dans le vide après la sortie du plasma, poursuivi
    # au-delà du plan récepteur y_R jusqu'au bord du graphe (transmis : vers le bas,
    # réfléchi : vers le haut). matplotlib découpe le dépassement latéral.
    y_edge = ax.get_ylim()[0] if py_out < 0 else ax.get_ylim()[1]
    if abs(py_out) > 1e-10 and (y_edge - y_out) / py_out > 0:
        t_end = (y_edge - y_out) / py_out
        x_vide = np.array([x_out + px_out * t_end])
        y_vide = np.array([y_edge])
    else:
        x_vide = np.array([])
        y_vide = np.array([])

    ax.plot(np.concatenate([[x0_src, xe0], traj[:, 0], x_vide]),
            np.concatenate([[y0_src, ye0], traj[:, 1], y_vide]),
            '-', color=color, linewidth=lw, alpha=alpha_plot)

# historique pour le graphe d'amplitude
hist_alpha_deg = []
hist_ampl      = []

_theta0 = np.arcsin((D_PRESETS[0] / 2) / _R_geom_presets[0])  # angle initial (preset d=20 cm)

# ── état du scan θ ───────────────────────────────────────────────────────────
_view_zoom = [1.0]   # zoom : 1 = vue par défaut, >1 = zoom avant
_view_cx   = [0.0]   # centre x (m)
_view_cy   = [a]     # centre y (m)

_delta_alpha_theta = [0.0]  # angle relatif fixé au démarrage du scan θ
_updating          = [False] # verrou pour éviter double-update lors du scan θ
_export_data       = {'ray': None, 'int': None, 'amp': None}
_ray_mode          = ['decale']  # 'decale' ou 'isopuissance'

def update(val):
    if _updating[0]:
        return
    theta = np.radians(sl_theta.val)
    alpha = np.radians(sl_alpha.val)
    N     = int(sl_N.val)
    n_rays = 2 * N + 1

    # ── positions E et R (symétrie bilatérale, ψ=0) ───────────────────────────
    x_E = -_R_geom[0] * np.sin(theta)
    y_E =  a - _R_geom[0] * np.cos(theta)
    x_R =  _R_geom[0] * np.sin(theta)
    y_R =  y_E

    # ── recherche de ω via la table 2D interpolée ─────────────────────────────
    omega_val = float(omega_2d_interps[(current_profile_name[0], _d_idx[0])]([[alpha, theta]])[0])
    omega_approx = np.isnan(omega_val)
    if omega_approx:
        omega_val = float(omega_interp[0](alpha))

    f_GHz = omega_val / (2e9 * np.pi)

    # ── CDFs (TE₁₀ et sinc²) ─────────────────────────────────────────────────
    lam_emit  = 2 * np.pi * c / omega_val
    L_ant     = 2 * W_antenne
    _th_grid  = np.linspace(-np.pi / 2, np.pi / 2, 8000)
    _v        = (L_ant / lam_emit) * np.sin(_th_grid)
    _num      = np.cos(np.pi * _v)
    _den      = 1.0 - (2.0 * _v) ** 2
    _sing     = np.isclose(_den, 0.0, atol=1e-9)
    _P_te10   = np.where(_sing, (np.pi / 4.0) ** 2,
                         (_num / np.where(_sing, 1.0, _den)) ** 2)
    _P_sinc2  = np.sinc(_v) ** 2   # np.sinc normalise : sinc(x)=sin(πx)/(πx)

    if _ray_mode[0] == 'iso_sinc2':
        _P_cdf = _P_sinc2
    else:
        _P_cdf = _P_te10

    _cdf      = cumulative_trapezoid(_P_cdf, _th_grid, initial=0)
    _cdf     /= _cdf[-1]
    _inv_cdf  = interp1d(_cdf, _th_grid, kind='linear', bounds_error=False,
                         fill_value=(_th_grid[0], _th_grid[-1]))
    probs     = (np.arange(n_rays) + 0.5) / n_rays
    theta_k   = _inv_cdf(probs)   # θ_k[N] = 0 par symétrie

    perp_x, perp_y = -np.sin(alpha), np.cos(alpha)

    # ── mode de lancement des rayons ─────────────────────────────────────────
    if _ray_mode[0] == 'decale':
        # rayons parallèles (même angle α) depuis 2N+1 positions sur l'ouverture
        offsets = np.linspace(-W_antenne, W_antenne, n_rays)
        sources = [(x_E + s*perp_x, y_E + s*perp_y) for s in offsets]
        angles  = [alpha] * n_rays
    else:
        # isopuissance (TE₁₀ ou sinc²) : segments uniformes, angles par CDF
        offsets = -W_antenne + (np.arange(n_rays) + 0.5) * L_ant / n_rays
        sources = [(x_E + s*perp_x, y_E + s*perp_y) for s in offsets]
        angles  = list(alpha + theta_k)

    # ── simule tous les rayons ────────────────────────────────────────────────
    ax_ray.cla()
    draw_static(ax_ray)
    hw = 0.25 / _view_zoom[0]
    ax_ray.set_xlim(_view_cx[0] - hw, _view_cx[0] + hw)
    ax_ray.set_ylim(_view_cy[0] - hw, _view_cy[0] + hw)
    draw_cone(ax_ray, alpha, x_E, y_E, x_R, y_R)

    x_out_all  = []
    phi_all    = []
    tau_all    = []
    _trajs_exp = []
    for k in range(n_rays):
        x0_s, y0_s = sources[k]
        x_fin, traj, tau_k, phi_k = simuler_rayon(angles[k], omega_val, x0=x0_s, y0=y0_s, y_target=y_R,
                                                   force_draw=omega_approx)
        x_out_all.append(x_fin)
        phi_all.append(phi_k)
        tau_all.append(tau_k)
        if traj is not None:
            is_central = (k == N)
            lw_r  = 1.4 if is_central else 0.5
            ap_r  = 1.0 if is_central else 0.4
            col   = 'tab:red' if is_central else 'tomato'
            tracer_traj(ax_ray, traj, x0_s, y0_s, x_R, y_R, col, lw_r, ap_r)
            _trajs_exp.append((traj, x0_s, y0_s, col, lw_r, ap_r))

    _export_data['ray'] = {
        'trajs': _trajs_exp, 'x_E': x_E, 'y_E': y_E, 'x_R': x_R, 'y_R': y_R,
        'alpha': alpha, 'f_GHz': f_GHz, 'omega_approx': omega_approx, 'N': N,
        'R_geom': _R_geom[0],
    }

    # ── calcul du profil d'intensité locale (méthode tube, mode décalé seul) ───
    valid = [(offsets[k], x_out_all[k], 1.0)
             for k in range(n_rays) if x_out_all[k] is not None]
    valid.sort(key=lambda t: t[0])

    ampl_rx = None
    if len(valid) >= 2:
        # trier par x_out (pas s_in) pour que x_mid soit monotone même si des rayons se croisent
        valid_sorted = sorted(valid, key=lambda t: t[1])
        s_in   = np.array([t[0] for t in valid_sorted])
        x_out  = np.array([t[1] for t in valid_sorted])
        emit_w = np.array([t[2] for t in valid_sorted])

        ds_in  = np.abs(np.diff(s_in))   # |ds_in| entre voisins triés par x_out
        dx_out = np.diff(x_out)           # toujours >= 0 (trié)
        # I_k = |ds_in / dx_out| — densité de rayons à la sortie, normalisée à 1 en espace libre
        I_k   = ds_in / np.where(dx_out < 1e-9, np.inf, dx_out)
        x_mid = 0.5 * (x_out[:-1] + x_out[1:])

        _export_data['int'] = {
            'x_mid': x_mid, 'I_k': I_k, 'dx_out': dx_out, 'x_R': x_R, 'n_rays': n_rays,
        }

        # # ── panneau intensité ─────────────────────────────────────────────────
        # ax_int.cla()
        # ax_int.bar(x_mid * 100, I_k, width=dx_out * 100,
        #            color='steelblue', alpha=0.7, edgecolor='none')
        # ax_int.axvline(x_R * 100, color='red', lw=1.5, linestyle='--', label="récepteur")
        # ax_int.axhline(1.0, color='k', lw=0.8, linestyle=':', alpha=0.5, label="réf. libre")
        # ax_int.set_xlabel("x à la réception (cm)")
        # ax_int.set_ylabel("I locale  (|ds_in| / dx_out)")
        # ax_int.set_title(f"Profil d'intensité  (2N+1 = {n_rays} rayons)")
        # ax_int.set_ylim(0, 5)
        # ax_int.legend(fontsize=7)
        # ax_int.grid(True, linestyle=':')

        # puissance au récepteur
        # décalé   : fraction des rayons parallèles atterrissant dans l'ouverture
        # isopuissance : fraction des rayons transmis (≠ réfléchis) — observable
        #               pertinent car la condition |x_out-x_R|≤W ne capte qu'1 rayon
        #               pour un diagramme sinc² large (L/λ≈1) quelle que soit α
        if _ray_mode[0] == 'decale':
            hits = sum(1 for xo in x_out_all if xo is not None and abs(xo - x_R) <= W_antenne)
        else:
            hits = sum(1 for xo in x_out_all if xo is not None)
        ampl_rx = hits / n_rays

    # ── panneau front de phase ────────────────────────────────────────────────
    phase_pairs = [(x_out_all[k], phi_all[k])
                   for k in range(n_rays)
                   if x_out_all[k] is not None and phi_all[k] is not None]
    ax_phase.cla()
    if len(phase_pairs) >= 2:
        x_arr  = np.array([p[0] for p in phase_pairs])
        phi_a  = np.array([p[1] for p in phase_pairs])
        # position le long de l'ouverture du détecteur (direction ⊥ α)
        # s_det = (x_arr - x_R) · perp_x  avec perp_x = -sin(alpha)
        s_det  = (x_arr - x_R) * (-np.sin(alpha)) * 100   # cm
        i_ref  = np.argmin(np.abs(s_det))                  # rayon le plus proche du centre
        phi_rel = phi_a - phi_a[i_ref]
        _export_data['phase'] = {'s_det': s_det, 'phi_rel': phi_rel,
                                  'N': int(sl_N.val), 'alpha_deg': np.degrees(alpha)}
        ax_phase.plot(s_det, phi_rel, 'o-', color='darkorange', lw=1.5, ms=3)
        ax_phase.axvline(0, color='k', lw=0.8, linestyle=':', alpha=0.5)
        ax_phase.axhline(0, color='k', lw=0.5, linestyle=':', alpha=0.4)
        ax_phase.set_xlabel("s (cm)", fontsize=7)
        ax_phase.set_ylabel("Δφ (rad)", fontsize=7)
        ax_phase.tick_params(labelsize=6)
        ax_phase.grid(True, linestyle=':')
    ax_phase.set_xlim(-W_antenne * 100 * 1.1, W_antenne * 100 * 1.1)

    # ── panneau temps de vol ──────────────────────────────────────────────────
    tau_pairs = [(x_out_all[k], tau_all[k])
                 for k in range(n_rays)
                 if x_out_all[k] is not None and tau_all[k] is not None]
    ax_tau.cla()
    if len(tau_pairs) >= 2:
        x_arr_t = np.array([p[0] for p in tau_pairs])
        tau_a   = np.array([p[1] for p in tau_pairs])
        s_det_t = (x_arr_t - x_R) * (-np.sin(alpha)) * 100   # cm le long du détecteur
        tau_ns  = tau_a * 1e9                                  # ns
        _export_data['tau'] = {'s_det': s_det_t, 'tau_ns': tau_ns,
                               'N': int(sl_N.val), 'alpha_deg': np.degrees(alpha)}
        ax_tau.plot(s_det_t, tau_ns, 'o-', color='mediumseagreen', lw=1.5, ms=3)
        ax_tau.axvline(0, color='k', lw=0.8, linestyle=':', alpha=0.5)
        ax_tau.set_xlabel("s (cm)", fontsize=7)
        ax_tau.set_ylabel("τ (ns)", fontsize=7)
        ax_tau.tick_params(labelsize=6)
        ax_tau.grid(True, linestyle=':')
    ax_tau.set_xlim(-W_antenne * 100 * 1.1, W_antenne * 100 * 1.1)

    # ── titre panneau rayons ──────────────────────────────────────────────────
    title = f"α = {np.degrees(alpha):.1f}°   f = {f_GHz:.2f} GHz"
    if omega_approx:
        title += "  ⚠ ω approx"
    _is_iso = (_ray_mode[0] != 'decale')
    _amp_label = "A = P_rec/P_tot" if _is_iso else "puissance norm."
    _amp_title_val = "A" if _is_iso else "puissance"

    if ampl_rx is not None:
        title += f"\n{_amp_title_val} = {ampl_rx:.3f}"
    ax_ray.set_title(title, fontsize=9)

    # ── historique amplitude ──────────────────────────────────────────────────
    if ampl_rx is not None:
        hist_alpha_deg.append(np.degrees(alpha))
        hist_ampl.append(ampl_rx)
        _export_data['amp'] = {
            'alpha_hist': list(hist_alpha_deg), 'ampl_hist': list(hist_ampl),
            'alpha_cur': np.degrees(alpha), 'ampl_cur': ampl_rx,
        }

    ax_amp.cla()
    if len(hist_alpha_deg) > 1:
        order = np.argsort(hist_alpha_deg)
        ax_amp.plot(np.array(hist_alpha_deg)[order],
                    np.array(hist_ampl)[order], 'b.-')
    ax_amp.axhline(1.0, color='k', linestyle='--', alpha=0.4, label="référence")
    if ampl_rx is not None:
        ax_amp.scatter([np.degrees(alpha)], [ampl_rx], color='red', zorder=5, s=60)
    ax_amp.set_xlabel("α (°)", fontsize=7)
    ax_amp.set_ylabel(_amp_label, fontsize=7)
    ax_amp.legend(fontsize=6)
    ax_amp.tick_params(labelsize=6)
    ax_amp.grid(True, linestyle=':')

    # ── diagramme de rayonnement ──────────────────────────────────────────────
    ax_rad.cla()
    lam       = 2 * np.pi * c / omega_val
    theta_deg = np.linspace(-90, 90, 2000)
    _v_r = (2 * W_antenne / lam) * np.sin(np.radians(theta_deg))
    if _ray_mode[0] == 'isopuissance':
        _num_r  = np.cos(np.pi * _v_r)
        _den_r  = 1.0 - (2.0 * _v_r) ** 2
        _sing_r = np.isclose(_den_r, 0.0, atol=1e-9)
        P_rad   = np.where(_sing_r, (np.pi / 4.0) ** 2,
                           (_num_r / np.where(_sing_r, 1.0, _den_r)) ** 2)
        _label_rad = 'TE₁₀'
    else:   # décalé et iso_sinc2 : illumination uniforme → sinc²
        P_rad      = np.sinc(_v_r) ** 2
        _label_rad = 'sinc²'
    P_rad_dB = 10 * np.log10(np.maximum(P_rad, 1e-10))

    # ── FWHM (points à −3 dB autour du pic) ──────────────────────────────────
    i_center = np.argmax(P_rad_dB)
    half_pow = P_rad_dB[i_center] - 3.0
    theta_left  = theta_deg[0]
    theta_right = theta_deg[-1]
    for i in range(i_center, 0, -1):
        if P_rad_dB[i] < half_pow:
            frac = (half_pow - P_rad_dB[i+1]) / (P_rad_dB[i] - P_rad_dB[i+1])
            theta_left = theta_deg[i+1] + frac * (theta_deg[i] - theta_deg[i+1])
            break
    for i in range(i_center, len(P_rad_dB) - 1):
        if P_rad_dB[i+1] < half_pow:
            frac = (half_pow - P_rad_dB[i]) / (P_rad_dB[i+1] - P_rad_dB[i])
            theta_right = theta_deg[i] + frac * (theta_deg[i+1] - theta_deg[i])
            break
    fwhm_val = theta_right - theta_left

    # ── premier lobe secondaire (min local → max local à droite du pic) ──────
    sll_dB = None
    for i in range(i_center + 2, len(P_rad_dB) - 2):
        if P_rad_dB[i-1] >= P_rad_dB[i] <= P_rad_dB[i+1]:  # minimum local (null)
            for j in range(i + 2, len(P_rad_dB) - 1):
                if P_rad_dB[j-1] <= P_rad_dB[j] >= P_rad_dB[j+1]:  # maximum local
                    sll_dB = P_rad_dB[j]
                    break
            break

    ax_rad.plot(theta_deg, P_rad_dB, color='steelblue', lw=2.0, label=_label_rad)
    ax_rad.axhline(-3,  color='gray', lw=0.8, linestyle='--', alpha=0.6, label='−3 dB')
    ax_rad.axvline(0,   color='k',   lw=0.6, linestyle=':',  alpha=0.4)

    # marqueurs FWHM
    ax_rad.axvline(theta_left,  color='crimson', lw=0.9, linestyle='--', alpha=0.7)
    ax_rad.axvline(theta_right, color='crimson', lw=0.9, linestyle='--', alpha=0.7)
    ax_rad.annotate('', xy=(theta_right, half_pow), xytext=(theta_left, half_pow),
                    arrowprops=dict(arrowstyle='<->', color='crimson', lw=1.2))
    ax_rad.text(0, half_pow - 1.2, f"FWHM = {fwhm_val:.1f}°",
                ha='center', va='top', fontsize=7.5, color='crimson',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, ec='none'))

    # niveau du lobe secondaire
    if sll_dB is not None:
        ax_rad.axhline(sll_dB, color='darkorange', lw=0.9, linestyle=':', alpha=0.9)
        ax_rad.text(87, sll_dB + 0.8, f"SLL = {sll_dB:.1f} dB",
                    ha='right', va='bottom', fontsize=7.5, color='darkorange',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, ec='none'))

    ax_rad.set_xlim(-90, 90)
    ax_rad.set_ylim(-30, 1)
    ax_rad.set_xlabel("θ par rapport à la normale (°)", fontsize=8)
    ax_rad.set_ylabel("P(θ) / P(0)  (dB)", fontsize=8)
    ax_rad.set_title(
                     f"f = {f_GHz:.2f} GHz,  2W/λ = {2*W_antenne/lam:.2f}", fontsize=9)
    ax_rad.legend(fontsize=7)
    ax_rad.grid(True, linestyle=':')

    # ── distribution angulaire des rayons (fig4) ─────────────────────────────
    ax_distrib.cla()
    k_arr      = np.arange(n_rays)
    angles_rel = np.degrees(np.array(angles) - alpha)
    ax_distrib.plot(k_arr, angles_rel, 'o-',
                    color='steelblue', lw=1.5, ms=4)
    ax_distrib.axhline(0, color='k', lw=0.8, linestyle=':')
    ax_distrib.set_xlabel("indice du rayon  k", fontsize=9)
    ax_distrib.set_ylabel("θ_k  (°)", fontsize=9)
    ax_distrib.set_title(
        f"2N+1 = {n_rays}", fontsize=9)
    ax_distrib.set_xlim(-0.5, n_rays - 0.5)
    ax_distrib.set_ylim(-90, 90)
    ax_distrib.grid(True, linestyle=':', axis='y')
    fig4.canvas.draw_idle()

    fig.canvas.draw_idle()
    #fig2.canvas.draw_idle()
    fig3.canvas.draw_idle()

# ── slider θ (demi-angle de séparation E-R sur le cercle de rayon R_geom) ────
_theta0_deg = float(np.degrees(_theta0))
ax_sl_theta = fig.add_axes([0.300, 0.172, 0.250, 0.028])
sl_theta = widgets.Slider(ax_sl_theta, 'θ (°)', 5, 175,
                          valinit=_theta0_deg, valstep=1)
sl_theta.valtext.set_visible(False)

def on_theta_change(v):
    if not _updating[0]:
        tb_theta.set_val(f"{v:.0f}")
        update(v)

sl_theta.on_changed(on_theta_change)

# ── slider α ─────────────────────────────────────────────────────────────────
_atab0 = omega_interps[('parabolique', 0)].x
ax_sl = fig.add_axes([0.300, 0.125, 0.250, 0.028])
sl_alpha = widgets.Slider(ax_sl, 'α (°)', 0, 90,
    valinit=np.degrees(_atab0[len(_atab0) // 2]), valstep=0.2)
sl_alpha.valtext.set_visible(False)

def on_alpha_change(v):
    if not _updating[0]:
        tb_alpha.set_val(f"{v:.0f}")
        update(v)

sl_alpha.on_changed(on_alpha_change)

# ── slider N ─────────────────────────────────────────────────────────────────
ax_sl_N = fig.add_axes([0.300, 0.078, 0.250, 0.028])
sl_N = widgets.Slider(ax_sl_N, 'N', 1, 25, valinit=5, valstep=1)
sl_N.valtext.set_visible(False)
sl_N.on_changed(lambda v: (tb_N.set_val(str(int(v))), update(sl_alpha.val)))

# ── slider W (demi-largeur antenne) ──────────────────────────────────────────
ax_sl_W = fig.add_axes([0.300, 0.030, 0.250, 0.028])
sl_W = widgets.Slider(ax_sl_W, 'W (cm)', 0.5, 10.0,
                      valinit=W_antenne * 100, valstep=0.5)
sl_W.valtext.set_visible(False)

def on_W_change(v):
    global W_antenne
    W_antenne = v / 100.0
    tb_W.set_val(f"{v:.1f}")
    update(sl_alpha.val)

sl_W.on_changed(on_W_change)

# ── TextBox W ─────────────────────────────────────────────────────────────────
ax_tb_W = fig.add_axes([0.558, 0.027, 0.028, 0.028])
tb_W = widgets.TextBox(ax_tb_W, '', initial=f"{W_antenne*100:.1f}")
tb_W.text_disp.set_fontsize(7)

def on_W_submit(text):
    try:
        val = float(text)
        sl_W.set_val(np.clip(val, sl_W.valmin, sl_W.valmax))
    except ValueError:
        pass

tb_W.on_submit(on_W_submit)

# ── sliders vue (zoom / x / y) ────────────────────────────────────────────────
# ── TextBoxes vue (zoom / x₀ / y₀) ──────────────────────────────────────────
fig.text(0.605, 0.158, 'Zoom',    fontsize=7, ha='left', color='gray')
fig.text(0.605, 0.111, 'x₀ (cm)', fontsize=7, ha='left', color='gray')
fig.text(0.605, 0.064, 'y₀ (cm)', fontsize=7, ha='left', color='gray')

ax_tb_zoom = fig.add_axes([0.655, 0.143, 0.060, 0.028])
tb_zoom = widgets.TextBox(ax_tb_zoom, '', initial='1.0')
tb_zoom.text_disp.set_fontsize(7)

ax_tb_vx = fig.add_axes([0.655, 0.096, 0.060, 0.028])
tb_vx = widgets.TextBox(ax_tb_vx, '', initial='0.0')
tb_vx.text_disp.set_fontsize(7)

ax_tb_vy = fig.add_axes([0.655, 0.049, 0.060, 0.028])
tb_vy = widgets.TextBox(ax_tb_vy, '', initial=f'{a*100:.1f}')
tb_vy.text_disp.set_fontsize(7)

def on_zoom_submit(text):
    try:
        _view_zoom[0] = max(0.1, float(text))
        update(sl_alpha.val)
    except ValueError:
        pass

def on_vx_submit(text):
    try:
        _view_cx[0] = float(text) / 100.0
        update(sl_alpha.val)
    except ValueError:
        pass

def on_vy_submit(text):
    try:
        _view_cy[0] = float(text) / 100.0
        update(sl_alpha.val)
    except ValueError:
        pass

tb_zoom.on_submit(on_zoom_submit)
tb_vx.on_submit(on_vx_submit)
tb_vy.on_submit(on_vy_submit)

# ── TextBox θ ─────────────────────────────────────────────────────────────────
ax_tb_theta = fig.add_axes([0.558, 0.169, 0.028, 0.028])
tb_theta = widgets.TextBox(ax_tb_theta, '', initial=f"{sl_theta.val:.0f}")
tb_theta.text_disp.set_fontsize(7)

def on_theta_submit(text):
    try:
        val = float(text)
        sl_theta.set_val(np.clip(val, sl_theta.valmin, sl_theta.valmax))
    except ValueError:
        pass

tb_theta.on_submit(on_theta_submit)

# ── RadioButtons mode de lancement des rayons ────────────────────────────────
ax_radio_mode = fig.add_axes([0.02, 0.565, 0.18, 0.115])
ax_radio_mode.set_title('Mode rayons', fontsize=7, pad=2)
radio_mode = widgets.RadioButtons(ax_radio_mode,
                                  ['Décalés', 'Iso. sinc²', 'Iso. TE₁₀'],
                                  activecolor='steelblue')
for lbl in radio_mode.labels:
    lbl.set_fontsize(7)

def on_mode_change(label):
    if label == 'Décalés':
        _ray_mode[0] = 'decale'
    elif label == 'Iso. sinc²':
        _ray_mode[0] = 'iso_sinc2'
    else:
        _ray_mode[0] = 'isopuissance'
    update(sl_alpha.val)

radio_mode.on_clicked(on_mode_change)

# ── RadioButtons d ────────────────────────────────────────────────────────────
ax_radio_d = fig.add_axes([0.02, 0.70, 0.18, 0.11])
ax_radio_d.set_title('$d_{geom}$ (m)', fontsize=7, pad=2)
_d_labels = [f"{int(d_*100)} cm" for d_ in D_PRESETS]
radio_d = widgets.RadioButtons(ax_radio_d, _d_labels, activecolor='steelblue')
for lbl in radio_d.labels:
    lbl.set_fontsize(7)

def on_d_change(label):
    di = _d_labels.index(label)
    _d_idx[0]  = di
    _R_geom[0] = _R_geom_presets[di]
    omega_interp[0] = omega_interps[(current_profile_name[0], di)]
    hist_alpha_deg.clear()
    hist_ampl.clear()
    _init_scan_ranges()
    update(sl_alpha.val)

radio_d.on_clicked(on_d_change)


# ── bouton Défaut (θ et α optimaux pour le profil et le d courants) ───────────
ax_btn_default = fig.add_axes([0.02, 0.232, 0.18, 0.030])
btn_default = widgets.Button(ax_btn_default, 'Défaut', color='#e8e8f8')
btn_default.label.set_fontsize(8)

def set_default_params(event=None):
    d = _defaults()
    sl_theta.set_val(np.clip(d['theta'], sl_theta.valmin, sl_theta.valmax))
    sl_alpha.set_val(np.clip(d['alpha'], sl_alpha.valmin, sl_alpha.valmax))
    _scan_theta_min[0] = float(d['theta_min'])
    _scan_theta_max[0] = float(d['theta_max'])
    _scan_alpha_min[0] = float(d['alpha_min'])
    _scan_alpha_max[0] = float(d['alpha_max'])
    _scan_N_min[0]     = int(d['N_min'])
    _scan_N_max[0]     = int(d['N_max'])
    tb_sth_min.set_val(str(d['theta_min']))
    tb_sth_max.set_val(str(d['theta_max']))
    tb_sal_min.set_val(str(d['alpha_min']))
    tb_sal_max.set_val(str(d['alpha_max']))
    tb_sN_min.set_val(str(d['N_min']))
    tb_sN_max.set_val(str(d['N_max']))
    update(sl_alpha.val)

btn_default.on_clicked(set_default_params)

# ── TextBox α ─────────────────────────────────────────────────────────────────
ax_tb_alpha = fig.add_axes([0.558, 0.122, 0.028, 0.028])
tb_alpha = widgets.TextBox(ax_tb_alpha, '', initial=f"{sl_alpha.val:.0f}")
tb_alpha.text_disp.set_fontsize(7)

def on_alpha_submit(text):
    try:
        val = float(text)
        sl_alpha.set_val(np.clip(val, sl_alpha.valmin, sl_alpha.valmax))
    except ValueError:
        pass

tb_alpha.on_submit(on_alpha_submit)

# ── TextBox N ─────────────────────────────────────────────────────────────────
ax_tb_N = fig.add_axes([0.558, 0.075, 0.028, 0.028])
tb_N = widgets.TextBox(ax_tb_N, '', initial=str(int(sl_N.val)))
tb_N.text_disp.set_fontsize(7)

def on_N_submit(text):
    try:
        val = int(float(text))
        sl_N.set_val(np.clip(val, sl_N.valmin, sl_N.valmax))
    except ValueError:
        pass

tb_N.on_submit(on_N_submit)

# ── plages de scan configurables ─────────────────────────────────────────────
# ── paramètres par défaut par profil (à modifier ici) ────────────────────────
PROFILE_DEFAULTS = {
    'parabolique': dict(theta=45, alpha=28,
                        theta_min=25, theta_max=60,
                        alpha_min=13.0, alpha_max=39.0,
                        N_min=1, N_max=25),
    'gaussien':    dict(theta=45, alpha=20,
                        theta_min=25, theta_max=60,
                        alpha_min=10.0, alpha_max=35.0,
                        N_min=1, N_max=25),
    'gigogne':     dict(theta=45, alpha=25,
                        theta_min=25, theta_max=60,
                        alpha_min=10.0, alpha_max=40.0,
                        N_min=1, N_max=25),
}

def _defaults():
    return PROFILE_DEFAULTS[current_profile_name[0]]

_scan_alpha_min = [float(_defaults()['alpha_min'])]
_scan_alpha_max = [float(_defaults()['alpha_max'])]
_scan_theta_min = [float(_defaults()['theta_min'])]
_scan_theta_max = [float(_defaults()['theta_max'])]
_scan_N_min     = [int(_defaults()['N_min'])]
_scan_N_max     = [int(_defaults()['N_max'])]

# ── scan automatique ─────────────────────────────────────────────────────────
scan_timers = [None, None, None]   # [scan_α, scan_N, scan_rot]

def _stop_scan(idx, btn, label):
    if scan_timers[idx] is not None:
        scan_timers[idx].stop()
        scan_timers[idx] = None
    btn.label.set_text(label)
    fig.canvas.draw_idle()

def scan_alpha_step():
    nv = round(sl_alpha.val + 0.4, 1)
    lim = float(np.clip(_scan_alpha_max[0], sl_alpha.valmin, sl_alpha.valmax))
    if nv >= lim:
        sl_alpha.set_val(lim)
        _stop_scan(0, btn_scan_alpha, 'Scan α')
    else:
        sl_alpha.set_val(nv)

def scan_N_step():
    nv = int(sl_N.val) + 1
    lim = int(np.clip(_scan_N_max[0], sl_N.valmin, sl_N.valmax))
    if nv >= lim:
        sl_N.set_val(lim)
        _stop_scan(1, btn_scan_N, 'Scan N')
    else:
        sl_N.set_val(nv)

def toggle_scan_alpha(event):
    if scan_timers[0] is not None:
        _stop_scan(0, btn_scan_alpha, 'Scan α')
    else:
        start = float(np.clip(_scan_alpha_min[0], sl_alpha.valmin, sl_alpha.valmax))
        sl_alpha.set_val(start)
        t = fig.canvas.new_timer(interval=300)
        t.add_callback(scan_alpha_step)
        scan_timers[0] = t
        t.start()
        btn_scan_alpha.label.set_text('Stop α')
        fig.canvas.draw_idle()

def toggle_scan_N(event):
    if scan_timers[1] is not None:
        _stop_scan(1, btn_scan_N, 'Scan N')
    else:
        start = int(np.clip(_scan_N_min[0], sl_N.valmin, sl_N.valmax))
        sl_N.set_val(start)
        t = fig.canvas.new_timer(interval=600)
        t.add_callback(scan_N_step)
        scan_timers[1] = t
        t.start()
        btn_scan_N.label.set_text('Stop N')
        fig.canvas.draw_idle()

ax_scan_a = fig.add_axes([0.02, 0.318, 0.18, 0.030])
btn_scan_alpha = widgets.Button(ax_scan_a, 'Scan α', color='#e8e8f8')
btn_scan_alpha.label.set_fontsize(8)
btn_scan_alpha.on_clicked(toggle_scan_alpha)

ax_scan_N = fig.add_axes([0.02, 0.284, 0.18, 0.030])
btn_scan_N = widgets.Button(ax_scan_N, 'Scan N', color='#e8e8f8')
btn_scan_N.label.set_fontsize(8)
btn_scan_N.on_clicked(toggle_scan_N)

# ── scan θ (θ : min→max, α adapté pour maintenir l'angle relatif E→centre) ───
def _stop_scan_theta():
    if scan_timers[2] is not None:
        scan_timers[2].stop()
        scan_timers[2] = None
    btn_scan_theta.label.set_text('Scan θ')
    fig.canvas.draw_idle()

def scan_theta_step():
    lim = float(np.clip(_scan_theta_max[0], sl_theta.valmin, sl_theta.valmax))
    nv = min(sl_theta.val + 1, lim)
    theta_new = np.radians(nv)
    alpha_new = np.pi / 2 - theta_new + _delta_alpha_theta[0]
    alpha_new_deg = float(np.clip(np.degrees(alpha_new), 0, 90))
    _updating[0] = True
    sl_alpha.set_val(alpha_new_deg)
    tb_alpha.set_val(f"{alpha_new_deg:.0f}")
    _updating[0] = False
    sl_theta.set_val(nv)
    if nv >= lim:
        _stop_scan_theta()

def toggle_scan_theta(event):
    if scan_timers[2] is not None:
        _stop_scan_theta()
    else:
        alpha_0 = np.radians(sl_alpha.val)
        _delta_alpha_theta[0] = alpha_0 - (np.pi / 2 - np.radians(sl_theta.val))
        start = float(np.clip(_scan_theta_min[0], sl_theta.valmin, sl_theta.valmax))
        sl_theta.set_val(start)
        t = fig.canvas.new_timer(interval=200)
        t.add_callback(scan_theta_step)
        scan_timers[2] = t
        t.start()
        btn_scan_theta.label.set_text('Stop θ')
        fig.canvas.draw_idle()

ax_scan_rot = fig.add_axes([0.02, 0.352, 0.18, 0.030])
btn_scan_theta = widgets.Button(ax_scan_rot, 'Scan θ', color='#e8e8f8')
btn_scan_theta.label.set_fontsize(8)
btn_scan_theta.on_clicked(toggle_scan_theta)

# ── plages de scan : TextBoxes min/max + bouton Auto pour α ──────────────────
fig.text(0.110, 0.535, 'plages scan', fontsize=6, ha='center', color='gray')
fig.text(0.068, 0.519, 'min', fontsize=6, ha='center', color='gray')
fig.text(0.152, 0.519, 'max', fontsize=6, ha='center', color='gray')
fig.text(0.022, 0.497, 'θ', fontsize=6, ha='left', color='gray')
fig.text(0.022, 0.452, 'α', fontsize=6, ha='left', color='gray')
fig.text(0.022, 0.412, 'N', fontsize=6, ha='left', color='gray')

ax_sth_min = fig.add_axes([0.038, 0.484, 0.065, 0.028])
tb_sth_min = widgets.TextBox(ax_sth_min, '', initial=str(_defaults()['theta_min']))
tb_sth_min.text_disp.set_fontsize(7)

ax_sth_max = fig.add_axes([0.118, 0.484, 0.065, 0.028])
tb_sth_max = widgets.TextBox(ax_sth_max, '', initial=str(_defaults()['theta_max']))
tb_sth_max.text_disp.set_fontsize(7)

ax_sal_min = fig.add_axes([0.038, 0.444, 0.065, 0.028])
tb_sal_min = widgets.TextBox(ax_sal_min, '', initial=str(_defaults()['alpha_min']))
tb_sal_min.text_disp.set_fontsize(7)

ax_sal_max = fig.add_axes([0.118, 0.444, 0.065, 0.028])
tb_sal_max = widgets.TextBox(ax_sal_max, '', initial=str(_defaults()['alpha_max']))
tb_sal_max.text_disp.set_fontsize(7)

ax_sN_min = fig.add_axes([0.038, 0.404, 0.065, 0.028])
tb_sN_min = widgets.TextBox(ax_sN_min, '', initial=str(_defaults()['N_min']))
tb_sN_min.text_disp.set_fontsize(7)

ax_sN_max = fig.add_axes([0.118, 0.404, 0.065, 0.028])
tb_sN_max = widgets.TextBox(ax_sN_max, '', initial=str(_defaults()['N_max']))
tb_sN_max.text_disp.set_fontsize(7)


def _get_valid_alpha_range():
    key = (current_profile_name[0], _d_idx[0])
    if key not in _omega_2d_raw:
        return sl_alpha.valmin, sl_alpha.valmax
    om_2d  = _omega_2d_raw[key]
    a_pts  = _alpha_pts_raw[key]
    th_pts = _theta_pts_raw[key]
    j = int(np.argmin(np.abs(th_pts - np.radians(sl_theta.val))))
    valid = ~np.isnan(om_2d[:, j])
    if valid.any():
        return float(np.degrees(a_pts[valid].min())), float(np.degrees(a_pts[valid].max()))
    return sl_alpha.valmin, sl_alpha.valmax

def _get_valid_theta_range():
    key = (current_profile_name[0], _d_idx[0])
    if key not in _omega_2d_raw:
        return sl_theta.valmin, sl_theta.valmax
    om_2d  = _omega_2d_raw[key]
    a_pts  = _alpha_pts_raw[key]
    th_pts = _theta_pts_raw[key]
    i = int(np.argmin(np.abs(a_pts - np.radians(sl_alpha.val))))
    valid = ~np.isnan(om_2d[i, :])
    if valid.any():
        return float(np.degrees(th_pts[valid].min())), float(np.degrees(th_pts[valid].max()))
    return sl_theta.valmin, sl_theta.valmax

def _init_scan_ranges():
    mn_a, mx_a = _get_valid_alpha_range()
    _scan_alpha_min[0] = mn_a
    _scan_alpha_max[0] = mx_a
    tb_sal_min.set_val(f"{mn_a:.1f}")
    tb_sal_max.set_val(f"{mx_a:.1f}")
    mn_t, mx_t = _get_valid_theta_range()
    _scan_theta_min[0] = mn_t
    _scan_theta_max[0] = mx_t
    tb_sth_min.set_val(f"{int(round(mn_t))}")
    tb_sth_max.set_val(f"{int(round(mx_t))}")

def _on_sth_min(text):
    try:
        _scan_theta_min[0] = float(np.clip(float(text), sl_theta.valmin, sl_theta.valmax))
    except ValueError:
        pass

def _on_sth_max(text):
    try:
        _scan_theta_max[0] = float(np.clip(float(text), sl_theta.valmin, sl_theta.valmax))
    except ValueError:
        pass

def _on_sal_min(text):
    try:
        _scan_alpha_min[0] = float(np.clip(float(text), sl_alpha.valmin, sl_alpha.valmax))
    except ValueError:
        pass

def _on_sal_max(text):
    try:
        _scan_alpha_max[0] = float(np.clip(float(text), sl_alpha.valmin, sl_alpha.valmax))
    except ValueError:
        pass

def _on_sN_min(text):
    try:
        _scan_N_min[0] = int(np.clip(int(float(text)), sl_N.valmin, sl_N.valmax))
    except ValueError:
        pass

def _on_sN_max(text):
    try:
        _scan_N_max[0] = int(np.clip(int(float(text)), sl_N.valmin, sl_N.valmax))
    except ValueError:
        pass

tb_sth_min.on_submit(_on_sth_min)
tb_sth_max.on_submit(_on_sth_max)
tb_sal_min.on_submit(_on_sal_min)
tb_sal_max.on_submit(_on_sal_max)
tb_sN_min.on_submit(_on_sN_min)
tb_sN_max.on_submit(_on_sN_max)

# ── menu profil ───────────────────────────────────────────────────────────────
ax_radio = fig.add_axes([0.02, 0.84, 0.18, 0.11])
ax_radio.set_title('Profil', fontsize=8, pad=2)
radio = widgets.RadioButtons(ax_radio, list(PROFILES.keys()), activecolor='steelblue')
for lbl in radio.labels:
    lbl.set_fontsize(8)

def on_profile_change(label):
    current_derivees[0]    = PROFILES[label][0]
    current_n_boundary[0]  = PROFILES[label][1]
    current_profile_name[0] = label
    omega_interp[0]        = omega_interps[(label, _d_idx[0])]
    hist_alpha_deg.clear()
    hist_ampl.clear()
    set_default_params()

radio.on_clicked(on_profile_change)

# ── bouton reset ──────────────────────────────────────────────────────────────
ax_btn = fig.add_axes([0.02, 0.194, 0.18, 0.030])
btn_reset = widgets.Button(ax_btn, 'Reset', color='#e8e8f8')
btn_reset.label.set_fontsize(8)

def reset(event):
    hist_alpha_deg.clear()
    hist_ampl.clear()
    update(sl_alpha.val)

btn_reset.on_clicked(reset)

# ── boutons export PDF (un par panneau) ──────────────────────────────────────

from matplotlib.figure import Figure as _Figure

_EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rapport")
os.makedirs(_EXPORT_DIR, exist_ok=True)

def _ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _export_path(fname):
    return os.path.join(_EXPORT_DIR, fname)

def export_ray(event):
    rd = _export_data['ray']
    if rd is None:
        print("Rien à exporter (rayons)")
        return
    profil = current_profile_name[0]
    alpha_val = int(sl_alpha.val)
    fig_e = _Figure(figsize=(5, 6), constrained_layout=True)
    ax_e = fig_e.add_subplot(111)
    draw_static(ax_e)
    for traj, x0, y0, col, lw, ap in rd['trajs']:
        tracer_traj(ax_e, traj, x0, y0, rd['x_R'], rd['y_R'], col, lw, ap)
    draw_cone(ax_e, rd['alpha'], rd['x_E'], rd['y_E'], rd['x_R'], rd['y_R'])
    etitle = f"α = {np.degrees(rd['alpha']):.1f}°   f = {rd['f_GHz']:.2f} GHz"
    if rd['omega_approx']:
        etitle += "  ⚠ ω approx"
    ax_e.set_title(etitle)
    fname = f"ray_tracing_{profil}_a{alpha_val}deg_{_ts()}.pdf"
    fig_e.savefig(_export_path(fname), format='pdf', dpi=300)
    print(f"Exporté : {fname}")

def export_phase(event):
    pd = _export_data.get('phase')
    if pd is None:
        print("Rien à exporter (phase)")
        return
    profil = current_profile_name[0]
    fig_e = _Figure(figsize=(5, 3.5), constrained_layout=True)
    ax_e = fig_e.add_subplot(111)
    ax_e.plot(pd['s_det'], pd['phi_rel'], 'o-', color='darkorange', lw=1.5, ms=3)
    ax_e.axvline(0, color='k', lw=0.8, ls=':', alpha=0.5)
    ax_e.axhline(0, color='k', lw=0.5, ls=':', alpha=0.4)
    ax_e.set_xlabel("position le long du détecteur (cm)")
    ax_e.set_ylabel("Δφ (rad)")
    ax_e.set_xlim(-W_antenne*100*1.1, W_antenne*100*1.1)
    ax_e.set_title(f"Front de phase  ({profil},  α={pd['alpha_deg']:.1f}°,  N={pd['N']})")
    ax_e.grid(True, ls=':')
    fname = f"phase_{profil}_a{int(pd['alpha_deg'])}deg_N{pd['N']}_{_ts()}.pdf"
    fig_e.savefig(_export_path(fname), format='pdf', dpi=300)
    print(f"Exporté : {fname}")

def export_tau(event):
    td = _export_data.get('tau')
    if td is None:
        print("Rien à exporter (temps de vol)")
        return
    profil = current_profile_name[0]
    fig_e = _Figure(figsize=(5, 3.5), constrained_layout=True)
    ax_e = fig_e.add_subplot(111)
    ax_e.plot(td['s_det'], td['tau_ns'], 'o-', color='mediumseagreen', lw=1.5, ms=3)
    ax_e.axvline(0, color='k', lw=0.8, ls=':', alpha=0.5)
    ax_e.set_xlabel("position le long du détecteur (cm)")
    ax_e.set_ylabel("τ (ns)")
    ax_e.set_xlim(-W_antenne*100*1.1, W_antenne*100*1.1)
    ax_e.set_title(f"Temps de vol  ({profil},  α={td['alpha_deg']:.1f}°,  N={td['N']})")
    ax_e.grid(True, ls=':')
    fname = f"temps_vol_{profil}_a{int(td['alpha_deg'])}deg_N{td['N']}_{_ts()}.pdf"
    fig_e.savefig(_export_path(fname), format='pdf', dpi=300)
    print(f"Exporté : {fname}")

def export_pui(event):
    ad = _export_data.get('amp')
    if ad is None:
        print("Rien à exporter (puissance)")
        return
    profil = current_profile_name[0]
    fig_e = _Figure(figsize=(5, 3.5), constrained_layout=True)
    ax_e = fig_e.add_subplot(111)
    if len(ad['alpha_hist']) > 1:
        order = np.argsort(ad['alpha_hist'])
        ax_e.plot(np.array(ad['alpha_hist'])[order],
                  np.array(ad['ampl_hist'])[order], 'b.-')
    ax_e.axhline(1.0, color='k', ls='--', alpha=0.4, label="référence")
    if ad['ampl_cur'] is not None:
        ax_e.scatter([ad['alpha_cur']], [ad['ampl_cur']], color='red', zorder=5, s=60)
    ax_e.set_xlabel("α (°)")
    ax_e.set_ylabel("Puissance normalisée au récepteur")
    ax_e.set_title(f"Puissance normalisée vs α  ({profil})")
    ax_e.legend()
    ax_e.grid(True, ls=':')
    fname = f"puissance_{profil}_{_ts()}.pdf"
    fig_e.savefig(_export_path(fname), format='pdf', dpi=300)
    print(f"Exporté : {fname}")

ax_btn_ray = fig.add_axes([0.02, 0.156, 0.18, 0.030])
ax_btn_pha = fig.add_axes([0.02, 0.118, 0.18, 0.030])
ax_btn_tau = fig.add_axes([0.02, 0.080, 0.18, 0.030])
ax_btn_pui = fig.add_axes([0.02, 0.042, 0.18, 0.030])
btn_ray = widgets.Button(ax_btn_ray, 'Export Rayons',  color='#e8e8f8')
btn_pha = widgets.Button(ax_btn_pha, 'Export Phase',   color='#e8e8f8')
btn_tau = widgets.Button(ax_btn_tau, 'Export τ',       color='#e8e8f8')
btn_pui = widgets.Button(ax_btn_pui, 'Export Puiss.',  color='#e8e8f8')
for _b in (btn_ray, btn_pha, btn_tau, btn_pui):
    _b.label.set_fontsize(8)
for b in (btn_ray, btn_pha, btn_tau, btn_pui):
    b.label.set_fontsize(7)
btn_ray.on_clicked(export_ray)
btn_pha.on_clicked(export_phase)
btn_tau.on_clicked(export_tau)
btn_pui.on_clicked(export_pui)

draw_static(ax_ray)
update(sl_alpha.val)
plt.show()
