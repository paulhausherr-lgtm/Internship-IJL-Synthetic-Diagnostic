import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

choix_bruit = input("bruit sur tau ? [0] aucun  [1] pic  [2] aléatoire : ").strip()
amplitude_pic, niveau_bruit = 0.0, 0.0
if choix_bruit == "1":
    amplitude_pic = float(input("amplitude pic (ex: 0.1 = +10%) : "))
elif choix_bruit == "2":
    niveau_bruit = float(input("niveau bruit relatif σ (ex: 0.02 = 2%) : "))

c = 3e8
e = 1.6e-19
me = 9.11e-31
eps0 = 8.85e-12

a = 0.2  # rayon plasma (m)
d = 0.4  # distance émetteur-récepteur (m)
n0 = 1e18  # densité électronique centrale (m^-3)

omega_p0 = np.sqrt((n0 * e**2) / (me * eps0))  # pulsation plasma centrale (rad/s)

# profil gigogne : N couches paraboliques de rayon uniforme entre a et a_min
# dn de chaque couche uniforme tel que la densité centrale totale vaut n0
N_gigogne = 5
a_min_gigogne = 0.02  # rayon de la couche la plus interne (m)
radii_gigogne = np.linspace(a, a_min_gigogne, N_gigogne)
profil_gigogne = [{'dn': n0 / N_gigogne, 'a': float(ai)} for ai in radii_gigogne]


alpha_c_deg = np.degrees(np.arctan(2 * a / d))

phi_geom = np.arctan(d / (2 * a))  # angle émetteur-centre plasma par rapport à l'horizontale
R_geom   = np.sqrt(a**2 + (d / 2)**2)  # distance émetteur-centre plasma (m)

# bornes du scan en paramètre d'impact (on évite les rayons tangents au centre, singuliers)
p_min = R_geom * np.cos(np.radians(alpha_c_deg - 0.2) + phi_geom)  # p proche du centre
p_max = R_geom * np.cos(np.radians(10) + phi_geom)  # p à alpha=10° (bord)
t = np.linspace(0, 1, 50)**0.5        # sub-linéaire : dense près de p_min (centre)
p_scan = p_max - (p_max - p_min) * t  # → plus de points d'inversion proches du centre

alphas_valides = np.arccos(p_scan / R_geom) - phi_geom  # angles correspondants (rad)

# calcule le point d'entrée (xe, ye)
def calc_point_entree(alpha):
    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)
    cot_a = cos_a / sin_a
    sqrt_term = np.sqrt(a**2 + a * d * cot_a - (d**2) / 4)
    y_e = (sin_a**2) * (a + (d / 2) * cot_a - sqrt_term)
    x_e = y_e * cot_a - (d / 2)
    return x_e, y_e

# gradient du système hamiltonien optique
def derivees_rk4_conique(Y, omega_val):
    x, y, px, py = Y
    r_c = np.sqrt(x**2 + (y - a)**2)  # distance au centre du plasma (0, a)
    X_val = omega_p0**2 / omega_val**2

    if r_c < a:
        r_reg = np.sqrt(x**2 + (y - a)**2 + 1e-10)  # régularisation de la pointe du cône
        # gradient de l'indice conique n^2 = 1 - X*(1 - r/a)
        dpx = (X_val / (2 * a)) * (x / r_reg)
        dpy = (X_val / (2 * a)) * ((y - a) / r_reg)
    else:
        dpx = 0.0  # dans le vide
        dpy = 0.0

    return np.array([px, py, dpx, dpy])

def derivees_rk4_gauss(Y, omega_val, w=0.07):
    x, y, px, py = Y
    X_val = omega_p0**2 / omega_val**2
    r2 = x**2 + (y - a)**2

    # gradient de n² = 1 - X*exp(-r²/w²)
    exp_term = X_val * np.exp(-r2 / w**2)
    dpx = (exp_term / w**2) * x
    dpy = (exp_term / w**2) * (y - a)

    return np.array([px, py, dpx, dpy])

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


# un pas de RK4
def rk4_step(Y, dsigma, omega_val):
    k1 = derivees_rk4_gauss(Y, omega_val)
    k2 = derivees_rk4_gauss(Y + 0.5 * dsigma * k1, omega_val)
    k3 = derivees_rk4_gauss(Y + 0.5 * dsigma * k2, omega_val)
    k4 = derivees_rk4_gauss(Y + dsigma * k3, omega_val)
    #k1 = derivees_rk4_conique(Y, omega_val)
    #k2 = derivees_rk4_conique(Y + 0.5 * dsigma * k1, omega_val)
    #k3 = derivees_rk4_gauss(Y + 0.5 * dsigma * k2, omega_val)
    #k4 = derivees_rk4_gauss(Y + dsigma * k3, omega_val)
    
    return Y + (dsigma / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# lance un rayon et retourne son point d'impact sur y=0 et la trajectoire
def simuler_rayon(alpha, omega_val, dsigma=1e-3):
    x_e, y_e = calc_point_entree(alpha)

    # condition initiale : n(a)=1 exactement pour gigogne/conique/parabolique (densité nulle au bord)
    Y = np.array([x_e, y_e, np.cos(alpha), np.sin(alpha)])
    # profil gaussien : n(a) ≠ 1, il faut corriger l'impulsion
    #wp2_a = omega_p0**2 * np.exp(-a**2 / 0.07**2)
    #n_a = np.sqrt(max(1.0 - wp2_a / omega_val**2, 1e-12))
    #Y = np.array([x_e, y_e, n_a * np.cos(alpha), n_a * np.sin(alpha)])
    Y[:2] += 1e-6 * np.array([np.cos(alpha), np.sin(alpha)])  # micro-poussée

    trajectoire = [Y.copy()]

    while True:
        Y = rk4_step(Y, dsigma, omega_val)
        trajectoire.append(Y.copy())

        x, y, px, py = Y
        r_c = np.sqrt(x**2 + (y - a)**2)

        if r_c >= a:  # sortie du plasma
            break
        if len(trajectoire) > 5000:  # sécurité : trop d'itérations
            return None, None, None

    x_out, y_out, px_out, py_out = Y
    if py_out >= 0:
        return None, None, None

    # projection balistique dans le vide jusqu'au récepteur (y = 0)
    x_final = x_out - y_out * (px_out / py_out)
    # temps de vol aller-retour : tau = sigma_plasma / c  (ds = n*dsigma, v_g = c*n)
    tau_rt = (len(trajectoire) - 1) * dsigma / c
    return x_final, np.array(trajectoire), tau_rt

# ajuste omega pour que le rayon frappe exactement x = d/2 (shooting)
# les solutions sont juste au-dessus du cutoff -> on scanne omega au plus près de omega_p0
# et on détecte le changement de signe de (x_final - d/2) parmi les rayons qui atteignent y=0
def trouver_omega(alpha):
    def x_err(omega_test):
        x_fin = simuler_rayon(alpha, omega_test)[0]
        return None if x_fin is None else x_fin - (d / 2)

    omegas = np.linspace(0.5 * omega_p0, 1.3 * omega_p0, 40)
    errs = [x_err(om) for om in omegas]

    for j in range(len(omegas) - 1):
        e0, e1 = errs[j], errs[j + 1]
        if e0 is not None and e1 is not None and e0 * e1 < 0:  # changement de signe (rayons valides)
            try:
                return brentq(lambda om: simuler_rayon(alpha, om)[0] - d/2, omegas[j], omegas[j + 1])
            except (ValueError, TypeError):
                continue  # racine sur un None -> on passe au changement de signe suivant
    return np.nan  # aucun changement de signe -> pas de solution

#main
donnees_export = []

plt.figure(figsize=(10, 8))
theta_cercle = np.linspace(0, 2*np.pi, 200)
plt.plot(a * np.cos(theta_cercle), a + a * np.sin(theta_cercle), 'k-', alpha=0.5, label="bord plasma")
for couche in profil_gigogne:
    ai = couche['a']
    plt.plot(ai * np.cos(theta_cercle), a + ai * np.sin(theta_cercle),
             '--', alpha=0.4, label=f"couche a={ai*100:.0f} cm")
plt.plot([-d/2, d/2], [0, 0], 'ko', markersize=8, label="antennes E/R")

print(f"{len(alphas_valides)} rayons...")

for i, alpha in enumerate(alphas_valides):
    alpha_deg = np.degrees(alpha)

    omega_opti = trouver_omega(alpha)
    if np.isnan(omega_opti):
        print(f"  [!] échec convergence pour alpha = {alpha_deg:.2f}°")
        continue

    print(f"  rayon {i+1} | alpha = {alpha_deg:.2f}° | f = {(omega_opti/(2*np.pi))/1e9:.2f} GHz")

    # simulation finale pour la trajectoire complète et le temps de vol
    x_final, traj, tau_rt = simuler_rayon(alpha, omega_opti)

    # segment d'entrée dans le vide (émetteur -> point d'entrée)
    xe0, ye0 = traj[0, 0], traj[0, 1]
    t_in = np.linspace(-ye0 / np.sin(alpha), 0, 10)
    x_in = xe0 + np.cos(alpha) * t_in
    y_in = ye0 + np.sin(alpha) * t_in

    # segment balistique de sortie dans le vide (point de sortie -> récepteur)
    x_out, y_out, px_out, py_out = traj[-1]
    t_vide = np.linspace(0, -y_out/py_out, 10)
    x_vide = x_out + px_out * t_vide
    y_vide = y_out + py_out * t_vide

    plt.plot(np.concatenate([x_in, traj[:, 0], x_vide]),
             np.concatenate([y_in, traj[:, 1], y_vide]),
             '-', linewidth=0.8)
    donnees_export.append([alpha_deg, omega_opti, tau_rt / 2])

if donnees_export:
    donnees_bruit = [row[:] for row in donnees_export]
    if choix_bruit == "1":
        idx_pic = len(donnees_bruit) // 2
        donnees_bruit[idx_pic][2] *= (1 + amplitude_pic)
        print(f"pic ajouté : rayon {idx_pic+1}, tau +{amplitude_pic*100:.0f}%  (alpha={donnees_bruit[idx_pic][0]:.2f}°)")
    elif choix_bruit == "2":
        for row in donnees_bruit:
            row[2] *= (1 + niveau_bruit * np.random.randn())
        print(f"bruit gaussien σ={niveau_bruit*100:.1f}% appliqué sur {len(donnees_bruit)} points")
    else:
        print("aucun bruit appliqué")
    np.savetxt("donnees_inversion_gen.txt", donnees_bruit, fmt=["%.6f", "%.12e", "%.12e"],
               header="alpha_deg \t omega_rad_s \t tau_plasma_s")
else:
    print("aucun rayon convergé — pas d'export")

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Ray tracing RK4")
plt.xlim(-0.12, 0.12)
plt.ylim(-0.01, 0.22)
plt.grid(True, linestyle=':')
plt.legend()
plt.show()
