import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib.pyplot as plt

c = 3e8          # vitesse de la lumière (m/s)
e = 1.602176634e-19  # charge élémentaire (C)
me = 9.1093837e-31    # masse de l'électron (kg)
eps0 = 8.854187817e-12  # permittivité du vide (F/m)

a = 0.2   # rayon plasma (m)
d = 0.4  # distance émetteur-récepteur (m)
n0 = 1e18   # densité centrale théorique (m^-3)

omega_p0 = np.sqrt(n0 * e**2 / (me * eps0))  # pulsation plasma centrale (rad/s)

# profil gigogne — doit rester synchronisé avec general_ray_tracing.py
N_gigogne = 5
a_min_gigogne = 0.02
radii_gigogne = np.linspace(a, a_min_gigogne, N_gigogne)
profil_gigogne = [{'dn': n0 / N_gigogne, 'a': float(ai)} for ai in radii_gigogne]

def calc_point_entree(alpha):
    # coordonnées du point d'entrée du rayon sur le bord du plasma
    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)
    cot_a = cos_a / sin_a
    sqrt_term = np.sqrt(a**2 + a * d * cot_a - (d**2) / 4)
    y_e = (sin_a**2) * (a + (d / 2) * cot_a - sqrt_term)
    x_e = y_e * cot_a - (d / 2)
    return x_e, y_e

def convertir_donnees(alpha_deg, omega):
    # convertit (alpha, tau_plasma, omega) -> (p, omega, Theta) purement depuis la géométrie de alpha
    p_list = []
    Theta_list = []
    omega_list = []
    for i in range(len(alpha_deg)):
        alpha = np.radians(alpha_deg[i])
        x_e, y_e = calc_point_entree(alpha)
        p_val = np.abs((d / 2) * np.sin(alpha) - a * np.cos(alpha))  # invariant de Bouguer
        num = -(x_e**2) + (y_e - a)**2
        den =  x_e**2  + (y_e - a)**2
        Theta_val = np.arccos(np.clip(num / den, -1.0, 1.0))    # angle balayé dans le plasma
        p_list.append(p_val)
        Theta_list.append(Theta_val)
        omega_list.append(omega[i])
    return p_list, omega_list, Theta_list

def convertir_donnees_2(alpha_deg, omega, tau, wp2_bord=0.0):
    # convertit (alpha, omega, tau) -> (p, omega, tau) en calculant p depuis la géométrie de alpha
    # wp2_bord : wp² au bord r=a — corrige p_vac → p_réel = n(a)·p_vac (0.0 pour profil à bord nul)
    p_list = []; omega_list = []; tau_list = []
    for i in range(len(alpha_deg)):
        alpha = np.radians(alpha_deg[i])
        p_vac = np.abs((d / 2) * np.sin(alpha) - a * np.cos(alpha))
        n_a = np.sqrt(max(1.0 - wp2_bord / omega[i]**2, 1e-12))  # n(a) dans le milieu
        p_val = n_a * p_vac        # invariant de Bouguer réel = n(a)·p_vac
        #p_val = p_vac             # invariant vide (correct si n(a)=1 : profil conique/parabolique)
        p_list.append(p_val); omega_list.append(omega[i]); tau_list.append(2 * tau[i])
    return p_list, omega_list, tau_list

def invert_onion(p, omega, Theta, n_quad=5):
    """
    Reconstruit le profil de densité électronique n_e(r) par la méthode de pelure d'oignon.

    Pour chaque rayon mesuré produisant les paramètres p_i, omega_i et Theta_i, on cherche le wp^2_i local qui reproduit
    l'angle Theta_i observé. Pour se faire on optimise à partir du profil déjà reconstruit. 
    On va de l'extérieur (r=a) vers le centre.

    Paramètres:
    p: liste des paramètres d'impact (m)
    omega: liste des pulsations (rad/s)
    Theta: liste des angles balayés dans le plasma (rad)
    n_quad: nombre de points pour l'intégration numérique (quad)

    Retourne
    r_rebroussement : liste des rayons de rebroussement (m), triés par r croissant
    n_e: liste des densités électroniques associées (m^-3)
    """

    # on commencer par trier les données dans l'ordre décroissant (bord --> centre) 
    ordre = sorted(range(len(p)), key=lambda i: -p[i])
    p_trie  = [p[i] for i in ordre]  # paramètres d'impact triés
    omega_trie = [omega[i] for i in ordre]  # pulsations associées
    Theta_trie = [Theta[i] for i in ordre]  # angles balayés associés

    # condition au bord n_e(r=a) = 0: ancre l'interpolation et évite d'extrapoler dans le vide
    couches_r   = [a]
    couches_wp2 = [0.0]

    n_total = len(p_trie)  # nombre total couches

    # début de la boucle de peeling
    for i_couche in range(n_total):
        p_i = p_trie[i_couche]
        wi = omega_trie[i_couche]
        Th_obs = Theta_trie[i_couche]
        print(f"couche {i_couche + 1}/{n_total}", end="\r", flush=True)

        
        # fonction qui permet de générer un profil de wp^2 à partir des couches déjà déterminées par interpolation 
        # on génère aussi une valeur de wp^2_i au point d'entrée
        def wp2_profil(r, extra_r, extra_wp2):
            r_tries = couches_r + [extra_r]
            wp2_tries = couches_wp2 + [extra_wp2]
            ordre = np.argsort(r_tries)
            r_tries  = [r_tries[i] for i in ordre]
            wp2_tries = [wp2_tries[i] for i in ordre]
            return np.interp(r, r_tries, wp2_tries, left=wp2_tries[0], right=0.0)

        #fonction qui pour une valeur de wp2_i donnée, calcule l'angle de rebroussement prédit par le modèle
        def modele_Theta(wp2_i):
            r_re = p_i / np.sqrt(max(1.0 - wp2_i / wi**2, 1e-12))  # rayon de rebroussement estimé
            r_re = min(r_re, a - 1e-6)

            #changement de variable pour éviter les problèmes de singularité à p_i
            def zeta2(r):
                return r**2 * (1.0 - wp2_profil(r, r_re, wp2_i) / wi**2) 

            u_max = np.sqrt(max(a**2 - p_i**2, 0.0))  # borne supérieure 

            def integ(u):
                z  = np.sqrt(p_i**2 + u**2)  # z = N(r)·r 
                
                #gestion erreur pour la singularité à p_i: on trouve le r qui correspond à ce z
                try:
                    r = brentq(lambda r: zeta2(r) - z**2, r_re, a)
                except ValueError:
                    r = r_re

                wp2 = wp2_profil(r, r_re, wp2_i)
                dr = 1e-7 * a
                wp2p = (wp2_profil(r + dr, r_re, wp2_i) - wp2_profil(r - dr, r_re, wp2_i)) / (2 * dr)  # dérivée de wp^2 par différences finies
                dz2dr = 2 * r - (2 * r * wp2 + r**2 * wp2p) / wi**2  # dérivée de zeta^2 par rapport à r
                if dz2dr <= 0:
                    dz2dr = 2 * r
                return 4 * p_i / (r * dz2dr) #forme finale de l'intégrande

            Theta_pred, err_quad = quad(integ, 0.0, u_max, limit=n_quad) #intégration numérique
            return Theta_pred, r_re

        hi = wi**2 * (1.0 - (p_i / a)**2) * 0.9999  # borne supérieure pour wp2_i(mesuré)
        
        wp2_i = brentq(lambda x: modele_Theta(x)[0] - Th_obs, 0.0, hi)  # wp^2 local qui reproduit Theta observé
    
    
        _, r_re = modele_Theta(wp2_i)
       #append au listes finales
        couches_r.append(r_re) 
        couches_wp2.append(wp2_i)

    print(f"couche {n_total}/{n_total} — terminé")

    n_e = []
    for wp2 in couches_wp2:
        n_e.append(max(wp2 * eps0 * me / e**2, 0.0))  # conversion wp^2 ---> n_e(r)

    # trier par r_re croissant au cas il n'est pas strictement décroissant d'une couche à l'autre.
    #permettra de tester d'autre profile de densité par la suite
    ordre_final = np.argsort(couches_r)
    r_rebroussement = [couches_r[i] for i in ordre_final]
    n_e  = [n_e[i]  for i in ordre_final]

    return r_rebroussement, n_e

def invert_onion_tau(p, omega, tau, n_quad=20, wp2_bord=0.0):
    """
    Identique à invert_onion mais utilise le temps de vol tau comme observable.
    wp2_bord : valeur de wp² au bord r=a (0.0 pour profil à bord nul, sinon passer la vraie valeur)
    """
    ordre = sorted(range(len(p)), key=lambda i: -p[i])
    p_trie = [p[i] for i in ordre]
    omega_trie = [omega[i] for i in ordre]
    tau_trie   = [tau[i]   for i in ordre]

    couches_r   = [a]
    couches_wp2 = [wp2_bord]
    residus  = []

    n_total = len(p_trie)

    for i_couche in range(n_total):
        p_i = p_trie[i_couche]
        wi = omega_trie[i_couche]
        tau_obs = tau_trie[i_couche]
        print(f"couche {i_couche + 1}/{n_total}", end="\r", flush=True)

        def modele_tau(wp2_i):
            r_re = p_i / np.sqrt(max(1.0 - wp2_i / wi**2, 1e-12))
            r_re = min(r_re, a - 1e-6)

            # noeuds = couches connues + candidat (r_re, wp2_i), triés par r
            r_all   = couches_r + [r_re]
            wp2_all = couches_wp2 + [wp2_i]
            ordre_n = np.argsort(r_all)
            r_arr   = np.array([r_all[j]   for j in ordre_n])
            wp2_arr = np.array([wp2_all[j] for j in ordre_n])
            
            def wp2_val(r):
                return float(np.interp(r, r_arr, wp2_arr, left=wp2_arr[0], right=0.0))
            def wp2_der(r):
                if r <= r_arr[0] or r >= r_arr[-1]:
                    return 0.0
                k = min(max(int(np.searchsorted(r_arr, r)) - 1, 0), len(r_arr) - 2)
                return (wp2_arr[k + 1] - wp2_arr[k]) / (r_arr[k + 1] - r_arr[k])

            def zeta2(r):
                return r**2 * (1.0 - wp2_val(r) / wi**2)

            # n²(a) peut être < 1 si le profil ne s'annule pas au bord (ex. gaussienne)
            u_max = np.sqrt(max(a**2 * (1.0 - wp2_val(a) / wi**2) - p_i**2, 0.0))

            def integ(u):
                z2 = p_i**2 + u**2
                try:
                    r = brentq(lambda r: zeta2(r) - z2, r_re, a, xtol=1e-7)
                except ValueError:
                    r = r_re
                wp2  = wp2_val(r)
                wp2p = wp2_der(r)   # pente linéaire par morceaux — exacte pour l'interpolant, O(log n)
                dz2dr = 2 * r - (2 * r * wp2 + r**2 * wp2p) / wi**2
                if dz2dr <= 0:
                    dz2dr = 2 * r
                return (4 / c) * r / dz2dr

            tau_pred, _ = quad(integ, 0.0, u_max, limit=n_quad)
            return tau_pred, r_re

        hi = wi**2 * (1.0 - (p_i / a)**2) * 0.9999

        # modele_tau(wp2_i) est non-monotone.
        # donc brentq sur [0,hi] ne satisfait pas le théorème des valeurs intermédiaires. 
        # Ducoup on scan la zone, et on détecte tous les changements de signe pour séléctionner la plus petite racine
        #  physique et valider le changement de signe. 

        wp2_prev = couches_wp2[-1]
        xs = np.linspace(0.0, hi, 100)
        f_scan = np.array([modele_tau(x)[0] - tau_obs for x in xs])

        racines = []
        for j in range(len(xs) - 1):
            if f_scan[j] == 0.0:
                racines.append(xs[j])
            if f_scan[j] * f_scan[j + 1] < 0.0: # changement de signe
                racines.append(brentq(lambda x: modele_tau(x)[0] - tau_obs,
                                      xs[j], xs[j + 1]))

        # branche physique : plus petite racine >= wp2_prev (densité monotone croissante)
        candidates = [r for r in racines if r >= wp2_prev - 1e-6]
        if candidates:
            wp2_i = min(candidates)
            converge = True
        elif racines: # pas de branche physique -> racine la plus profonde
            wp2_i = max(racines)
            converge = True
        else:
            wp2_i = 0.0
            converge = False

        tau_pred, r_re = modele_tau(wp2_i)
        residus.append((r_re, tau_pred - tau_obs, converge))
        if wp2_i != 0.0:   # on laisse tomber les couches dégénérées (bracket échoué)
            couches_r.append(r_re)
            couches_wp2.append(wp2_i)

    print(f"couche {n_total}/{n_total} — terminé")

    ordre_final = np.argsort(couches_r)
    r_tri   = np.array([couches_r[i] for i in ordre_final])
    wp2_tri = np.array([couches_wp2[i] for i in ordre_final])

    n_e = [max(w * eps0 * me / e**2, 0.0) for w in wp2_tri]
    r_rebroussement = list(r_tri)
    return r_rebroussement, n_e, residus



# donnees = np.loadtxt("donnees_inversion_5.txt", skiprows=1)   # 2 colonnes : alpha, omega
# alpha_deg = donnees[:, 0]; omega = donnees[:, 1]
# p, omega, Theta = convertir_donnees(alpha_deg, omega)
# r_rebroussement_onion, n_e_onion = invert_onion(p, omega, Theta)
# residus = []   # invert_onion ne retourne pas de résidus

donnees = np.loadtxt("donnees_inversion_6.txt", skiprows=1)
alpha_deg = donnees[:, 0]  
omega  = donnees[:, 1]  
tau   = donnees[:, 2]  

w_gauss = 0.07  # largeur gaussienne (m)

# condition au bord selon le profil utilisé pour générer les données :
wp2_bord = 0.0                                         # profil gigogne / conique / parabolique (n(a)=1)
#wp2_bord = omega_p0**2 * np.exp(-a**2 / w_gauss**2)  # profil gaussien (n(a) ≠ 1)

p, omega, tau = convertir_donnees_2(alpha_deg, omega, tau, wp2_bord=wp2_bord)

r_rebroussement_onion, n_e_onion, residus = invert_onion_tau(p, omega, tau, wp2_bord=wp2_bord)

print(f"\na = {a*100:.1f} cm  |  d = {d*100:.1f} cm")
echecs = [(i, r[0]) for i, r in enumerate(residus) if not r[2]]
print(f"brentq échoué : {len(echecs)}/{len(residus)} couches")
if echecs:
    for i, r_re in echecs:
        print(f"  couche {i+1:3d}  r_re = {r_re*100:.2f} cm")

succes = [(i, r[0]) for i, r in enumerate(residus) if r[2]]
if succes:
    i_last, r_last = succes[-1]
    print(f"dernière couche convergeé : couche {i_last+1}  r_re = {r_last*100:.2f} cm")

r_theo = np.linspace(0, a, 200)
n_e_theo = np.zeros_like(r_theo)                          # profil gigogne : somme des paraboles
for couche in profil_gigogne:
    ai, dn_i = couche['a'], couche['dn']
    mask = r_theo < ai
    n_e_theo[mask] += dn_i * (1 - r_theo[mask]**2 / ai**2)
#n_e_theo = n0 * np.exp(-r_theo**2 / w_gauss**2)  # profil gaussien de référence
n_e_theo = n0 * (1 - r_theo**2 / a**2)           # profil parabolique de référence
#n_e_theo = n0 * (1 - r_theo / a)                 # profil conique de référence

r_filtre, ne_filtre, eps_filtre = [], [], []
for r, ne in zip(r_rebroussement_onion, n_e_onion):
    ne_ref = float(np.interp(r, r_theo, n_e_theo))
    if ne_ref == 0:
        continue
    eps = abs(ne - ne_ref) / ne_ref
    if eps <= 0.10:
        r_filtre.append(r)
        ne_filtre.append(ne)
        eps_filtre.append(eps)

if eps_filtre:
    print(f"erreur relative moyenne : {np.mean(eps_filtre)*100:.2f}%")
    print(f"erreur relative max     : {np.max(eps_filtre)*100:.2f}%  (r = {r_filtre[np.argmax(eps_filtre)]*100:.2f} cm)")

plt.figure()
plt.plot(r_theo, n_e_theo, 'k--', label="profil théorique")
plt.plot(r_filtre, ne_filtre, 'ro-', ms=4, label="inversion")
plt.xlabel("rayon $r$ (m)")
plt.ylabel("$n_e$ (m$^{-3}$)")
plt.grid(True)
plt.legend()

# graphe des résidus et erreur cumulée
r_res = [x[0] for x in residus]
res  = [x[1] for x in residus]
conv  = [x[2] for x in residus]
cumul = np.cumsum(res)

# plt.figure()
# #plt.subplot(2, 1, 1)
# colors = ['b' if c else 'r' for c in conv]
# plt.bar(range(len(res)), [abs(r)*1e9 for r in res], color=colors)
# plt.ylabel("|résidu| (ns)")
# plt.title("Résidu par couche  (bleu=convergé, rouge=échec brentq)")
# plt.xticks(range(len(res)), [f"{r*100:.1f}" for r in r_res], rotation=90, fontsize=7)
# plt.xlabel("r_re (cm)")
# plt.grid(True)


plt.show()
