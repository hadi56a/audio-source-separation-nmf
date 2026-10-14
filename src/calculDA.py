# Algo de GS
import numpy as np
import scipy.linalg
from sklearn.decomposition import NMF
from matplotlib import pyplot as plt

'''
-- Fonction calculDA_viaNMF() --

Brief:
Applique une factorisation NMF (Non-negative Matrix Factorization) sur un spectrogramme
ou toute matrice non-négative M. La fonction utilise l'initialisation NNDSVD pour 
améliorer la convergence et obtient la matrice des sources (D) et la matrice d'activation des sources (A).

Retourne:
- D : matrice des spectres caractéristiques (m * k)
- A : matrice d'activation des sources dans le temps (k * n)

Paramètres:
- M : matrice d'entrée non-négative (ex: spectrogramme)
- e : tolérance pour la convergence de l'algorithme NMF (par défaut 1e-8)
- nCompos : nombre de composantes (sources) à extraire (par défaut 5)
'''
def calculDA_viaNMF(M, e=1e-8, nCompos=5):

    model = NMF(n_components=nCompos, init='nndsvd', random_state=0, tol=e)
    D = model.fit_transform(M)
    A = model.components_

    return D, A






'''
-- Fonction calculDA_viaMoindresCarres() --

Brief:
Applique une factorisation approximative de la matrice non-négative M sous la forme
M ≈ D @ A en utilisant une méthode de moindres carrés alternés (ALS) [méthode analytique != additive ou multiplicative]. À chaque
itération, A est mise à jour en fonction de D, puis D est mise à jour en fonction de A.
Les deux matrices sont forcées à rester non-négatives après chaque mise à jour.

Retourne:
- D : matrice des bases (motifs fréquentiels) de taille (m * k)
- A : matrice d’activation des composantes de taille (k * n)
- iter : nombre d’itérations effectuées avant convergence
- normesObtenues : Liste des normes obtenues au fil des itérations

Paramètres:
- M : matrice d’entrée non-négative (ex: spectrogramme)
- nCompos : nombre de composantes à extraire
- eps : tolérance de convergence basée sur la variation entre itérations
- max_iter : nombre maximum d’itérations

Précision:
Ici, nous avons préféré utiliser la méthode analytique au lieu des méthodes additive et multiplicative. Cela s'explique par le fait
que c'est la solution la plus adaptée à notre problème. Nous cherchons à converger vite et trouver le minimum et non pas à explorer.
'''
def calculDA_viaMoindresCarres(M, nCompos, eps=1e-8, max_iter=250):
    
    freq, temps = M.shape

    # Pour que D*A soit proche de M dès le début et accélere la convergence
    moy = np.sqrt(M.mean() / nCompos)

    D = np.random.rand(freq, nCompos) * moy
    A = np.random.rand(nCompos, temps) * moy

    # Pour garder la contrainte de >=0
    D = np.maximum(D, 1e-10)
    A = np.maximum(A, 1e-10)

    # On initialise conv à Infini pour être certain d'entrer dans la boucle
    conv = float(np.inf)
    iter = 0

    normesObtenues = []

    while conv > eps and iter < max_iter:
        Dold = D
        Aold = A

        Dt_D = D.T@D
        Dt_M = D.T@M
        A = np.linalg.pinv(Dt_D)@Dt_M # Assurer la stabilité de l'inversion
        A = np.maximum(A, 1e-10)

        M_At = M@A.T
        A_At = A@A.T
        D = M_At@np.linalg.pinv(A_At) # Assurer la stabilité de l'inversion
        D = np.maximum(D, 1e-10)        

        # Calcul du résultat de la convergence via la norme de Frobenius
        conv = scipy.linalg.norm(Dold@Aold - D@A)
        normesObtenues.append(conv)

        iter+=1

    return D, A, iter, normesObtenues








'''
-- Fonction calculDA_viaKullbackLeibler() --

Brief:
Applique une factorisation approximative de la matrice non-négative M sous la forme
M ≈ D @ A en utilisant une méthode de divergence de Kullback Leibler (KLD) pour optimiser
la convergence. À chaque itération, A est mise à jour en fonction de D, puis D est 
mise à jour en fonction de A. Les deux matrices sont forcées à rester non-négatives après chaque mise à jour.

Retourne:
- D : matrice des bases (motifs fréquentiels) de taille (m * k)
- A : matrice d’activation des composantes de taille (k * n)
- iter : nombre d’itérations effectuées avant convergence
- normesObtenues : Liste des normes obtenues au fil des itérations

Paramètres:
- M : matrice d’entrée non-négative (ex: spectrogramme)
- nCompos : nombre de composantes à extraire
- eps : tolérance de convergence basée sur la variation entre itérations
- max_iter : nombre maximum d’itérations
'''
def calculDA_viaKullbackLeibler(M, nCompos, eps=1e-8, max_iter=250):
    
    freq, temps = M.shape

    # Pour que D*A soit proche de M dès le début et accélere la convergence
    moy = np.sqrt(M.mean() / nCompos)

    D = np.random.rand(freq, nCompos) * moy
    A = np.random.rand(nCompos, temps) * moy

    # Pour garder la contrainte de >=0
    D = np.maximum(D, 1e-10)
    A = np.maximum(A, 1e-10)

    # On initialise conv à Infini pour être certain d'entrer dans la boucle
    conv = float(np.inf)
    iter = 0

    normesObtenues = []

    while conv > eps and iter < max_iter:
        Dold = D
        Aold = A

        # Formule de calcul de A selon KL Divergence
        # Il faut calculer la dérivée de (xlog(yx​)−x+y) par rapport à y => 1 - x/y

        diviseur_DA = D@A + 1e-10
        # D.T pour avoir la bonne taille
        numeA = D.T@(M/diviseur_DA)
        denumA = np.sum(D, axis=0)[:, None]     # <=> D.T@np.ones_like(M) mais moins couteux (optimisation trouvée par IA)
        A = A * (numeA/denumA)

        A = np.maximum(A, 1e-10)

        # Formule de calcul de D selon KL Divergence
        diviseur_DA = D@A + 1e-10
        numeD = (M/diviseur_DA)@A.T
        denumD = np.sum(A, axis=1)[None, :]    # <=> np.ones_like(M)@A.T mais moins couteux (optimisation trouvée par IA)
        D = D * (numeD/denumD)

        D = np.maximum(D, 1e-10)        

        conv = scipy.linalg.norm(Dold@Aold - D@A)
        normesObtenues.append(conv)

        iter+=1

    return D, A, iter, normesObtenues





'''
-- Fonction calculDA_viaItakuraSaito() --

Brief:
Applique une factorisation approximative de la matrice non-négative M sous la forme
M ≈ D @ A en utilisant une méthode de divergence de Itakura Saito (ISD) pour optimiser
la convergence. À chaque itération, A est mise à jour en fonction de D, puis D est 
mise à jour en fonction de A. Les deux matrices sont forcées à rester non-négatives après chaque mise à jour.

Retourne:
- D : matrice des bases (motifs fréquentiels) de taille (m * k)
- A : matrice d’activation des composantes de taille (k * n)
- iter : nombre d’itérations effectuées avant convergence
- normesObtenues : Liste des normes obtenues au fil des itérations

Paramètres:
- M : matrice d’entrée non-négative (ex: spectrogramme)
- nCompos : nombre de composantes à extraire
- eps : tolérance de convergence basée sur la variation entre itérations
- max_iter : nombre maximum d’itérations
'''
def calculDA_viaItakuraSaito(M, nCompos, eps=1e-8, max_iter=250):
    
    freq, temps = M.shape

    # Pour que D*A soit proche de M dès le début et accélere la convergence
    moy = np.sqrt(M.mean() / nCompos)

    D = np.random.rand(freq, nCompos) * moy
    A = np.random.rand(nCompos, temps) * moy

    # Pour garder la contrainte de >=0
    D = np.maximum(D, 1e-10)
    A = np.maximum(A, 1e-10)

    # On initialise conv à Infini pour être certain d'entrer dans la boucle
    conv = float(np.inf)
    iter = 0

    normesObtenues = []

    while conv > eps and iter < max_iter:
        Dold = D
        Aold = A

        # Il faut dérivé (x/y - log(x/y) - y) par rapport à y => y-x / y**2
        # On obtient ainsi 1/y - x/y**2
        diviseur_DA = (D@A)

        # D.T pour avoir la bonne taille
        numeA = D.T@(M/(diviseur_DA**2)) # M/(diviseur_DA**2) <=> x/y*2
        denumA = D.T@(1/diviseur_DA) # M/(diviseur_DA) <=> 1/y
        A = A * (numeA/denumA)
        A = np.maximum(A, 1e-10)

        # Formule de calcul de D selon IS Divergence
        diviseur_DA = (D@A)
        numeD = (M/(diviseur_DA**2))@A.T
        denumD = (1/diviseur_DA)@A.T
        D = D * (numeD/denumD)
        D = np.maximum(D, 1e-10)        


        conv = scipy.linalg.norm(Dold@Aold - D@A)
        normesObtenues.append(conv)

        iter+=1

    return D, A, iter, normesObtenues





"""
-- Fonction comparaisonMeilleurAlgo() --

Brief:
Comparaison des différents algorithmes MNF via la formule Ps/Pb avec la puissance du Signal M et la puissance du Bruit M-D@A.
Affichage ensuite du graphique de comparaison pour une meilleure visualisation des résultats.

Retourne:
- resultats : Dictionnaire des résultats {"Abréviation des noms d'algos": valeur du ratio}

Paramètres:
- M : matrice d’entrée non-négative (ex: spectrogramme)
- nCompos : nombre de composantes à extraire
- eps : tolérance de convergence basée sur la variation entre itérations
- max_iter : nombre maximum d’itérations
"""
def comparaisonMeilleurAlgo(M, nCompos, eps=1e-8, max_iter=250):

    # On récupère chaque résultat des fonctions
    D_MC, A_MC, _, _ = calculDA_viaMoindresCarres(M, nCompos, eps, max_iter)
    D_KL, A_KL, _, _ = calculDA_viaKullbackLeibler(M, nCompos, eps, max_iter)
    D_IS, A_IS, _, _ = calculDA_viaItakuraSaito(M, nCompos, eps, max_iter)

    # On calcule l'erreur pour chaque fonction
    Erreur_MC = M - (D_MC@A_MC)
    Erreur_KL = M - (D_KL@A_KL)
    Erreur_IS = M - (D_IS@A_IS)

    # Pour avoir la comparaison, on va utiliser Ps / Pb donc on convertit en puissance:

    Puissance_SignalM = np.sum(M**2) # calcul de Ps

    Puissance_Bruit_MC = np.sum(Erreur_MC**2)
    Puissance_Bruit_KL = np.sum(Erreur_KL**2)
    Puissance_Bruit_IS = np.sum(Erreur_IS**2)

    
    # On traduit en décibel pour avoir des valeurs plus compréhensible

    ratio_MC = 10 * np.log10(Puissance_SignalM / Puissance_Bruit_MC)
    ratio_KL = 10 * np.log10(Puissance_SignalM / Puissance_Bruit_KL)
    ratio_IS = 10 * np.log10(Puissance_SignalM / Puissance_Bruit_IS)


    # Maintenant qu'on a les valeurs, on en fait un dictionnaire
    resultats = {
        "MC": ratio_MC,
        "KL": ratio_KL,
        "IS": ratio_IS
    }

    # Ensuite on va faire les affichages par liste

    noms = resultats.keys()
    ratios = resultats.values()

    # Gestion de l'affichage (faites via IA pour une meilleure qualité)
    barres = plt.bar(noms, ratios, color=['blue', 'green', 'red'])

    plt.ylabel('Ratio Ps/Pb (dB)')
    plt.title('Comparaison des différents algorithmes MNF')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Afficher les valeurs sur les barres
    for b in barres:
        yval = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, yval + 0.5, f"{yval:.1f} dB", ha='center', va='bottom', fontweight='bold')
        
    plt.show()

    
    # on renvoie le dictionnaire
    return resultats