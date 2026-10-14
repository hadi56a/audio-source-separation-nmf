import numpy as np
import scipy.signal

'''
-- Fonction recomposition() --

Brief:
Reconstruit les signaux temporels à partir de la factorisation NMF d'un spectrogramme.
À partir des matrices A (activations temporelles) et D (spectres sources), la fonction applique
un masque spectral à la STFT complexe originale pour reconstituer chaque source dans le temps.

Retourne: 
- Une liste de signaux temporels reconstruits (un signal par source).

Paramètres:
- A : matrice d'activation des sources (m * k)
- D : matrice liste des sources (k * n)
- MatriceComplexeSpectrogramme : Matrice complexe correspondant à la STFT (Short Time Fourier Transform) du signal d'origine
- param... : doivent correspondrent aux paramètres utilisés lors de la STFT de base.

'''
def recomposition(A, D, MatriceComplexeSpectrogramme, paramsFS=44100, paramWindow="hann", paramNperseg=1024, paramNoverlap=None):
    
    if D.shape[1] != A.shape[0]:
        raise Exception(f"Erreur, taille de matrice différente : ({A.shape[0]}) et ({D.shape[1]})")
    
    if paramNoverlap is None:
        paramNoverlap = paramNperseg // 2

    signauxFinaux = []

    nbSources = D.shape[1]
    V = D@A + 1e-12 # pour éviter la / par 0 dans le calcul du masque

    for i in range(nbSources):
        V_i = np.outer(D[:, i], A[i, :])

        masque = V_i / V # Masque pour avoir une qualité des audios améliorées

        # On ajoute la phase à notre signal pour l'istft (on présume que la phase du signal d'origine suffit)
        z = masque * MatriceComplexeSpectrogramme

        _, signal_audio_Reconstruit = scipy.signal.istft(z, fs=paramsFS, window=paramWindow, nperseg=paramNperseg, noverlap=paramNoverlap)

        signauxFinaux.append(signal_audio_Reconstruit)

    return signauxFinaux                                      