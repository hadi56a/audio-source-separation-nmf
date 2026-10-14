import numpy as np
import scipy as sc


def STFT_Hann(x,f):
    # Normalisation
    x = x.astype(float)
    x /= np.max(np.abs(x))

    # Paramètres de STFT
    taille_fenetre=2048
    freq, t, Zxx = sc.signal.stft(x,fs=f,nperseg=taille_fenetre,noverlap=taille_fenetre//2,window='hann')

    #Valeur absolue
    S = np.abs(Zxx)

    # Passage en dB
    S_db = 20 * np.log10(S + 1e-12)

    return t, freq, S_db, S



def STFT_Gauss(x,f):
    # Normalisation
    x = x.astype(float)
    x /= np.max(np.abs(x))

    # Paramètres de STFT
    fenetre = sc.signal.windows.gaussian(50, std=8, sym=True)
    freq, t, Zxx = sc.signal.stft(x,fs=f,nperseg=len(fenetre),noverlap=len(fenetre)//2,window=fenetre)

    #Valeur absolue
    S = np.abs(Zxx)

    # Passage en dB
    S_db = 20 * np.log10(S + 1e-12)

    return t, freq, S_db



def STFT_calculerParametres(signal_data):
    """
    Choisit simplement nperseg et noverlap en fonction de la longueur du signal.
      - fenêtre = 5% de la longueur du signal
      - nperseg arrondi à une puissance de 2
      - noverlap = moitié de nperseg
    """

    # longueur du signal en échantillons
    N = len(signal_data)

    # fenêtre = 5% du signal
    nperseg = int(0.05 * N)

    # éviter les valeurs trop petites ou trop grandes
    nperseg = max(256, min(nperseg, 8192))

    # arrondir à la puissance de 2 la plus proche
    nperseg = 2**int(np.round(np.log2(nperseg)))

    # recouvrement = moitié de la fenêtre
    noverlap = nperseg // 2

    print(f"[STFT] nperseg = {nperseg}, noverlap = {noverlap}")

    return nperseg, noverlap