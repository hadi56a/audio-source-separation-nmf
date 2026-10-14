from matplotlib import pyplot as plt
import numpy as np

def afficher_NMF(W, A, freq, t):
    """
    W : matrice (freq × k)
    A : matrice (k × temps)
    M : spectrogramme original (freq × temps)
    freq : vecteur des fréquences STFT
    t : vecteur des temps STFT
    """
    
    k = A.shape[0]   # nombre de composantes
    
    # Reconstruction pour l'affichage
    M_recons = W @ A

    # --- 1) Activations temporelles A ---
    axA = plt.subplot2grid((5,4), (0,1), colspan=3)
    axA.imshow(A, aspect='auto', origin='lower',extent=[t[0], t[-1], 0, k], cmap='jet')
    plt.title("Temporal activations H")
    plt.ylabel("Composante")
    
    # --- 2) Dictionnaire W ---
    plt.imshow(W, aspect='auto', origin='lower',extent=[0, k, freq[0], freq[-1]], cmap='jet')
    plt.title("Dictionary W")
    plt.xlabel("Composante")
    plt.ylabel("Hz")
    plt.ylim(0, 3000)

    # --- 3) Spectrogramme reconstruit ---
    axG = plt.subplot2grid((5,4), (1,1), colspan=3, rowspan=4)
    axG.imshow(M_recons, aspect='auto', origin='lower',extent=[t[0], t[-1], freq[0], freq[-1]], cmap='jet')
    plt.title("Reconstructed Spectrogram W × H")
    plt.xlabel("Time")
    plt.ylabel("Hz")
    plt.ylim(0, 3000)
    plt.tight_layout()
    plt.show()








def tracer_signaux(signaux, fs):
    plt.figure(figsize=(14, 8))

    nb = len(signaux)
    for i, sig in enumerate(signaux):
        t = np.linspace(0, len(sig)/fs, len(sig))

        plt.subplot(nb, 1, i+1)
        plt.plot(t, sig)
        plt.title(f"Source {i}")
        plt.xlabel("Temps (s)")
        plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()