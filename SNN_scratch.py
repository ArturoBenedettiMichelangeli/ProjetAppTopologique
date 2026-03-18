import numpy as np

class SimplicialConvolution:
    def __init__(self, nb_filtres, K, activation='relu'):
        self.nb_filtres = nb_filtres
        self.K = K  # Degré du polynôme (le "support")
        self.activation = activation

        # Les poids et biais seront initialisés selon le nombre de canaux d'entrée
        self.W = None
        self.biais = None

    def forward(self, input_signals, Lk):

        N, nb_canaux_in = input_signals.shape

        # Initialisation des poids : un poids par degré du polynôme, par canal, par filtre
        if self.W is None:
            # Shape: (nb_filtres, nb_canaux_in, K + 1)
            # On a K+1 coefficients pour le polynôme : w_0*L^0 + w_1*L^1 + ... + w_K*L^K
            self.W = np.random.randn(self.nb_filtres, nb_canaux_in, self.K + 1) * 0.1
            self.biais = np.zeros((self.nb_filtres,))

        # Initialisation de la sortie : (N, nb_filtres)
        output = np.zeros((N, self.nb_filtres))

        # Pré-calcul des puissances du Laplacien appliquées au signal
        # On calcule [L^0*f, L^1*f, ..., L^K*f]
        # Cela remplace le "glissement de fenêtre" par une "diffusion locale"
        signaux_diffusés = []
        current_f = input_signals # L^0 * f
        signaux_diffusés.append(current_f)
        
        for k in range(1, self.K + 1):
            # On propage le signal au voisinage n+1 via le Laplacien
            current_f = np.dot(Lk, current_f)
            signaux_diffusés.append(current_f)

        # Application des filtres
        for f_idx in range(self.nb_filtres):
            s = np.zeros(N)
            
            for c in range(nb_canaux_in):
                # Pour chaque canal d'entrée, on applique le polynôme de poids
                for k in range(self.K + 1):
                    # Poids spécifique au filtre f, au canal c et à la puissance k
                    poids = self.W[f_idx, c, k]
                    
                    # On ajoute la contribution du signal diffusé
                    s += poids * signaux_diffusés[k][:, c]
            
            # Ajout du biais pour le filtre f
            s += self.biais[f_idx]

            # Activation
            if self.activation == 'relu':
                s = np.maximum(0, s)
            
            output[:, f_idx] = s

        return output