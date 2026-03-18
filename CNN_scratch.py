import numpy as np

class SimpleConv2D:
    def __init__(self, nb_filtres, kernel_size, strides=1, padding=0, activation='relu'):
        self.nb_filtres = nb_filtres
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation

        self.W = None
        self.biais = None

    def produit_frobenius(self, A, B):
        return np.sum(A * B)

    def forward(self, input):
        h_in, l_in, nb_canaux = input.shape

        if self.W is None:
            self.W = np.zeros((self.nb_filtres, nb_canaux, self.kernel_size, self.kernel_size))
            self.biais = np.zeros((self.nb_filtres,))

        h_out = int((h_in - self.kernel_size + 2 * self.padding) / self.strides + 1)
        l_out = int((l_in - self.kernel_size + 2 * self.padding) / self.strides + 1)

        output = np.zeros((h_out, l_out, self.nb_filtres))

        # glissement de la fenetre de convolution (en supposant strides = 1)
        for i in range(h_out):
            for j in range(l_out):
                f_conv = input[i:i+self.kernel_size, j:j+self.kernel_size, :]
                
                for filtre in range(self.nb_filtres):
                    W_filtre = self.W[filtre]
                    s = 0.0
                    
                    for c in range(nb_canaux):
                        s += self.produit_frobenius(
                            f_conv[:, :, c],
                            W_filtre[c, :, :]
                        )
                    
                    s += self.biais[filtre]

                    # activation ReLU
                    if self.activation == 'relu':
                        s = max(0, s)

                    output[i, j, filtre] = s

        return output
