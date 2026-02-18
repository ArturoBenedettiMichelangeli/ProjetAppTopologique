import numpy as np
from collections import defaultdict



# Chargement du graphe
def charger_snap_dblp(chemin_fichier, max_aretes=1500):
    aretes = []
    noeuds = set()
    print(f"Lecture du fichier : {chemin_fichier}")
    try:
        compteur = 0
        with open(chemin_fichier, 'r') as f:
            for ligne in f:
                if ligne.startswith('#') or ligne.startswith('From'):
                    continue
                u, v = map(int, ligne.split())
                if u == v: continue 
                aretes.append(tuple(sorted((u, v))))
                noeuds.add(u)
                noeuds.add(v)
                compteur += 1
                if compteur >= max_aretes: break
        return list(set(aretes)), sorted(list(noeuds))
    except FileNotFoundError:
        print("Erreur : Fichier introuvable.")
        return None, None

def extraire_triangles(aretes):
    voisins = defaultdict(set)
    for u, v in aretes:
        voisins[u].add(v)
        voisins[v].add(u)
    tris = set()
    for u, v in aretes:
        intersection = voisins[u].intersection(voisins[v])
        for w in intersection:
            tris.add(tuple(sorted((u, v, w))))
    return list(tris)


# Calcul du laplacien simplicial (L1)
def construire_laplacien_L1(aretes, triangles, noeuds):
    n_aretes = len(aretes)
    n_noeuds = len(noeuds)
    n_tri = len(triangles)
    
    map_noeuds = {n: i for i, n in enumerate(noeuds)}
    map_aretes = {e: i for i, e in enumerate(aretes)}

    B1 = np.zeros((n_noeuds, n_aretes))
    for j, (u, v) in enumerate(aretes):
        B1[map_noeuds[u], j] = -1
        B1[map_noeuds[v], j] = 1

    B2 = np.zeros((n_aretes, n_tri))
    for k, (u, v, w) in enumerate(triangles):
        e1, e2, e3 = tuple(sorted((u, v))), tuple(sorted((v, w))), tuple(sorted((u, w)))
        if e1 in map_aretes: B2[map_aretes[e1], k] = 1
        if e2 in map_aretes: B2[map_aretes[e2], k] = 1
        if e3 in map_aretes: B2[map_aretes[e3], k] = -1

    L1 = (B1.T @ B1) + (B2 @ B2.T)

    # Normalisation spectrale (Symmetric Normalization)
    diag = np.abs(np.diag(L1))
    d_inv_sqrt = np.power(diag + 1e-7, -0.5)
    D = np.diag(d_inv_sqrt)
    return D @ L1 @ D


class CoucheSimpliciale:
    def __init__(self, in_dim, out_dim, K):
        self.K = K
        # Initialisation plus stable (proche de Xavier)
        limit = np.sqrt(6 / (in_dim + out_dim))
        self.poids = [np.random.uniform(-limit, limit, (in_dim, out_dim)) * 0.1 for _ in range(K + 1)]
        self.bias = np.zeros(out_dim)

    def forward(self, x, L):
        self.x_diffuses = [x]
        for k in range(self.K):
            # On applique L et on peut optionnellement stabiliser l'amplitude
            next_x = L @ self.x_diffuses[-1]
            self.x_diffuses.append(next_x)
            
        res = np.zeros((x.shape[0], self.poids[0].shape[1]))
        for k in range(self.K + 1):
            res += self.x_diffuses[k] @ self.poids[k]
        return res + self.bias

    def backward(self, d_out, learning_rate):
        # Gradient clipping pour éviter l'explosion (Nan/Inf)
        d_out = np.clip(d_out, -1.0, 1.0)
        
        for k in range(self.K + 1):
            grad_w = self.x_diffuses[k].T @ d_out
            # Clipping du gradient des poids
            grad_w = np.clip(grad_w, -1.0, 1.0)
            self.poids[k] -= learning_rate * grad_w
        
        grad_b = np.sum(d_out, axis=0)
        self.bias -= learning_rate * grad_b



def lancer_experience():
    np.random.seed(42)
    
    FICHIER = "com-dblp.ungraph.txt" 
    aretes, noeuds = charger_snap_dblp(FICHIER, max_aretes=1500)
    if not aretes: return

    tris = extraire_triangles(aretes)
    L1 = construire_laplacien_L1(aretes, tris, noeuds)
    
    # Calcul du signal cible (Z-score normalization)
    degres = defaultdict(int)
    for u, v in aretes:
        degres[u] += 1
        degres[v] += 1
    
    y_raw = np.array([degres[u] + degres[v] for u, v in aretes], dtype=float).reshape(-1, 1)
    mean_y = np.mean(y_raw)
    std_y = np.std(y_raw) + 1e-9
    y_true = (y_raw - mean_y) / std_y

    # Masquage (30% des données connues, 70% à imputer)
    masque = np.random.rand(len(aretes)) < 0.3
    x_entrainement = np.zeros_like(y_true)
    x_entrainement[masque] = y_true[masque] 

    # Paramètres d'apprentissage
    modele = CoucheSimpliciale(in_dim=1, out_dim=1, K=2)
    LR = 0.001  # Réduit pour la stabilité
    EPOCHS = 300

    print(f"Entraînement sur {len(aretes)} arêtes...")
    for e in range(EPOCHS + 1):
        preds = modele.forward(x_entrainement, L1)
        
        # Erreur sur les données connues uniquement (le masque contient les indices connus)
        erreur = preds - y_true
        d_out = np.zeros_like(erreur)
        # On ne rétropropage que sur les indices connus
        d_out[masque] = erreur[masque] / np.sum(masque)
        
        loss = np.mean(erreur[masque]**2)
        
        if np.isnan(loss) or np.isinf(loss):
            print(f"Arrêt précoce : Loss instable à l'époque {e}")
            break
            
        modele.backward(d_out, LR)

        if e % 20 == 0:
            # MAE sur les données manquantes (imputation)
            mae_test = np.mean(np.abs(preds[~masque] - y_true[~masque]))
            # On ré-inverse la normalisation pour une MAE lisible (optionnel)
            mae_denorm = mae_test * std_y
            print(f"Epoque {e:3d} | Loss: {loss:.6f} | MAE Imputation (Norm): {mae_test:.6f}")

if __name__ == "__main__":
    lancer_experience()