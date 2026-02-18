Utilisation des LLM dans le code:
* pour comprendre le papier
* pour optimiser la stabilité numérique du modèle (notamment l'initialisation des poids, le gradient clipping et la normalisation spectrale du Laplacien)
* pour l'implémentation de l'algorithme d'extraction des triangles afin d'en améliorer l'efficacité computationnelle

Ce projet présente une implémentation d'un Réseau de Neurones Simplicial (SNN) dédié à la prédiction de valeurs manquantes sur les collaborations au sein d'un réseau de co-autorship. L'objectif est d'exploiter la structure topologique du complexe simplicial pour imputer ces valeurs manquantes en considérant non seulement les liens binaires, mais aussi les relations d'ordre supérieur.

Le dataset utilisé pour tester le modèle est DBLP (com-dblp.ungraph), issu de la collection SNAP de l'Université de Stanford. Ce choix se justifie par des contraintes matérielles, le dataset du papier étant trop volumineux pour nos ordinateurs. Le fichier com-dblp.ungraph.txt contient une liste d'adjacence représentant les collaborations scientifiques où chaque nœud est un auteur. Pour construire le complexe simplicial, nous extrayons de ce réseau les arêtes (1-simplexes) et nous identifions les triangles (2-simplexes) formés par les cliques de trois auteurs.

L'architecture du SNN repose sur une généralisation des réseaux de neurones convolutionnels (CNN) appliqués à des structures non-euclidiennes. Contrairement aux CNN classiques opérant sur des grilles de pixels régulières, le SNN traite des données structurées sous forme de complexes. Dans un CNN, la convolution s'appuie sur le glissement d'un noyau local effectuant un produit de Frobenius. Dans notre modèle, cette opération est remplacée par un mécanisme de diffusion spectrale utilisant le Laplacien de Hodge d'ordre 1 ($L_1$). Ce dernier intègre à la fois la connexité par les nœuds et par les triangles. Mathématiquement, la convolution simpliciale est définie par un polynôme du Laplacien $\sum w_k \cdot L^k \cdot Input$, où les poids appris $w_k$ déterminent l'influence de chaque niveau de voisinage, à l'image des coefficients d'un filtre spatial traditionnel.

La méthodologie d'entraînement repose sur l'application d'un masque qui occulte 70% des données de collaboration. Le réseau est entraîné à reconstruire le signal original en diffusant l'information connue à travers le complexe. Pour garantir la stabilité numérique face aux puissances de matrices, l'implémentation utilise une normalisation spectrale et une technique de gradient clipping. Les résultats obtenus démontrent une convergence efficace du modèle. Partant d'une perte initiale élevée, le réseau parvient à une erreur absolue moyenne (MAE) normalisée d'environ 1.23 après 300 époques, comme illustré par les logs d'entraînement suivants :

Epoque   0 | Loss: 3269.720194 | MAE Imputation (Norm): 23.039417
Epoque  20 | Loss: 2504.157778 | MAE Imputation (Norm): 20.323193
Epoque  40 | Loss: 1841.996479 | MAE Imputation (Norm): 17.601517
Epoque  60 | Loss: 1282.515075 | MAE Imputation (Norm): 14.875012
Epoque  80 | Loss: 825.024630 | MAE Imputation (Norm): 12.152148
Epoque 100 | Loss: 468.867155 | MAE Imputation (Norm): 9.439717
Epoque 120 | Loss: 213.414313 | MAE Imputation (Norm): 6.728693
Epoque 140 | Loss: 58.066176 | MAE Imputation (Norm): 4.047900
Epoque 160 | Loss: 2.250043 | MAE Imputation (Norm): 1.529460
Epoque 180 | Loss: 2.046531 | MAE Imputation (Norm): 1.375724
Epoque 200 | Loss: 1.966143 | MAE Imputation (Norm): 1.349909
Epoque 220 | Loss: 1.889019 | MAE Imputation (Norm): 1.324619
Epoque 240 | Loss: 1.815028 | MAE Imputation (Norm): 1.299893
Epoque 260 | Loss: 1.744041 | MAE Imputation (Norm): 1.275672
Epoque 280 | Loss: 1.675936 | MAE Imputation (Norm): 1.251970
Epoque 300 | Loss: 1.610595 | MAE Imputation (Norm): 1.231575
