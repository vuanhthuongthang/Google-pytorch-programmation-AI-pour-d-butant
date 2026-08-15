# Cours de PyTorch pour débutant : De la Physique à l'IA

## Présentation
Ce dépôt compile l'intégralité du cours détaillant le lien entre l'énergie thermodynamique, la rétropropagation, et la résolution de systèmes physiques linéaires (AX = Y). Ces travaux sont proposés par Thang Formation - IA et sont optimisés pour des environnements Windows 11 et Linux Mint 22.3.

## Théorie Mathématique
Le cœur de ce cours démontre que la rétropropagation minimise l'énergie d'un système par la règle de la chaîne.
* La fonction d'énergie (Loss) s'écrit avec la norme de Frobenius : `L(A) = 1/2 ||AX - Y||²_F`
* Le calcul du gradient par rapport à la matrice A donne : `∇_A L = (AX - Y)Xᵀ`
* À l'équilibre thermodynamique, le gradient s'annule, menant à l'équation `AXXᵀ = YXᵀ` et trouvant la solution optimale de la matrice A.

## Contenu des Scripts PyTorch
Les scripts fournis sont exécutables directement sur le CPU sans nécessiter de carte graphique lourde :
* **identification_raideurs_pont.py** : Identification des raideurs d'un pont (Système mécanique).
* **reconstruction_mnist.py** : Reconstruction d'une image MNIST réelle en simulant un milieu hétérogène.
* **retropropagation_paint_linux.py** : Rétropropagation sur un dessin Paint optimisé pour Linux Mint 22.3 en mode "Headless" (génération PNG sans interface graphique).
* **retropropagation_paint_windows.py** : Rétropropagation sur un dessin Paint optimisé pour l'environnement natif de Windows 11 (affichage pop-up direct).

## Licence
Cours sous licence MIT pour Thang Formation – IA.
