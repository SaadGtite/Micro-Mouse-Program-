Solutions Micromouse – Résolution de Labyrinthes (2024-2025)
Ce dépôt propose deux algorithmes performants pour la résolution automatique de labyrinthes (Micromouse), adaptés aux compétitions avec des contraintes modernes : support du format 32×32 avec zone d’arrivée 3×3 non centrée et unique ouverture.

1. micro-mouse-solution-plus-court.py
Description
Ce script Python est conçu pour trouver le chemin le plus court dans un labyrinthe fourni sous forme d’image. Il s’appuie sur :

Détection automatique des murs (OpenCV)

Exploration du labyrinthe basée sur le Flood Fill (remplissage par propagation) optimisée

Calcul du chemin optimal avec algorithme A*

Respect des dernières contraintes Micromouse : zone d’arrivée étendue, détection dynamique des points de départ et d’arrivée

Fonctionnement
Chargement de l’image du labyrinthe (noir/blanc, format PNG/JPG acceptés)

Conversion de l’image en grille de murs (N/E/S/O)

Exploration initiale pour découvrir la zone but

Calcul retour (pour simuler l’aller-retour réel du robot)

Speed run optimal en simulant le run le plus rapide possible

Affichage graphique des solutions et sauvegarde des résultats

Comment utiliser
Lancer simplement :

text
python3 Micro-mouse-solution-plus-court.py
Saisir le chemin de l’image et la taille du labyrinthe (16 ou 32)

Toutes les étapes sont commentées dans le script

Dépendances nécessaires :
opencv-python — numpy — matplotlib

2. Micro-mouse-solution-plus-rapide.ipynb
Description
Ce notebook Jupyter offre une visualisation avancée et animée de l’exploration et de la résolution du labyrinthe. Principales caractéristiques :

Lecture d’image comme dans le script précédent

Exploration intelligente : détection des cellules, découverte dynamique de la zone but

Simulation animée de l’exploration, du calcul du retour, et du speed run via matplotlib

Statistiques détaillées : longueur des chemins, nombre de virages, efficacité, découverte des zones

Fonctionnement
Ouvrir le notebook dans Jupyter

Modifier si besoin la variable imagepath pour pointer vers le bon fichier image

Exécuter toutes les cellules une à une

Suivre sur les figures les différentes étapes : détection des murs, exploration, chemin optimal

Animations et images exportables intégrées pour des rapports visuels

Dépendances nécessaires :
opencv-python — numpy — matplotlib — Pillow (pour les animations GIF)

Résumé des étapes communes :
Image noir/blanc du labyrinthe, n’importe quelle dimension typique (16×16, 32×32)

Détection automatique : murs, départ, arrivée (zone spéciale 3×3)

Affichage interactif avec explications à chaque étape

Adaptés pour tutoriel, compétition, ou amélioration de l’algorithme

Conseils d’utilisation
Privilégiez des images nettes avec bon contraste

Adapter le seuil (threshold) dans les scripts si votre image a une inversion de couleurs

Testé sur Windows, Linux et Mac (via Anaconda/Jupyter ou Python natif)