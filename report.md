# Rapport - Estimation du mouvement d'un objet unique

## 1. Introduction

Ce projet etudie l'estimation du mouvement apparent d'une voiture dans une sequence d'images. L'objectif est de construire une chaine classique de traitement d'images capable de produire un champ de mouvement, une trajectoire globale, puis une analyse de vitesse et de direction.

## 2. Contexte du projet

La sequence represente une voiture observee dans une scene exterieure. La voiture est consideree comme un objet unique et rigide: sa forme globale reste suffisamment stable pour que le centre de sa bounding box estimee soit utilise comme representation de sa position.

## 3. Dataset

Les frames sont chargees depuis `data/car/car-11/img/`. Les fichiers sont tries par nom pour conserver l'ordre temporel. Cette etape permet de connaitre le nombre total de frames, les dimensions des images et la qualite visuelle generale de la sequence.

**Interpretation.** Une sequence ordonnee est indispensable, car chaque deplacement est calcule entre deux frames successives. La qualite generale est suffisante pour des methodes classiques, mais les ombres, l'eclairage et les objets voisins peuvent perturber la segmentation.

## 4. Problematique

La problematique de ce projet est d'estimer le mouvement apparent d'une voiture dans une sequence d'images, en utilisant uniquement les variations d'intensite entre frames successives. Il s'agit d'isoler la voiture, d'estimer son champ de mouvement par flot optique, puis d'extraire sa trajectoire globale et d'analyser sa vitesse et sa direction. Cette estimation depend fortement des hypotheses du flot optique : conservation de l'intensite, petits deplacements et coherence spatiale.

## 5. Methodologie proposee

Le pipeline suit l'ordre suivant:

```text
chargement frames -> pretraitement -> contraste par histogramme
-> ROI manuelle -> segmentation -> morphologie -> Canny
-> difference d'images -> Lucas-Kanade -> trajectoire
-> vitesse/direction -> visualisation et interface
```

Le pipeline final est autonome: les resultats viennent uniquement des images et de l'initialisation manuelle.

## 6. Pretraitement

**Description.** Chaque frame est convertie en niveaux de gris pour travailler sur la luminance. Un filtre gaussien reduit le bruit local. Une amelioration du contraste aide a mieux distinguer la voiture de la route.

**Resultats generes.** La comparaison est sauvegardee dans `results/preprocessing/preprocessing_histogram_comparison.png` et dans `results/final_visualization/preprocessing_histogram_comparison.png`.

**Interpretation.** Le grayscale simplifie le traitement, le flou gaussien stabilise le seuillage et Lucas-Kanade, et le stretching ameliore la dynamique des niveaux de gris. Une amelioration trop forte peut cependant modifier les intensites et influencer l'hypothese d'illumination constante du flot optique.

## 7. Amelioration du contraste par histogramme

**Description.** L'histogramme decrit la distribution des intensites. Le stretching applique `I' = 255 * (I - Imin) / (Imax - Imin)` pour etendre les niveaux de gris sur `[0,255]`. L'egalisation d'histogramme redistribue plus fortement les intensites.

**Resultats generes.** Les notebooks affichent les histogrammes avant/apres et sauvegardent les comparaisons dans `results/preprocessing/`.

**Interpretation.** Le stretching occupe mieux l'intervalle `[0,255]` sans etre aussi agressif que l'egalisation. L'egalisation est utile pour comparer, mais elle peut rendre les intensites moins coherentes d'une frame a l'autre. Le pipeline utilise `stretching` par defaut.

## 8. Segmentation Otsu/adaptative

**Description.** La segmentation sert a isoler la voiture de l'arriere-plan. Otsu calcule un seuil global dans la ROI; le seuillage adaptatif calcule des seuils locaux. On travaille dans une ROI pour eviter que la route, les arbres ou les autres voitures dominent le seuillage.

**Resultats generes.** Les masques sont sauvegardes dans `results/segmentation/`, notamment `mask_otsu_initial.png`, `mask_adaptive_initial.png` et les overlays `mask_overlay_*.png`.

**Interpretation.** Otsu est souvent stable quand la ROI encadre bien la voiture. Le seuillage adaptatif peut aider si l'eclairage varie, mais il peut aussi produire des details parasites. Un bon masque facilite la detection de points pertinents pour Lucas-Kanade.

## 9. Morphologie

**Description.** L'ouverture supprime les petits bruits par erosion puis dilatation. La fermeture remplit les petits trous par dilatation puis erosion. La plus grande composante connectee conserve l'objet principal.

**Resultats generes.** Les masques nettoyes sont sauvegardes dans `results/morphology/`, par exemple `mask_cleaned_initial.png` et `largest_component_initial.png`.

**Interpretation.** La morphologie rend le masque plus propre et plus compact. Cela reduit les points parasites et ameliore la stabilite du champ de mouvement Lucas-Kanade. Si la segmentation echoue, le pipeline garde la bbox courante comme fallback.

## 10. Detection de contours Canny

**Description.** Canny detecte les contours forts de l'image. Ici, il sert a valider visuellement les limites de la voiture dans la ROI.

**Resultats generes.** Les overlays sont sauvegardes dans `results/edge_detection/`.

**Interpretation.** Lorsque les contours correspondent au contour de la voiture, ils confirment que la ROI et la segmentation sont coherentes. Des contours parasites peuvent venir de la route, des ombres ou des details de texture. Canny n'est pas la methode principale du tracking.

## 11. Detection de mouvement par difference d'images

**Description.** La difference absolue `absdiff` entre deux frames successives met en evidence les pixels qui changent. Un seuillage produit ensuite une carte binaire de mouvement.

**Resultats generes.** Les images `difference_*.png`, `motion_mask_*.png` et `motion_overlay_*.png` sont sauvegardees dans `results/motion_detection/`.

**Interpretation.** Si la zone mobile correspond a la voiture, la difference d'images confirme le deplacement avant Lucas-Kanade. Cette methode reste sensible a l'eclairage, aux ombres et aux variations d'intensite non liees au mouvement.

## 12. Estimation du champ de mouvement par Lucas-Kanade

**Description.** Lucas-Kanade suit des points caracteristiques entre deux frames. Chaque vecteur relie une position ancienne et une position nouvelle; l'ensemble des vecteurs visualise le champ de mouvement demande. Le vecteur moyen donne `dx_global` et `dy_global`.

**Resultats generes.** Les images avec vecteurs sont sauvegardees dans `results/optical_flow/`, et l'image finale dans `results/final_visualization/final_motion_field_lk.png`.

**Interpretation.** Si les vecteurs sont nombreux et orientes de facon coherente, le champ de mouvement suit correctement la voiture. Les erreurs possibles sont la perte de points, une texture faible, des ombres, une segmentation imparfaite ou un deplacement trop grand entre deux frames.

## 13. Extraction de la trajectoire globale

**Description.** La bbox est deplacee par le mouvement moyen Lucas-Kanade, puis ajustee par segmentation quand celle-ci est fiable. Le centre de chaque bbox estimee est stocke. Les centres successifs sont relies pour former la trajectoire globale.

**Resultats generes.** Le fichier `results/trajectory.csv` contient frame, centre, bbox, deplacement, vitesse, direction et nombre de points suivis. Les images sont sauvegardees dans `results/trajectory/` et `results/final_visualization/final_trajectory.png`.

**Interpretation.** Une trajectoire lisse indique un suivi stable. Les ruptures ou oscillations peuvent indiquer une perte de points, une erreur de segmentation ou une bbox initiale trop large.

## 14. Analyse de la vitesse

**Description.** La vitesse est calculee par la distance entre deux positions successives: `distance = sqrt(dx^2 + dy^2)`. Elle est exprimee en pixels/frame. Si le FPS reel est inconnu, aucune vitesse physique n'est inventee.

**Resultats generes.** Le graphe est sauvegarde dans `results/graphs/speed.png`.

**Interpretation.** La vitesse moyenne resume le mouvement global, tandis que la vitesse maximale signale les frames ou le deplacement estime est le plus fort. Des pics peuvent venir d'un vrai changement de mouvement ou d'une erreur de suivi.

## 15. Analyse de la direction

**Description.** La direction est calculee avec `atan2(dy, dx)` puis convertie en degres. Dans une image, l'axe `y` est oriente vers le bas, ce qui modifie l'interpretation geometrique habituelle.

**Resultats generes.** Le graphe est sauvegarde dans `results/graphs/direction.png`.

**Interpretation.** La direction moyenne indique l'orientation generale du deplacement dans le plan image. Des changements brusques de direction peuvent etre dus a des variations de trajectoire, a la segmentation ou a la perte de points.

## 16. Interface Tkinter

**Description.** L'interface `app_tkinter.py` permet de choisir le dataset, entrer une bbox ou selectionner une ROI, choisir les methodes, lancer le suivi et afficher les sorties principales.

**Resultats generes.** L'interface appelle le meme pipeline et sauvegarde les resultats dans `results/`.

**Interpretation.** Le resume textuel affiche dans l'interface donne le nombre de frames traitees, la distance totale, la vitesse moyenne, la vitesse maximale, la direction moyenne, le nombre moyen de points Lucas-Kanade et les limites.

## 17. Resultats et interpretation

Le fichier `results/interpretation_results.txt` est genere automatiquement a la fin du pipeline. Il reprend les descriptions principales, les sorties sauvegardees et l'interpretation numerique: nombre de frames, distance totale, vitesse moyenne, vitesse maximale, direction moyenne, deplacement cumule et nombre moyen de points suivis.

Les sorties visuelles obligatoires sont produites:

- comparaison pretraitement/histogramme;
- masque Otsu;
- masque adaptatif;
- masque apres morphologie;
- contours Canny;
- difference d'images;
- motion mask;
- champ de mouvement Lucas-Kanade;
- trajectoire globale;
- graphe vitesse;
- graphe direction;
- resume texte.

## 18. Limites

Le suivi depend de la ROI initiale. Les variations d'eclairage, les ombres, le faible contraste, les erreurs de segmentation, la presence d'autres objets, la perte de points et les grands deplacements entre frames peuvent degrader les resultats. Les valeurs de vitesse sont en pixels/frame en l'absence de calibration et de FPS reel.

## 19. Conclusion

Ce projet montre qu'une chaine classique de traitement d'images permet d'estimer le mouvement d'un objet unique sans utiliser de deep learning ni annotations groundtruth. La voiture est initialisee par une ROI manuelle, puis segmentee par seuillage et nettoyee par morphologie. Canny permet de valider les contours, la difference d'images met en evidence les zones mobiles, et Lucas-Kanade fournit le champ de mouvement sous forme de vecteurs. La trajectoire globale est extraite a partir des centres successifs de l'objet, tandis que la vitesse et la direction permettent d'analyser quantitativement le deplacement. Les resultats restent sensibles a l'eclairage, au contraste, aux erreurs de segmentation et a la perte de points suivis.
