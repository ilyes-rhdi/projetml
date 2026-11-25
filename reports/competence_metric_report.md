# Rapport — métrique de voisinage fondée sur la compétence des experts

## Décision de recherche

La piste initiale de gating dépendant des caractéristiques est déjà une famille connue : stacking à poids dépendant des covariables, sélection dynamique de régresseurs et sélection dynamique conditionnée par validation existent déjà. Les comparaisons pertinentes incluent [Feature-Weighted Linear Stacking](https://arxiv.org/abs/0911.0460), [MINE pour la sélection dynamique de régresseurs](https://doi.org/10.1016/j.ins.2020.07.056) et le cadre de 2026 de [sélection dynamique de régression conditionnée par validation](https://ieeexplore.ieee.org/abstract/document/11608383). La sélection de voisins adaptée à la compétence existe également en classification, par exemple [DES-FA](https://arxiv.org/abs/1811.00669).

Ces travaux empêchent de présenter le gating local ou la distance adaptative comme une nouveauté en soi. La variante examinée ici est plus précise : apprendre une métrique diagonale de voisinage à partir des erreurs hors pli des experts, puis ne retenir les variables de compétence que si leur information mutuelle dépasse un seuil de permutation corrigé pour toutes les variables. Les recherches effectuées n'ont pas confirmé cette combinaison exacte pour le local stacking de régression; elles ne constituent toutefois pas une revue bibliographique exhaustive. La nouveauté reste donc une **hypothèse à valider avec l'enseignant**, pas une revendication établie.

## Méthode testée

1. Produire des prédictions hors pli des experts sur le seul ensemble d'entraînement.
2. Pour chaque ligne, déterminer quel expert a la plus faible erreur absolue, après normalisation par l'erreur moyenne de chaque expert.
3. Mesurer l'information mutuelle entre chaque variable et cette identité de compétence. Un test par permutations des étiquettes, fondé sur le maximum des scores entre variables, écarte les signaux faibles et corrige les faux positifs multiples.
4. Utiliser les scores restants pour pondérer la distance euclidienne et former le voisinage du local stacking.
5. Choisir le niveau de shrinkage par validation interne. Une règle à une erreur standard conserve le global lorsque le gain local n'est pas assez marqué.

Les plis externes ne servent jamais à choisir la métrique, le shrinkage ou le repli. Le prototype est dans [competence_metric_local_stacking.py](../competence_metric_local_stacking.py) et son exécuteur dans [run_competence_metric_experiments.py](../run_competence_metric_experiments.py).

## Résultats

RMSE moyen par pli; les comparaisons de plis répétés sont descriptives. « Victoires/nuls/défaites » compare le candidat à la distance euclidienne sur les mêmes plis.

| Jeu | Validation externe | Stacking global | Local euclidien | Métrique compétence | V/N/D vs euclidien | Local choisi |
|---|---:|---:|---:|---:|---:|---:|
| Simulation, spécialisation régionale + 25 variables aléatoires | 5 plis × 2 répétitions | 1,5913 | 1,5923 | **1,5645** | 4/5/1 | 6/10 |
| Simulation, compétence homogène + 25 variables aléatoires | 5 × 2 | 0,4971 | 0,4971 | 0,4971 | 0/10/0 | 0/10 |
| UCI CCPP (9 568 lignes, 4 variables) | 5 plis | 3,3666 | 3,3668 | **3,3497** | 4/1/0 | 4/5 |
| UCI Student Performance — math (395 lignes, 58 colonnes après encodage) | 5 plis | 1,6646 | 1,6646 | 1,6646 | 0/5/0 | 0/5 |
| UCI Bike Sharing — quotidien (731 jours, 36 colonnes après encodage) | 5 × 2 | 655,93 | 644,54 | **643,99** | 1/9/0 | 8/10 |
| UCI Appliances Energy (échantillon fixe de 3 000 lignes, 27 variables) | 5 plis | 86,6211 | 86,6211 | 86,6211 | 0/5/0 | 0/5 |
| UCI Concrete Slump Test (103 lignes, 7 entrées) | 5 plis | 7,0821 | 7,0821 | 7,0821 | 0/5/0 | 0/5 |

### Lecture des résultats

- Sur le contrôle synthétique où la compétence change réellement selon une région, le candidat baisse le RMSE de 1,7 % par rapport au global. Il gagne 4 plis, en perd 2 et égale le global dans 4 plis; le résultat est encourageant mais limité à une simulation construite.
- Sur le contrôle homogène, la sélection choisit toujours le global; aucune perte locale n'est imposée.
- Sur cinq jeux UCI, le repli global évite une dégradation par rapport au global sur les 30 plis évalués. L'amélioration contre le voisinage euclidien apparaît surtout sur CCPP (environ 0,5 % de RMSE) et très légèrement sur Bike Sharing (environ 0,08 % sur dix plis). Student, Appliances et Slump restent au global.
- La plus grande part du gain sur Bike vient du local stacking déjà présent dans le projet : le candidat métrique ne fait mieux que l'euclidien que dans 1 pli sur 10. Sur CCPP, il le fait dans 4 plis sur 5.
- Aucun test de significativité n'est reporté : les plis de validation croisée répétés ne sont pas indépendants, et les gains sont petits.

Les jeux proviennent du dépôt UCI : [CCPP](https://archive.ics.uci.edu/dataset/294/combined%2Bcycle%2Bpower%2Bplant), [Student Performance](https://archive.ics.uci.edu/dataset/320/student%2Bperformance), [Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike-sharing-dataset), [Appliances Energy](https://archive.ics.uci.edu/dataset/374/appliances-energy-prediction) et [Concrete Slump Test](https://archive.ics.uci.edu/dataset/182/concrete%2Bslump%2Btest). Le jeu Bike est évalué par plis aléatoires : cela teste l'interpolation entre jours observés, pas une prévision chronologique. Appliances est sous-échantillonné pour le temps de calcul; ses deux variables aléatoires UCI restent conservées.

## Conclusion pratique

Cette piste combine une différence méthodologique ciblée et des résultats réels modestes : elle améliore CCPP, ne dégrade pas les autres cas testés grâce au repli, mais revient souvent à la distance euclidienne ou au global. On ne peut pas en déduire une probabilité générale d'amélioration sur de nouveaux jeux. Les données soutiennent plutôt une chance modérée de gain lorsque la compétence des experts dépend de quelques variables identifiables, et une probabilité élevée de repli sans changement autrement; ces appréciations sont qualitatives, pas des estimations statistiques.

Avant de la présenter comme contribution nouvelle, il faut compléter la recherche bibliographique autour de DES-FA, MINE, du metric learning pour la régression et du local stacking, puis faire valider la formulation par l'enseignant. Les deux variantes de gating déjà implémentées restent des références de comparaison, pas des revendications de nouveauté.
