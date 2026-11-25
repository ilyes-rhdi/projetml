# Ablation — métrique orientée gain (V3)

## Question

La métrique de voisinage actuelle (V1) est apprise sur un **proxy** : l'identité de l'expert localement le plus compétent. V3 remplace ce proxy par l'**objectif final** : la RMSE du stacking local. Est-ce que cela améliore la métrique et les prédictions ?

## Méthode testée

`gain_metric_local_stacking.py` :

1. les poids diagonaux sont paramétrés par `exp(strength * theta)`, normalisés à une moyenne de 1 ;
2. une descente par coordonnées choisit `theta` pour minimiser la MSE de validation du stacking local ;
3. la MSE est calculée sur 30 % des lignes d'un sous-échantillon, les poids locaux étant ajustés sur les 70 % restants ;
4. plusieurs découpages peuvent être moyennés.

Le reste est identique à V1 : mêmes experts, mêmes plis externes, même règle de shrinkage et de repli. La seule différence est la cible de la métrique.

## Protocole

- **Jeu principal :** UCI CCPP (9 568 lignes, 4 variables), 5 plis externes, 2 plis internes, graine 411.
- **Comparaison à shrinkage forcé `lambda = 1`** : isole la qualité de la métrique de la règle de repli.
- **Comparaison avec sélection honnête** : le shrinkage est choisi dans le pli d'entraînement, comme dans le protocole final.
- **Diagnostic synthétique :** simulation régionale + 25 variables de bruit, un pli, comparaison à une métrique oracle qui sait que seule la variable 0 définit la région.
- **Variantes :** global, local euclidien, V1 (MI/argmin), V3 (gain).

## Résultats

### CCPP, shrinkage forcé à `lambda = 1` (5 plis)

| Méthode | RMSE moyen | Écart-type | vs global | vs V1 |
|---|---:|---:|---:|---:|
| Global | 3,3844 | 0,1796 | — | — |
| Local euclidien | 3,3777 | 0,1812 | −0,20 % | +0,35 % |
| **V1 (MI/argmin)** | **3,3659** | **0,1792** | **−0,55 %** | — |
| V3, 2 passes, 1 découpage | 3,3766 | 0,1810 | −0,23 % | +0,32 % |
| V3, 6 passes, 3 découpages | 3,3790 | 0,1799 | −0,16 % | +0,39 % |

- V3 bat Euclide de **0,03 %** seulement, et perd contre V1 dans **4 plis sur 5**.
- Une recherche plus profonde **aggrave** V3 : 3,3790, soit **pire qu'Euclide**. La descente sur-apprend la MSE de validation.
- V3 gagne 1 pli sur 5 contre V1, et 4 plis sur 5 contre Euclide avec la recherche courte.

### CCPP, sélection honnête du shrinkage (5 plis)

| Méthode | RMSE moyen | Shrinkage choisi |
|---|---:|---|
| Global | 3,3844 | — |
| V1 | 3,3783 | local 2/5, global 3/5 |
| V3 | 3,3844 | **global 5/5** |

Avec la règle de repli, V3 ne déclenche jamais le mode local : sa métrique n'est jamais utilisée. V1 reste locale dans 2 plis et bat le global.

### Diagnostic synthétique régional (1 pli, `lambda = 1`)

| Métrique | RMSE |
|---|---:|
| Uniforme (Euclide) | 1,7702 |
| V3 apprise | 1,7718 |
| Oracle : variable 0 × 10 | **1,7505** |
| Oracle : variable 0 × 3 | 1,7586 |
| Global | 1,7572 |

V3 trouve bien la variable 0 en tête, mais avec un poids trop faible (1,78) et en gonflant aussi des variables de bruit. La métrique apprise est légèrement **pire** qu'Euclide, alors que la métrique oracle fait mieux que le global. Le problème n'est donc pas l'absence de signal, mais la difficulté à l'exploiter sans sur-apprendre.

## Interprétation

1. **La cible est trop bruitée.** La MSE finale dépend de la réalisation des erreurs sur un petit échantillon de validation ; l'identité du meilleur expert est une cible plus stable. Le proxy de V1 agit comme un **réducteur de variance**, pas comme un simple raccourci.
2. **L'effet est de second ordre.** Le stacking local est déjà régularisé vers les poids globaux. Changer le voisinage ne change les prédictions que marginalement, et le bruit de la métrique peut annuler le gain.
3. **La recherche sur-apprend.** Plus de passes et plus de découpages devraient stabiliser ; ici, ils dégradent. Les poids appris varient selon le découpage et sélectionnent parfois la variable 3 (RH) au lieu de la variable 1 (V), ce qui coûte cher sur CCPP.
4. **La nouveauté est plus faible que prévu.** L'idée d'apprendre une métrique en minimisant l'erreur de régression existe déjà :
   - Weinberger & Tesauro, *Metric Learning for Kernel Regression*, AISTATS 2007 ;
   - Perrot & Habrard, *Regressive Virtual Metric Learning*, NeurIPS 2015.

   V3 n'est donc pas un objectif nouveau ; c'est une **application** de cet objectif au stacking local. C'est une contribution d'intégration, plus étroite que ce que le tableau initial laissait penser.

## Décision

- **Ne pas retenir V3 comme contribution principale.** Elle est dominée par V1 empiriquement et plus faible en nouveauté.
- **La garder comme ablation négative** dans le rapport : elle montre que le proxy de compétence n'est pas un détail d'implémentation, mais un choix de réduction de variance. C'est un résultat honnête et utile.
- **Ne pas brancher V3 dans le runner principal.** Le code et les tests restent disponibles pour l'ablation.
- **Prochaine variante : V2 uniquement**, et seulement si le temps le permet. Sinon, consolider V1 et le protocole (3 répétitions, baselines, tests de significativité, méthode publiée proche).

## Fichiers

| Fichier | Rôle |
|---|---|
| `gain_metric_local_stacking.py` | Implémentation de V3 |
| `run_gain_metric_experiments.py` | Runner de criblage V1 vs V3 |
| `tests/test_gain_metric.py` | 3 tests de V3 (12 tests au total avec la référence) |
| `results/gain_metric/` | CSV bruts des runs V3 |
