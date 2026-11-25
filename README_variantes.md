# Projet ML — plan de travail du groupe

**Cours :** ING IA 4 — USTHB — Dr. Bida
**Échéance :** 1er janvier 2027, 23:59
**Groupe :** 8 personnes (à compléter)
**Dossier :** `C:\Users\PC\Downloads\test`

> Ce README est le point d'entrée pour diviser le travail. Il contient le tableau des variantes, le protocole, la carte des fichiers et les lots de travail. La colonne **Responsable** est à remplir en réunion.

---

## 1. Objectif

Produire un projet de régression ensembliste qui respecte les contraintes du cours :

- au moins **5 jeux UCI** de régression ;
- validation croisée **répétée au moins 3 fois** pour les résultats finaux ;
- **nouveauté clairement énoncée et testée** ;
- **implémentation propre** de la partie nouvelle par le groupe ;
- **au moins 6 papiers** dans la revue de littérature ;
- comparaison au **stacking global**, à **au moins 2 baselines classiques** et aux **méthodes publiées les plus proches** ;
- **rapport honnête**, pertes et cas de repli inclus ;
- **pas de deep learning**, **pas de fuite de données**.

Le projet ne doit pas seulement avoir un petit gain : il doit aussi pouvoir expliquer **ce qui est nouveau** et **ce qui ne l'est pas**.

---

## 2. État actuel

### 2.1 Méthode de référence

`local_stacking.py` implémente le stacking local :

1. plusieurs experts sont entraînés et leurs prédictions hors pli (OOF) sont conservées ;
2. un stacking global apprend des poids non négatifs à partir de ces OOF ;
3. pour chaque requête, les poids sont réappris sur ses `k = 50` plus proches voisins, avec une pénalité ridge vers les poids globaux ;
4. le shrinkage `lambda` est choisi sur les données d'entraînement, avec un repli global possible ;
5. la règle stricte ne passe en local que si le gain dépasse son erreur standard.

### 2.2 Pistes déjà testées

1. **Feature-error gating** : un modèle apprend à moduler la correction locale à partir des variables.
2. **Stability gating** : le voisinage est bootstrappé et la correction est réduite quand il est instable.
3. **Métrique de compétence** : une métrique diagonale est apprise à partir des erreurs OOF, puis utilisée pour le KNN du stacking local.

La piste actuellement retenue comme prototype est la **métrique de compétence** (`competence_metric_local_stacking.py`).

### 2.3 Résultats actuels

RMSE moyen par pli. Les chiffres viennent de `reports/competence_metric_report.md` et des CSV de `results/competence_metric/`.

| Jeu | Global | Local euclidien | Métrique compétence | Lecture |
|---|---:|---:|---:|---|
| Simulation régionale + 25 variables bruitées | 1,5913 | 1,5923 | **1,5645** | +1,7 % vs global ; 4 victoires / 5 nuls / 1 défaite vs euclidien |
| Simulation homogène + 25 variables bruitées | 0,4971 | 0,4971 | 0,4971 | aucun gain, repli global |
| UCI CCPP | 3,3666 | 3,3668 | **3,3497** | +0,5 % vs global ; 4 plis sur 5 |
| UCI Student Performance — math | 1,6646 | 1,6646 | 1,6646 | repli global |
| UCI Bike Sharing — quotidien | 655,93 | 644,54 | **643,99** | +1,8 % vs global, mais +0,08 % vs euclidien ; 1 pli sur 10 |
| UCI Appliances Energy | 86,6211 | 86,6211 | 86,6211 | repli global |
| UCI Concrete Slump Test | 7,0821 | 7,0821 | 7,0821 | repli global |

**Lecture honnête :**

- le gain spécifique à la métrique est réel sur CCPP et sur la simulation régionale ;
- sur Bike, l'essentiel du gain vient du **local stacking seul**, pas de la métrique ;
- sur Student, Appliances et Slump, le repli global est choisi ;
- la plupart des jeux ne sont qu'à **1 répétition** : ce n'est pas encore le protocole final ;
- aucun test de significativité n'est reporté ;
- la nouveauté n'est **pas prouvée** : la chaîne exacte n'a pas été trouvée dans la littérature, mais chaque brique existe déjà.

---

## 3. Tableau des variantes

C'est le tableau à utiliser pour diviser le travail. `V1` est la méthode actuelle ; `V2` à `V6` sont les pistes à tester.

| ID | Variante | Ce qui change | Nouveauté | Coût | Risque | Responsable | Statut |
|---|---|---|---|---|---|---|---|
| **V1** | Méthode actuelle | `argmin` de l'erreur normalisée → information mutuelle + permutations → métrique diagonale → KNN local → shrinkage + repli | Combinaison inédite (à confirmer) | Déjà implémenté | Gains modestes ; 1 répétition sur la plupart des jeux | *(à remplir)* | Prototype + rapport |
| **V2** | Compétence soft / margin-aware | On remplace l'étiquette `argmin` par un **rang attendu** sous une distribution softmax des erreurs normalisées ; la métrique est apprise par information mutuelle sur cette cible continue | Moyenne : nouveau target, même cadre MI + permutations | Faible | Gain probablement petit ; peut être vu comme une variante de mesures de compétence | *(à remplir)* | À implémenter — lot 1 |
| **V3** | Métrique orientée gain | Poids diagonaux optimisés pour **minimiser la RMSE du stacking local**, au lieu de la MI avec le proxy | **Faible-moyenne** : l'objectif downstream existe déjà (MLKR 2007, RVML 2015) ; seule l'application au stacking local est nouvelle | Moyen | Sur-apprentissage ; sélection instable des variables | *(à remplir)* | **Testé — rejeté comme contribution principale ; gardé en ablation** |
| **V4** | Métrique pairwise | La cible devient « quel expert bat quel autre, et de combien », au lieu du seul gagnant | Moyenne-high : ranking de compétence | Moyen | Proche de la littérature ranking / performance prediction | *(à remplir)* | Après V3 |
| **V5** | Criblage stabilisé | Bootstrap du test de permutations ; on ne garde que les variables sélectionnées de façon stable | Faible : robustesse, pas un nouveau principe | Faible | Peut être vu comme une correction méthodologique | *(à remplir)* | Ablation |
| **V6** | Voisinage par profils d'erreur | Les voisins sont choisis dans l'espace des **erreurs prédites** des experts, pas dans l'espace des features | Moyenne : voisinage différent | Moyen | Recouvrement possible avec MINE / performance prediction | *(à remplir)* | À vérifier avant investissement |

### Verdict V3 (testé le 25/11/2026)

**Question :** remplacer la cible proxy (identité du meilleur expert) par l'objectif final (RMSE du stacking local) améliore-t-il la métrique ?

**Résultat sur CCPP, 5 plis, `lambda = 1` forcé :**

| Méthode | RMSE moyen | vs global | vs V1 |
|---|---:|---:|---:|
| Global | 3,3844 | — | — |
| Local euclidien | 3,3777 | −0,20 % | +0,35 % |
| **V1 (MI/argmin)** | **3,3659** | **−0,55 %** | — |
| V3 (2 passes) | 3,3766 | −0,23 % | +0,32 % |
| V3 (6 passes, 3 découpages) | 3,3790 | −0,16 % | +0,39 % |

- V3 bat Euclide de 0,03 %, mais perd contre V1 dans 4 plis sur 5.
- Une recherche plus profonde **aggrave** V3 : elle sur-apprend la MSE de validation et devient pire qu'Euclide.
- Avec la sélection honnête du shrinkage, V3 choisit le **repli global 5 fois sur 5** ; V1 reste locale 2 fois sur 5 et bat le global.
- Sur la simulation régionale, V3 trouve la bonne variable (variable 0) mais avec un poids trop faible et en gonflant du bruit ; la métrique apprise fait 1,7718 contre 1,7702 pour Euclide, alors qu'une métrique oracle atteint 1,7505.

**Pourquoi :** la MSE finale est une cible très bruitée ; l'identité du meilleur expert est un proxy de moindre variance. Le stacking local étant déjà régularisé vers les poids globaux, l'effet de la métrique est de second ordre.

**Nouveauté :** apprendre une métrique en minimisant l'erreur de régression existe déjà (MLKR 2007 ; RVML 2015). V3 est une application au stacking local, pas un objectif nouveau.

**Décision :** ne pas retenir V3 comme contribution principale. La garder comme **ablation négative** : elle montre que le proxy n'est pas un raccourci, mais un choix de réduction de variance. Détails dans [`reports/gain_metric_ablation.md`](./reports/gain_metric_ablation.md).

### Priorités

- **Lot 1 : V2 seule.** V2 est rapide et teste si l'`argmin` est le goulot d'étranglement. V3 a été testée et rejetée comme contribution principale (voir le verdict ci-dessus) ; elle reste une ablation.
- **V4** seulement si V2 est prometteuse : le pairwise peut améliorer, mais il ajoute du risque.
- **V5** en ablation de robustesse, pas comme contribution principale.
- **V6** à vérifier bibliographiquement avant d'investir : si le recouvrement avec la prédiction de performance est trop fort, on l'écarte.

---

## 4. Décisions de cadrage

1. **Nouveauté d'abord** : on élimine les variantes trop proches de Zhou & Lin (2026), Warner et al. (2026), Wakayama & Sugasawa (2025), Coscrato et al. (2020) et Moura et al. (2019–2021), même si elles gagnent plus.
2. **Le gain départage** ensuite, entre variantes déjà jugées originales.
3. **Niveau de nouveauté ouvert** : combinaison inédite, composant nouveau, ou principe nouveau selon ce qui tient empiriquement.
4. **Pas de sollicitation de l'enseignant** : on assume une nouveauté de combinaison, avec une formulation prudente et le risque documenté.
5. **Environ 1 semaine de tests**, puis **3 semaines pour le rapport**.
6. **5 UCI × 5 plis × 3 répétitions** pour la confirmation finale.

---

## 5. Protocole

### 5.1 Criblage

- Jeux : simulation régionale + 25 bruit, UCI CCPP, UCI Bike Sharing.
- Validation : 5 plis × 1 répétition.
- Variantes : V1, V2, V3 (puis V4/V5 si le temps).
- Baselines : global, local euclidien.
- But : classer les variantes, pas conclure.

### 5.2 Confirmation

- Jeux : les 5 UCI du rapport (CCPP, Student, Bike, Appliances, Slump) + les 2 simulations de contrôle.
- Validation : 5 plis × **3 répétitions**.
- Variantes : la meilleure variante et un challenger.
- Baselines : meilleur expert seul, moyenne simple, stacking global, local euclidien, méthode actuelle.
- Méthode publiée proche : FWLS (Feature-Weighted Linear Stacking) ou une version simplifiée de MINE, à défaut une variante DES.
- But : résultat défendable pour le rapport.

### 5.3 Métriques

- RMSE moyen et écart-type par pli ;
- victoires / nuls / défaites contre le global et contre le local euclidien ;
- taux de sélection locale ;
- nombre de variables actives dans la métrique ;
- temps d'entraînement et de prédiction.

### 5.4 Règle d'arrêt

On ne garde une variante que si :

- elle bat la méthode actuelle sur **au moins 2 des 3 jeux de criblage** ;
- elle ne dégrade pas le contrôle négatif (simulation homogène) ;
- elle reste défendable en nouveauté après la revue de littérature.

Sinon, on revient à V1, qui reste le fallback déjà documenté.

### 5.5 Anti-fuite

- La métrique, le shrinkage et le repli sont choisis **uniquement dans les plis d'entraînement**.
- Les plis externes ne servent jamais à choisir une variante, un seuil ou un hyperparamètre.
- Les résultats de criblage ne doivent pas être réutilisés pour choisir les hyperparamètres de la confirmation.
- Si un jeu sert au criblage, il ne doit pas servir à sélectionner la variante finale **et** à la confirmer sans le dire.

---

## 6. Répartition du travail

Charge indicative : environ **17 jours-personne**, soit ~2 jours par membre. À ajuster en réunion.

| WP | Tâche | Fichiers / livrable | Charge | Responsable | Dépend de |
|---|---|---|---:|---|---|
| **WP0** | Coordination : planning, README, réunions, intégration | `README_variantes.md` | 0,5 j | *(à remplir)* | — |
| **WP1** | Revue de littérature : 6+ papiers, tableau, note de nouveauté | `reports/`, note d'une page | 2 j | *(à remplir)* | — |
| **WP2** | V2 soft / margin-aware : implémentation + tests | `competence_metric_local_stacking.py`, `tests/` | 2 j | *(à remplir)* | WP1 |
| **WP3** | V3 métrique orientée gain : implémentation + tests | nouveau module, `tests/` | 3 j | *(à remplir)* | WP1 |
| **WP4** | V4 pairwise (si V3 prometteuse) | nouveau module, `tests/` | 2 j | *(à remplir)* | WP3 |
| **WP5** | Benchmark : runner variants, baselines, 3 répétitions, stats | `run_competence_metric_experiments.py` | 3 j | *(à remplir)* | WP2, WP3 |
| **WP6** | Rapport : rédaction, figures, limites, formulation de nouveauté | `reports/` | 3 j | *(à remplir)* | WP5 |
| **WP7** | Vérification : tests, seeds, CSV, anti-fuite, relecture | `tests/`, `results/` | 1,5 j | *(à remplir)* | WP5 |

### Répartition suggérée pour 8 personnes

- 1 personne : WP0 + WP7 (coordination et vérification) ;
- 1 personne : WP1 (littérature et nouveauté) ;
- 1 personne : WP2 (V2) ;
- 1 à 2 personnes : WP3 (V3) ;
- 1 personne : WP4 (V4) si V3 prometteuse ;
- 2 personnes : WP5 (benchmark) ;
- 1 à 2 personnes : WP6 (rapport).

---

## 7. Fichiers du projet

| Fichier / dossier | Rôle | Statut |
|---|---|---|
| `local_stacking.py` | Méthode de référence : stacking global + local + shrinkage + règle stricte | **Ne pas modifier sans accord** |
| `competence_metric_local_stacking.py` | Prototype actuel de la métrique de compétence | Base pour V2/V3 |
| `run_competence_metric_experiments.py` | Runner de validation imbriquée du prototype | À étendre pour les variantes |
| `gain_metric_local_stacking.py` | Variante V3 : métrique diagonale optimisée pour la RMSE locale | **Ablation** — ne pas retenir comme contribution |
| `run_gain_metric_experiments.py` | Runner de criblage V1 vs V3 | Ablation |
| `tests/test_gain_metric.py` | 3 tests de V3 | Passent |
| `results/gain_metric/` | CSV bruts du criblage V3 | **Ne pas modifier** |
| `reports/gain_metric_ablation.md` | Rapport d'ablation V3 | Référence |
| `stability_gated_local_stacking.py` | Experts, OOF et variante stability gate | Importé par le prototype |
| `feature_gated_stacking.py` | Ancienne variante feature-error gate | Comparaison, pas la nouveauté |
| `run_stability_gating_experiments.py` | Chargeurs UCI + simulations + ancien harness | Importé par le runner |
| `compare_on_dataset.py` | Comparaison repeated CV + Wilcoxon de la référence | Utile pour les baselines |
| `tests/test_local_stacking.py` | 9 tests de la méthode de référence | À compléter |
| `reports/` | Rapports actuels et antérieurs | À mettre à jour |
| `results/` | CSV bruts des expériences | **Ne pas modifier** |
| `docs/` | PDF du cours (feuille de route, synthèse) | Référence |
| `README.md` | README de la méthode de référence | Ne pas confondre avec ce plan |
| `README_variantes.md` | Ce fichier : plan de travail du groupe | À maintenir |

### Dettes techniques connues

1. `run_competence_metric_experiments.py` appelle `predict_components` même quand le repli global est choisi ; il faut brancher `predict_global` pour éviter la recherche de voisins.
2. Aucun test ne couvre `CompetenceMetricLocalStack` ; ajouter des tests de non-régression.
3. Les valeurs par défaut de la classe (`permutations=24`, `metric_strength=0.75`) ne correspondent pas à celles du runner (`12`, `0.5`) ; aligner ou documenter.
4. La plupart des jeux ne sont qu'à 1 répétition ; la confirmation doit être relancée en 3 répétitions.
5. Le KNN par requête reste coûteux ; le repli global doit court-circuiter la recherche de voisins.

---

## 8. Calendrier

| Période | Objectif |
|---|---|
| **J1–J2** | WP1 + implémentation V2/V3 + tests |
| **J3–J4** | Criblage sur 3 jeux (5 plis × 1 répétition) |
| **J5** | Confirmation sur 5 UCI × 5 plis × 3 répétitions |
| **J6–J7** | Analyse, mise à jour du rapport, gel des résultats |
| **Semaines 2–4** | Rapport final, figures, vérification, relecture |
| **31 décembre** | Buffer |
| **1er janvier 2027, 23:59** | Dépôt |

---

## 9. Règles du groupe

- Ne jamais choisir un hyperparamètre, un seuil ou une variante sur les plis externes.
- Toujours rapporter les pertes, les cas de repli et les jeux où la méthode ne fait rien.
- Utiliser **3 répétitions** pour les résultats finaux.
- Fixer les seeds et sauvegarder les CSV.
- Citer les papiers ; ne pas présenter une brique connue comme nouvelle.
- Pas de deep learning.
- Chaque membre doit pouvoir expliquer sa partie et défendre les choix.
- Ne pas modifier les CSV bruts ; créer de nouveaux fichiers pour les nouveaux runs.
- Si une variante échoue, le dire : un résultat négatif honnête vaut mieux qu'un gain non reproductible.

---

## 10. Formulation de nouveauté (à utiliser avec prudence)

> « Nous proposons une variante de stacking local pour la régression dans laquelle la métrique de voisinage est apprise à partir des profils d'erreur out-of-fold via un criblage par information mutuelle à seuil de permutation corrigé, avec shrinkage vers les poids globaux et repli validé. À notre connaissance, cette combinaison précise n'a pas été publiée, mais chacun de ses composants est établi et des travaux récents en couvrent des versions partielles. »

**Ne pas écrire** « nouvelle méthode » sans cette nuance. La revue de littérature n'a pas trouvé la chaîne exacte, mais elle n'a pas prouvé son absence.

Papiers à citer comme limites :

- Zhou & Lin, *When Does Dynamic Ensembling Pay Off?*, arXiv 2026.
- Warner et al., *Validation-Conditioned Dynamic Ensemble Regression*, IEEE ICAD 2026.
- Wakayama & Sugasawa, stacking covariate-dépendant, 2025.
- Coscrato et al., *NN-Stacking*, Neurocomputing 2020.
- Moura et al., MINE, *Information Sciences* 2020 ; mesures de compétence, 2019–2021.
- Pérez-Godoy et al., DESReg, *Neurocomputing* 2024.
- Cruz et al., revue DES, *Information Fusion* 2017.
- Sill et al., *Feature-Weighted Linear Stacking*, 2009.
- Weinberger & Tesauro, *Metric Learning for Kernel Regression*, AISTATS 2007.
- Perrot & Habrard, *Regressive Virtual Metric Learning*, NeurIPS 2015.

---

## 11. Prochaines étapes

- [ ] Réunion : remplir la colonne **Responsable** du tableau des variantes et des WP.
- [ ] WP1 : tableau de littérature + note de nouveauté d'une page.
- [ ] WP2 : implémenter V2 + tests.
- [x] WP3 : implémenter V3 + tests.
- [x] Criblage V3 sur CCPP + simulation régionale ; verdict : rejetée comme contribution principale.
- [ ] Décider si V2 vaut le temps restant ; sinon consolider V1 et le protocole.
- [ ] WP5 : lancer le criblage V1/V2 sur les 3 jeux (si V2 est retenue).
- [ ] Intégrer l'ablation V3 dans le rapport final.
- [ ] Décider V4/V5/V6 seulement si V2 est prometteuse.
- [ ] Confirmation 5 UCI × 5 plis × 3 répétitions.
- [ ] Mettre à jour `reports/competence_metric_report.md`.
- [ ] Relecture finale et dépôt.
