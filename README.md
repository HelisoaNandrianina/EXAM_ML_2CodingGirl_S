# **Rapport de Projet — Atlantic Haven Hotels**

## **Examen Final Machine Learning & Data Science — M1**

Réalisé au sein de **ISPM — Madagascar** ([www.ispm-edu.com](https://www.ispm-edu.com))

---

### **1. Informations sur le Groupe**

#### Membre 1

- nom : RADIMIMANANA
- prénom(s) : Nandrianina Rivomahefa Helisoa
- classe : IGGLIA 4
- numéro : 05
- rôle : Responsable EDA & baseline & présentation video (analyse exploratoire, valeurs manquantes, régression logistique de référence)

#### Membre 2

- nom :RAKOTOARIMALALA
- prénom(s) : Fanomezantsoa Ianissa
- classe : IGGLIA 4
- numéro : 55 
- rôle : Responsable modélisation & feature engineering (validation temporelle, RandomForest/HistGradientBoosting, étude d'ablation)

#### Membre 3

- nom :RAKOTO
- prénom(s) : Jean De Rivaze Jocelyn 
- classe : IGGLIA 4
- numéro : 38
- rôle : Responsable analyse d'erreurs, README (interprétation, faux positifs/négatifs, rédaction du rapport)

---

### **2. Résumé du Travail**

#### Problématique

Atlantic Haven Hotels exploite des établissements dans dix régions italiennes, avec des réservations issues de canaux très variés (plateforme en ligne, agence, téléphone, entreprise, hôtel direct). Une annulation tardive laisse une chambre inoccupée et perturbe la planification opérationnelle. L'enjeu est donc de détecter, dès la réservation, les dossiers présentant un risque élevé d'annulation  suffisamment tôt pour permettre une action commerciale proportionnée  sans pénaliser les clients qui, in fine, maintiennent leur séjour.

#### Méthodologie adoptée

1. **EDA** : cible déséquilibrée (~26 % d'annulations), valeurs manquantes traitées selon leur sémantique (`enfants`/`demandes_speciales` → 0, `agent_id` manquant → réservation directe, `prix_moyen_nuit_eur` → imputation médiane apprise sur train), corrélations numériques faibles, signal concentré sur les variables commerciales (`type_acompte`, `canal_reservation`, `tarif_remboursable`).
2. **Validation temporelle** : le jeu de test est postérieur au train (aucun chevauchement de dates) — split chronologique au 80ᵉ centile de `date_reservation` (pas de K-Fold aléatoire, qui aurait fuité de l'information future).
3. **Baseline** : régression logistique (`class_weight="balanced"`), prétraitements appris uniquement sur le train.
4. **Comparaison de modèles** : régression logistique, Random Forest et HistGradientBoosting, sur le même split temporel, avec seuil de décision optimisé (F1 maximal) pour chaque modèle plutôt qu'un seuil fixe à 0.5.
5. **Feature engineering** : variables de composition de séjour, écart de prix à la moyenne de la ville, historique client, et encodages cible (taux d'annulation moyen appris sur train) pour `type_acompte`×`tarif_remboursable`, `canal_reservation`, `segment_client`. Gain vérifié par étude d'ablation et par comparaison avant/après FE sur les trois familles de modèles.
6. **Modèle final** : HistGradientBoosting + feature engineering, seuil 0.413, ré-entraîné sur les 8000 lignes d'entraînement avant prédiction sur le test.

#### Résultats obtenus

- Meilleur F1-score de validation (temporelle) : **0.483** (HistGradientBoosting + feature engineering, seuil 0.413), contre **0.470** pour la baseline régression logistique (features brutes, seuil par défaut 0.5).
- Precision ≈ 0.35, rappel ≈ 0.77, ROC-AUC ≈ 0.65 sur le fold de validation (1605 réservations les plus récentes du train).
- Découverte majeure : le risque d'annulation est porté avant tout par les **conditions commerciales** de la réservation (`type_acompte`, `canal_reservation`, `tarif_remboursable`), pas par la destination, la saison ou la composition du séjour — les corrélations des variables numériques brutes avec la cible dépassent rarement 0.11.

#### Mots-clés

classification binaire, annulation hôtelière, validation temporelle, F1-score, feature engineering, encodage cible, HistGradientBoosting, déséquilibre de classes

---

### **3. Contenu du Repository**

- **notebook.ipynb** : code complet de l'EDA, du prétraitement, de la modélisation et de l'évaluation (exécuté de bout en bout, graine fixée) ;
- **submission.csv** : prédictions sur `reservations_test.csv` ;
- **README.md** : présent rapport ;
- **requirements.txt** : dépendances nécessaires à la reproduction du projet ;
- **LICENSE** : licence MIT applicable au code de ce dépôt (voir section 9) ;
- **ressources/** : données et canevas fournis par l'organisation.

**🔗 Liens utiles :**

- [**LIEN VERS LA VIDÉO DE PRÉSENTATION** — https://drive.google.com/file/d/1ZPnaFtKZU0h9jU_MFZCGcnND6aogtGnL/view?usp=drivesdk]

---

### **4. Résultats de Modélisation**

Tous les scores ci-dessous sont mesurés sur le **même fold de validation temporelle** (les 1605 réservations les plus récentes du train, `date_reservation` ≥ 80ᵉ centile), avec un seuil de décision optimisé (F1 maximal) propre à chaque ligne.

| Modèle | Paramètres principaux | F1-score | Précision | Rappel | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Régression logistique — baseline (features brutes, seuil 0.5) | `class_weight="balanced"` | 0.455 | 0.363 | 0.609 | 0.647 |
| Régression logistique + FE | `class_weight="balanced"`, seuil 0.358 | 0.471 | 0.331 | 0.819 | 0.651 |
| Random Forest + FE | `n_estimators=300, max_depth=6, min_samples_leaf=30`, seuil 0.465 | 0.482 | 0.352 | 0.764 | 0.651 |
| **Modèle final — HistGradientBoosting + FE** | `max_iter=100, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0, learning_rate=0.05`, seuil **0.413** | **0.483** | 0.352 | 0.769 | 0.649 |

**Seuil de décision retenu :** 0.413 (au lieu de 0.5), déterminé en maximisant le F1 sur la courbe précision-rappel du fold de validation.

**Justification du choix du modèle final :**

HistGradientBoosting avec feature engineering obtient le meilleur F1 au seuil optimal (0.483) parmi les modèles testés, avec un écart faible mais présent par rapport à Random Forest (0.482) et à la régression logistique (0.471). Les trois modèles ont été volontairement régularisés (arbres peu profonds / peu de feuilles, `min_samples_leaf` élevé) après un constat expérimental : sur seulement 6400 lignes d'entraînement et un signal modéré (ROC-AUC plafonnant autour de 0.65 quel que soit le modèle), des arbres profonds par défaut sur-apprennent et généralisent moins bien. HistGradientBoosting est également retenu pour sa gestion native des valeurs manquantes et sa robustesse générale sur données tabulaires de taille modeste. La stabilité (F1 proche entre HGB et Random Forest) et l'absence de gain démesuré d'un modèle sur l'autre confortent le choix d'un modèle correctement régularisé plutôt que le plus complexe disponible.

---

### **5. Réponses aux Questions d'Analyse**

#### **Q1. Pourquoi utilise-t-on principalement le F1-score plutôt que l'accuracy pour cette tâche ?**

La cible est déséquilibrée (~74 % de réservations maintenues, ~26 % d'annulations). Un modèle prédisant systématiquement « maintenue » atteindrait ~74 % d'accuracy tout en étant inutile opérationnellement (rappel nul sur la classe qui intéresse l'hôtel). Le F1-score, moyenne harmonique de la précision et du rappel sur la classe « annulation », pénalise ce comportement dégénéré et reflète mieux la capacité réelle du modèle à identifier les annulations sans produire un nombre excessif de fausses alertes.

#### **Q2. Dans ce contexte, qu'est-ce qui est le plus grave : un faux positif ou un faux négatif ?**

- **Faux négatif** (annulation réelle prédite comme maintenue) : l'hôtel ne prend aucune mesure et découvre l'annulation tardivement chambre inoccupée, perte de revenu, replanification en urgence.
- **Faux positif** (réservation maintenue prédite comme annulée) : l'hôtel engage une action préventive (contact, relance) auprès d'un client qui honore finalement son séjour coût faible si l'action reste une simple relance, mais risque réel si elle est intrusive (ex. sur-réservation de sa chambre).

Le faux négatif est généralement **plus coûteux opérationnellement** (perte sèche de revenu, replanification tardive) qu'un faux positif traité par une action réversible et à faible friction. C'est ce qui justifie de privilégier un seuil favorisant le rappel (0.413, en dessous de 0.5) plutôt qu'un seuil maximisant uniquement la précision une réponse nuancée reste toutefois nécessaire : si l'action déclenchée par un faux positif est intrusive (overbooking), son coût peut devenir comparable à celui d'un faux négatif.

#### **Q3. Quelles variables créées par feature engineering ont le plus amélioré votre modèle par rapport à la régression logistique de référence ?**

L'étude d'ablation (notebook, §4.2) montre que l'**historique client** (`taux_annulation_passee`, `client_nouveau`) et les **encodages de risque** (`risque_acompte_score`, `risque_canal_score`, `risque_segment_score` taux d'annulation moyen par catégorie, appris sur le train) apportent les gains individuels les plus nets sur le F1 au seuil optimal et l'average precision. Les variables calendaires (mois/jour de semaine/saison d'arrivée) et les indicateurs binaires grossiers testés (délai long, week-end en haute saison) n'ont apporté aucun gain mesurable et ont été écartés du jeu final.

#### **Q4. Pourquoi un découpage aléatoire simple peut-il produire une évaluation trompeuse sur ce dataset ?**

Le jeu de test (`date_reservation` du 24/05/2025 au 31/12/2025) est strictement postérieur au jeu d'entraînement (01/01/2023 au 24/05/2025), sans chevauchement. Le taux d'annulation mensuel varie dans le temps. Un K-Fold aléatoire mélangerait des réservations « futures » dans les folds d'entraînement et permettrait au modèle d'apprendre indirectement des tendances qu'il ne connaîtrait pas en production, donnant une estimation de performance optimiste et non représentative. Le protocole retenu trie les données par `date_reservation` et réserve les 20 % les plus récents (à partir du 28/11/2024) comme validation, reproduisant la même logique temporelle que le split train/test officiel.

#### **Q5. Quels profils ou scénarios de réservation sont les plus fréquemment associés aux annulations dans vos analyses ?**

- Réservations **sans acompte** (`type_acompte = "aucun"`) et à **tarif remboursable** : combinaison la plus à risque (34 % d'annulation pour "aucun acompte", vs 10 % pour "acompte total").
- Réservations passées par une **plateforme en ligne** (30 %) ou une **agence** (28 %), contre 14 % pour les réservations « entreprise ».
- Segments **groupe** (31 %) et **famille** (28 %), contre 23 % pour le segment affaires.
- Réservations avec un **délai de réservation long** (réservées très à l'avance) et **sans historique client** (`reservations_passees = 0`).

*Ces facteurs relèvent des conditions commerciales et du contexte de réservation, pas d'une caractéristique intrinsèque d'une région ou d'une population. les variables `region_hotel`, `ville` et `type_destination` encodent d'ailleurs la même partition en 10 groupes dans ce jeu synthétique.*

#### **Q6. Comment votre pipeline traite-t-il les valeurs manquantes et les catégories jamais observées pendant l'entraînement ?**

- Valeurs manquantes numériques (`prix_moyen_nuit_eur`) : imputation par la **médiane apprise sur le fold d'entraînement** (`SimpleImputer` scikit-learn), jamais recalculée sur validation/test.
- Valeurs manquantes sémantiques (`enfants`, `demandes_speciales` → 0 ; `agent_id` → "DIRECT" ; `marche_origine` → "inconnu") : remplacement par une constante documentée, appliqué identiquement à train/val/test.
- Catégories jamais vues à l'entraînement : `OneHotEncoder(handle_unknown="ignore")` — la ligne reçoit un vecteur nul pour cette variable plutôt que de provoquer une erreur.
- L'ensemble de ces traitements (imputation, encodages de risque, moyennes de prix) est encapsulé dans des `Pipeline`/`ColumnTransformer` scikit-learn dont le `fit()` ne porte que sur les données d'entraînement — aucune fuite d'information du test.

#### **Q7. Selon vous, quelle action l'hôtel devrait-il entreprendre lorsqu'une réservation en cours présente une forte probabilité d'annulation ?**

Une action proportionnée et réversible plutôt qu'une décision automatique : email/SMS de confirmation, appel de courtoisie, offre incitative légère (surclassement, flexibilité) pour les profils identifiés à risque (acompte faible, plateforme en ligne, sans historique). Un overbooking contrôlé peut être envisagé uniquement pour les probabilités très élevées proches de la date d'arrivée, en s'appuyant sur des statistiques de no-show historiques jamais une annulation unilatérale de la réservation du client.

#### **Q8. Votre modèle présente-t-il des performances comparables selon les régions ou les types de destination ?**

Non : le F1 par région varie de ~0.41 (Campania, n=169) à ~0.61 (Sicilia, n=121) sur le fold de validation. Cette dispersion est en grande partie un **effet de taille d'échantillon** (sous-groupes de 68 à 258 lignes, F1 intrinsèquement instable sur un tel volume) plutôt qu'une différence de fond dans le comportement d'annulation propre à une région. Elle ne doit pas être interprétée comme une caractéristique intrinsèque d'une population ou d'un territoire.

#### **Q9. Analyse des erreurs**

- **Faux positifs** : profils quasi systématiquement `type_acompte = "aucun"`, délai de réservation long, sans historique (`reservations_passees = 0`) un profil statistiquement à risque que le modèle signale à juste titre, mais qui ne s'est pas concrétisé en annulation pour ces réservations précises.
- **Faux négatifs** : profils `type_acompte = "total"` ou `"partiel"` (normalement protecteurs), certains avec un historique de 1 à 2 annulations passées que le modèle sous-pondère face au poids dominant de `type_acompte`.
- **Raisons possibles** : le modèle apprend une hiérarchie de risque dominée par 2-3 variables commerciales fortes, ce qui le rend moins sensible aux cas où ces variables « protectrices » sont contredites par un historique client à risque.
- **Piste d'amélioration** : tester une variable croisant explicitement `type_acompte` et `annulations_passees` (plutôt que le seul `taux_annulation_passee` agrégé), et collecter davantage de features comportementales (délai entre modifications, historique de paiement) pour mieux discriminer ces cas limites.

---

### **6. Conclusion et Recommandations**

Le modèle final (HistGradientBoosting + feature engineering, seuil 0.413) atteint un F1 de validation de 0.483, en amélioration mesurable et reproductible par rapport à la baseline régression logistique (0.470). Le signal disponible dans ce jeu de données reste modéré (ROC-AUC ≈ 0.65 quel que soit le modèle), porté essentiellement par les conditions commerciales de la réservation. Le modèle est adapté à un usage d'**aide à la décision** (priorisation des relances, ciblage des actions commerciales) plutôt qu'à une automatisation complète des décisions d'annulation.

**Recommandation opérationnelle finale :** utiliser la probabilité produite comme signal de priorisation pour des actions réversibles et proportionnées (relance, offre de flexibilité), réserver l'overbooking contrôlé aux probabilités extrêmes proches de la date d'arrivée, et re-calibrer périodiquement le seuil de décision à mesure que de nouvelles données de test/production deviennent disponibles.

---

### **7. Reproductibilité**

- version de Python : 3.12 (compatible ≥ 3.10)
- principales bibliothèques et versions : pandas ≥ 2.0, numpy ≥ 1.26, scikit-learn ≥ 1.3, matplotlib ≥ 3.8, seaborn ≥ 0.13 (voir `requirements.txt`)
- graine(s) aléatoire(s) : `SEED = 42`, fixée pour tous les modèles et splits
- commande ou procédure d'exécution : `jupyter nbconvert --to notebook --execute --inplace notebook.ipynb` (ou exécution cellule par cellule depuis un noyau vierge)
- durée approximative d'entraînement : < 1 minute (ensemble du notebook, EDA incluse)
- environnement utilisé : local (Python/Anaconda)

---

### **8. Bibliographie**

- Documentation scikit-learn — `Pipeline`, `ColumnTransformer`, `HistGradientBoostingClassifier`, `permutation_importance` : [scikit-learn.org](https://scikit-learn.org)
- Sujet et canevas fournis par l'ISPM : `sujet.pdf`, `readme-model.md`, `data_dictionary.csv`
- Outil d'IA générative utilisé : Claude (Anthropic) assistance à la rédaction du pipeline scikit-learn et à la mise en forme du présent rapport ; toutes les métriques et sorties ont été vérifiées par exécution effective du notebook.

---

### **9. Licence**

Le code source de ce dépôt (notebook, scripts, documentation) est distribué sous **licence MIT** — voir le fichier [`LICENSE`](./LICENSE). Elle autorise la réutilisation, la modification et la redistribution du code, y compris à des fins commerciales, à condition de conserver la mention des auteurs.

Cette licence **ne couvre pas** le dossier [`ressources/`](./ressources/) : les données et le canevas fournis par l'ISPM restent des données **entièrement synthétiques, à usage strictement pédagogique**, sans droit de réutilisation commerciale accordé par ce dépôt.
