# Évaluation et baseline (Phase 0)

Ce module rejoue le moteur de prédiction sur des dates passées et compare
aux observations réelles pour produire des métriques de performance.

## Pourquoi

Avant de modifier l'algorithme (Phase 2 spatial layer, Phase 3 score de
pression), on a besoin d'une **baseline mesurable** : sans chiffres précis
sur la qualité du modèle actuel, on ne saura pas si les évolutions futures
améliorent ou dégradent les prédictions.

## Architecture

```
pasdevelib/eval/
├── backtest.py     Rejoue predict() avec un historique tronqué (anti-triche)
├── baseline.py     Baseline climatology (moyenne par station, heure)
├── metrics.py      MAE, decision accuracy, Brier, calibration, coverage
└── runner.py       Orchestration et upload vers la release `eval`

scripts/
├── run_backtest_rolling.py    Quotidien, 7 derniers jours
└── run_backtest_full.py       Manuel, 90 jours en baseline figée

.github/workflows/
├── eval-rolling.yml           Cron 06h UTC quotidien
└── eval-full.yml              workflow_dispatch (manuel)
```

## Anti-triche temporelle

Le `predict.py` actuel exclut uniquement la date cible des candidats
analogues. Pour un backtest **honnête**, on doit exclure **toutes les
dates ≥ cible** : sinon, l'algorithme pourrait piocher comme analogue
un jour postérieur à la cible (ce qu'il n'aurait pas pu faire en
production au moment de la prédiction).

C'est ce que fait `backtest.backtest_single_day()` : il filtre
`hourly_history` et `calendar_df` à `date < target_date` strictement.

## Métriques produites

### Régression sur `fill_rate`

- **MAE** (Mean Absolute Error) : erreur absolue moyenne sur le `fill_rate`
  prédit (p50). Unité = part de la capacité (0-1). MAE = 0,1 = on se trompe
  en moyenne de 10% du nombre de docks.

### Décision binaire (la métrique vraiment utile pour l'utilisateur)

- **Decision Accuracy `velib`** : taux de bonnes décisions "y aura-t-il un
  vélo ?" (prédit `proba_velib > 0.5` vs réel `has_velib`).
- **Decision Accuracy `place`** : idem côté docks libres.

### Calibration

- **Brier score** : `mean((proba - réalité)²)`. Plus c'est bas, mieux c'est.
  - 0 = parfait
  - 0,25 = "je dis toujours 50%"
  - Au-delà = pire qu'une pièce.
- **Coverage 50%** : taux d'observations dans `[p25, p75]`. Devrait être
  proche de 0,50 si les quantiles sont bien calibrés. Plus proche de 1 = on
  surévalue l'incertitude. Plus proche de 0 = on la sous-évalue.

### Baseline climatology

Pour chaque (station, heure), la moyenne historique inconditionnelle
(ignore météo, calendrier, etc.). C'est le baseline le plus simple : si
notre k-NN ne le bat pas, il n'apporte aucune valeur.

**Critère de succès minimum** : `algo.MAE < baseline.MAE` ET
`algo.decision_accuracy > baseline.decision_accuracy` sur la fenêtre
de 7 jours.

## Stockage

Tout passe par une release GitHub `eval` (séparée de `aggregates` et
`history`) :

| Asset | Description | Mise à jour |
|---|---|---|
| `rolling_metrics.json` | Résumé global + détail journalier + top/bottom 10 | Quotidienne |
| `rolling_per_station.parquet` | Métriques par station × jour | Quotidienne |
| `rolling_calibration.parquet` | Buckets de calibration par jour | Quotidienne |
| `baseline_90d_metrics.json` | Snapshot 90 jours (figé) | Manuel |
| `baseline_90d_per_station.parquet` | Détail station 90j (figé) | Manuel |

## Repères pour interpréter les chiffres

Sans baseline historique, voici des repères absolus pour Vélib' Paris :

| Métrique | Mauvais | Acceptable | Bon |
|---|---|---|---|
| MAE fill_rate | > 0,25 | 0,15 - 0,25 | < 0,15 |
| Decision Accuracy | < 0,70 | 0,70 - 0,85 | > 0,85 |
| Brier | > 0,20 | 0,12 - 0,20 | < 0,12 |
| Coverage 50% | < 0,30 ou > 0,70 | 0,40 - 0,60 | 0,45 - 0,55 |

Ces seuils sont indicatifs : la vraie référence est l'écart par rapport à
la baseline climatology.

## Quand tourner quoi

- **Rolling 7j** : automatique tous les jours à 06:00 UTC. Suit le drift
  dans le temps.
- **Full 90j** : à lancer manuellement via Actions → "Eval full baseline" :
  - Une première fois maintenant pour figer le baseline d'avant Phase 2
  - Une seconde fois après chaque modification algorithmique majeure
  - Comparer les deux JSON pour voir l'impact

## Webapp

La dashboard `/internal/eval` (à venir) consomme `rolling_metrics.json`
côté client pour afficher l'évolution dans le temps. Elle est non listée
(noindex, hors sitemap) : accessible via URL directe uniquement.
