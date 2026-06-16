# pasdevelib-bot

[![Licence](https://img.shields.io/badge/licence-PolyForm%20Noncommercial-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Scrape](https://github.com/pasdevelib/pasdevelib-bot/actions/workflows/scrape.yml/badge.svg)](https://github.com/pasdevelib/pasdevelib-bot/actions/workflows/scrape.yml)
[![Site](https://img.shields.io/badge/site-pasdevelib.fr-6C4CF1)](https://pasdevelib.fr)

Bot de collecte et de prédiction pour [PasDeVélib.fr](https://pasdevelib.fr) — une app citoyenne qui prédit la disponibilité des ~1 500 stations Vélib' Métropole à Paris.

Zéro serveur, zéro base de données. Tout passe par GitHub Actions et GitHub Releases.

## Architecture

```
Vélib' GBFS API ──┐
                  ├──> pasdevelib-bot ──> GitHub Releases ──> pasdevelib-webapp
Open-Meteo API ───┘                       (parquet files)       (Next.js, Vercel)
```

## Stockage

Les données sont stockées dans trois releases GitHub utilisées comme buckets :

| Release tag  | Contenu                                        | Fréquence         |
|--------------|------------------------------------------------|-------------------|
| `live`       | `current_day.parquet`, `stations.json`         | Toutes les 5 min  |
| `history`    | `YYYY-MM-DD.parquet` par jour                  | 1x/jour à 3h      |
| `aggregates` | `medians.parquet`, `analog_index.parquet`, ... | 1x/semaine        |
| `backup-*`   | Snapshot quotidien complet                     | 1x/jour à 3h      |

## Modèle prédictif

Méthode des **journées analogues** (k-NN temporel).

Pour prédire la station `X` à demain 19h :

1. On récupère les conditions cibles : jour de semaine, mois, météo prévue (Open-Meteo Forecast)
2. On cherche dans l'historique les K jours les plus similaires sur :
   - Jour de semaine, fenêtre saisonnière
   - Température, pluie, vent
   - Statut calendaire (vacances, jour férié)
3. Sur ces voisins analogues, on calcule :
   - `proba_velib` = % de voisins avec au moins un vélo
   - `proba_place` = % de voisins avec au moins une place
   - Quantiles p25/p50/p75 du taux de remplissage

Implémenté dans `pasdevelib/predict.py`.

## Itinéraire vélo A → B

Module `pasdevelib/routing.py`. Combine disponibilité live ou prédite, pré-filtrage géographique et deeplinks Apple Plans / Google Maps.

```python
from pasdevelib.routing import Coord, plan_trip, TripMode

plans = plan_trip(
    origin=Coord(48.8580, 2.3475),       # Châtelet
    destination=Coord(48.8531, 2.3692),  # Bastille
    stations=stations_json,
    departure_states=live_or_predicted,
    mode=TripMode.NOW,
)
```

Variable d'env optionnelle : `ORS_API_KEY` ([openrouteservice.org](https://openrouteservice.org/dev/#/signup), 500 matrices/jour). Sinon, fallback Haversine.

## Workflows GitHub Actions

| Workflow         | Déclencheur             | Rôle                                    |
|------------------|-------------------------|-----------------------------------------|
| `scrape.yml`     | `*/5 * * * *`           | Fetch GBFS + append au parquet du jour  |
| `consolidate.yml`| `0 3 * * *`             | Archive le jour dans `history`          |
| `backup.yml`     | `0 3 * * *`             | Snapshot quotidien complet              |
| `aggregate.yml`  | `0 4 * * 1`             | Reconstruit les tables de prédiction    |
| `forecast.yml`   | `0 4 * * *`             | Calcule les prévisions J+7              |
| `eval-daily.yml` | `30 3 * * *`            | Métriques de précision quotidiennes     |
| `bootstrap.yml`  | `workflow_dispatch`     | Import one-shot de l'historique initial |

## Installation

```bash
git clone https://github.com/pasdevelib/pasdevelib-bot
cd pasdevelib-bot
pip install -e .
```

Variables d'environnement requises dans GitHub Actions : `GITHUB_TOKEN` (automatique).

## Sources de données

- **Vélib' GBFS** : `https://velib-metropole-opendata.smoove.pro/opendata/Velib_Metropole/` — [ODbL](https://opendatacommons.org/licenses/odbl/)
- **Historique d'amorçage** : [`lovasoa/historique-velib-opendata`](https://github.com/lovasoa/historique-velib-opendata)
- **Météo** : [Open-Meteo](https://open-meteo.com/) — licence CC BY 4.0
- **Calendrier** : [`etalab/jours-feries-france`](https://github.com/etalab/jours-feries-france) + vacances data.gouv.fr

## Signaler un problème

Ouvrez une [issue GitHub](https://github.com/pasdevelib/pasdevelib-bot/issues) pour tout bug ou question technique relative au bot.

Pour les retours sur l'app, rendez-vous sur [pasdevelib.fr/contributions](https://pasdevelib.fr/contributions).

## Licence

Le code source de ce bot est publié sous licence **PolyForm Noncommercial 1.0.0** — utilisation libre à des fins non commerciales. Voir [LICENSE](LICENSE).

Les données Vélib' Métropole utilisées sont publiées sous licence **ODbL** par Vélib' Métropole. PasDeVélib n'est pas affilié à Vélib' Métropole ni à Smovengo.
