# pasdevelib-bot

Bot de collecte et de prédiction pour [PasDeVélib.fr](https://pasdevelib.fr).

Récupère toutes les 5 minutes l'état des ~1500 stations Vélib' Métropole via l'API GBFS publique, archive l'historique sur GitHub Releases (zéro infra), enrichit avec la météo Open-Meteo, et publie une table de prédiction utilisée par le front Next.js.

## Architecture

```
   Vélib' GBFS API ──┐
                     ├──> pasdevelib-bot ──> GitHub Releases ──> pasdevelib-webapp
   Open-Meteo API ───┘                       (parquet files)        (Next.js)
```

Pas de base de données, pas de serveur. Tout passe par des assets de GitHub Releases téléchargés au build du front (ou lus à la volée via API GitHub).

## Stockage

Trois releases servent de "buckets":

| Release tag       | Contenu                                          | Mise à jour       |
|-------------------|--------------------------------------------------|-------------------|
| `live`            | `current_day.parquet`, `stations.json`           | Toutes les 5 min  |
| `history`         | `YYYY-MM-DD.parquet` par jour archivé            | 1x/jour à 03:00   |
| `aggregates`      | `medians.parquet`, `weather.parquet`             | 1x/semaine        |

## Modèle prédictif

Méthode des **journées analogues** (k-NN temporel), comme les forecasts retail.

Pour prédire la station `X` à demain 19h:

1. On récupère les conditions cibles : jour de semaine, mois, météo prévue (Open-Meteo Forecast)
2. On cherche dans l'historique les K=20 jours les plus similaires sur:
   - Même jour de semaine
   - Fenêtre saisonnière ±15 jours
   - Température ±3°C, pluie oui/non, vent ±5 km/h
   - Statut calendaire (vacances, jour férié)
3. Sur ces 20 voisins à 19h, on calcule:
   - `proba_velib` = % de voisins avec `num_bikes >= 1`
   - `proba_place` = % de voisins avec `num_docks_available >= 1`
   - Bande p25-p75 du remplissage pour le graphe

Implémenté dans `pasdevelib/predict.py`.

## Itinéraire vélo A → B

Module `pasdevelib/routing.py`. Trouve les meilleurs trajets Vélib' entre deux points en combinant:

1. Pré-filtrage Haversine des stations dans un rayon de marche (par défaut 600 m)
2. Filtre disponibilité : `proba_velib >= 0.5` au départ, `proba_place >= 0.5` à l'arrivée
3. Raffinement OpenRouteService : matrice de distances piétonnes réelles
4. Scoring `confiance / durée totale` puis tri
5. Génération de **deeplinks** Apple Plans (`maps.apple.com`) et Google Maps pour chaque segment (marche → vélo → marche)

Deux modes :

- **`TripMode.NOW`** : utilise les disponibilités live GBFS
- **`TripMode.LATER`** : utilise les prédictions du modèle analogue à l'heure cible

```python
from pasdevelib.routing import Coord, plan_trip, TripMode

plans = plan_trip(
    origin=Coord(48.8580, 2.3475),         # Châtelet
    destination=Coord(48.8531, 2.3692),    # Bastille
    stations=stations_json,
    departure_states=live_or_predicted_states,
    mode=TripMode.NOW,
    n_results=3,
    min_proba=0.5,
)
```

Variable d'env requise pour le raffinement ORS : `ORS_API_KEY` (gratuit, [openrouteservice.org](https://openrouteservice.org/dev/#/signup), 500 matrices/jour). Si absente, fallback automatique sur Haversine.

## Workflows GitHub Actions

| Workflow         | Trigger                  | Rôle                                       |
|------------------|--------------------------|--------------------------------------------|
| `scrape.yml`     | `*/5 * * * *`            | Fetch GBFS + append au parquet du jour     |
| `consolidate.yml`| `0 3 * * *`              | Archive le jour précédent dans `history`   |
| `weather.yml`    | `0 5 * * *`              | Append météo de la veille (Open-Meteo)     |
| `aggregate.yml`  | `0 4 * * 1`              | Reconstruit les tables de prédiction       |
| `bootstrap.yml`  | `workflow_dispatch`      | Import one-shot du dump lovasoa            |

## Setup initial

```bash
# 1. Clone + install
git clone https://github.com/pasdevelib/pasdevelib-bot
cd pasdevelib-bot
pip install -e .

# 2. Bootstrap historique (one-shot, ~10 min)
gh workflow run bootstrap.yml

# 3. Vérifier que le scrape tourne
gh run list --workflow=scrape.yml
```

## Sources de données

- **Vélib' GBFS**: `https://velib-metropole-opendata.smoove.pro/opendata/Velib_Metropole/`
- **Historique d'amorçage**: [`lovasoa/historique-velib-opendata`](https://github.com/lovasoa/historique-velib-opendata)
- **Météo**: [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api) + [Forecast API](https://open-meteo.com/en/docs)
- **Calendrier**: [`etalab/jours-feries-france`](https://github.com/etalab/jours-feries-france) + vacances scolaires data.gouv

## Licence

AGPL-3.0 (cohérence avec l'écosystème civic tech).
