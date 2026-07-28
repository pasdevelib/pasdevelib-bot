# pasdevelib-bot

[![Licence](https://img.shields.io/badge/licence-PolyForm%20Noncommercial-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Site](https://img.shields.io/badge/site-pasdevelib.app-6C4CF1)](https://pasdevelib.app)

Bot de collecte et de prédiction pour [PasDeVélib](https://pasdevelib.app) — une app citoyenne qui prédit la disponibilité des stations Vélib' Métropole.

Zéro serveur, zéro base de données propriétaire. Infrastructure entièrement gratuite.

## En bref

Ce dépôt récupère régulièrement les données publiques du réseau Vélib' Métropole, les archive, et calcule des prédictions de disponibilité consommées par [pasdevelib.app](https://pasdevelib.app).

Détails d'implémentation (méthode de prédiction, fréquences exactes, format de stockage) volontairement non documentés publiquement.

## Sources de données

- **Vélib' GBFS** — API publique, licence [ODbL](https://opendatacommons.org/licenses/odbl/)
- **Météo** — [Open-Meteo](https://open-meteo.com/), licence CC BY 4.0
- **Calendrier** — [`etalab/jours-feries-france`](https://github.com/etalab/jours-feries-france) et données vacances scolaires data.gouv.fr

## Signaler un problème

Ouvrez une [issue GitHub](https://github.com/pasdevelib/pasdevelib-bot/issues) pour un bug ou une question technique.

Pour les retours sur l'application, rendez-vous sur [pasdevelib.app/contributions](https://pasdevelib.app/contributions).

## Licence

Code source sous licence **PolyForm Noncommercial 1.0.0** — utilisation libre à des fins non commerciales. Voir [LICENSE](LICENSE).

Les données Vélib' Métropole utilisées sont publiées sous licence **ODbL** par Vélib' Métropole. PasDeVélib n'est pas affilié à Vélib' Métropole ni à Smovengo.
