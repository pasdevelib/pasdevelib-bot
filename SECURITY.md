# Signalement de vulnérabilités

Si vous découvrez une vulnérabilité de sécurité dans ce projet, **n'ouvrez pas d'issue publique**.

Contactez-nous directement par email : hello@pasdevelib.fr

Nous nous engageons à répondre sous 72 heures et à corriger toute vulnérabilité confirmée dans les meilleurs délais.

## Périmètre

Ce dépôt contient le bot de collecte et de prédiction. Il tourne exclusivement via GitHub Actions et n'expose aucun service réseau directement.

Les données traitées sont des données publiques (API GBFS Vélib' Métropole, Open-Meteo). Aucune donnée personnelle n'est manipulée dans ce bot.

## Ce qui n'est pas dans le périmètre

- Vulnérabilités dans les dépendances tierces (signalez-les directement aux mainteneurs concernés)
- Problèmes liés à l'application web : voir [pasdevelib-webapp](https://github.com/tmcws2/pasdevelib-webapp)
