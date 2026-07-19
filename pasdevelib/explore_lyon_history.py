"""explore_lyon_history.py — Script exploratoire (temporaire).

Interroge le service REST rdata de data.grandlyon.com pour voir la forme
reelle des donnees disponibles sur jcd_jcdecaux.jcdvelov — objectif :
savoir si cette table contient un vrai historique temporel (plusieurs
lignes par station a des dates differentes) ou seulement l'etat courant,
et quels sont les noms de colonnes exacts a utiliser pour ecrire un vrai
script de recuperation retroactive.

Usage : python -m pasdevelib.explore_lyon_history

A supprimer une fois l'exploration terminee et le vrai script de backfill
ecrit (ce fichier n'est pas destine a rester dans le projet).
"""
from __future__ import annotations

import json

import requests

URL = "https://data.grandlyon.com/fr/datapusher/ws/rdata/jcd_jcdecaux.jcdvelov/all.json"


def main() -> None:
    print(f"[explore] GET {URL}?compact=false&maxfeatures=5")
    r = requests.get(URL, params={"compact": "false", "maxfeatures": 5}, timeout=30)
    print(f"[explore] status: {r.status_code}")
    print(f"[explore] content-type: {r.headers.get('content-type')}")

    try:
        data = r.json()
    except Exception as e:
        print(f"[explore] Reponse non-JSON ({e}), 2000 premiers caracteres:")
        print(r.text[:2000])
        return

    print("[explore] Cles racine:", list(data.keys()) if isinstance(data, dict) else type(data))

    results = data.get("results") if isinstance(data, dict) else None
    if results:
        print(f"[explore] {len(results)} resultats recus")
        print("[explore] Premier resultat (toutes cles) :")
        print(json.dumps(results[0], indent=2, ensure_ascii=False))
        if len(results) > 1:
            print("[explore] Deuxieme resultat (pour comparer les dates/timestamps) :")
            print(json.dumps(results[1], indent=2, ensure_ascii=False))
    else:
        print("[explore] Pas de champ 'results' — dump complet (tronque a 3000 caracteres) :")
        print(json.dumps(data, indent=2, ensure_ascii=False)[:3000])

    # Deuxieme essai : meme station a un autre "start" pour voir si les
    # dates changent (preuve d'un vrai historique vs un instantane unique).
    print()
    print("[explore] Deuxieme appel avec start=1000 (pour comparer) :")
    r2 = requests.get(URL, params={"compact": "false", "maxfeatures": 3, "start": 1000}, timeout=30)
    try:
        data2 = r2.json()
        results2 = data2.get("results") if isinstance(data2, dict) else None
        if results2:
            print(json.dumps(results2[0], indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[explore] Erreur second appel: {e}")


if __name__ == "__main__":
    main()
