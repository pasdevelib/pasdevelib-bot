"""Stockage sur GitHub Releases via le CLI `gh`.

Convention :
- release `live`        : current_day.parquet, stations.json
- release `history`     : YYYY-MM-DD.parquet (un par jour)
- release `aggregates`  : medians.parquet, weather.parquet, calendar.parquet
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

REPO = os.environ.get("GITHUB_REPOSITORY", "pasdevelib/pasdevelib-bot")

# Mapping logique
RELEASE_LIVE = "live"
RELEASE_HISTORY = "history"
RELEASE_AGGREGATES = "aggregates"


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Exécute une commande shell."""
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def ensure_release(tag: str, title: str | None = None) -> None:
    """Crée la release si elle n'existe pas."""
    result = _run(["gh", "release", "view", tag, "--repo", REPO], check=False)
    if result.returncode != 0:
        _run([
            "gh", "release", "create", tag,
            "--repo", REPO,
            "--title", title or tag.capitalize(),
            "--notes", f"Auto-generated bucket for `{tag}` data.",
        ])


def upload_asset(tag: str, local_path: Path, asset_name: str | None = None) -> None:
    """Upload un fichier comme asset de la release. Écrase si existe."""
    name = asset_name or local_path.name
    # `--clobber` remplace l'asset s'il existe déjà
    _run([
        "gh", "release", "upload", tag,
        f"{local_path}#{name}",
        "--repo", REPO,
        "--clobber",
    ])


def download_asset(tag: str, asset_name: str, dest: Path) -> Path | None:
    """Télécharge un asset. Retourne None s'il n'existe pas."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = _run([
        "gh", "release", "download", tag,
        "--repo", REPO,
        "--pattern", asset_name,
        "--output", str(dest),
        "--clobber",
    ], check=False)
    return dest if result.returncode == 0 else None


def list_assets(tag: str) -> list[str]:
    """Liste les assets d'une release."""
    result = _run(
        ["gh", "release", "view", tag, "--repo", REPO, "--json", "assets"],
        check=False,
    )
    if result.returncode != 0:
        return []
    data = json.loads(result.stdout)
    return [a["name"] for a in data.get("assets", [])]


def append_to_parquet(tag: str, asset_name: str, new_rows: pd.DataFrame) -> None:
    """Télécharge le parquet courant, append, et ré-uploade."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / asset_name
        existing = download_asset(tag, asset_name, tmp_path)
        if existing and existing.exists() and existing.stat().st_size > 0:
            df = pd.read_parquet(tmp_path)
            df = pd.concat([df, new_rows], ignore_index=True)
        else:
            df = new_rows
        df.to_parquet(tmp_path, compression="zstd", index=False)
        upload_asset(tag, tmp_path, asset_name)
