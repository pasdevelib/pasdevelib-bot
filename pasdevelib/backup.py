"""Daily backup of historical parquets to a separate GitHub release.

Strategy
--------
Every day, this script downloads the current `aggregates` and `history`
release assets from the main bot repo and re-uploads them to a `backup-YYYYMMDD`
release. The most recent 30 daily snapshots are kept; older ones are pruned.

This gives us:
  - Recovery from accidental deletion or corruption of the main releases
  - A point-in-time view of historical aggregations
  - Off-the-main-release-track storage so the active pipeline can't trample
    the backups

Environment variables
---------------------
GITHUB_TOKEN      Required. Token with `contents: write` on the bot repo.
GITHUB_REPOSITORY Set automatically by Actions (e.g. `pasdevelib/pasdevelib-bot`).

Usage
-----
Run via the workflow `.github/workflows/backup.yml`, scheduled daily at 04:00 UTC.
Can also be triggered manually via `workflow_dispatch`.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path
from typing import Iterable

import requests

GITHUB_API = "https://api.github.com"
SOURCE_RELEASES = ["aggregates", "history"]
BACKUP_TAG_PREFIX = "backup-"
KEEP_BACKUPS = 30
TIMEOUT = 60


def _now_utc() -> dt.datetime:
    """Timezone-aware UTC datetime (replaces deprecated utcnow())."""
    return dt.datetime.now(dt.UTC)


def _now_iso_z() -> str:
    """ISO 8601 timestamp with 'Z' suffix (Zulu time), e.g. '2026-05-05T21:27:08Z'."""
    return _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def get_release(repo: str, tag: str, token: str) -> dict | None:
    r = requests.get(
        f"{GITHUB_API}/repos/{repo}/releases/tags/{tag}",
        headers=_headers(token),
        timeout=TIMEOUT,
    )
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def list_releases(repo: str, token: str) -> list[dict]:
    out: list[dict] = []
    page = 1
    while True:
        r = requests.get(
            f"{GITHUB_API}/repos/{repo}/releases",
            headers=_headers(token),
            params={"per_page": 100, "page": page},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        page += 1
    return out


def create_release(repo: str, tag: str, name: str, token: str) -> dict:
    r = requests.post(
        f"{GITHUB_API}/repos/{repo}/releases",
        headers=_headers(token),
        json={
            "tag_name": tag,
            "name": name,
            "body": f"Daily backup created at {_now_iso_z()}",
            "draft": False,
            "prerelease": True,  # keeps backups out of the "Latest" badge
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def delete_release(repo: str, release_id: int, tag: str, token: str) -> None:
    # Delete release
    r = requests.delete(
        f"{GITHUB_API}/repos/{repo}/releases/{release_id}",
        headers=_headers(token),
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    # Also delete the underlying tag so it doesn't pile up
    r = requests.delete(
        f"{GITHUB_API}/repos/{repo}/git/refs/tags/{tag}",
        headers=_headers(token),
        timeout=TIMEOUT,
    )
    if r.status_code not in (204, 422):  # 422 = tag already gone, fine
        r.raise_for_status()


def download_assets(release: dict, token: str, dest: Path) -> Iterable[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    headers = {**_headers(token), "Accept": "application/octet-stream"}
    for asset in release.get("assets", []):
        target = dest / asset["name"]
        print(f"  downloading {asset['name']} ({asset['size']:,} bytes)")
        with requests.get(
            asset["url"], headers=headers, stream=True, timeout=TIMEOUT
        ) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        yield target


def upload_asset(repo: str, release: dict, path: Path, token: str) -> None:
    upload_url = release["upload_url"].split("{")[0]
    headers = {
        **_headers(token),
        "Content-Type": "application/octet-stream",
    }
    with open(path, "rb") as f:
        r = requests.post(
            upload_url,
            headers=headers,
            params={"name": path.name},
            data=f,
            timeout=TIMEOUT * 5,  # large parquets
        )
        r.raise_for_status()
    print(f"  uploaded {path.name}")


def main() -> int:
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    today = _now_utc().strftime("%Y%m%d")
    backup_tag = f"{BACKUP_TAG_PREFIX}{today}"

    print(f"== Backup pipeline: repo={repo} tag={backup_tag} ==")

    # 1. Skip if today's backup already exists
    existing = get_release(repo, backup_tag, token)
    if existing is not None:
        print(f"Backup {backup_tag} already exists, skipping.")
        return 0

    # 2. Create today's backup release
    print(f"Creating release {backup_tag}…")
    backup_release = create_release(
        repo, backup_tag, f"Backup {today}", token
    )

    # 3. For each source release, download assets and re-upload to backup
    workdir = Path("/tmp/pasdevelib-backup")
    for src_tag in SOURCE_RELEASES:
        print(f"\n>> Source release: {src_tag}")
        src_release = get_release(repo, src_tag, token)
        if src_release is None:
            print(f"  source release '{src_tag}' not found, skipping.")
            continue
        if not src_release.get("assets"):
            print(f"  no assets in '{src_tag}', skipping.")
            continue
        src_dir = workdir / src_tag
        for asset_path in download_assets(src_release, token, src_dir):
            # Prefix the filename with the source tag to avoid collisions
            renamed = asset_path.with_name(f"{src_tag}__{asset_path.name}")
            asset_path.rename(renamed)
            upload_asset(repo, backup_release, renamed, token)

    # 4. Prune old backups (keep only the most recent KEEP_BACKUPS)
    print(f"\n>> Pruning old backups (keeping {KEEP_BACKUPS} most recent)…")
    all_releases = list_releases(repo, token)
    backups = sorted(
        [r for r in all_releases if r["tag_name"].startswith(BACKUP_TAG_PREFIX)],
        key=lambda r: r["tag_name"],
        reverse=True,
    )
    to_delete = backups[KEEP_BACKUPS:]
    for r in to_delete:
        print(f"  deleting old backup: {r['tag_name']}")
        try:
            delete_release(repo, r["id"], r["tag_name"], token)
        except requests.HTTPError as e:
            print(f"    failed to delete {r['tag_name']}: {e}")

    print(f"\n== Backup complete: {backup_tag} ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
