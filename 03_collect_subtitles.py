import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

DATA = Path("data")
PROCESSED = DATA / "processed"
RAW_SUBS = DATA / "raw" / "subtitles"

RAW_SUBS.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

OS_API_BASE = "https://api.opensubtitles.com/api/v1"
OS_API_KEY = ""
OS_USER_AGENT = "imdb-entropy-analysis v1"
OS_USERNAME = ""
OS_PASSWORD = ""

SUB_LANGUAGES = "en"
SUB_SAMPLE_SIZE = 200
SUB_OVERWRITE = False
SUB_TIMEOUT = 30


def _slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text)).strip("_")
    return text or "untitled"


def _imdb_digits(imdb_id: str) -> str:
    return str(imdb_id).replace("tt", "")


def _headers(token: str | None = None) -> dict[str, str]:
    headers = {
        "Api-Key": OS_API_KEY,
        "User-Agent": OS_USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _login_token() -> str | None:
    if not OS_USERNAME or not OS_PASSWORD:
        return None

    resp = requests.post(
        f"{OS_API_BASE}/login",
        headers=_headers(),
        json={"username": OS_USERNAME, "password": OS_PASSWORD},
        timeout=SUB_TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("token")


def _search_subtitle(imdb_id: str, year: int | float | str) -> dict[str, Any] | None:
    params = {
        "imdb_id": _imdb_digits(imdb_id),
        "languages": SUB_LANGUAGES,
        "type": "movie",
        "year": int(year),
        "order_by": "download_count",
        "order_direction": "desc",
    }

    resp = requests.get(
        f"{OS_API_BASE}/subtitles",
        headers=_headers(),
        params=params,
        timeout=SUB_TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", [])

    if not data:
        return None

    best = data[0]
    attrs = best.get("attributes", {})
    files = attrs.get("files", [])
    if not files:
        return None

    file_id = files[0].get("file_id")
    if not file_id:
        return None

    return {
        "file_id": file_id,
        "language": attrs.get("language") or attrs.get("language_code"),
        "release": attrs.get("release"),
    }


def _download_subtitle(file_id: int, token: str) -> tuple[bytes, str | None]:
    resp = requests.post(
        f"{OS_API_BASE}/download",
        headers=_headers(token),
        json={"file_id": file_id},
        timeout=SUB_TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()
    link = payload.get("link")
    suggested_name = payload.get("file_name")
    if not link:
        raise RuntimeError("Resposta sem link de download")

    file_resp = requests.get(link, timeout=SUB_TIMEOUT)
    file_resp.raise_for_status()
    return file_resp.content, suggested_name


def _save_subtitle_bytes(content: bytes, out_base: Path) -> Path:
    if zipfile.is_zipfile(BytesIO(content)):
        with zipfile.ZipFile(BytesIO(content)) as zf:
            srt_names = [
                name for name in zf.namelist() if name.lower().endswith(".srt")
            ]
            if not srt_names:
                raise RuntimeError("Zip sem arquivo .srt")
            target = srt_names[0]
            data = zf.read(target)
            out_path = out_base.with_suffix(".srt")
            out_path.write_bytes(data)
            return out_path

    out_path = out_base.with_suffix(".srt")
    out_path.write_bytes(content)
    return out_path


def save_log(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED / "subtitle_collection_log.csv", index=False)


def main():
    if not OS_API_KEY:
        raise RuntimeError("Defina OS_API_KEY diretamente no 03_collect_subtitles.py")

    movies = pd.read_csv(PROCESSED / "movies.csv")

    movies = movies.head(SUB_SAMPLE_SIZE).copy()
    token = _login_token()

    rows = []

    for _, row in movies.iterrows():
        imdb_id = row["imdb_id"]
        title = row["title"]
        year = row["year"]

        try:
            movie_dir = RAW_SUBS / str(imdb_id)
            movie_dir.mkdir(parents=True, exist_ok=True)

            existing = list(movie_dir.glob("*.srt"))
            if existing and not SUB_OVERWRITE:
                rows.append(
                    {
                        "imdb_id": imdb_id,
                        "title": title,
                        "year": year,
                        "subtitle_found": True,
                        "subtitle_filename": existing[0].name,
                        "provider": "local_cache",
                        "language": None,
                        "error": None,
                    }
                )
                print(f"[CACHE] {imdb_id} - {title}")
                continue

            match = _search_subtitle(imdb_id, year)

            if not match:
                rows.append(
                    {
                        "imdb_id": imdb_id,
                        "title": title,
                        "year": year,
                        "subtitle_found": False,
                        "subtitle_filename": None,
                        "provider": "opensubtitles",
                        "language": None,
                        "error": "subtitle_not_found",
                    }
                )
                print(f"[SEM LEGENDA] {imdb_id} - {title}")
                continue

            if not token:
                rows.append(
                    {
                        "imdb_id": imdb_id,
                        "title": title,
                        "year": year,
                        "subtitle_found": False,
                        "subtitle_filename": None,
                        "provider": "opensubtitles",
                        "language": match.get("language"),
                        "error": "missing_login_credentials",
                    }
                )
                print(f"[LOGIN NECESSARIO] {imdb_id} - {title}")
                continue

            file_id = int(match["file_id"])
            content, suggested_name = _download_subtitle(file_id, token)
            filename_root = _slug(Path(suggested_name or f"{imdb_id}_{file_id}").stem)
            out_file = _save_subtitle_bytes(content, movie_dir / filename_root)

            rows.append(
                {
                    "imdb_id": imdb_id,
                    "title": title,
                    "year": year,
                    "subtitle_found": True,
                    "subtitle_filename": str(out_file.relative_to(DATA)),
                    "provider": "opensubtitles",
                    "language": match.get("language"),
                    "error": None,
                }
            )

            print(f"[OK] {imdb_id} - {title} -> {out_file.name}")

        except Exception as e:
            rows.append(
                {
                    "imdb_id": imdb_id,
                    "title": title,
                    "year": year,
                    "subtitle_found": False,
                    "subtitle_filename": None,
                    "provider": "opensubtitles",
                    "language": None,
                    "error": str(e),
                }
            )
            print(f"[ERRO] {imdb_id} - {title}: {e}")

    save_log(rows)
    print("Log salvo com sucesso.")


if __name__ == "__main__":
    main()
