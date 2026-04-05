from pathlib import Path

import pandas as pd

DATA = Path("data")
RAW_SUBTITLES_CSV = DATA / "raw" / "subtitles" / "movies_subtitles.csv"
MOVIES_CSV = DATA / "processed" / "movies.csv"
OUTPUT_CSV = DATA / "processed" / "movies_with_subtitle_stats.csv"

REQUIRED_COLUMNS = ["start_time", "end_time", "text", "imdb_id"]


def _word_count(value: object) -> int:
    if pd.isna(value):
        return 0
    return len(str(value).split())


def load_and_aggregate_subtitles(path: Path, chunk_size: int = 200_000) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    missing = [col for col in REQUIRED_COLUMNS if col not in header.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em movies_subtitles.csv: {missing}")

    aggregated_chunks = []

    for chunk in pd.read_csv(path, usecols=REQUIRED_COLUMNS, chunksize=chunk_size):
        chunk["start_time"] = pd.to_numeric(chunk["start_time"], errors="coerce")
        chunk["end_time"] = pd.to_numeric(chunk["end_time"], errors="coerce")

        chunk = chunk.dropna(subset=["imdb_id", "start_time", "end_time"])
        chunk["duration_seconds"] = (chunk["end_time"] - chunk["start_time"]).clip(
            lower=0
        )
        chunk["text_chars"] = chunk["text"].fillna("").astype(str).str.len()
        chunk["text_words"] = chunk["text"].map(_word_count)

        grouped = chunk.groupby("imdb_id", as_index=False).agg(
            subtitle_segments=("text", "size"),
            subtitle_duration_seconds=("duration_seconds", "sum"),
            subtitle_text_chars=("text_chars", "sum"),
            subtitle_text_words=("text_words", "sum"),
        )
        aggregated_chunks.append(grouped)

    if not aggregated_chunks:
        return pd.DataFrame(
            columns=[
                "imdb_id",
                "subtitle_segments",
                "subtitle_duration_seconds",
                "subtitle_text_chars",
                "subtitle_text_words",
            ]
        )

    subtitles = pd.concat(aggregated_chunks, ignore_index=True)
    subtitles = subtitles.groupby("imdb_id", as_index=False).sum()
    return subtitles


def main() -> None:
    if not RAW_SUBTITLES_CSV.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {RAW_SUBTITLES_CSV}")
    if not MOVIES_CSV.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {MOVIES_CSV}")

    movies = pd.read_csv(MOVIES_CSV)
    subtitles = load_and_aggregate_subtitles(RAW_SUBTITLES_CSV)

    merged = movies.merge(subtitles, on="imdb_id", how="inner")
    merged["has_subtitle"] = True

    merged["subtitle_segments"] = merged["subtitle_segments"].astype(int)
    merged["subtitle_text_chars"] = merged["subtitle_text_chars"].astype(int)
    merged["subtitle_text_words"] = merged["subtitle_text_words"].astype(int)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)

    subtitle_unique_ids = int(subtitles["imdb_id"].nunique())

    print(f"IMDb IDs com legenda em movies_subtitles.csv: {subtitle_unique_ids}")
    print(f"Filmes no CSV final (apenas com legenda): {len(merged)}")
    print(f"CSV final salvo em: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
