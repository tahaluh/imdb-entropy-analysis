from pathlib import Path

import pandas as pd

DATA = Path("data")
RAW = DATA / "raw" / "imdb"
PROCESSED = DATA / "processed"

RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

IMDB_BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
IMDB_RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"

BASICS_PATH = RAW / "title.basics.tsv.gz"
RATINGS_PATH = RAW / "title.ratings.tsv.gz"


def load_tsv_gz(url: str, local_path: Path) -> pd.DataFrame:
    pickle_path = local_path.with_suffix(".pkl")

    if pickle_path.exists():
        return pd.read_pickle(pickle_path)

    df = pd.read_csv(url, sep="\t", compression="gzip", low_memory=False)
    df.to_pickle(pickle_path)
    return df


def main():
    basics = load_tsv_gz(IMDB_BASICS_URL, BASICS_PATH)
    ratings = load_tsv_gz(IMDB_RATINGS_URL, RATINGS_PATH)

    basics = basics.replace("\\N", pd.NA)
    ratings = ratings.replace("\\N", pd.NA)

    df = basics.merge(ratings, on="tconst", how="inner")

    df = df[
        (df["titleType"] == "movie")
        & (df["isAdult"] == 0)
        & (df["startYear"].notna())
        & (df["runtimeMinutes"].notna())
        & (df["averageRating"].notna())
        & (df["numVotes"].notna())
    ].copy()

    df["startYear"] = pd.to_numeric(df["startYear"], errors="coerce")
    df["runtimeMinutes"] = pd.to_numeric(df["runtimeMinutes"], errors="coerce")
    df["numVotes"] = pd.to_numeric(df["numVotes"], errors="coerce")
    df["averageRating"] = pd.to_numeric(df["averageRating"], errors="coerce")

    df = df[(df["numVotes"] >= 10000) & (df["startYear"] >= 1990)].copy()

    df = df[
        [
            "tconst",
            "primaryTitle",
            "originalTitle",
            "startYear",
            "runtimeMinutes",
            "genres",
            "averageRating",
            "numVotes",
        ]
    ].rename(
        columns={
            "tconst": "imdb_id",
            "primaryTitle": "title",
            "originalTitle": "original_title",
            "startYear": "year",
            "averageRating": "rating",
            "numVotes": "votes",
        }
    )

    df = df.sort_values(["votes", "rating"], ascending=[False, False])
    df.to_csv(PROCESSED / "movies.csv", index=False)

    print(f"Salvo: {PROCESSED / 'movies.csv'}")
    print(f"Total de filmes: {len(df)}")


if __name__ == "__main__":
    main()
