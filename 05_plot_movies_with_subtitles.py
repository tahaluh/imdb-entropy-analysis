from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA = Path("data")
PROCESSED = DATA / "processed"
PLOTS = PROCESSED / "plots"
INPUT_CSV = PROCESSED / "movies_with_subtitle_stats.csv"


def _prepare_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "has_subtitle" in df.columns:
        df = df[df["has_subtitle"] == True].copy()

    for col in ["year", "rating", "votes", "runtimeMinutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["year", "rating", "votes"])
    df["year"] = df["year"].astype(int)
    df = df[df["year"] > 0].copy()
    return df


def plot_rating_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(df["rating"], bins=25, color="#2a9d8f", edgecolor="white")
    plt.title("Distribuicao de notas IMDb (com legenda)")
    plt.xlabel("Nota")
    plt.ylabel("Quantidade de filmes")
    plt.tight_layout()
    plt.savefig(out_dir / "rating_distribution_with_subtitles.png", dpi=180)
    plt.close()


def plot_movies_per_year(df: pd.DataFrame, out_dir: Path) -> None:
    yearly_count = df.groupby("year", as_index=False)["imdb_id"].count()
    yearly_count = yearly_count.rename(columns={"imdb_id": "movie_count"})

    plt.figure(figsize=(12, 6))
    plt.bar(yearly_count["year"], yearly_count["movie_count"], color="#264653")
    plt.title("Quantidade de filmes por ano (com legenda)")
    plt.xlabel("Ano")
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.savefig(out_dir / "movies_per_year_with_subtitles.png", dpi=180)
    plt.close()


def plot_rating_by_year(df: pd.DataFrame, out_dir: Path) -> None:
    yearly_stats = df.groupby("year", as_index=False).agg(
        avg_rating=("rating", "mean"),
        median_rating=("rating", "median"),
        movie_count=("imdb_id", "count"),
    )

    filtered = yearly_stats[yearly_stats["movie_count"] >= 10].copy()

    plt.figure(figsize=(12, 6))
    plt.plot(
        filtered["year"],
        filtered["avg_rating"],
        label="Media",
        color="#e76f51",
        linewidth=2,
    )
    plt.plot(
        filtered["year"],
        filtered["median_rating"],
        label="Mediana",
        color="#457b9d",
        linewidth=2,
    )
    plt.title("Nota por ano (com legenda)")
    plt.xlabel("Ano")
    plt.ylabel("Nota")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "rating_by_year_with_subtitles.png", dpi=180)
    plt.close()

    yearly_stats.to_csv(out_dir / "yearly_stats_with_subtitles.csv", index=False)


def plot_votes_vs_rating(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(df["votes"], df["rating"], alpha=0.25, color="#1d3557", s=15)
    plt.xscale("log")
    plt.title("Relacao votos x nota (com legenda)")
    plt.xlabel("Votos (log)")
    plt.ylabel("Nota")
    plt.tight_layout()
    plt.savefig(out_dir / "votes_vs_rating_with_subtitles.png", dpi=180)
    plt.close()


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {INPUT_CSV}. Rode primeiro o 04_merge_subtitles_with_movies.py"
        )

    df = _prepare_dataframe(INPUT_CSV)
    if df.empty:
        raise RuntimeError("Nao ha filmes com legenda para plotar no CSV informado.")

    plot_rating_distribution(df, PLOTS)
    plot_movies_per_year(df, PLOTS)
    plot_rating_by_year(df, PLOTS)
    plot_votes_vs_rating(df, PLOTS)

    print(f"Graficos salvos em: {PLOTS}")
    print(f"Total de filmes analisados (com legenda): {len(df)}")


if __name__ == "__main__":
    main()
