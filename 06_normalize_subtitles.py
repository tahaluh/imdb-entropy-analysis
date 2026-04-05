import re
from pathlib import Path

import pandas as pd

DATA = Path("data")

RAW_SUBTITLES_CSV = DATA / "raw" / "subtitles" / "movies_subtitles.csv"
MOVIES_CSV = DATA / "processed" / "movies.csv"

NORMALIZED_OUTPUT_CSV = DATA / "processed" / "movies_subtitles_normalized.csv"
MERGED_OUTPUT_CSV = DATA / "processed" / "movies_with_subtitles.csv"

REQUIRED_SUBTITLE_COLUMNS = ["start_time", "end_time", "text", "imdb_id"]
REQUIRED_MOVIE_COLUMNS = [
    "imdb_id",
    "title",
    "original_title",
    "year",
    "runtimeMinutes",
    "genres",
    "rating",
    "votes",
]


def clean_text(text: str) -> str:
    """Normaliza e limpa texto de legenda."""
    if pd.isna(text):
        return ""

    text = str(text)

    # quebra de linha -> espaço
    text = text.replace("\n", " ").replace("\r", " ")

    # lowercase
    text = text.lower()

    # remove tags html
    text = re.sub(r"<[^>]+>", " ", text)

    # remove conteúdo entre [] e ()
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)

    # remove speaker tags do tipo "BOY:" ou "VOICE BOX:"
    text = re.sub(r"^\s*[a-z][a-z\s\-']{0,40}:\s*", " ", text)

    # remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # mantém palavras/números/espaços, remove pontuação
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # remove underscores
    text = text.replace("_", " ")

    # colapsa múltiplos espaços
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def calculate_basic_stats(text: str) -> dict:
    """Calcula estatísticas básicas do texto já limpo."""
    if not text:
        return {
            "clean_text_length": 0,
            "clean_text_words": 0,
            "unique_words": 0,
            "avg_word_length": 0.0,
            "avg_words_per_segment_proxy": 0.0,
        }

    words = text.split()
    unique_words = len(set(words))
    total_words = len(words)
    total_chars = len(text)
    avg_word_length = (
        sum(len(word) for word in words) / total_words if total_words else 0.0
    )

    return {
        "clean_text_length": total_chars,
        "clean_text_words": total_words,
        "unique_words": unique_words,
        "avg_word_length": avg_word_length,
        "avg_words_per_segment_proxy": 0.0,  # preenchido depois
    }


def validate_columns(path: Path, required_columns: list[str]) -> None:
    header = pd.read_csv(path, nrows=0)
    missing = [col for col in required_columns if col not in header.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em {path.name}: {missing}")


def load_and_normalize_subtitles(path: Path, chunk_size: int = 200_000) -> pd.DataFrame:
    """Lê, limpa e agrega legendas por imdb_id."""
    validate_columns(path, REQUIRED_SUBTITLE_COLUMNS)

    aggregated_chunks = []
    chunk_num = 0

    for chunk in pd.read_csv(
        path, usecols=REQUIRED_SUBTITLE_COLUMNS, chunksize=chunk_size
    ):
        chunk_num += 1
        chunk = chunk.dropna(subset=["imdb_id"]).copy()

        chunk["text"] = chunk["text"].fillna("").astype(str)
        chunk["clean_text"] = chunk["text"].apply(clean_text)
        chunk["original_text_chars"] = chunk["text"].str.len()
        chunk["clean_segment_words"] = chunk["clean_text"].apply(
            lambda x: len(x.split()) if x else 0
        )

        grouped = chunk.groupby("imdb_id", as_index=False).agg(
            full_clean_text=(
                "clean_text",
                lambda x: " ".join(part for part in x if part).strip(),
            ),
            segment_count=("text", "size"),
            original_text_chars=("original_text_chars", "sum"),
            clean_segment_words_sum=("clean_segment_words", "sum"),
        )

        aggregated_chunks.append(grouped)
        print(f"  Chunk {chunk_num}: {len(grouped)} filmes processados")

    if not aggregated_chunks:
        return pd.DataFrame(
            columns=[
                "imdb_id",
                "full_clean_text",
                "segment_count",
                "original_text_chars",
                "clean_text_length",
                "clean_text_words",
                "unique_words",
                "vocabulary_richness",
                "avg_word_length",
                "avg_words_per_segment",
            ]
        )

    normalized = pd.concat(aggregated_chunks, ignore_index=True)

    # Se um imdb_id apareceu em mais de um chunk, agrega de novo
    normalized = normalized.groupby("imdb_id", as_index=False).agg(
        full_clean_text=(
            "full_clean_text",
            lambda x: " ".join(part for part in x if part).strip(),
        ),
        segment_count=("segment_count", "sum"),
        original_text_chars=("original_text_chars", "sum"),
        clean_segment_words_sum=("clean_segment_words_sum", "sum"),
    )

    # limpa novamente após concatenação final
    normalized["full_clean_text"] = normalized["full_clean_text"].apply(clean_text)

    stats = normalized["full_clean_text"].apply(calculate_basic_stats)
    stats_df = pd.DataFrame(stats.tolist())

    normalized = pd.concat([normalized, stats_df], axis=1)

    normalized["vocabulary_richness"] = normalized["unique_words"] / normalized[
        "clean_text_words"
    ].clip(lower=1)

    normalized["avg_words_per_segment"] = normalized[
        "clean_segment_words_sum"
    ] / normalized["segment_count"].clip(lower=1)

    normalized = normalized.drop(
        columns=["clean_segment_words_sum", "avg_words_per_segment_proxy"]
    )

    normalized["clean_text_length"] = normalized["clean_text_length"].astype(int)
    normalized["clean_text_words"] = normalized["clean_text_words"].astype(int)
    normalized["unique_words"] = normalized["unique_words"].astype(int)
    normalized["segment_count"] = normalized["segment_count"].astype(int)
    normalized["original_text_chars"] = normalized["original_text_chars"].astype(int)

    normalized["avg_word_length"] = normalized["avg_word_length"].round(4)
    normalized["avg_words_per_segment"] = normalized["avg_words_per_segment"].round(4)
    normalized["vocabulary_richness"] = normalized["vocabulary_richness"].round(6)

    return normalized


def merge_with_movies(subtitles_df: pd.DataFrame, movies_path: Path) -> pd.DataFrame:
    """Une as legendas normalizadas com os metadados dos filmes."""
    validate_columns(movies_path, REQUIRED_MOVIE_COLUMNS)

    movies = pd.read_csv(movies_path)
    merged = movies.merge(subtitles_df, on="imdb_id", how="inner")

    merged = merged[
        [
            "imdb_id",
            "title",
            "original_title",
            "year",
            "runtimeMinutes",
            "genres",
            "rating",
            "votes",
            "segment_count",
            "original_text_chars",
            "clean_text_length",
            "clean_text_words",
            "unique_words",
            "vocabulary_richness",
            "avg_word_length",
            "avg_words_per_segment",
            "full_clean_text",
        ]
    ].copy()

    return merged


def print_summary(normalized: pd.DataFrame, merged: pd.DataFrame) -> None:
    total_unique_ids = int(normalized["imdb_id"].nunique())
    avg_clean_length = normalized["clean_text_length"].mean()
    avg_words = normalized["clean_text_words"].mean()
    avg_unique = normalized["unique_words"].mean()
    avg_vocabulary = normalized["vocabulary_richness"].mean()

    min_segments = int(normalized["segment_count"].min())
    max_segments = int(normalized["segment_count"].max())
    avg_segments = normalized["segment_count"].mean()
    median_segments = normalized["segment_count"].median()
    p25_segments = normalized["segment_count"].quantile(0.25)
    p75_segments = normalized["segment_count"].quantile(0.75)

    top_most = normalized.nlargest(3, "segment_count")[["imdb_id", "segment_count"]]
    top_least = normalized.nsmallest(3, "segment_count")[["imdb_id", "segment_count"]]

    print("\nResumo de normalização:")
    print(f"  Total de imdb_ids com legenda: {total_unique_ids}")
    print(f"\n  Distribuição de linhas de legenda (segment_count) por filme:")
    print(f"    Mínimo: {min_segments}")
    print(f"    P25: {p25_segments:.0f}")
    print(f"    Mediana: {median_segments:.0f}")
    print(f"    Média: {avg_segments:.0f}")
    print(f"    P75: {p75_segments:.0f}")
    print(f"    Máximo: {max_segments}")
    print(f"\n  Top 3 filmes com MAIS linhas de legenda:")
    for idx, row in top_most.iterrows():
        print(f"    {row['imdb_id']}: {int(row['segment_count'])} linhas")
    print(f"\n  Top 3 filmes com MENOS linhas de legenda:")
    for idx, row in top_least.iterrows():
        print(f"    {row['imdb_id']}: {int(row['segment_count'])} linhas")
    print(f"\n  Estatísticas de texto limpo:")
    print(f"    Média de caracteres: {avg_clean_length:.0f}")
    print(f"    Média de palavras: {avg_words:.0f}")
    print(f"    Média de palavras únicas: {avg_unique:.0f}")
    print(f"    Média de riqueza vocabular: {avg_vocabulary:.4f}")
    print(f"\n  Filmes após merge com movies.csv: {len(merged)}")


def main() -> None:
    if not RAW_SUBTITLES_CSV.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {RAW_SUBTITLES_CSV}")

    if not MOVIES_CSV.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {MOVIES_CSV}")

    print("Lendo, limpando e agregando legendas por imdb_id...")
    normalized = load_and_normalize_subtitles(RAW_SUBTITLES_CSV)

    print("Realizando merge com movies.csv...")
    merged = merge_with_movies(normalized, MOVIES_CSV)

    print(f"\nFilmes antes do filtro de segment_count: {len(merged)}")

    merged = merged[
        (merged["segment_count"] >= 500) & (merged["segment_count"] <= 6000)
    ].copy()

    print(f"Filmes após filtro (500 <= segment_count <= 6000): {len(merged)}")

    NORMALIZED_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(NORMALIZED_OUTPUT_CSV, index=False)
    merged.to_csv(MERGED_OUTPUT_CSV, index=False)

    print_summary(normalized, merged)

    print(f"\nCSV de legendas normalizadas salvo em: {NORMALIZED_OUTPUT_CSV}")
    print(f"CSV final para análise salvo em: {MERGED_OUTPUT_CSV}")


if __name__ == "__main__":
    main()
