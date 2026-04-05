import bz2
import gzip
import lzma
import math
from collections import Counter
from pathlib import Path

import pandas as pd

DATA = Path("data")
INPUT_CSV = DATA / "processed" / "movies_filtered.csv"
OUTPUT_CSV = DATA / "processed" / "movies_information_features.csv"


def char_entropy(text: str) -> float:
    """Calcula a entropia de Shannon por caractere."""
    if not text:
        return 0.0

    counts = Counter(text)
    total = len(text)

    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def bigram_entropy(text: str) -> float:
    """Calcula a entropia de bigramas de caracteres."""
    if len(text) < 2:
        return 0.0

    bigrams = [text[i : i + 2] for i in range(len(text) - 1)]
    counts = Counter(bigrams)
    total = len(bigrams)

    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def trigram_entropy(text: str) -> float:
    """Calcula a entropia de trigramas de caracteres."""
    if len(text) < 3:
        return 0.0

    trigrams = [text[i : i + 3] for i in range(len(text) - 2)]
    counts = Counter(trigrams)
    total = len(trigrams)

    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def gzip_size(text: str) -> int:
    if not text:
        return 0
    return len(gzip.compress(text.encode("utf-8")))


def bz2_size(text: str) -> int:
    if not text:
        return 0
    return len(bz2.compress(text.encode("utf-8")))


def lzma_size(text: str) -> int:
    if not text:
        return 0
    return len(lzma.compress(text.encode("utf-8")))


def word_entropy(text: str) -> float:
    """Entropia baseada em tokens/palavras."""
    if not text:
        return 0.0

    words = text.split()
    if not words:
        return 0.0

    counts = Counter(words)
    total = len(words)

    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    if "full_clean_text" not in df.columns:
        raise ValueError("A coluna 'full_clean_text' não existe no dataset de entrada.")

    df["full_clean_text"] = df["full_clean_text"].apply(safe_text)

    print("Calculando bytes originais...")
    df["original_bytes"] = df["full_clean_text"].apply(lambda x: len(x.encode("utf-8")))

    print("Calculando entropia de caracteres...")
    df["char_entropy"] = df["full_clean_text"].apply(char_entropy)

    print("Calculando entropia de bigramas...")
    df["bigram_entropy"] = df["full_clean_text"].apply(bigram_entropy)

    print("Calculando entropia de trigramas...")
    df["trigram_entropy"] = df["full_clean_text"].apply(trigram_entropy)

    print("Calculando entropia de palavras...")
    df["word_entropy"] = df["full_clean_text"].apply(word_entropy)

    print("Calculando compressão gzip...")
    df["gzip_bytes"] = df["full_clean_text"].apply(gzip_size)

    print("Calculando compressão bz2...")
    df["bz2_bytes"] = df["full_clean_text"].apply(bz2_size)

    print("Calculando compressão lzma...")
    df["lzma_bytes"] = df["full_clean_text"].apply(lzma_size)

    # razões de compressão
    df["gzip_ratio"] = df["gzip_bytes"] / df["original_bytes"].clip(lower=1)
    df["bz2_ratio"] = df["bz2_bytes"] / df["original_bytes"].clip(lower=1)
    df["lzma_ratio"] = df["lzma_bytes"] / df["original_bytes"].clip(lower=1)

    # economia de compressão
    df["gzip_saving"] = 1 - df["gzip_ratio"]
    df["bz2_saving"] = 1 - df["bz2_ratio"]
    df["lzma_saving"] = 1 - df["lzma_ratio"]

    # bits por byte/caractere aproximados
    df["gzip_bits_per_byte"] = (df["gzip_bytes"] * 8) / df["original_bytes"].clip(
        lower=1
    )
    df["bz2_bits_per_byte"] = (df["bz2_bytes"] * 8) / df["original_bytes"].clip(lower=1)
    df["lzma_bits_per_byte"] = (df["lzma_bytes"] * 8) / df["original_bytes"].clip(
        lower=1
    )

    # arredondamentos
    round_cols = [
        "char_entropy",
        "bigram_entropy",
        "trigram_entropy",
        "word_entropy",
        "gzip_ratio",
        "bz2_ratio",
        "lzma_ratio",
        "gzip_saving",
        "bz2_saving",
        "lzma_saving",
        "gzip_bits_per_byte",
        "bz2_bits_per_byte",
        "lzma_bits_per_byte",
    ]
    df[round_cols] = df[round_cols].round(6)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nResumo das features informacionais:")
    print(f"  Total de filmes: {len(df)}")
    print(f"  Média char_entropy: {df['char_entropy'].mean():.4f}")
    print(f"  Média bigram_entropy: {df['bigram_entropy'].mean():.4f}")
    print(f"  Média trigram_entropy: {df['trigram_entropy'].mean():.4f}")
    print(f"  Média word_entropy: {df['word_entropy'].mean():.4f}")
    print(f"  Média gzip_ratio: {df['gzip_ratio'].mean():.4f}")
    print(f"  Média bz2_ratio: {df['bz2_ratio'].mean():.4f}")
    print(f"  Média lzma_ratio: {df['lzma_ratio'].mean():.4f}")
    print(f"\nCSV final salvo em: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
