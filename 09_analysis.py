from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA = Path("data")
INPUT_CSV = DATA / "processed" / "movies_information_features.csv"
OUTPUT_DIR = DATA / "analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid")


def load_data():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    return df


def compute_correlations(df: pd.DataFrame):
    print("\n📊 Correlação com rating (Pearson):")
    pearson = df.corr(numeric_only=True)["rating"].sort_values(ascending=False)
    print(pearson)

    print("\n📊 Correlação com rating (Spearman):")
    spearman = df.corr(method="spearman", numeric_only=True)["rating"].sort_values(
        ascending=False
    )
    print(spearman)

    pearson.to_csv(OUTPUT_DIR / "correlation_pearson.csv")
    spearman.to_csv(OUTPUT_DIR / "correlation_spearman.csv")


def scatter_plot(df, x, y="rating"):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=x, y=y, alpha=0.4)
    plt.title(f"{y} vs {x}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"scatter_{y}_vs_{x}.png")
    plt.close()


def generate_scatter_plots(df):
    features = [
        "char_entropy",
        "bigram_entropy",
        "trigram_entropy",
        "word_entropy",
        "gzip_ratio",
        "bz2_ratio",
        "lzma_ratio",
    ]

    for feature in features:
        print(f"Gerando scatter: {feature}")
        scatter_plot(df, feature)


def create_rating_groups(df):
    df["rating_group"] = pd.cut(
        df["rating"],
        bins=[0, 6, 7.5, 10],
        labels=["ruim", "medio", "bom"],
    )
    return df


def boxplot_feature(df, feature):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="rating_group", y=feature)
    plt.title(f"{feature} por grupo de rating")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"boxplot_{feature}.png")
    plt.close()


def generate_boxplots(df):
    features = [
        "char_entropy",
        "bigram_entropy",
        "trigram_entropy",
        "word_entropy",
        "gzip_ratio",
    ]

    for feature in features:
        print(f"Gerando boxplot: {feature}")
        boxplot_feature(df, feature)


def summary_by_group(df):
    summary = df.groupby("rating_group").agg(
        count=("rating", "count"),
        mean_rating=("rating", "mean"),
        mean_char_entropy=("char_entropy", "mean"),
        mean_word_entropy=("word_entropy", "mean"),
        mean_gzip_ratio=("gzip_ratio", "mean"),
    )

    print("\n📊 Resumo por grupo de rating:")
    print(summary)

    summary.to_csv(OUTPUT_DIR / "group_summary.csv")


def summary_by_rating_bin_1pt(df: pd.DataFrame):
    bin_edges = list(range(0, 11))
    labels = [f"{i}-{i + 1}" for i in range(0, 10)]

    df = df.copy()
    df["rating_bin_1pt"] = pd.cut(
        df["rating"],
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    summary = df.groupby("rating_bin_1pt", observed=False).agg(
        count=("rating", "count"),
        mean_rating=("rating", "mean"),
        mean_word_entropy=("word_entropy", "mean"),
        mean_gzip_ratio=("gzip_ratio", "mean"),
        mean_bz2_ratio=("bz2_ratio", "mean"),
        mean_lzma_ratio=("lzma_ratio", "mean"),
    )

    print("\n📊 Resumo por faixa de rating (1 em 1):")
    print(summary)

    summary.to_csv(OUTPUT_DIR / "group_summary_1pt_bins.csv")

    summary_plot = summary.reset_index()

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=summary_plot,
        x="rating_bin_1pt",
        y="mean_word_entropy",
        marker="o",
        label="word_entropy",
    )
    plt.title("Media de word_entropy por faixa de rating (1 em 1)")
    plt.xlabel("Faixa de rating")
    plt.ylabel("Media")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "line_word_entropy_by_rating_1pt.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=summary_plot,
        x="rating_bin_1pt",
        y="mean_gzip_ratio",
        marker="o",
        label="gzip_ratio",
    )
    sns.lineplot(
        data=summary_plot,
        x="rating_bin_1pt",
        y="mean_bz2_ratio",
        marker="o",
        label="bz2_ratio",
    )
    sns.lineplot(
        data=summary_plot,
        x="rating_bin_1pt",
        y="mean_lzma_ratio",
        marker="o",
        label="lzma_ratio",
    )
    plt.title("Media de razoes de compressao por faixa de rating (1 em 1)")
    plt.xlabel("Faixa de rating")
    plt.ylabel("Media da razao")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "line_compression_ratio_by_rating_1pt.png")
    plt.close()

    filtered = df.dropna(subset=["rating_bin_1pt", "word_entropy", "gzip_ratio"])

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=filtered, x="rating_bin_1pt", y="word_entropy")
    plt.title("Distribuicao de word_entropy por faixa de rating (1 em 1)")
    plt.xlabel("Faixa de rating")
    plt.ylabel("word_entropy")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "boxplot_word_entropy_by_rating_1pt.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=filtered, x="rating_bin_1pt", y="gzip_ratio")
    plt.title("Distribuicao de gzip_ratio por faixa de rating (1 em 1)")
    plt.xlabel("Faixa de rating")
    plt.ylabel("gzip_ratio")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "boxplot_gzip_ratio_by_rating_1pt.png")
    plt.close()


def interpret_results(df):
    print("\n🧠 Insights automáticos:")

    corr = df.corr(numeric_only=True)["rating"]

    important = corr.abs().sort_values(ascending=False)

    print("\nTop relações com rating:")
    print(important.head(10))

    print("\nInterpretação:")

    if corr["char_entropy"] > 0:
        print("- Filmes com maior entropia de caracteres tendem a ter notas maiores.")
    else:
        print("- Não há evidência de que maior entropia de caracteres aumente a nota.")

    if corr["gzip_ratio"] < 0:
        print("- Filmes mais compressíveis tendem a ter notas maiores.")
    else:
        print("- Compressibilidade não indica diretamente qualidade do filme.")

    print("\nObservação:")
    print("Mesmo correlações fracas são válidas — o importante é a análise.")


def main():
    print("Carregando dados...")
    df = load_data()

    print("Calculando correlações...")
    compute_correlations(df)

    print("Criando grupos de rating...")
    df = create_rating_groups(df)

    print("Gerando scatter plots...")
    generate_scatter_plots(df)

    print("Gerando boxplots...")
    generate_boxplots(df)

    print("Resumo por grupo...")
    summary_by_group(df)

    print("Resumo por faixa 1 em 1...")
    summary_by_rating_bin_1pt(df)

    print("Gerando insights...")
    interpret_results(df)

    print(f"\n✅ Análises salvas em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
