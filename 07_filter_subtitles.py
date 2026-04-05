from pathlib import Path

import pandas as pd

DATA = Path("data")
INPUT_CSV = DATA / "processed" / "movies_with_subtitles.csv"
OUTPUT_CSV = DATA / "processed" / "movies_filtered.csv"


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {INPUT_CSV}. Execute 06_normalize_subtitles.py primeiro."
        )

    print(f"Lendo {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    print(f"Total de filmes antes do filtro: {len(df)}")

    filtered = df[(df["segment_count"] >= 500) & (df["segment_count"] <= 6000)]

    print(
        f"Total de filmes após filtro (500 <= segment_count <= 6000): {len(filtered)}"
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(OUTPUT_CSV, index=False)

    print(f"\nCSV filtrado salvo em: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
