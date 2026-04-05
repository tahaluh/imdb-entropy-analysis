from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA = Path("data")
INPUT_CSV = DATA / "processed" / "movies_information_features.csv"
OUTPUT_DIR = DATA / "analysis"
OUTPUT_METRICS = OUTPUT_DIR / "regression_metrics.csv"
OUTPUT_PREDICTIONS = OUTPUT_DIR / "regression_predictions.csv"
OUTPUT_PLOT = OUTPUT_DIR / "regression_pred_vs_real.png"


FEATURE_ALIASES = {
    "entropy": ["entropy", "char_entropy"],
    "gzip_ratio": ["gzip_ratio"],
    "word_count": ["word_count", "clean_text_words"],
}
TARGET = "rating"


def resolve_feature(df: pd.DataFrame, logical_name: str) -> str:
    for candidate in FEATURE_ALIASES[logical_name]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Nenhuma coluna encontrada para '{logical_name}'. Opcoes: {FEATURE_ALIASES[logical_name]}"
    )


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    entropy_col = resolve_feature(df, "entropy")
    gzip_col = resolve_feature(df, "gzip_ratio")
    word_count_col = resolve_feature(df, "word_count")

    used_features = [entropy_col, gzip_col, word_count_col]
    print("Features usadas:", used_features)

    model_df = df[used_features + [TARGET]].dropna().copy()
    if model_df.empty:
        raise RuntimeError("Sem dados apos dropna para regressao.")

    X = model_df[used_features]
    y = model_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    coef_df = pd.DataFrame(
        {
            "feature": used_features,
            "coefficient": model.coef_,
        }
    )
    intercept_df = pd.DataFrame(
        [{"feature": "intercept", "coefficient": model.intercept_}]
    )
    coef_df = pd.concat([coef_df, intercept_df], ignore_index=True)
    coef_df.to_csv(OUTPUT_DIR / "regression_coefficients.csv", index=False)

    metrics_df = pd.DataFrame(
        [
            {
                "rows_used": len(model_df),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
                "features": ",".join(used_features),
            }
        ]
    )
    metrics_df.to_csv(OUTPUT_METRICS, index=False)

    pred_df = X_test.copy()
    pred_df["rating_real"] = y_test.values
    pred_df["rating_pred"] = y_pred
    pred_df["abs_error"] = (pred_df["rating_real"] - pred_df["rating_pred"]).abs()
    pred_df.to_csv(OUTPUT_PREDICTIONS, index=False)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    low = min(y_test.min(), y_pred.min())
    high = max(y_test.max(), y_pred.max())
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Rating real")
    plt.ylabel("Rating previsto")
    plt.title("Regressao Linear: Real vs Previsto")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=180)
    plt.close()

    print("\nResumo da regressao linear")
    print(f"  Linhas usadas: {len(model_df)}")
    print(f"  R2: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Coeficientes salvos em: {OUTPUT_DIR / 'regression_coefficients.csv'}")
    print(f"  Metricas salvas em: {OUTPUT_METRICS}")
    print(f"  Predicoes salvas em: {OUTPUT_PREDICTIONS}")
    print(f"  Grafico salvo em: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
