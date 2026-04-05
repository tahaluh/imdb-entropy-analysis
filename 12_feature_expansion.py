from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_validate

DATA = Path("data")
INPUT_CSV = DATA / "processed" / "movies_information_features.csv"
OUTPUT_DIR = DATA / "analysis"

TARGET = "rating"

BASE_CANDIDATE_FEATURES = [
    "char_entropy",
    "bigram_entropy",
    "trigram_entropy",
    "word_entropy",
    "gzip_ratio",
    "bz2_ratio",
    "lzma_ratio",
    "gzip_bits_per_byte",
    "bz2_bits_per_byte",
    "lzma_bits_per_byte",
    "segment_count",
    "clean_text_words",
    "clean_text_length",
    "unique_words",
    "vocabulary_richness",
    "avg_word_length",
    "avg_words_per_segment",
]


def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    available = [c for c in BASE_CANDIDATE_FEATURES if c in df.columns]

    work = df[available + [TARGET]].copy()

    # Transformacoes para reduzir escala assimetrica em contagens grandes.
    for col in [
        "clean_text_words",
        "clean_text_length",
        "segment_count",
        "unique_words",
    ]:
        if col in work.columns:
            work[f"log1p_{col}"] = np.log1p(work[col].clip(lower=0))

    # Interacoes simples entre complexidade e tamanho de texto.
    if "char_entropy" in work.columns and "clean_text_words" in work.columns:
        work["entropy_x_words"] = work["char_entropy"] * np.log1p(
            work["clean_text_words"].clip(lower=0)
        )

    if "word_entropy" in work.columns and "segment_count" in work.columns:
        work["word_entropy_x_segments"] = work["word_entropy"] * np.log1p(
            work["segment_count"].clip(lower=0)
        )

    work = work.dropna()

    y = work[TARGET].copy()
    X = work.drop(columns=[TARGET])

    return X, y, list(X.columns)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    X, y, feature_names = build_feature_matrix(df)

    if X.empty:
        raise RuntimeError("Matriz de features vazia apos preparacao.")

    print(f"Linhas usadas: {len(X)}")
    print(f"Total de features usadas: {len(feature_names)}")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        print("XGBoost detectado: incluido na comparacao.")
    except Exception:
        print("XGBoost nao instalado: seguindo sem ele.")

    rows = []
    best_name = None
    best_r2 = -1e9

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={
                "r2": "r2",
                "mae": "neg_mean_absolute_error",
                "mse": "neg_mean_squared_error",
            },
            n_jobs=-1,
        )

        mean_r2 = scores["test_r2"].mean()
        mean_mae = -scores["test_mae"].mean()
        mean_rmse = (-scores["test_mse"].mean()) ** 0.5

        rows.append(
            {
                "model": name,
                "rows_used": len(X),
                "feature_count": len(feature_names),
                "cv_folds": 5,
                "r2_mean": mean_r2,
                "mae_mean": mean_mae,
                "rmse_mean": mean_rmse,
            }
        )

        print(f"[{name}] R2={mean_r2:.4f} | MAE={mean_mae:.4f} | RMSE={mean_rmse:.4f}")

        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_name = name

    ranking = pd.DataFrame(rows).sort_values("r2_mean", ascending=False)
    ranking.to_csv(OUTPUT_DIR / "12_model_comparison_cv.csv", index=False)

    if best_name is None:
        raise RuntimeError("Nao foi possivel selecionar melhor modelo.")

    best_model = models[best_name]
    y_pred_cv = cross_val_predict(best_model, X, y, cv=cv, n_jobs=-1)

    pred_df = X.copy()
    pred_df["rating_real"] = y.values
    pred_df["rating_pred_cv"] = y_pred_cv
    pred_df["abs_error"] = (pred_df["rating_real"] - pred_df["rating_pred_cv"]).abs()
    pred_df.to_csv(OUTPUT_DIR / "12_best_model_cv_predictions.csv", index=False)

    final_r2 = r2_score(y, y_pred_cv)
    final_mae = mean_absolute_error(y, y_pred_cv)
    final_rmse = rmse(y, y_pred_cv)

    summary = pd.DataFrame(
        [
            {
                "best_model": best_name,
                "rows_used": len(X),
                "feature_count": len(feature_names),
                "r2": final_r2,
                "mae": final_mae,
                "rmse": final_rmse,
            }
        ]
    )
    summary.to_csv(OUTPUT_DIR / "12_best_model_summary.csv", index=False)

    feature_list = pd.DataFrame({"feature": feature_names})
    feature_list.to_csv(OUTPUT_DIR / "12_features_used.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.scatter(y, y_pred_cv, alpha=0.35)
    low = min(y.min(), y_pred_cv.min())
    high = max(y.max(), y_pred_cv.max())
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Rating real")
    plt.ylabel("Rating previsto (CV)")
    plt.title(f"12 - Melhor modelo ({best_name})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "12_best_model_cv_pred_vs_real.png", dpi=180)
    plt.close()

    best_model.fit(X, y)
    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": best_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        fi.to_csv(OUTPUT_DIR / "12_best_model_feature_importance.csv", index=False)

    print("\nResumo final do 12")
    print(f"  Melhor modelo: {best_name}")
    print(f"  R2 (CV preditivo): {final_r2:.4f}")
    print(f"  MAE (CV preditivo): {final_mae:.4f}")
    print(f"  RMSE (CV preditivo): {final_rmse:.4f}")
    print(f"  Ranking salvo em: {OUTPUT_DIR / '12_model_comparison_cv.csv'}")
    print(f"  Resumo salvo em: {OUTPUT_DIR / '12_best_model_summary.csv'}")


if __name__ == "__main__":
    main()
