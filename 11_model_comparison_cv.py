from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_validate

DATA = Path("data")
INPUT_CSV = DATA / "processed" / "movies_information_features.csv"
OUTPUT_DIR = DATA / "analysis"

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


def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {INPUT_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    entropy_col = resolve_feature(df, "entropy")
    gzip_col = resolve_feature(df, "gzip_ratio")
    word_count_col = resolve_feature(df, "word_count")
    used_features = [entropy_col, gzip_col, word_count_col]

    model_df = df[used_features + [TARGET]].dropna().copy()
    if model_df.empty:
        raise RuntimeError("Sem dados apos dropna para comparacao de modelos.")

    X = model_df[used_features]
    y = model_df[TARGET]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    }

    # XGBoost is optional. If installed, include it in the comparison.
    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
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
                "rows_used": len(model_df),
                "features": ",".join(used_features),
                "cv_folds": 5,
                "r2_mean": mean_r2,
                "mae_mean": mean_mae,
                "rmse_mean": mean_rmse,
            }
        )

        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_name = name

        print(f"[{name}] R2={mean_r2:.4f} | MAE={mean_mae:.4f} | RMSE={mean_rmse:.4f}")

    metrics_df = pd.DataFrame(rows).sort_values("r2_mean", ascending=False)
    metrics_df.to_csv(OUTPUT_DIR / "model_comparison_cv.csv", index=False)

    if best_name is None:
        raise RuntimeError("Nenhum modelo valido para gerar previsoes.")

    best_model = models[best_name]
    y_pred_cv = cross_val_predict(best_model, X, y, cv=cv, n_jobs=-1)

    pred_df = X.copy()
    pred_df["rating_real"] = y.values
    pred_df["rating_pred_cv"] = y_pred_cv
    pred_df["abs_error"] = (pred_df["rating_real"] - pred_df["rating_pred_cv"]).abs()
    pred_df.to_csv(OUTPUT_DIR / "best_model_cv_predictions.csv", index=False)

    final_r2 = r2_score(y, y_pred_cv)
    final_mae = mean_absolute_error(y, y_pred_cv)
    final_rmse = rmse(y, y_pred_cv)

    plt.figure(figsize=(7, 5))
    plt.scatter(y, y_pred_cv, alpha=0.4)
    low = min(y.min(), y_pred_cv.min())
    high = max(y.max(), y_pred_cv.max())
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Rating real")
    plt.ylabel("Rating previsto (CV)")
    plt.title(f"Melhor modelo ({best_name}) - Real vs Previsto")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_model_cv_pred_vs_real.png", dpi=180)
    plt.close()

    best_model.fit(X, y)
    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame(
            {
                "feature": used_features,
                "importance": best_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        fi.to_csv(OUTPUT_DIR / "best_model_feature_importance.csv", index=False)

        plt.figure(figsize=(7, 4))
        plt.bar(fi["feature"], fi["importance"])
        plt.title(f"Importancia de features ({best_name})")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "best_model_feature_importance.png", dpi=180)
        plt.close()

    summary = pd.DataFrame(
        [
            {
                "best_model": best_name,
                "rows_used": len(model_df),
                "r2": final_r2,
                "mae": final_mae,
                "rmse": final_rmse,
                "features": ",".join(used_features),
            }
        ]
    )
    summary.to_csv(OUTPUT_DIR / "best_model_summary.csv", index=False)

    print("\nResumo final")
    print(f"  Melhor modelo: {best_name}")
    print(f"  R2 (CV preditivo): {final_r2:.4f}")
    print(f"  MAE (CV preditivo): {final_mae:.4f}")
    print(f"  RMSE (CV preditivo): {final_rmse:.4f}")
    print(f"  Ranking salvo em: {OUTPUT_DIR / 'model_comparison_cv.csv'}")
    print(f"  Predicoes salvas em: {OUTPUT_DIR / 'best_model_cv_predictions.csv'}")


if __name__ == "__main__":
    main()
