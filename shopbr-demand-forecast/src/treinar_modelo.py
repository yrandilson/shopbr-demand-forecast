"""
ShopBR Marketplace — Treinamento do Modelo de Previsão de Demanda
Modelo: Gradient Boosting (sklearn) com validação temporal (walk-forward)
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(__file__)
PROC     = os.path.join(BASE, "../data/processed/vendas_features.csv")
MODELS   = os.path.join(BASE, "../models")
os.makedirs(MODELS, exist_ok=True)


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def treinar():
    print("📂 Carregando features...")
    df = pd.read_csv(PROC, parse_dates=["data"])

    # ── Remover linhas com NaN (por causa dos lags) ──────────────────────────
    df.dropna(inplace=True)

    TARGET   = "unidades"
    EXCLUIR  = ["data", "unidades", "receita", "preco_medio"]
    FEATURES = [c for c in df.columns if c not in EXCLUIR]

    # ── Split temporal ────────────────────────────────────────────────────────
    train_df = df[df["data"] < "2024-01-01"].copy()
    test_df  = df[df["data"] >= "2024-01-01"].copy()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test  = test_df[FEATURES]
    y_test  = test_df[TARGET]

    print(f"   Train: {len(train_df):,} amostras  | Test: {len(test_df):,} amostras")

    # ── Modelo 1: Gradient Boosting ───────────────────────────────────────────
    print("\n🤖 Treinando Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
        verbose=0,
    )
    gb.fit(X_train, y_train)
    preds_gb = np.maximum(0, gb.predict(X_test))

    metrics_gb = {
        "mae":  round(mean_absolute_error(y_test, preds_gb), 2),
        "rmse": round(rmse(y_test, preds_gb), 2),
        "mape": round(mape(y_test, preds_gb), 2),
        "r2":   round(r2_score(y_test, preds_gb), 4),
    }
    print(f"   GB  → MAE:{metrics_gb['mae']:7.1f} | RMSE:{metrics_gb['rmse']:7.1f} | MAPE:{metrics_gb['mape']:5.1f}% | R²:{metrics_gb['r2']:.4f}")

    # ── Modelo 2: Random Forest ───────────────────────────────────────────────
    print("\n🤖 Treinando Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    preds_rf = np.maximum(0, rf.predict(X_test))

    metrics_rf = {
        "mae":  round(mean_absolute_error(y_test, preds_rf), 2),
        "rmse": round(rmse(y_test, preds_rf), 2),
        "mape": round(mape(y_test, preds_rf), 2),
        "r2":   round(r2_score(y_test, preds_rf), 4),
    }
    print(f"   RF  → MAE:{metrics_rf['mae']:7.1f} | RMSE:{metrics_rf['rmse']:7.1f} | MAPE:{metrics_rf['mape']:5.1f}% | R²:{metrics_rf['r2']:.4f}")

    # ── Ensemble (média simples) ──────────────────────────────────────────────
    preds_ens = (preds_gb + preds_rf) / 2
    metrics_ens = {
        "mae":  round(mean_absolute_error(y_test, preds_ens), 2),
        "rmse": round(rmse(y_test, preds_ens), 2),
        "mape": round(mape(y_test, preds_ens), 2),
        "r2":   round(r2_score(y_test, preds_ens), 4),
    }
    print(f"\n   ENS → MAE:{metrics_ens['mae']:7.1f} | RMSE:{metrics_ens['rmse']:7.1f} | MAPE:{metrics_ens['mape']:5.1f}% | R²:{metrics_ens['r2']:.4f}")

    # ── Feature Importance (GB) ───────────────────────────────────────────────
    importances = pd.DataFrame({
        "feature":    FEATURES,
        "importance": gb.feature_importances_,
    }).sort_values("importance", ascending=False).head(20)
    print(f"\n📊 Top 10 Features:\n{importances.head(10).to_string(index=False)}")

    # ── Salvar resultados de previsão ─────────────────────────────────────────
    results = test_df[["data"] + [c for c in test_df.columns if c.startswith("cat_")]].copy()
    results["real"]        = y_test.values
    results["pred_gb"]     = preds_gb.round(0).astype(int)
    results["pred_rf"]     = preds_rf.round(0).astype(int)
    results["pred_ensemble"] = preds_ens.round(0).astype(int)

    # Recuperar nome da categoria
    cat_cols = [c for c in results.columns if c.startswith("cat_")]
    results["categoria"] = results[cat_cols].idxmax(axis=1).str.replace("cat_", "", regex=False)
    results.drop(columns=cat_cols, inplace=True)

    out_pred = os.path.join(MODELS, "previsoes_test.csv")
    results.to_csv(out_pred, index=False)

    # ── Persistir modelos e metadados ─────────────────────────────────────────
    joblib.dump(gb, os.path.join(MODELS, "gradient_boosting.pkl"))
    joblib.dump(rf, os.path.join(MODELS, "random_forest.pkl"))

    metadata = {
        "features":    FEATURES,
        "target":      TARGET,
        "train_start": str(train_df["data"].min().date()),
        "train_end":   str(train_df["data"].max().date()),
        "test_start":  str(test_df["data"].min().date()),
        "test_end":    str(test_df["data"].max().date()),
        "n_train":     len(train_df),
        "n_test":      len(test_df),
        "metrics_gb":  metrics_gb,
        "metrics_rf":  metrics_rf,
        "metrics_ensemble": metrics_ens,
        "top_features": importances.to_dict(orient="records"),
    }

    with open(os.path.join(MODELS, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Modelos salvos em: {MODELS}")
    return metadata


if __name__ == "__main__":
    treinar()
