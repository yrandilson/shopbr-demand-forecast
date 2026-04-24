"""
ShopBR Marketplace — Dashboard de Previsão de Demanda
Flask + Chart.js | Dark theme
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import json
import os

app = Flask(__name__)

BASE   = os.path.dirname(__file__)
MODELS = os.path.join(BASE, "../models")
RAW    = os.path.join(BASE, "../data/raw/vendas.csv")


def _load():
    preds   = pd.read_csv(os.path.join(MODELS, "previsoes_test.csv"), parse_dates=["data"])
    raw     = pd.read_csv(RAW, parse_dates=["data"])
    with open(os.path.join(MODELS, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return preds, raw, meta


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/overview")
def api_overview():
    preds, raw, meta = _load()
    m = meta["metrics_rf"]
    return jsonify({
        "mae":           m["mae"],
        "rmse":          m["rmse"],
        "mape":          m["mape"],
        "r2":            m["r2"],
        "total_real":    int(preds["real"].sum()),
        "total_pred":    int(preds["pred_rf"].sum()),
        "train_start":   meta["train_start"],
        "train_end":     meta["train_end"],
        "test_start":    meta["test_start"],
        "test_end":      meta["test_end"],
        "n_train":       meta["n_train"],
        "n_test":        meta["n_test"],
        "categorias":    sorted(preds["categoria"].unique().tolist()),
    })


@app.route("/api/serie_temporal")
def api_serie_temporal():
    cat   = request.args.get("categoria", "Todas")
    preds, raw, _ = _load()

    if cat != "Todas":
        preds = preds[preds["categoria"] == cat]

    grp = preds.groupby("data").agg(
        real=("real", "sum"),
        pred=("pred_rf", "sum"),
    ).reset_index()

    # Agregar por semana para o gráfico ficar mais limpo
    grp["semana"] = grp["data"].dt.to_period("W").dt.start_time
    grp_sem = grp.groupby("semana").agg(real=("real","sum"), pred=("pred","sum")).reset_index()

    return jsonify({
        "labels": grp_sem["semana"].dt.strftime("%d/%m/%Y").tolist(),
        "real":   grp_sem["real"].tolist(),
        "pred":   grp_sem["pred"].tolist(),
    })


@app.route("/api/por_categoria")
def api_por_categoria():
    preds, raw, _ = _load()
    grp = preds.groupby("categoria").agg(
        real=("real", "sum"),
        pred=("pred_rf", "sum"),
    ).reset_index()

    return jsonify({
        "categorias": grp["categoria"].tolist(),
        "real":       grp["real"].tolist(),
        "pred":       grp["pred"].tolist(),
    })


@app.route("/api/historico_vendas")
def api_historico_vendas():
    _, raw, _ = _load()
    cat = request.args.get("categoria", "Todas")
    if cat != "Todas":
        raw = raw[raw["categoria"] == cat]

    grp = raw.groupby("data").agg(unidades=("unidades","sum"), receita=("receita","sum")).reset_index()
    grp["semana"] = grp["data"].dt.to_period("W").dt.start_time
    grp_sem = grp.groupby("semana").agg(unidades=("unidades","sum"), receita=("receita","sum")).reset_index()

    return jsonify({
        "labels":    grp_sem["semana"].dt.strftime("%d/%m/%Y").tolist(),
        "unidades":  grp_sem["unidades"].tolist(),
        "receita":   [round(r/1000, 1) for r in grp_sem["receita"].tolist()],
    })


@app.route("/api/feature_importance")
def api_feature_importance():
    _, _, meta = _load()
    top = meta["top_features"][:12]
    label_map = {
        "lag_1d": "Lag 1 dia", "lag_7d": "Lag 7 dias", "lag_14d": "Lag 14 dias",
        "lag_21d": "Lag 21 dias", "lag_28d": "Lag 28 dias", "lag_35d": "Lag 35 dias",
        "roll_mean_7d": "Média 7d", "roll_std_7d": "Desvio 7d", "roll_min_7d": "Mín 7d",
        "roll_min_14d": "Mín 14d", "roll_mean_14d": "Média 14d",
        "semana_ano": "Semana do Ano", "dia_mes": "Dia do Mês",
        "dia_semana": "Dia da Semana", "diff_28d": "Δ 28 dias",
        "mes": "Mês", "mes_sin": "Mês (sin)", "trend_idx": "Tendência",
    }
    return jsonify({
        "features":    [label_map.get(r["feature"], r["feature"]) for r in top],
        "importances": [round(r["importance"]*100, 2) for r in top],
    })


@app.route("/api/residuos")
def api_residuos():
    preds, _, _ = _load()
    res = (preds["real"] - preds["pred_rf"]).tolist()
    # Histograma manual
    hist, edges = np.histogram(res, bins=30)
    centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]
    return jsonify({
        "bins":   [round(c, 1) for c in centers],
        "counts": hist.tolist(),
    })


@app.route("/api/tabela")
def api_tabela():
    preds, raw, _ = _load()

    # Juntar receita do período de teste
    rec = raw[raw["data"] >= "2024-01-01"].groupby(["data","categoria"])["receita"].sum().reset_index()
    merged = preds.merge(rec, on=["data","categoria"], how="left")

    grp = merged.groupby("categoria").agg(
        real_total   = ("real",    "sum"),
        pred_total   = ("pred_rf", "sum"),
        receita_real = ("receita", "sum"),
    ).reset_index()

    grp["erro_pct"] = ((grp["pred_total"] - grp["real_total"]) / grp["real_total"] * 100).round(1)
    grp["receita_real"] = grp["receita_real"].round(0)

    return jsonify(grp.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True, port=5050)
