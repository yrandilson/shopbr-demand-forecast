"""
ShopBR Marketplace — Feature Engineering
Gera features de séries temporais para o modelo de previsão de demanda.
"""

import pandas as pd
import numpy as np
import os


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["data"] = pd.to_datetime(df["data"])
    df.sort_values(["categoria", "data"], inplace=True)

    resultado = []

    for cat, grupo in df.groupby("categoria"):
        grupo = grupo.copy().reset_index(drop=True)
        serie = grupo["unidades"]

        # ── Features de calendário ─────────────────────────────────────────
        grupo["dia_semana"]     = grupo["data"].dt.dayofweek
        grupo["mes"]            = grupo["data"].dt.month
        grupo["semana_ano"]     = grupo["data"].dt.isocalendar().week.astype(int)
        grupo["dia_mes"]        = grupo["data"].dt.day
        grupo["trimestre"]      = grupo["data"].dt.quarter
        grupo["is_fim_semana"]  = (grupo["dia_semana"] >= 5).astype(int)

        # ── Sazonalidade cíclica (sin/cos) ────────────────────────────────
        grupo["mes_sin"]        = np.sin(2 * np.pi * grupo["mes"] / 12)
        grupo["mes_cos"]        = np.cos(2 * np.pi * grupo["mes"] / 12)
        grupo["dow_sin"]        = np.sin(2 * np.pi * grupo["dia_semana"] / 7)
        grupo["dow_cos"]        = np.cos(2 * np.pi * grupo["dia_semana"] / 7)

        # ── Flags de feriados comerciais ──────────────────────────────────
        def flag_feriado(row):
            m, d = row["mes"], row["dia_mes"]
            if m == 11 and 20 <= d <= 30: return 1   # Black Friday
            if m == 12 and 15 <= d <= 31: return 1   # Natal
            if m == 5  and  8 <= d <= 14: return 1   # Dia das Mães
            if m == 6  and  8 <= d <= 14: return 1   # Dia dos Namorados
            if m == 8  and  8 <= d <= 14: return 1   # Dia dos Pais
            return 0

        grupo["is_feriado_comercial"] = grupo.apply(flag_feriado, axis=1)

        # ── Lags ──────────────────────────────────────────────────────────
        for lag in [1, 7, 14, 21, 28, 35]:
            grupo[f"lag_{lag}d"] = serie.shift(lag)

        # ── Rolling stats ─────────────────────────────────────────────────
        for janela in [7, 14, 28]:
            grupo[f"roll_mean_{janela}d"] = serie.shift(1).rolling(janela).mean()
            grupo[f"roll_std_{janela}d"]  = serie.shift(1).rolling(janela).std()
            grupo[f"roll_max_{janela}d"]  = serie.shift(1).rolling(janela).max()
            grupo[f"roll_min_{janela}d"]  = serie.shift(1).rolling(janela).min()

        # ── Diferença (primeira ordem) ────────────────────────────────────
        grupo["diff_1d"]  = serie.shift(1).diff(1)
        grupo["diff_7d"]  = serie.shift(1).diff(7)
        grupo["diff_28d"] = serie.shift(1).diff(28)

        # ── Trend (posição relativa na série) ─────────────────────────────
        grupo["trend_idx"] = np.arange(len(grupo))

        # ── Receita histórica (rolling) ───────────────────────────────────
        grupo["roll_receita_7d"]  = grupo["receita"].shift(1).rolling(7).mean()
        grupo["roll_receita_28d"] = grupo["receita"].shift(1).rolling(28).mean()

        resultado.append(grupo)

    out = pd.concat(resultado).reset_index(drop=True)

    # One-hot encoding de categoria
    out = pd.get_dummies(out, columns=["categoria"], prefix="cat", drop_first=False)

    return out


if __name__ == "__main__":
    raw_path  = os.path.join(os.path.dirname(__file__), "../data/raw/vendas.csv")
    proc_path = os.path.join(os.path.dirname(__file__), "../data/processed/vendas_features.csv")

    df = pd.read_csv(raw_path, parse_dates=["data"])
    df_feat = criar_features(df)
    df_feat.to_csv(proc_path, index=False)
    print(f"✅ Features geradas: {df_feat.shape} → {proc_path}")
    print(f"   Features: {list(df_feat.columns)}")
