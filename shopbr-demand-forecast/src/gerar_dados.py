"""
ShopBR Marketplace — Gerador de Dados Sintéticos
Simula vendas diárias de 2022-01-01 a 2024-03-31 com sazonalidade realista.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ── Configurações ─────────────────────────────────────────────────────────────
CATEGORIAS = {
    "Eletrônicos":    {"base": 320, "preco_medio": 850.0,  "trend": 0.0003},
    "Moda":           {"base": 480, "preco_medio": 180.0,  "trend": 0.0001},
    "Casa & Jardim":  {"base": 210, "preco_medio": 260.0,  "trend": 0.0002},
    "Beleza":         {"base": 390, "preco_medio": 95.0,   "trend": 0.0004},
    "Esportes":       {"base": 155, "preco_medio": 320.0,  "trend": 0.0002},
}

# Feriados comerciais brasileiros (mês, semana_do_mes) -> multiplicador
FERIADOS = {
    (5, 2):  1.45,   # Dia das Mães
    (6, 2):  1.30,   # Dia dos Namorados
    (8, 2):  1.25,   # Dia dos Pais
    (11, 4): 2.80,   # Black Friday
    (12, 2): 1.60,   # Natal (compras antecipadas)
    (12, 3): 1.85,   # Natal
    (12, 4): 1.40,   # pós-natal / presente
    (1, 1):  0.55,   # Janeiro fraco
    (2, 1):  0.60,   # Carnaval
}

SAZONALIDADE_MES = {
    1: 0.72, 2: 0.68, 3: 0.85, 4: 0.90,
    5: 1.05, 6: 1.02, 7: 0.95, 8: 1.00,
    9: 0.92, 10: 0.97, 11: 1.55, 12: 1.70,
}

SAZONALIDADE_DIA = {0: 0.88, 1: 0.85, 2: 0.90, 3: 0.93, 4: 1.05, 5: 1.20, 6: 1.18}


def gerar_vendas():
    datas = pd.date_range("2022-01-01", "2024-03-31", freq="D")
    registros = []

    for cat, cfg in CATEGORIAS.items():
        base     = cfg["base"]
        preco    = cfg["preco_medio"]
        trend    = cfg["trend"]

        for i, data in enumerate(datas):
            mes   = data.month
            dow   = data.dayofweek
            semana_mes = (data.day - 1) // 7 + 1

            mult_sazonal = SAZONALIDADE_MES[mes]
            mult_dow     = SAZONALIDADE_DIA[dow]
            mult_feriado = FERIADOS.get((mes, semana_mes), 1.0)
            mult_trend   = 1 + trend * i

            # Ruído multiplicativo
            ruido = np.random.lognormal(0, 0.12)

            unidades = int(base * mult_sazonal * mult_dow * mult_feriado * mult_trend * ruido)
            unidades = max(1, unidades)

            # Variação de preço ±8%
            preco_venda = round(preco * np.random.uniform(0.92, 1.08), 2)
            receita     = round(unidades * preco_venda, 2)

            registros.append({
                "data":       data,
                "categoria":  cat,
                "unidades":   unidades,
                "preco_medio": preco_venda,
                "receita":    receita,
            })

    df = pd.DataFrame(registros)
    df.sort_values(["data", "categoria"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == "__main__":
    df = gerar_vendas()
    out = os.path.join(os.path.dirname(__file__), "../data/raw/vendas.csv")
    df.to_csv(out, index=False)
    print(f"✅ Dataset gerado: {len(df):,} registros → {out}")
    print(df.describe())
