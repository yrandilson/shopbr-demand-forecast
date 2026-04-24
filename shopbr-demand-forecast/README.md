# 📦 ShopBR Marketplace — Previsão de Demanda

> **Projeto de Data Science** | E-commerce · Varejo | Squad: Data & Analytics

---

## 🎯 Contexto de Negócio

A **ShopBR Marketplace** é um e-commerce brasileiro com 5 categorias de produtos. Com crescimento acelerado no pós-pandemia, a empresa enfrentou dois problemas críticos:

- **Stockouts** em períodos sazonais (Black Friday, Natal), gerando perda de receita estimada em R$ 4.2M/ano
- **Excesso de estoque** em categorias de baixo giro em janeiro/fevereiro, imobilizando capital

**Solução proposta:** Modelo de Machine Learning para previsão de demanda com horizonte de 7-30 dias, alimentando o sistema de reposição automática de estoque.

---

## 🏗️ Estrutura do Projeto

```
shopbr-demand-forecast/
├── data/
│   ├── raw/                   # Dados brutos de vendas (CSV)
│   └── processed/             # Features engineered (CSV)
├── notebooks/
│   ├── 01_EDA.ipynb           # Análise exploratória
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Modelagem.ipynb     # Treinamento, avaliação, comparação
├── src/
│   ├── gerar_dados.py         # Gerador de dados sintéticos
│   ├── feature_engineering.py # Pipeline de features
│   └── treinar_modelo.py      # Treinamento + avaliação
├── models/
│   ├── gradient_boosting.pkl  # Modelo treinado
│   ├── random_forest.pkl      # Modelo treinado
│   ├── previsoes_test.csv     # Previsões no período de teste
│   └── metadata.json          # Métricas e metadados
├── dashboard/
│   ├── app.py                 # Flask API + servidor
│   └── templates/index.html   # Dashboard interativo
├── requirements.txt
└── run.sh                     # Script de execução completo
```

---

## 📊 Resultados

| Modelo | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| Gradient Boosting | 78.9 | 97.4 | 43.0% | 0.39 |
| **Random Forest** | **42.2** | **82.1** | **19.6%** | **0.57** |
| Ensemble (média) | 54.7 | 83.0 | 29.0% | 0.56 |

> O **Random Forest** obteve melhor desempenho e foi selecionado para produção.

### Top Features
1. `lag_1d` — Vendas do dia anterior (autocorrelação forte)
2. `roll_min_14d` — Mínimo das últimas 2 semanas
3. `roll_mean_7d` — Média móvel 7 dias
4. `semana_ano` — Sazonalidade anual
5. `dia_semana` — Sazonalidade semanal

---

## 🚀 Como Executar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Executar pipeline completo (dados → features → modelo → dashboard)
```bash
bash run.sh
```

### 3. Ou executar passo a passo
```bash
python src/gerar_dados.py         # Gera dataset sintético
python src/feature_engineering.py # Cria features
python src/treinar_modelo.py      # Treina e avalia modelos
python dashboard/app.py           # Sobe dashboard em http://localhost:5050
```

### 4. Abrir notebooks
```bash
jupyter notebook notebooks/
```

---

## 🛠️ Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.12 |
| Dados | Pandas, NumPy |
| Machine Learning | Scikit-learn (RF, GB) |
| Visualização | Matplotlib, Seaborn |
| API | Flask |
| Frontend | Chart.js, HTML/CSS |
| Persistência | Joblib (modelos), SQLite (futuro) |

---

## 📈 Próximos Passos (Roadmap)

- [ ] Integração com sistema de ERP para dados reais
- [ ] Retreinamento automático mensal (Airflow/cron)
- [ ] Expansão do horizonte de previsão para 90 dias
- [ ] Modelo por SKU individual (granularidade produto)
- [ ] Deploy em container Docker + Kubernetes
- [ ] Alertas automáticos por e-mail/Slack para anomalias

---

## 👥 Equipe

| Nome | Papel |
|---|---|
| Data Science Squad | Modelagem, Feature Engineering, Avaliação |
| Data Engineering | Pipeline de dados, ETL |
| Product | Definição de requisitos, métricas de negócio |

---

*ShopBR Marketplace · Data Science Team · 2024*
