#!/bin/bash
# ShopBR Marketplace — Pipeline Completo
set -e

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   ShopBR · Demand Forecasting Pipeline   ║"
echo "╚══════════════════════════════════════════╝"
echo ""

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "📦 [1/4] Instalando dependências..."
pip install -r requirements.txt -q

echo ""
echo "🗃️  [2/4] Gerando dataset sintético..."
python src/gerar_dados.py

echo ""
echo "⚙️  [3/4] Criando features..."
python src/feature_engineering.py

echo ""
echo "🤖 [4/4] Treinando modelos..."
python src/treinar_modelo.py

echo ""
echo "✅ Pipeline concluído!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Iniciando dashboard em http://localhost:5050"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python dashboard/app.py
