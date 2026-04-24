@echo off
chcp 65001 >nul
echo.
echo ╔══════════════════════════════════════════╗
echo ║   ShopBR · Demand Forecasting Pipeline   ║
echo ╚══════════════════════════════════════════╝
echo.

cd /d "%~dp0"

echo [1/4] Instalando dependencias...
pip install -r requirements.txt -q
if errorlevel 1 ( echo ERRO ao instalar dependencias & pause & exit /b 1 )

echo.
echo [2/4] Gerando dataset sintetico...
python src\gerar_dados.py
if errorlevel 1 ( echo ERRO ao gerar dados & pause & exit /b 1 )

echo.
echo [3/4] Criando features...
python src\feature_engineering.py
if errorlevel 1 ( echo ERRO no feature engineering & pause & exit /b 1 )

echo.
echo [4/4] Treinando modelos...
python src\treinar_modelo.py
if errorlevel 1 ( echo ERRO no treinamento & pause & exit /b 1 )

echo.
echo ==========================================
echo  Pipeline concluido com sucesso!
echo  Abrindo dashboard em http://localhost:5050
echo ==========================================
echo.
echo Pressione CTRL+C para encerrar o servidor.
echo.

python dashboard\app.py
pause
