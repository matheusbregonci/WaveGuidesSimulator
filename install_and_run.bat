@echo off
title Waveguide Simulator - Instalacao e Execucao

echo ================================================
echo    WAVEGUIDE SIMULATOR - INSTALACAO AUTOMATICA
echo ================================================
echo.

echo [1/4] Verificando Python...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Python nao encontrado!
    echo Por favor, instale Python 3.8+ antes de continuar.
    pause
    exit /b 1
)
echo ✓ Python encontrado!

echo.
echo [2/4] Criando ambiente virtual...
if exist venv rmdir /s /q venv
python -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Falha ao criar ambiente virtual!
    pause
    exit /b 1
)
echo ✓ Ambiente virtual criado!

echo.
echo [3/4] Ativando ambiente e instalando dependencias...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Falha ao instalar dependencias!
    pause
    exit /b 1
)
echo ✓ Dependencias instaladas!

echo.
echo [4/4] Iniciando aplicativo...
echo.
echo ================================================
echo    WAVEGUIDE SIMULATOR INICIADO COM SUCESSO!
echo ================================================
echo.
echo O aplicativo abriu no seu navegador?
echo Se nao, acesse: http://localhost:8501
echo.
echo Para parar o aplicativo: Ctrl+C
echo ================================================

python run_app.py

echo.
echo Aplicativo finalizado.
pause