@echo off
title Waveguide Simulator - Build Executavel

echo ================================================
echo    CRIANDO EXECUTAVEL - WAVEGUIDE SIMULATOR
echo ================================================
echo.

echo [1/5] Verificando ambiente virtual...
if not exist venv (
    echo ERRO: Ambiente virtual nao encontrado!
    echo Execute primeiro: install_and_run.bat
    pause
    exit /b 1
)
echo ✓ Ambiente virtual encontrado!

echo.
echo [2/5] Ativando ambiente virtual...
call venv\Scripts\activate.bat

echo.
echo [3/5] Instalando cx_Freeze para build...
pip install cx_Freeze
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Falha ao instalar cx_Freeze!
    pause
    exit /b 1
)
echo ✓ cx_Freeze instalado!

echo.
echo [4/5] Construindo executavel...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo Tentando build com cx_Freeze...
python setup.py build
if %ERRORLEVEL% NEQ 0 (
    echo AVISO: cx_Freeze falhou. Isso e normal para Streamlit.
    echo Use uma das alternativas:
    echo 1. create_portable.bat (RECOMENDADO)
    echo 2. build_pyinstaller.bat
    pause
    exit /b 1
)
echo ✓ Executavel construido!

echo.
echo [5/5] Finalizando...
echo ================================================
echo    BUILD CONCLUIDO COM SUCESSO!
echo ================================================
echo.
echo Executavel criado em: build\exe.win-amd64-3.xx\
echo.
echo Para executar:
echo 1. Va na pasta build\exe.win-amd64-3.xx\
echo 2. Execute WaveguideSimulator.exe
echo.
echo NOTA: Copie toda a pasta para distribuir!
echo ================================================

pause