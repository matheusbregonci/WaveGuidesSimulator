@echo off
title Waveguide Simulator - Build PyInstaller

echo ================================================
echo    CRIANDO EXECUTAVEL - PYINSTALLER
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
echo [3/5] Instalando PyInstaller...
pip install pyinstaller
if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Falha ao instalar PyInstaller!
    pause
    exit /b 1
)
echo ✓ PyInstaller instalado!

echo.
echo [4/5] Construindo executavel com PyInstaller...
if exist dist rmdir /s /q dist
if exist build_temp rmdir /s /q build_temp

pyinstaller --onedir ^
    --console ^
    --name "WaveguideSimulator" ^
    --distpath "dist" ^
    --workpath "build_temp" ^
    --add-data "launch.py;." ^
    --add-data "src;src" ^
    --add-data "models_old;models_old" ^
    --add-data "streamlit_config.toml;.streamlit" ^
    --hidden-import "streamlit" ^
    --hidden-import "plotly" ^
    --hidden-import "matplotlib" ^
    --hidden-import "reportlab" ^
    --collect-all "streamlit" ^
    --collect-all "plotly" ^
    main_executable.py

if %ERRORLEVEL% NEQ 0 (
    echo ERRO: Falha ao construir executavel!
    pause
    exit /b 1
)
echo ✓ Executavel construido!

echo.
echo [5/5] Finalizando...
echo ================================================
echo    BUILD PYINSTALLER CONCLUIDO!
echo ================================================
echo.
echo Executavel criado em: dist\WaveguideSimulator\
echo.
echo Para executar:
echo 1. Va na pasta dist\WaveguideSimulator\
echo 2. Execute WaveguideSimulator.exe
echo.
echo NOTA: Copie toda a pasta dist\WaveguideSimulator\ para distribuir!
echo ================================================

pause