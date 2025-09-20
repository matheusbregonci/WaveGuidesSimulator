@echo off
title Waveguide Simulator - Criar Versao Portavel

echo ================================================
echo    CRIANDO VERSAO PORTAVEL - STREAMLIT
echo ================================================
echo.

echo [1/4] Verificando ambiente virtual...
if not exist venv (
    echo ERRO: Ambiente virtual nao encontrado!
    echo Execute primeiro: install_and_run.bat
    pause
    exit /b 1
)
echo ✓ Ambiente virtual encontrado!

echo.
echo [2/4] Criando pasta portavel...
if exist WaveguideSimulator_Portable rmdir /s /q WaveguideSimulator_Portable
mkdir WaveguideSimulator_Portable
echo ✓ Pasta criada!

echo.
echo [3/4] Copiando arquivos...
xcopy /E /I venv WaveguideSimulator_Portable\venv
copy launch.py WaveguideSimulator_Portable\
copy run_app.py WaveguideSimulator_Portable\
copy streamlit_config.toml WaveguideSimulator_Portable\
xcopy /E /I src WaveguideSimulator_Portable\src
xcopy /E /I models_old WaveguideSimulator_Portable\models_old
copy README.md WaveguideSimulator_Portable\
echo ✓ Arquivos copiados!

echo.
echo [4/4] Criando launcher portavel...
echo @echo off > WaveguideSimulator_Portable\START_WAVEGUIDE_SIMULATOR.bat
echo title Waveguide Simulator - Versao Portavel >> WaveguideSimulator_Portable\START_WAVEGUIDE_SIMULATOR.bat
echo echo ================================================ >> WaveguideSimulator_Portable\START_WAVEGUIDE_SIMULATOR.bat
echo echo    WAVEGUIDE SIMULATOR - VERSAO PORTAVEL >> WaveguideSimulator_Portable\START_WAVEGUIDE_SIMULATOR.bat
echo echo ================================================ >> WaveguideSimulator_Portable\START_WAVEGUIDE_SIMULATOR.bat
echo echo. >> WaveguideSimulator_Portable\START_WAVEGUIDE_SIMULATOR.bat
echo echo Iniciando aplicativo... >> WaveguideSimulator_Portable\START_WAVEGUIDE_SIMULATOR.bat
echo echo O navegador abrira automaticamente >> WaveguideSimulator_Portable\START_WAVEGUIDE_SIMULATOR.bat
echo echo Para parar: Ctrl+C nesta janela >> WaveguideSimulator_Portable\START_WAVEGUIDE_SIMULATOR.bat
echo echo ================================================ >> WaveguideSimulator_Portable\START_WAVEGUIDE_SIMULATOR.bat
echo echo. >> WaveguideSimulator_Portable\START_WAVEGUIDE_SIMULATOR.bat
echo call venv\Scripts\activate.bat >> WaveguideSimulator_Portable\START_WAVEGUIDE_SIMULATOR.bat
echo python run_app.py >> WaveguideSimulator_Portable\START_WAVEGUIDE_SIMULATOR.bat
echo pause >> WaveguideSimulator_Portable\START_WAVEGUIDE_SIMULATOR.bat

echo ✓ Launcher criado!

echo.
echo ================================================
echo    VERSAO PORTAVEL CRIADA COM SUCESSO!
echo ================================================
echo.
echo Pasta: WaveguideSimulator_Portable\
echo.
echo Para usar:
echo 1. Copie a pasta WaveguideSimulator_Portable para qualquer local
echo 2. Execute: START_WAVEGUIDE_SIMULATOR.bat
echo.
echo Esta versao funciona sem instalacao adicional!
echo ================================================

pause