"""
Setup script para criar executável do Waveguide Simulator
"""

from cx_Freeze import setup, Executable
import sys
import os

# Configurações para o build
build_exe_options = {
    "packages": [
        "streamlit", "numpy", "matplotlib", "plotly", "scipy",
        "reportlab", "PIL", "io", "base64", "datetime", "typing",
        "pathlib", "sys", "os", "subprocess", "webbrowser"
    ],
    "excludes": ["tkinter", "test", "unittest"],
    "include_files": [
        ("launch.py", "launch.py"),  # Incluir launch.py na raiz
        ("src/", "src/"),
        ("models_old/", "models_old/"),
        ("streamlit_config.toml", ".streamlit/config.toml"),
    ],
    "zip_include_packages": [],  # Não zipar pacotes importantes
    "zip_exclude_packages": ["*"],  # Excluir tudo do zip
    "optimize": 1
}

base = None
if sys.platform == "win32":
    base = "Console"  # Use "Win32GUI" para remover console

setup(
    name="WaveguideSimulator",
    version="2.0",
    description="Simulador de Guias de Onda Eletromagnéticas",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "main_executable.py",
            base=base,
            target_name="WaveguideSimulator.exe",
            icon=None  # Adicione um ícone .ico se desejar
        )
    ]
)