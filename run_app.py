#!/usr/bin/env python3
"""
Launcher principal para o Waveguide Simulator
Este é o ponto de entrada para o executável
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Função principal que inicia o app Streamlit"""
    try:
        # Configurar paths
        current_dir = Path(__file__).parent
        launch_script = current_dir / "launch.py"

        # Verificar se o launch.py existe
        if not launch_script.exists():
            print(f"❌ Erro: Arquivo {launch_script} não encontrado!")
            input("Pressione Enter para sair...")
            return

        # Executar streamlit
        print("🚀 Iniciando Waveguide Simulator...")
        print("🌐 O aplicativo abrirá no seu navegador padrão")
        print("📋 Para parar o aplicativo, pressione Ctrl+C neste terminal")
        print("-" * 60)

        # Comando para executar streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(launch_script)]

        # Executar o comando
        subprocess.run(cmd, cwd=current_dir)

    except KeyboardInterrupt:
        print("\n👋 Aplicativo interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro ao iniciar aplicativo: {e}")
        input("Pressione Enter para sair...")

if __name__ == "__main__":
    main()