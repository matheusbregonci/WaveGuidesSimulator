#!/usr/bin/env python3
"""
Arquivo principal para o executável do Waveguide Simulator
Este arquivo substitui o run_app.py para executável cx_Freeze
"""

import sys
import os
import subprocess
from pathlib import Path
import webbrowser
import time

def main():
    """Função principal que inicia o app Streamlit"""
    try:
        # Configurar paths - para executável
        if getattr(sys, 'frozen', False):
            # Executável cx_Freeze
            current_dir = Path(sys.executable).parent
            launch_script = current_dir / "launch.py"
        else:
            # Desenvolvimento
            current_dir = Path(__file__).parent
            launch_script = current_dir / "launch.py"

        print("🚀 Iniciando Waveguide Simulator...")
        print(f"📁 Diretório atual: {current_dir}")
        print(f"📄 Script launcher: {launch_script}")

        # Verificar se o launch.py existe
        if not launch_script.exists():
            print(f"❌ Erro: Arquivo {launch_script} não encontrado!")
            print("📁 Arquivos disponíveis:")
            for file in current_dir.iterdir():
                print(f"   - {file.name}")
            input("Pressione Enter para sair...")
            return

        print("✅ Arquivo launch.py encontrado!")
        print("🌐 O aplicativo abrirá no seu navegador padrão")
        print("📋 Para parar o aplicativo, feche esta janela")
        print("-" * 60)

        # Comando para executar streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", str(launch_script), "--server.headless", "true"]

        print(f"🔧 Executando comando: {' '.join(cmd)}")

        # Aguardar um pouco e abrir navegador
        def abrir_navegador():
            time.sleep(3)
            webbrowser.open('http://localhost:8501')

        import threading
        browser_thread = threading.Thread(target=abrir_navegador)
        browser_thread.daemon = True
        browser_thread.start()

        # Executar o comando
        process = subprocess.run(cmd, cwd=current_dir)

    except KeyboardInterrupt:
        print("\n👋 Aplicativo interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro ao iniciar aplicativo: {e}")
        print(f"📁 Diretório de trabalho: {os.getcwd()}")
        print(f"🐍 Python executable: {sys.executable}")
        input("Pressione Enter para sair...")

if __name__ == "__main__":
    main()