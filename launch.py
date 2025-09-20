"""
Launcher simples para o Simulador de Guias de Onda
Uso: streamlit run launch.py
"""

import sys
import os
from pathlib import Path

# Configuração de paths
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
models_dir = current_dir / "models"

# Função para adicionar path se não existir
def add_to_path(path):
    path_str = str(path.absolute())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Tentar nova estrutura
if src_dir.exists():
    print("Usando nova estrutura...")
    add_to_path(src_dir)
    try:
        from apps.app_streamlit_melhorias import main
        main()
    except ImportError as e:
        print(f"Erro na nova estrutura: {e}")
        # Fallback para original
        print("Tentando estrutura original...")
        add_to_path(models_dir)
        from app_streamlit_melhorias import main
        main()
else:
    # Usar estrutura original
    print("Usando estrutura original...")
    add_to_path(models_dir)
    from app_streamlit_melhorias import main
    main()