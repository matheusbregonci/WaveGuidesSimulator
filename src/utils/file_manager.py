"""
Sistema de Gerenciamento de Arquivos para o Simulador de Guias de Onda
Organiza automaticamente os arquivos gerados pelo programa
"""

import os
import shutil
import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json
import glob

class FileManager:
    """Gerenciador central de arquivos do simulador."""

    def __init__(self, base_path: Optional[str] = None):
        """
        Inicializa o gerenciador de arquivos.

        Args:
            base_path: Caminho base do projeto (opcional, usa o diretório atual)
        """
        if base_path:
            self.base_path = Path(base_path)
        else:
            # Assume que está no diretório models/ e vai para o parent
            self.base_path = Path(__file__).parent.parent

        self.outputs_path = self.base_path / 'outputs'
        self._criar_estrutura_diretorios()

    def _criar_estrutura_diretorios(self):
        """Cria a estrutura de diretórios necessária."""
        dirs = [
            'outputs/reports/retangular',
            'outputs/reports/cilindrica',
            'outputs/visualizations/static',
            'outputs/visualizations/interactive',
            'outputs/visualizations/animations',
            'outputs/data/json',
            'outputs/data/csv',
            'outputs/temp'
        ]

        for dir_path in dirs:
            full_path = self.base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Diretório criado/verificado: {full_path}")

    def get_report_path(self, tipo_simulacao: str, filename: str = None) -> Path:
        """
        Retorna o caminho para salvar relatórios.

        Args:
            tipo_simulacao: 'retangular' ou 'cilindrica'
            filename: Nome específico do arquivo (opcional)

        Returns:
            Path completo para o arquivo de relatório
        """
        if tipo_simulacao.lower() not in ['retangular', 'cilindrica']:
            tipo_simulacao = 'retangular'

        report_dir = self.outputs_path / 'reports' / tipo_simulacao.lower()

        if filename:
            return report_dir / filename
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"relatorio_{tipo_simulacao.lower()}_{timestamp}.pdf"
            return report_dir / filename

    def get_visualization_path(self, tipo: str, filename: str = None) -> Path:
        """
        Retorna o caminho para salvar visualizações.

        Args:
            tipo: 'static', 'interactive', ou 'animations'
            filename: Nome específico do arquivo (opcional)

        Returns:
            Path completo para o arquivo de visualização
        """
        if tipo not in ['static', 'interactive', 'animations']:
            tipo = 'static'

        viz_dir = self.outputs_path / 'visualizations' / tipo

        if filename:
            return viz_dir / filename
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            return viz_dir / f"visualizacao_{timestamp}"

    def get_data_path(self, tipo: str = 'json', filename: str = None) -> Path:
        """
        Retorna o caminho para salvar dados.

        Args:
            tipo: 'json' ou 'csv'
            filename: Nome específico do arquivo (opcional)

        Returns:
            Path completo para o arquivo de dados
        """
        if tipo not in ['json', 'csv']:
            tipo = 'json'

        data_dir = self.outputs_path / 'data' / tipo

        if filename:
            return data_dir / filename
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = 'json' if tipo == 'json' else 'csv'
            filename = f"dados_simulacao_{timestamp}.{extension}"
            return data_dir / filename

    def get_temp_path(self, filename: str) -> Path:
        """
        Retorna o caminho para arquivos temporários.

        Args:
            filename: Nome do arquivo temporário

        Returns:
            Path completo para o arquivo temporário
        """
        return self.outputs_path / 'temp' / filename

    def salvar_dados_simulacao(self, dados: Dict, tipo_simulacao: str) -> str:
        """
        Salva dados da simulação em JSON.

        Args:
            dados: Dicionário com dados da simulação
            tipo_simulacao: Tipo da simulação

        Returns:
            Caminho do arquivo salvo
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulacao_{tipo_simulacao.lower()}_{timestamp}.json"
        filepath = self.get_data_path('json', filename)

        # Limpar dados para serialização JSON
        dados_clean = self._limpar_dados_para_json(dados)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dados_clean, f, indent=2, ensure_ascii=False)

        print(f"💾 Dados salvos: {filepath}")
        return str(filepath)

    def _limpar_dados_para_json(self, dados: Dict) -> Dict:
        """Remove objetos não serializáveis dos dados."""
        dados_clean = {}
        for key, value in dados.items():
            if key == 'imagens':
                # Converter imagens para referências de arquivo
                dados_clean[key] = {}
                if isinstance(value, dict):
                    for img_key, img_value in value.items():
                        if isinstance(img_value, dict) and img_value.get('tipo') == 'arquivo':
                            dados_clean[key][img_key] = f"arquivo: {img_value.get('caminho')}"
                        elif isinstance(img_value, str):
                            dados_clean[key][img_key] = img_value[:100] + "..." if len(img_value) > 100 else img_value
            elif isinstance(value, (str, int, float, bool, list)):
                dados_clean[key] = value
            else:
                dados_clean[key] = str(value)

        return dados_clean

    def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        Remove arquivos temporários antigos.

        Args:
            max_age_hours: Idade máxima dos arquivos em horas
        """
        temp_dir = self.outputs_path / 'temp'
        if not temp_dir.exists():
            return

        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=max_age_hours)
        removed_count = 0

        for file_path in temp_dir.glob('*'):
            if file_path.is_file():
                file_time = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    try:
                        file_path.unlink()
                        removed_count += 1
                        print(f"🗑️ Arquivo temporário removido: {file_path.name}")
                    except Exception as e:
                        print(f"❌ Erro ao remover {file_path}: {e}")

        print(f"🧹 Limpeza concluída: {removed_count} arquivos removidos")

    def cleanup_old_reports(self, max_reports_per_type: int = 10):
        """
        Remove relatórios antigos, mantendo apenas os mais recentes.

        Args:
            max_reports_per_type: Número máximo de relatórios por tipo
        """
        for tipo in ['retangular', 'cilindrica']:
            report_dir = self.outputs_path / 'reports' / tipo
            if not report_dir.exists():
                continue

            # Listar todos os PDFs e ordenar por data de modificação
            pdf_files = list(report_dir.glob('*.pdf'))
            pdf_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Remover os mais antigos
            if len(pdf_files) > max_reports_per_type:
                files_to_remove = pdf_files[max_reports_per_type:]
                for file_path in files_to_remove:
                    try:
                        file_path.unlink()
                        print(f"🗑️ Relatório antigo removido: {file_path.name}")
                    except Exception as e:
                        print(f"❌ Erro ao remover {file_path}: {e}")

    def get_recent_files(self, file_type: str = 'reports', limit: int = 5) -> List[Path]:
        """
        Retorna lista dos arquivos mais recentes de um tipo.

        Args:
            file_type: 'reports', 'visualizations', ou 'data'
            limit: Número máximo de arquivos

        Returns:
            Lista de paths dos arquivos mais recentes
        """
        if file_type == 'reports':
            search_dirs = [
                self.outputs_path / 'reports' / 'retangular',
                self.outputs_path / 'reports' / 'cilindrica'
            ]
            pattern = '*.pdf'
        elif file_type == 'visualizations':
            search_dirs = [
                self.outputs_path / 'visualizations' / 'static',
                self.outputs_path / 'visualizations' / 'interactive',
                self.outputs_path / 'visualizations' / 'animations'
            ]
            pattern = '*'
        elif file_type == 'data':
            search_dirs = [
                self.outputs_path / 'data' / 'json',
                self.outputs_path / 'data' / 'csv'
            ]
            pattern = '*'
        else:
            return []

        all_files = []
        for search_dir in search_dirs:
            if search_dir.exists():
                all_files.extend(search_dir.glob(pattern))

        # Ordenar por data de modificação (mais recente primeiro)
        all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        return all_files[:limit]

    def get_storage_info(self) -> Dict[str, str]:
        """
        Retorna informações sobre uso de armazenamento.

        Returns:
            Dicionário com informações de armazenamento
        """
        def get_dir_size(path: Path) -> int:
            """Calcula tamanho total de um diretório."""
            total = 0
            if path.exists():
                for file_path in path.rglob('*'):
                    if file_path.is_file():
                        total += file_path.stat().st_size
            return total

        def format_bytes(bytes_count: int) -> str:
            """Formata bytes em unidades legíveis."""
            for unit in ['B', 'KB', 'MB', 'GB']:
                if bytes_count < 1024:
                    return f"{bytes_count:.1f} {unit}"
                bytes_count /= 1024
            return f"{bytes_count:.1f} TB"

        info = {}
        for subdir in ['reports', 'visualizations', 'data', 'temp']:
            dir_path = self.outputs_path / subdir
            size = get_dir_size(dir_path)
            info[subdir] = format_bytes(size)

        total_size = get_dir_size(self.outputs_path)
        info['total'] = format_bytes(total_size)

        return info

# Instância global do gerenciador
file_manager = FileManager()