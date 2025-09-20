import io
import os
import base64
import datetime
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.legends import Legend
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image as PILImage

class RelatorioPDF:
    """
    Classe para geração de relatórios completos em PDF das simulações de guias de onda.
    """

    def __init__(self):
        self.doc = None
        self.story = []
        self.styles = getSampleStyleSheet()
        self._configurar_estilos()

    def _configurar_estilos(self):
        """Configura estilos customizados para o PDF."""

        # Estilo do título principal
        self.styles.add(ParagraphStyle(
            name='TituloPrincipal',
            parent=self.styles['Title'],
            fontSize=20,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2E86AB'),
            fontName='Helvetica-Bold'
        ))

        # Estilo de seção
        self.styles.add(ParagraphStyle(
            name='TituloSecao',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=15,
            spaceBefore=20,
            textColor=colors.HexColor('#A23B72'),
            fontName='Helvetica-Bold'
        ))

        # Estilo de subseção
        self.styles.add(ParagraphStyle(
            name='TituloSubsecao',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
            textColor=colors.HexColor('#2E86AB'),
            fontName='Helvetica-Bold'
        ))

        # Estilo para dados técnicos
        self.styles.add(ParagraphStyle(
            name='DadosTecnicos',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceBefore=5,
            spaceAfter=5,
            fontName='Helvetica'
        ))

        # Estilo para cabeçalho
        self.styles.add(ParagraphStyle(
            name='Cabecalho',
            parent=self.styles['Normal'],
            fontSize=8,
            alignment=TA_RIGHT,
            textColor=colors.grey
        ))

        # Estilo para rodapé
        self.styles.add(ParagraphStyle(
            name='Rodape',
            parent=self.styles['Normal'],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.grey
        ))

    def criar_relatorio(self, dados_simulacao: Dict[str, Any], nome_arquivo: str = None) -> str:
        """
        Cria o relatório completo em PDF.

        Args:
            dados_simulacao: Dicionário com todos os dados da simulação
            nome_arquivo: Nome do arquivo PDF (opcional)

        Returns:
            Caminho do arquivo PDF gerado
        """

        if nome_arquivo is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            tipo_guia = dados_simulacao.get('tipo_guia', 'simulacao')
            nome_arquivo = f"relatorio_{tipo_guia}_{timestamp}.pdf"

        # Criar documento
        self.doc = SimpleDocTemplate(
            nome_arquivo,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )

        # Limpar story
        self.story = []

        # Adicionar conteúdo
        self._adicionar_cabecalho(dados_simulacao)
        self._adicionar_resumo_executivo(dados_simulacao)
        self._adicionar_parametros_simulacao(dados_simulacao)
        self._adicionar_resultados_visualizacoes(dados_simulacao)
        self._adicionar_analise_tecnica(dados_simulacao)
        # self._adicionar_conclusoes(dados_simulacao)
        self._adicionar_apendices(dados_simulacao)

        # Gerar PDF
        self.doc.build(self.story)

        # Limpar arquivos temporários de imagem
        self._limpar_arquivos_temporarios(dados_simulacao)

        return nome_arquivo

    def _limpar_arquivos_temporarios(self, dados: Dict[str, Any]):
        """Remove arquivos temporários de imagem após gerar o PDF."""
        if 'imagens' in dados:
            for img_data in dados['imagens'].values():
                if isinstance(img_data, dict) and img_data.get('tipo') == 'arquivo':
                    caminho = img_data.get('caminho')
                    if caminho and os.path.exists(caminho):
                        try:
                            os.remove(caminho)
                            print(f"Arquivo temporário removido: {caminho}")
                        except Exception as e:
                            print(f"Erro ao remover arquivo temporário {caminho}: {e}")

    def _adicionar_cabecalho(self, dados: Dict[str, Any]):
        """Adiciona cabeçalho do relatório."""

        # Título principal
        titulo = f"Relatório de Simulação - {dados.get('tipo_guia', 'Guia de Onda')}"
        self.story.append(Paragraph(titulo, self.styles['TituloPrincipal']))

        # Informações gerais
        timestamp = datetime.datetime.now().strftime("%d/%m/%Y às %H:%M")
        info_geral = f"""
        <b>Data de Geração:</b> {timestamp}<br/>
        <b>Tipo de Guia:</b> {dados.get('tipo_guia', 'N/A')}<br/>
        <b>Versão do Software:</b> Simulador de Guias de onda v2.0<br/>
        <b>Usuário:</b> {dados.get('usuario', 'Usuário Padrão')}
        """
        self.story.append(Paragraph(info_geral, self.styles['Normal']))
        self.story.append(Spacer(1, 20))

    def _adicionar_resumo_executivo(self, dados: Dict[str, Any]):
        """Adiciona resumo executivo."""

        self.story.append(Paragraph("1. Resumo Executivo", self.styles['TituloSecao']))

        tipo_guia = dados.get('tipo_guia', 'N/A')
        frequencia = dados.get('frequencia', 'N/A')
        material = dados.get('material', 'N/A')

        resumo = f"""
        Este relatório apresenta os resultados da simulação eletromagnética de uma {tipo_guia}
        operando na frequência de {frequencia} GHz. A simulação foi realizada considerando um meio
        dielétrico composto por {material}.<br/><br/>

        <b>Os resultados incluem:</b><br/>
        • Análise completa dos <b>campos elétricos (E)</b> e <b>magnéticos (H)</b><br/>
        • Comparação entre <b>modos TE (transversal elétrico)</b> e <b>TM (transversal magnético)</b><br/>
        • Distribuições espaciais de campo em diferentes componentes<br/>
        • Características de propagação e padrões de interferência<br/>
        • Visualizações 2D, vetoriais e de modos complementares
        """

        self.story.append(Paragraph(resumo, self.styles['Normal']))
        self.story.append(Spacer(1, 15))

    def _adicionar_parametros_simulacao(self, dados: Dict[str, Any]):
        """Adiciona seção de parâmetros da simulação."""

        self.story.append(Paragraph("2. Parâmetros da Simulação", self.styles['TituloSecao']))

        # Parâmetros gerais
        self.story.append(Paragraph("2.1 Configuração Geral", self.styles['TituloSubsecao']))

        parametros_gerais = [
            ['Parâmetro', 'Valor', 'Unidade'],
            ['Tipo de Guia', dados.get('tipo_guia', 'N/A'), '-'],
            ['Frequência', f"{dados.get('frequencia', 'N/A')}", 'GHz'],
            ['Material Dielétrico', dados.get('material', 'N/A'), '-'],
            ['Permissividade Relativa (ε<sub>r</sub>)', f"{dados.get('permissividade', 'N/A')}", '-'],
            ['Permeabilidade Relativa (μ<sub>r</sub>)', f"{dados.get('permeabilidade', 'N/A')}", '-'],
        ]

        # Adicionar parâmetros específicos do tipo de guia
        if dados.get('tipo_guia') == 'Guia Retangular':
            parametros_gerais.extend([
                ['Largura (a)', f"{dados.get('largura', 'N/A')}", 'mm'],
                ['Altura (b)', f"{dados.get('altura', 'N/A')}", 'mm'],
                ['Plano de Visualização', dados.get('plano', 'N/A'), '-'],
                ['Tipo de Campo', dados.get('campo', 'N/A'), '-'],
                ['Componente', dados.get('componente', 'N/A'), '-']
            ])
        elif dados.get('tipo_guia') == 'Guia Cilíndrica':
            parametros_gerais.extend([
                ['Raio (a)', f"{dados.get('raio', 'N/A')}", 'mm'],
                ['Comprimento', f"{dados.get('comprimento', 'N/A')}", 'mm'],
                ['Modo m', f"{dados.get('modo_m', 'N/A')}", '-'],
                ['Modo n', f"{dados.get('modo_n', 'N/A')}", '-']
            ])

        tabela = Table(parametros_gerais)
        tabela.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        self.story.append(tabela)
        self.story.append(Spacer(1, 20))

    def _adicionar_resultados_visualizacoes(self, dados: Dict[str, Any]):
        """Adiciona seção de resultados e visualizações."""

        self.story.append(Paragraph("3. Resultados e Visualizações", self.styles['TituloSecao']))

        # Adicionar descrição dos resultados
        self.story.append(Paragraph("3.1 Distribuição dos Campos", self.styles['TituloSubsecao']))

        descricao = """
        As visualizações a seguir mostram a distribuição dos campos eletromagnéticos na guia de onda.
        Os campos foram calculados usando as equações de Maxwell para a geometria e condições de
        contorno específicas da simulação.
        """
        self.story.append(Paragraph(descricao, self.styles['Normal']))

        # Adicionar imagens se disponíveis
        if 'imagens' in dados:
            for i, (titulo_img, img_data) in enumerate(dados['imagens'].items()):
                if img_data:
                    # Tratar mensagens informativas
                    if isinstance(img_data, str) and (img_data.startswith(("Erro", "Gráficos adicionais", "Animações completas", "Nota"))):
                        self.story.append(Paragraph(f"3.{i+2} {titulo_img}", self.styles['TituloSubsecao']))
                        self.story.append(Paragraph(img_data, self.styles['Normal']))
                        self.story.append(Spacer(1, 10))
                    else:
                        self.story.append(Paragraph(f"3.{i+2} {titulo_img}", self.styles['TituloSubsecao']))

                        # Converter imagem para PDF
                        img = self._processar_imagem(img_data)
                        if img:
                            self.story.append(img)
                            self.story.append(Spacer(1, 10))
                        else:
                            self.story.append(Paragraph("[Visualização não disponível no relatório]", self.styles['Normal']))
                            self.story.append(Spacer(1, 10))

        self.story.append(Spacer(1, 15))

    def _processar_imagem(self, img_data) -> Optional[Image]:
        """Processa e redimensiona imagem para inclusão no PDF."""
        try:
            # Verificar se é um dicionário com caminho de arquivo
            if isinstance(img_data, dict) and img_data.get('tipo') == 'arquivo':
                caminho_arquivo = img_data.get('caminho')
                if caminho_arquivo and os.path.exists(caminho_arquivo):
                    # Usar arquivo diretamente e manter proporção
                    img = Image(caminho_arquivo)
                    # Manter proporção da imagem original
                    img._restrictSize(4.5*inch, 4*inch)
                    return img
                else:
                    print(f"Arquivo não encontrado: {caminho_arquivo}")
                    return None

            elif isinstance(img_data, str):
                # Verificar se é uma mensagem de erro ou nota
                if img_data.startswith(("Erro", "Gráficos adicionais", "Animações completas", "Nota")):
                    return None  # Pular processamento para strings informativas

                # Verificar se o base64 não está muito longo (evitar códigos gigantescos)
                if len(img_data) > 1024 * 1024:  # 1MB como string base64
                    print(f"String base64 muito longa ({len(img_data)} chars), pulando...")
                    return None

                # Se é base64
                try:
                    img_bytes = base64.b64decode(img_data)
                    # Verificar se os bytes realmente formam uma imagem PNG válida
                    if len(img_bytes) > 10 and not img_bytes.startswith(b'\x89PNG'):
                        print("Dados decodificados não são PNG válido")
                        return None
                    img_buffer = io.BytesIO(img_bytes)
                    img = Image(img_buffer)
                    img.drawHeight = 3*inch
                    img.drawWidth = 4.5*inch
                    return img
                except Exception as e:
                    print(f"Erro ao decodificar base64: {e}")
                    return None

            elif hasattr(img_data, 'read'):
                # Se é um buffer
                img = Image(img_data)
                img.drawHeight = 3*inch
                img.drawWidth = 4.5*inch
                return img
            else:
                return None

        except Exception as e:
            print(f"Erro ao processar imagem: {e}")
            return None

    def _adicionar_analise_tecnica(self, dados: Dict[str, Any]):
        """Adiciona análise técnica dos resultados."""

        self.story.append(Paragraph("4. Análise Técnica", self.styles['TituloSecao']))

        # Características de propagação
        self.story.append(Paragraph("4.1 Características de Propagação", self.styles['TituloSubsecao']))

        if dados.get('tipo_guia') == 'Guia Retangular':
            analise = f"""
            Para uma guia retangular com dimensões {dados.get('largura', 'N/A')} × {dados.get('altura', 'N/A')} mm
            operando em {dados.get('frequencia', 'N/A')} GHz:

            • O modo dominante é o TE₁₀
            • A frequência de corte teórica é aproximadamente {self._calcular_freq_corte_retangular(dados)} GHz
            • A guia está operando {self._analisar_regime_operacao(dados)}
            • O campo {dados.get('campo', 'N/A')} na componente {dados.get('componente', 'N/A')} apresenta as características esperadas
            """
        elif dados.get('tipo_guia') == 'Guia Cilíndrica':
            modo_n = dados.get('modo_n', 'N/A')
            modo_m = dados.get('modo_m', 'N/A')
            analise = f"""
            Para uma guia cilíndrica com raio {dados.get('raio', 'N/A')} mm operando em {dados.get('frequencia', 'N/A')} GHz:
            <br/><br/>
            <b>Análise de Modos TE e TM:</b><br/>
            • <b>Modo TE<sub>{modo_n}{modo_m}</sub>:</b> Modo transversal elétrico (E<sub>z</sub> = 0)<br/>
            • <b>Modo TM<sub>{modo_n}{modo_m}</sub>:</b> Modo transversal magnético (H<sub>z</sub> = 0)<br/>
            • A distribuição radial segue as funções de Bessel J<sub>{modo_n}</sub><br/>
            • A distribuição azimutal segue cos({modo_n}φ) e sin({modo_n}φ)<br/>
            • As condições de contorno são satisfeitas nas paredes condutoras<br/>
            • Os campos TE e TM apresentam padrões de distribuição complementares
            """
        else:
            analise = "Análise técnica não disponível para este tipo de guia."

        self.story.append(Paragraph(analise, self.styles['Normal']))
        self.story.append(Spacer(1, 15))

    def _calcular_freq_corte_retangular(self, dados: Dict[str, Any]) -> str:
        """Calcula frequência de corte para guia retangular."""
        try:
            largura = float(dados.get('largura', 0)) / 1000  # mm para m
            if largura > 0:
                c = 3e8  # velocidade da luz
                fc = c / (2 * largura)
                return f"{fc/1e9:.2f}"
            return "N/A"
        except:
            return "N/A"

    def _analisar_regime_operacao(self, dados: Dict[str, Any]) -> str:
        """Analisa regime de operação da guia."""
        try:
            freq = float(dados.get('frequencia', 0))
            fc = float(self._calcular_freq_corte_retangular(dados))

            if freq > fc * 1.5:
                return "bem acima da frequência de corte (regime de propagação estável)"
            elif freq > fc:
                return "acima da frequência de corte (propagação permitida)"
            else:
                return "abaixo da frequência de corte (modo evanescente)"
        except:
            return "regime indeterminado"

    def _adicionar_conclusoes(self, dados: Dict[str, Any]):
        """Adiciona seção de conclusões."""

        self.story.append(Paragraph("5. Conclusões", self.styles['TituloSecao']))

        conclusoes = f"""
        A simulação da {dados.get('tipo_guia', 'guia de onda')} foi realizada com sucesso, apresentando
        resultados consistentes com a teoria eletromagnética. Os campos calculados mostram as
        distribuições esperadas para as condições de contorno e parâmetros especificados.

        As visualizações geradas permitem uma compreensão clara da distribuição espacial dos campos
        e podem ser utilizadas para análise de desempenho e otimização do sistema.

        Os parâmetros utilizados na simulação são adequados para a aplicação pretendida e os
        resultados podem ser considerados confiáveis para análises preliminares de projeto.
        """

        self.story.append(Paragraph(conclusoes, self.styles['Normal']))
        self.story.append(Spacer(1, 15))

    def _adicionar_apendices(self, dados: Dict[str, Any]):
        """Adiciona apêndices com informações técnicas adicionais."""

        self.story.append(PageBreak())
        self.story.append(Paragraph("6. Apêndices", self.styles['TituloSecao']))

        # Apêndice A - Equações utilizadas
        self.story.append(Paragraph("6.1 Equações Fundamentais", self.styles['TituloSubsecao']))

        equacoes = """
        <b>Equações de Maxwell:</b><br/>
        ∇ × E = -∂B/∂t<br/>
        ∇ × H = ∂D/∂t + J<br/>
        ∇ · D = ρ<br/>
        ∇ · B = 0<br/><br/>

        <b>Condições de contorno:</b><br/>
        • Componente tangencial do campo elétrico nula nas paredes condutoras<br/>
        • Componente normal do campo magnético nula nas paredes condutoras<br/>
        """

        self.story.append(Paragraph(equacoes, self.styles['DadosTecnicos']))

        # Apêndice B - Dados da simulação
        self.story.append(Paragraph("6.2 Dados Completos da Simulação", self.styles['TituloSubsecao']))

        dados_completos = f"""
        <b>Timestamp:</b> {datetime.datetime.now().isoformat()}<br/>
        <b>Configuração completa:</b><br/>
        {str(dados).replace(',', ',<br/>')}
        """

        self.story.append(Paragraph(dados_completos, self.styles['DadosTecnicos']))

def capturar_matplotlib_como_base64(fig) -> str:
    """Captura figura matplotlib como string base64."""
    try:
        buffer = io.BytesIO()
        # Configurações otimizadas para reduzir tamanho do arquivo
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none',
                   pad_inches=0.1, transparent=False)
        buffer.seek(0)
        img_data = buffer.getvalue()
        buffer.close()

        # Verificar se o arquivo não está muito grande
        if len(img_data) > 1024 * 1024:  # 1MB limite
            print(f"Imagem muito grande ({len(img_data)} bytes), reduzindo qualidade...")
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight',
                       facecolor='white', edgecolor='none',
                       pad_inches=0.1, transparent=False)
            buffer.seek(0)
            img_data = buffer.getvalue()
            buffer.close()

        img_base64 = base64.b64encode(img_data).decode()
        return img_base64

    except Exception as e:
        print(f"Erro ao capturar matplotlib: {e}")
        return ""

def capturar_plotly_como_base64(fig) -> str:
    """Captura figura plotly como string base64 com timeout."""
    try:
        # Usar engine mais rápido e configurações otimizadas
        img_bytes = pio.to_image(
            fig,
            format='png',
            engine='kaleido',
            width=800,  # Menor resolução para ser mais rápido
            height=600
        )
        img_base64 = base64.b64encode(img_bytes).decode()
        return img_base64
    except Exception as e:
        print(f"Erro ao capturar Plotly: {e}")
        # Retornar placeholder em caso de erro
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

def gerar_relatorio_completo(dados_simulacao: Dict[str, Any], nome_arquivo: str = None) -> str:
    """
    Função principal para gerar relatório completo.

    Args:
        dados_simulacao: Dicionário com todos os dados da simulação
        nome_arquivo: Nome do arquivo PDF (opcional)

    Returns:
        Caminho do arquivo PDF gerado
    """
    gerador = RelatorioPDF()
    return gerador.criar_relatorio(dados_simulacao, nome_arquivo)