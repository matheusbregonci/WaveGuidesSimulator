import streamlit as st
import streamlit.components.v1 as components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.TEmn_model import Modo_TEmn
from models.Cilindrico_model import Modo_Cilindrico
import plotly.graph_objects as go
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import datetime
from typing import Dict, Any
import importlib.util
import tempfile

# Importar CavityWall3D
spec = importlib.util.spec_from_file_location("cavity_model",
    os.path.join(os.path.dirname(__file__), '..', 'models', '3d_cavity_wall_model.py'))
cavity_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cavity_module)
CavityWall3D = cavity_module.CavityWall3D

# Importar CylindricalCavityWall3D
spec_cyl = importlib.util.spec_from_file_location("cylindrical_cavity_model",
    os.path.join(os.path.dirname(__file__), '..', 'models', 'cylindrical_cavity_wall_model.py'))
cylindrical_cavity_module = importlib.util.module_from_spec(spec_cyl)
spec_cyl.loader.exec_module(cylindrical_cavity_module)
CylindricalCavityWall3D = cylindrical_cavity_module.CylindricalCavityWall3D

try:
    from reports.relatorio_pdf import gerar_relatorio_completo, capturar_matplotlib_como_base64, capturar_plotly_como_base64
    from utils.file_manager import file_manager
    RELATORIO_DISPONIVEL = True
except ImportError:
    RELATORIO_DISPONIVEL = False
    st.warning("⚠️ Módulo de relatórios não disponível. Instale: pip install reportlab")

@st.cache_resource
def get_state():
    return {}
state = get_state()

def apply_custom_css():
    st.markdown("""
    <style>
    /* Cores principais baseadas no plano UX/UI */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --success-color: #F18F01;
        --warning-color: #C73E1D;
        --neutral-color: #F5F5F5;
        --text-color: #333333;
    }

    /* Dashboard cards styling */
    .dashboard-card {
        background: linear-gradient(135deg, var(--primary-color), #3498db);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(46, 134, 171, 0.3);
        color: white;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        margin: 1rem 0;
        border: none;
    }

    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(46, 134, 171, 0.4);
    }

    .dashboard-card h3 {
        margin: 0 0 1rem 0;
        font-size: 1.8rem;
        font-weight: 600;
    }

    .dashboard-card p {
        margin: 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }

    .dashboard-card.cilindrica {
        background: linear-gradient(135deg, var(--secondary-color), #e74c3c);
    }

    /* Progress indicator */
    .progress-indicator {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
        gap: 1rem;
    }

    .step {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: #ddd;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: #666;
        transition: all 0.3s ease;
    }

    .step.active {
        background: var(--primary-color);
        color: white;
        transform: scale(1.1);
    }

    .step.completed {
        background: var(--success-color);
        color: white;
    }

    .step-connector {
        width: 60px;
        height: 3px;
        background: #ddd;
        border-radius: 2px;
    }

    .step-connector.active {
        background: var(--primary-color);
    }

    /* Tooltips and help */
    .help-tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        margin-left: 0.5rem;
    }

    .help-tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: #555;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        top: 125%;
        left: 50%;
        margin-left: -150px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.9rem;
        line-height: 1.4;
    }

    .help-tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Success/Warning alerts */
    .custom-success {
        background: linear-gradient(135deg, var(--success-color), #ff9f00);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #ff9f00;
    }

    .custom-warning {
        background: linear-gradient(135deg, var(--warning-color), #e74c3c);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #e74c3c;
    }

    /* Sidebar improvements */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--primary-color), #2980b9);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), #3498db);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(46, 134, 171, 0.3);
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(135deg, var(--neutral-color), #ecf0f1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--primary-color);
        margin: 1.5rem 0 1rem 0;
    }

    .section-header h3 {
        margin: 0;
        color: var(--primary-color);
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# Função removida - tooltips agora usam o parâmetro help nativo do Streamlit

def coletar_dados_simulacao_retangular() -> Dict[str, Any]:
    """Coleta todos os dados da simulação retangular para o relatório."""
    if 'TEmn' not in state:
        return {}

    TEmn = state['TEmn']
    campo, componente = state.get('campo_componente', ['N/A', 'N/A'])

    dados = {
        'tipo_guia': 'Guia Retangular',
        'timestamp': datetime.datetime.now().isoformat(),
        'frequencia': TEmn.frequencia / 1e9,  # Hz para GHz
        'largura': TEmn.largura,
        'altura': TEmn.altura,
        'permissividade': getattr(TEmn, 'mu', 1.0),       # CORRIGIDO: TEmn.mu armazena permissividade (código original trocado)
        'permeabilidade': getattr(TEmn, 'epsilon', 1.0),  # CORRIGIDO: TEmn.epsilon armazena permeabilidade (código original trocado)
        'plano': TEmn.plano,
        'campo': campo,
        'componente': componente,
        'material': 'Material configurado',
        'imagens': {}
    }

    return dados

def coletar_dados_simulacao_cilindrica() -> Dict[str, Any]:
    """Coleta todos os dados da simulação cilíndrica para o relatório."""
    if 'cilindro' not in state:
        return {}

    cilindrico = state['cilindro']
    modo_m = state.get('modo_m', 'N/A')
    modo_n = state.get('modo_n', 'N/A')
    permissividade_val = state.get('permissividade_original', 'N/A')
    permeabilidade_val = state.get('permeabilidade_original', 'N/A')

    dados = {
        'tipo_guia': 'Guia Cilíndrica',
        'timestamp': datetime.datetime.now().isoformat(),
        'frequencia': cilindrico.frequencia / 1e9,  # Hz para GHz
        'raio': cilindrico.raio * 1000,  # m para mm
        'permissividade': permissividade_val,
        'permeabilidade': permeabilidade_val,
        'modo_m': modo_m,
        'modo_n': modo_n,
        'material': 'Material configurado',
        'imagens': {}
    }

    return dados

def capturar_graficos_retangular(TEmn, campo, componente) -> Dict[str, str]:
    """Captura gráficos da simulação retangular como base64."""
    imagens = {}

    try:
        # Capturar AMBOS os campos - elétrico e magnético
        for tipo_campo in ['eletrico', 'magnetico']:
            with st.spinner(f"Capturando campo {tipo_campo}..."):
                fig_2d = TEmn.plot3DField(campo=tipo_campo, componente=componente)
                if RELATORIO_DISPONIVEL:
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    temp_filename = file_manager.get_temp_path(f"campo_{tipo_campo}_{componente}_{timestamp}.png")
                    fig_2d.savefig(temp_filename, dpi=120, bbox_inches='tight',
                                 facecolor='white', edgecolor='none', format='png')

                    # Armazenar o caminho do arquivo
                    nome_campo = f"Campo {tipo_campo.capitalize()} - Componente {componente.upper()}"
                    imagens[nome_campo] = {'tipo': 'arquivo', 'caminho': str(temp_filename)}

                plt.close(fig_2d)

        # Capturar campo vetorial também
        with st.spinner("Capturando campo vetorial..."):
            fig_vetorial = TEmn.plota_campo_vetorial(campo)
            if RELATORIO_DISPONIVEL:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                temp_filename = file_manager.get_temp_path(f"campo_vetorial_{timestamp}.png")
                fig_vetorial.savefig(temp_filename, dpi=120, bbox_inches='tight',
                                   facecolor='white', edgecolor='none', format='png')

                imagens['Campo Vetorial'] = {'tipo': 'arquivo', 'caminho': str(temp_filename)}

            plt.close(fig_vetorial)

        # Adicionar nota informativa
        imagens['Nota'] = "Gráficos 3D interativos disponíveis na interface web"

    except Exception as e:
        st.error(f"Erro ao capturar gráficos: {e}")
        imagens['Erro'] = f"Erro na captura: {str(e)}"

    return imagens

def capturar_graficos_cilindrica(cilindrico, X, Y, Rho, Phi) -> Dict[str, str]:
    """Captura gráficos da simulação cilíndrica como base64."""
    imagens = {}

    try:
        # Capturar ambos os modos - TE e TM
        for modo in ['TE', 'TM']:
            # Capturar ambos os campos - elétrico e magnético
            for tipo_campo in ['eletrico', 'magnetico']:
                with st.spinner(f"Capturando campo {tipo_campo} {modo}..."):
                    fig_anim, ax = plt.subplots(figsize=(6, 6))

                    # Calcular campos para um frame específico
                    fase_phi = np.pi/4  # 45 graus
                    Phi_com_fase = Phi + fase_phi

                    # Calcular componentes do campo baseado no modo
                    try:
                        if modo == 'TE':
                            if tipo_campo == 'eletrico':
                                rho_campo = cilindrico.TE_E_rho(rho=Rho, phi=Phi_com_fase)
                                phi_campo = cilindrico.TE_E_phi(rho=Rho, phi=Phi_com_fase)
                                titulo = "Campo Elétrico TE"
                            else:
                                rho_campo = cilindrico.TE_H_rho(rho=Rho, phi=Phi_com_fase)
                                phi_campo = cilindrico.TE_H_phi(rho=Rho, phi=Phi_com_fase)
                                titulo = "Campo Magnético TE"
                        else:  # TM
                            if tipo_campo == 'eletrico':
                                rho_campo = cilindrico.TM_E_rho(rho=Rho, phi=Phi_com_fase)
                                phi_campo = cilindrico.TM_E_phi(rho=Rho, phi=Phi_com_fase)
                                titulo = "Campo Elétrico TM"
                            else:
                                rho_campo = cilindrico.TM_H_rho(rho=Rho, phi=Phi_com_fase)
                                phi_campo = cilindrico.TM_H_phi(rho=Rho, phi=Phi_com_fase)
                                titulo = "Campo Magnético TM"

                        # Converter para cartesianas
                        e_x = rho_campo * np.cos(Phi) - phi_campo * np.sin(Phi)
                        e_y = rho_campo * np.sin(Phi) + phi_campo * np.cos(Phi)

                        # Filtrar pontos (usar menos pontos para acelerar)
                        mask = Rho <= cilindrico.raio
                        X_masked = X[mask][::2]  # Usar apenas metade dos pontos
                        Y_masked = Y[mask][::2]
                        e_x = e_x[mask][::2]
                        e_y = e_y[mask][::2]

                        # Normalizar vetores
                        magnitude = np.sqrt(e_x**2 + e_y**2)
                        max_magnitude = np.max(magnitude) if np.max(magnitude) > 0 else 1e-10

                        magnitude_nonzero = np.where(magnitude > 1e-12, magnitude, 1e-12)
                        e_x_normalized = e_x / magnitude_nonzero
                        e_y_normalized = e_y / magnitude_nonzero

                        scale_factor = cilindrico.raio * 0.1
                        e_x_display = e_x_normalized * scale_factor
                        e_y_display = e_y_normalized * scale_factor

                        quiver = ax.quiver(X_masked, Y_masked, e_x_display, e_y_display,
                                          magnitude, cmap='viridis',
                                          scale=1, scale_units='xy', angles='xy',
                                          pivot='middle', alpha=0.8)

                        circle = plt.Circle((0, 0), cilindrico.raio, color='red', fill=False, linestyle='--', linewidth=1.5)
                        ax.add_patch(plt.Circle((0, 0), cilindrico.raio, color='lightgray', alpha=0.5, zorder=0))
                        ax.add_artist(circle)

                        ax.set_xlabel("X (m)")
                        ax.set_ylabel("Y (m)")
                        ax.set_title(titulo)
                        ax.axis('equal')
                        ax.set_xlim(-cilindrico.raio*1.2, cilindrico.raio*1.2)
                        ax.set_ylim(-cilindrico.raio*1.2, cilindrico.raio*1.2)

                        plt.colorbar(quiver, ax=ax, shrink=0.8, aspect=20, label='Intensidade do Campo')

                        if RELATORIO_DISPONIVEL:
                            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                            temp_filename = file_manager.get_temp_path(f"campo_{tipo_campo}_{modo}_{timestamp}.png")
                            fig_anim.savefig(temp_filename, dpi=120, bbox_inches='tight',
                                           facecolor='white', edgecolor='none', format='png')

                            # Armazenar o caminho do arquivo
                            nome_campo = f"Campo {tipo_campo.capitalize()} {modo}"
                            imagens[nome_campo] = {'tipo': 'arquivo', 'caminho': str(temp_filename)}

                    except AttributeError as e:
                        # Se algum método TM não existir, pular
                        if modo == 'TM':
                            st.warning(f"Modo TM não implementado para {tipo_campo}: {e}")
                            continue
                        else:
                            raise e

                    plt.close(fig_anim)

        # Adicionar nota informativa
        imagens['Nota'] = "Animações completas e visualizações 3D disponíveis na interface interativa"

    except Exception as e:
        st.error(f"Erro ao capturar gráficos cilíndricos: {e}")
        imagens['Erro'] = f"Erro na captura: {str(e)}"

    return imagens

def gerar_relatorio_pdf(tipo_simulacao: str) -> bool:
    """Gera relatório PDF completo da simulação atual."""

    if not RELATORIO_DISPONIVEL:
        st.error("❌ Sistema de relatórios não disponível. Instale a biblioteca reportlab.")
        return False

    try:
        # Adicionar timeout e melhor controle de progresso
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Coletando dados da simulação...")
        progress_bar.progress(10)

        if tipo_simulacao == "retangular":
            if 'TEmn' not in state:
                st.error("❌ Nenhuma simulação retangular encontrada.")
                return False

            dados = coletar_dados_simulacao_retangular()
            TEmn = state['TEmn']
            campo, componente = state.get('campo_componente', ['eletrico', 'x'])

            status_text.text("Capturando visualizações (modo otimizado)...")
            progress_bar.progress(30)

            # Capturar gráficos com timeout
            try:
                dados['imagens'] = capturar_graficos_retangular(TEmn, campo, componente)
            except Exception as e:
                st.warning(f"⚠️ Erro ao capturar algumas visualizações: {e}")
                dados['imagens'] = {'Erro': 'Algumas visualizações não puderam ser capturadas'}

        elif tipo_simulacao == "cilindrica":
            if 'cilindro' not in state:
                st.error("❌ Nenhuma simulação cilíndrica encontrada.")
                return False

            dados = coletar_dados_simulacao_cilindrica()
            cilindrico = state['cilindro']
            X, Y, Rho, Phi = state['X'], state['Y'], state['Rho'], state['Phi']

            status_text.text("Capturando visualizações (modo otimizado)...")
            progress_bar.progress(30)

            # Capturar gráficos com timeout
            try:
                dados['imagens'] = capturar_graficos_cilindrica(cilindrico, X, Y, Rho, Phi)
            except Exception as e:
                st.warning(f"⚠️ Erro ao capturar algumas visualizações: {e}")
                dados['imagens'] = {'Erro': 'Algumas visualizações não puderam ser capturadas'}

        else:
            st.error("❌ Tipo de simulação inválido.")
            return False

        status_text.text("Gerando arquivo PDF...")
        progress_bar.progress(70)

        # Gerar relatório usando o gerenciador de arquivos
        if RELATORIO_DISPONIVEL:
            nome_arquivo_path = file_manager.get_report_path(tipo_simulacao)
            nome_arquivo = gerar_relatorio_completo(dados, str(nome_arquivo_path))
        else:
            nome_arquivo = gerar_relatorio_completo(dados)

        status_text.text("Preparando download...")
        progress_bar.progress(90)

        # Oferecer download
        with open(nome_arquivo, "rb") as pdf_file:
            pdf_data = pdf_file.read()

        progress_bar.progress(100)
        status_text.text("Relatório concluído!")

        st.success(f"✅ Relatório gerado com sucesso: {nome_arquivo}")

        # Botão de download
        st.download_button(
            label="📥 Baixar Relatório PDF",
            data=pdf_data,
            file_name=nome_arquivo,
            mime="application/pdf",
            use_container_width=True
        )

        # Limpar componentes de progresso
        progress_bar.empty()
        status_text.empty()

        return True

    except Exception as e:
        st.error(f"❌ Erro ao gerar relatório: {e}")
        # Detalhes do erro para debug
        st.error(f"Detalhes: {str(e)[:200]}...")
        return False

def dashboard_principal():
    st.markdown("""<div class="section-header"><h3>🏠 Dashboard Principal - Simulador de Guias de Onda</h3></div>""", unsafe_allow_html=True)

    st.markdown("""<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
    Bem-vindo ao simulador de guias de onda eletromagnéticas. Selecione o tipo de guia para começar sua simulação.
    </p>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📰 Guia Retangular", key="btn_retangular", help="Simular campos em guias de onda retangulares (WR-series)", use_container_width=True):
            st.session_state.pagina_atual = "Guia Retangular"
            st.session_state.step = 1
            st.rerun()

        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 0.5rem;">
            <h5 style="margin: 0 0 0.5rem 0; color: #2E86AB;">📋 Características:</h5>
            <ul style="margin: 0; padding-left: 1.2rem; color: #666;">
                <li>Guias WR (WR-42, WR-62, WR-90, etc.)</li>
                <li>Modos TE e TM</li>
                <li>Bandas X, Ku, K, C</li>
                <li>Visualização 2D e 3D</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if st.button("⭕ Guia Cilíndrica", key="btn_cilindrica", help="Simular campos em guias de onda cilíndricas", use_container_width=True):
            st.session_state.pagina_atual = "Guia Cilíndrica"
            st.session_state.step = 1
            st.rerun()

        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 0.5rem;">
            <h5 style="margin: 0 0 0.5rem 0; color: #A23B72;">📋 Características:</h5>
            <ul style="margin: 0; padding-left: 1.2rem; color: #666;">
                <li>Guias circulares personalizadas</li>
                <li>Modos TEₘₙ e TMₘₙ</li>
                <li>Animações de fase</li>
                <li>Campos vetoriais 3D</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""<div class="section-header"><h3>📊 Últimas Simulações</h3></div>""", unsafe_allow_html=True)

    if 'TEmn' in state or 'cilindro' in state:
        if 'TEmn' in state:
            st.success("✅ Guia Retangular configurada e pronta para simulação")
        if 'cilindro' in state:
            st.success("✅ Guia Cilíndrica configurada e pronta para simulação")
    else:
        st.info("ℹ️ Nenhuma simulação configurada ainda. Use os botões acima para começar.")

    st.markdown("""<div class="section-header"><h3>📚 Recursos de Aprendizado</h3></div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
            <h5 style="color: #2E86AB; margin: 0 0 0.5rem 0;">📖 Tutorial</h5>
            <p style="font-size: 0.9rem; color: #666; margin: 0;">Guia passo-a-passo para iniciantes</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
            <h5 style="color: #2E86AB; margin: 0 0 0.5rem 0;">🔬 Exemplos</h5>
            <p style="font-size: 0.9rem; color: #666; margin: 0;">Configurações pré-definidas</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
            <h5 style="color: #2E86AB; margin: 0 0 0.5rem 0;">❓ Ajuda</h5>
            <p style="font-size: 0.9rem; color: #666; margin: 0;">Documentação e suporte</p>
        </div>
        """, unsafe_allow_html=True)

def show_progress_indicator(current_step, total_steps=4):
    steps = ["🏠 Início", "⚙️ Configuração", "📊 Simulação", "📈 Resultados"]

    progress_html = '<div class="progress-indicator">'

    for i in range(total_steps):
        step_class = "step"
        if i < current_step - 1:
            step_class += " completed"
        elif i == current_step - 1:
            step_class += " active"

        progress_html += f'<div class="{step_class}">{i + 1}</div>'

        if i < total_steps - 1:
            connector_class = "step-connector"
            if i < current_step - 1:
                connector_class += " active"
            progress_html += f'<div class="{connector_class}"></div>'

    progress_html += '</div>'
    progress_html += f'<div style="text-align: center; margin-bottom: 2rem; color: #666;"><strong>Passo {current_step}: {steps[current_step-1]}</strong></div>'

    st.markdown(progress_html, unsafe_allow_html=True)

def guia_retangular():
    st.markdown("""<div class="section-header"><h3>📰 Modelo de Simulação - Guias Retangulares</h3></div>""", unsafe_allow_html=True)

    # Sistema de navegação por etapas
    if 'step' not in st.session_state:
        st.session_state.step = 1

    if st.session_state.step == 1:
        configuracao_parametros_retangular()
    elif st.session_state.step == 2:
        simulacao_retangular()
    elif st.session_state.step == 3:
        resultados_retangular()

def configuracao_parametros_retangular():
    st.markdown("""<h4 style="color: #2E86AB;">⚙️ Configuração de Parâmetros</h4>""", unsafe_allow_html=True)

    # Biblioteca de dielétricos pré-configurados
    st.markdown('<h5 style="color: #2E86AB; margin-top: 2rem;">📋 Dielétricos Pré-configurados</h5>', unsafe_allow_html=True)

    dieletricos = {
        "Personalizado": {"permissividade": 1.0, "permeabilidade": 1.0},
        "Ar": {"permissividade": 1.0, "permeabilidade": 1.0},
        "PTFE - Politetrafluoroetileno": {"permissividade": 2.25, "permeabilidade": 1.0},
        "Teflon": {"permissividade": 2.08, "permeabilidade": 1.0},
        "Porcelana": {"permissividade": 5.04, "permeabilidade": 1.0},
        "Nylon": {"permissividade": 2.28, "permeabilidade": 1.0}
    }

    material_selecionado = st.selectbox("Selecione o material dielétrico:", list(dieletricos.keys()),
                                       help="Material que preenche o interior da guia de onda")

    # Biblioteca de guias retangulares pré-configuradas
    st.markdown('<h5 style="color: #2E86AB; margin-top: 2rem;">📜 Guias Retangulares Pré-configuradas</h5>', unsafe_allow_html=True)

    guias_retangulares = {
        "Personalizada": {"nomenclatura": "Custom", "banda": "Custom", "faixa_freq": "Custom",
                         "largura": 22.86, "altura": 10.16, "freq_min": 1.0, "freq_max": 50.0},
        "WR-42": {"nomenclatura": "WR-42", "banda": "K", "faixa_freq": "18.0 - 26.5 GHz",
                 "largura": 10.70, "altura": 4.30, "freq_min": 18.0, "freq_max": 26.5},
        "WR-62": {"nomenclatura": "WR-62", "banda": "Ku", "faixa_freq": "12.4 - 18.0 GHz",
                 "largura": 15.80, "altura": 7.90, "freq_min": 12.4, "freq_max": 18.0},
        "WR-90": {"nomenclatura": "WR-90", "banda": "X", "faixa_freq": "8.20 - 12.4 GHz",
                 "largura": 22.86, "altura": 10.16, "freq_min": 8.20, "freq_max": 12.4},
        "WR-112": {"nomenclatura": "WR-112", "banda": "W", "faixa_freq": "7.05 - 10.0 GHz",
                  "largura": 28.50, "altura": 12.62, "freq_min": 7.05, "freq_max": 10.0},
        "WR-137": {"nomenclatura": "WR-137", "banda": "C", "faixa_freq": "5.85 - 8.20 GHz",
                  "largura": 34.85, "altura": 15.80, "freq_min": 5.85, "freq_max": 8.20}
    }

    guia_selecionada = st.selectbox("Selecione a guia retangular:", list(guias_retangulares.keys()),
                                   help="Escolha uma guia padrão ou configure manualmente")

    # Mostrar informações da guia selecionada com validação
    if guia_selecionada != "Personalizada":
        guia_info = guias_retangulares[guia_selecionada]
        st.markdown(f"""
        <div class="custom-success">
            <strong>✅ {guia_info['nomenclatura']}</strong> | <strong>Banda:</strong> {guia_info['banda']} |
            <strong>Faixa:</strong> {guia_info['faixa_freq']} | <strong>Dimensões:</strong> {guia_info['largura']}×{guia_info['altura']} mm
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<h5 style="color: #2E86AB; margin-top: 2rem;">⚙️ Parâmetros Detalhados</h5>', unsafe_allow_html=True)

    # Parâmetros que dependem das seleções com validação
    col1, col2 = st.columns(2)

    with col1:
        if guia_selecionada == "Personalizada":
            largura_guia = st.number_input("Largura da Guia (mm) 📏", value=22.86, step=0.1, min_value=1.0,
                                         help="Dimensão maior da seção transversal da guia")
        else:
            guia_info = guias_retangulares[guia_selecionada]
            largura_guia = st.number_input("Largura da Guia (mm) 📏", value=guia_info["largura"], step=0.1, min_value=1.0,
                                         help=f"Valor padrão para {guia_info['nomenclatura']}")

    with col2:
        if guia_selecionada == "Personalizada":
            altura_guia = st.number_input("Altura da Guia (mm) 📐", value=10.16, step=0.1, min_value=1.0,
                                        help="Dimensão menor da seção transversal da guia")
        else:
            altura_guia = st.number_input("Altura da Guia (mm) 📐", value=guia_info["altura"], step=0.1, min_value=1.0,
                                        help=f"Valor padrão para {guia_info['nomenclatura']}")

    # Validação das dimensões
    if largura_guia <= altura_guia:
        st.markdown("""
        <div class="custom-warning">
            ⚠️ <strong>Atenção:</strong> A largura deve ser maior que a altura para guias retangulares convencionais.
        </div>
        """, unsafe_allow_html=True)

    # Slider de frequência adaptativo com validação melhorada
    guia_info = guias_retangulares[guia_selecionada]
    freq_min = guia_info["freq_min"]
    freq_max = guia_info["freq_max"]

    if guia_selecionada == "Personalizada":
        frequencia_onda = st.slider(
            "Frequência da Onda (GHz)",
            min_value=freq_min,
            max_value=freq_max,
            value=min(max(12.0, freq_min), freq_max),
            step=0.1,
            help="Frequência de operação da onda eletromagnética. Para guias personalizadas, você pode escolher qualquer valor."
        )
    else:
        # Calcular valor padrão no centro da banda
        valor_central = (freq_min + freq_max) / 2
        # Criar strings separadamente para evitar problemas com aspas aninhadas
        nomenclatura_guia = guia_info['nomenclatura']
        tooltip_text = f'Faixa operacional otimizada para a guia {nomenclatura_guia}. Operar fora desta faixa pode resultar em propagação inadequada.'

        frequencia_onda = st.slider(
            f"Frequência da Onda - Banda {guia_info['banda']} (GHz)",
            min_value=freq_min,
            max_value=freq_max,
            value=valor_central,
            step=0.1,
            help=f"Faixa operacional da {guia_info['nomenclatura']}: {guia_info['faixa_freq']}. Operar fora desta faixa pode resultar em propagação inadequada."
        )

        # Indicador visual melhorado da posição na banda
        posicao_na_banda = (frequencia_onda - freq_min) / (freq_max - freq_min) * 100
        if posicao_na_banda < 25:
            status_cor = "🔵"
            status_texto = "Início da banda"
            status_class = "custom-info"
        elif posicao_na_banda < 75:
            status_cor = "🟢"
            status_texto = "Centro da banda (ótimo)"
            status_class = "custom-success"
        else:
            status_cor = "🟡"
            status_texto = "Final da banda"
            status_class = "custom-warning"

        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 6px; margin: 0.5rem 0; border-left: 4px solid #2E86AB;">
            {status_cor} <strong>{frequencia_onda:.1f} GHz</strong> - {status_texto} ({posicao_na_banda:.1f}% da faixa)
        </div>
        """, unsafe_allow_html=True)

    # Parâmetros do meio com validação
    col1, col2 = st.columns(2)

    with col1:
        if material_selecionado == "Personalizado":
            permissividade_meio = st.number_input("Permissividade Relativa (εᵣ)",
                                                value=1.0, step=0.1, min_value=0.1,
                                                help="Propriedade elétrica do material. Valores maiores que 1 indicam materiais dielétricos. Ar tem εᵣ = 1.")
        else:
            material_props = dieletricos[material_selecionado]
            permissividade_meio = st.number_input("Permissividade Relativa (εᵣ)",
                                                value=material_props["permissividade"], step=0.01, min_value=0.1,
                                                help=f"Valor característico do {material_selecionado}")

    with col2:
        if material_selecionado == "Personalizado":
            permeabilidade_meio = st.number_input("Permeabilidade Relativa (μᵣ)",
                                                 value=1.0, step=0.1, min_value=0.1,
                                                 help="Propriedade magnética do material. Para a maioria dos materiais não magnéticos, μᵣ = 1.")
        else:
            permeabilidade_meio = st.number_input("Permeabilidade Relativa (μᵣ)",
                                                 value=material_props["permeabilidade"], step=0.01, min_value=0.1,
                                                 help=f"Valor característico do {material_selecionado}")

    # Escolha do plano e campo com tooltips
    col1, col2, col3 = st.columns(3)

    with col1:
        plano_opcoes = ['xy', 'xz', 'yz']
        plano = st.selectbox("Plano de Visualização",
                            plano_opcoes,
                            help="Plano em que o campo será visualizado. xy = vista frontal, xz = vista lateral, yz = vista superior")

    with col2:
        campo = st.selectbox("Tipo de Campo",
                           ["eletrico", "magnetico"],
                           help="Campo elétrico (E) ou magnético (H). Ambos existem simultaneamente na guia, mas podem ser analisados separadamente.")

    with col3:
        componente = st.selectbox("Componente",
                                ['x', 'y', 'z'],
                                help="Direção do vetor campo. x = horizontal, y = vertical, z = longitudinal (direção de propagação)")

    # Resumo melhorado dos parâmetros selecionados
    st.markdown('<h5 style="color: #2E86AB; margin-top: 2rem;">📊 Resumo da Configuração</h5>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #2E86AB;">
            <h6 style="color: #2E86AB; margin: 0 0 0.5rem 0;">🧪 Material Dielétrico</h6>
        </div>
        """, unsafe_allow_html=True)
        st.write(f"• **Material:** {material_selecionado}")
        st.write(f"• **Permissividade (εᵣ):** {permissividade_meio}")
        st.write(f"• **Permeabilidade (μᵣ):** {permeabilidade_meio}")

    with col2:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #A23B72;">
            <h6 style="color: #A23B72; margin: 0 0 0.5rem 0;">📰 Guia de Onda</h6>
        </div>
        """, unsafe_allow_html=True)
        if guia_selecionada != "Personalizada":
            guia_info = guias_retangulares[guia_selecionada]
            st.write(f"• **Tipo:** {guia_info['nomenclatura']} - Banda {guia_info['banda']}")
            st.write(f"• **Faixa:** {guia_info['faixa_freq']}")
            posicao_resumo = (frequencia_onda - guia_info['freq_min']) / (guia_info['freq_max'] - guia_info['freq_min']) * 100
            st.write(f"• **Frequência:** {frequencia_onda:.1f} GHz ({posicao_resumo:.0f}% da banda)")
        else:
            st.write(f"• **Tipo:** Configuração personalizada")
            st.write(f"• **Frequência:** {frequencia_onda:.1f} GHz")
        st.write(f"• **Dimensões:** {largura_guia:.2f} × {altura_guia:.2f} mm")
        st.write(f"• **Campo:** {campo.capitalize()} - Componente {componente.upper()} - Plano {plano.upper()}")

    state['campo_componente'] = [campo, componente]

    # Botões de ação
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("⚙️ Aplicar Parâmetros e Continuar", use_container_width=True, type="primary"):
            # Validação final antes de aplicar
            validacao_ok = True

            if largura_guia <= altura_guia:
                st.error("❌ Erro: A largura deve ser maior que a altura")
                validacao_ok = False

            if permissividade_meio <= 0 or permeabilidade_meio <= 0:
                st.error("❌ Erro: Permissividade e permeabilidade devem ser positivas")
                validacao_ok = False

            if validacao_ok:
                try:
                    TEmn = Modo_TEmn(
                        largura=largura_guia,
                        altura=altura_guia,
                        frequencia=frequencia_onda * 1e9,
                        permissividade=permissividade_meio,
                        permeabilidade=permeabilidade_meio,
                        plano=plano
                    )
                    TEmn.calcula_campos()
                    state['TEmn'] = TEmn
                    st.session_state.step = 2
                    st.markdown("""
                    <div class="custom-success">
                        ✅ <strong>Parâmetros aplicados com sucesso!</strong> Prosseguindo para a simulação...
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
                except Exception as e:
                    st.markdown(f"""
                    <div class="custom-warning">
                        ⚠️ <strong>Erro na configuração:</strong> {str(e)}
                    </div>
                    """, unsafe_allow_html=True)

def simulacao_retangular():
    st.markdown("""<h4 style="color: #2E86AB;">📊 Simulação e Visualização</h4>""", unsafe_allow_html=True)

    if 'TEmn' not in state:
        st.markdown("""
        <div class="custom-warning">
            ⚠️ <strong>Configuração não encontrada.</strong> Por favor, volte à etapa anterior.
        </div>
        """, unsafe_allow_html=True)
        if st.button("⬅️ Voltar à Configuração"):
            st.session_state.step = 1
            st.rerun()
        return

    # Criar abas para diferentes tipos de visualização
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Campo 3D Interativo", "Campo 2D", "Campo Vetorial", "Animação 3D Cavidade", "Análise"])

    with tab1:
        TEmn = state['TEmn']
        campo, componente = state['campo_componente']

        st.markdown("**Visualização 3D Interativa do Campo**")

        if st.button("🌍 Gerar Visualização 3D", use_container_width=True):
            with st.spinner("Gerando visualização 3D..."):
                # Obter os dados do campo
                TEmn.calcula_campos()
                if campo == 'magnetico':
                    if componente == 'x':
                        imagem = TEmn.Hx
                    elif componente == 'y':
                        imagem = TEmn.Hy
                    elif componente == 'z':
                        imagem = TEmn.Hz
                elif campo == 'eletrico':
                    if componente == 'x':
                        imagem = TEmn.Ex
                    elif componente == 'y':
                        imagem = TEmn.Ey
                    elif componente == 'z':
                        imagem = TEmn.Ez

                # Criar o gráfico 3D interativo melhorado
                fig = go.Figure(data=[go.Surface(
                    z=imagem,
                    x=TEmn.x[:, 0],
                    y=TEmn.y[0, :],
                    colorscale='Viridis',
                    showscale=True
                )])

                fig.update_layout(
                    title=f"Campo {campo.capitalize()} - Componente {componente.upper()}",
                    scene=dict(
                        xaxis_title="Posição X (mm)",
                        yaxis_title="Posição Y (mm)",
                        zaxis_title="Intensidade do Campo",
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                    ),
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div class="custom-success">
                    ✅ <strong>Visualização 3D gerada com sucesso!</strong> Use o mouse para rotacionar, zoom e pan.
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        TEmn = state['TEmn']
        campo, componente = state['campo_componente']

        st.markdown("**Visualização 2D da Intensidade do Campo**")

        if st.button("🗺️ Gerar Mapa 2D", use_container_width=True):
            with st.spinner("Gerando mapa 2D..."):
                fig = TEmn.plot3DField(campo=campo, componente=componente)
                st.pyplot(fig)
                plt.close(fig)

                st.markdown("""
                <div class="custom-success">
                    ✅ <strong>Mapa 2D gerado com sucesso!</strong> As cores representam a intensidade do campo.
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        TEmn = state['TEmn']
        campo, componente = state['campo_componente']

        st.markdown("**Visualização Vetorial do Campo**")

        if st.button("➡️ Gerar Campo Vetorial", use_container_width=True):
            with st.spinner("Gerando campo vetorial..."):
                fig = TEmn.plota_campo_vetorial(campo)
                st.pyplot(fig)
                plt.close(fig)

                st.markdown("""
                <div class="custom-success">
                    ✅ <strong>Campo vetorial gerado com sucesso!</strong> As setas indicam direção e intensidade.
                </div>
                """, unsafe_allow_html=True)

    with tab4:
        st.markdown("**Animação 3D da Cavidade - Visualização nas Paredes**")
        st.write("Esta visualização mostra a intensidade do campo nas paredes da cavidade tridimensional.")

        col1, col2 = st.columns(2)

        with col1:
            campo_cavidade = st.selectbox("Campo (Cavidade)", ["magnetico", "eletrico"], key="campo_cavidade")
            tipo_intensidade = st.selectbox(
                "Tipo de Intensidade",
                ["direcional", "total", "perpendicular"],
                key="tipo_intensidade",
                help="Direcional: componente específica | Total: magnitude total | Perpendicular: componente perpendicular à parede"
            )

        with col2:
            direcao_vetor = st.selectbox(
                "Direção do Vetor",
                ["x", "y", "z"],
                key="direcao_vetor",
                help="Direção da componente quando tipo_intensidade='direcional'"
            )
            resolucao_cavidade = st.slider(
                "Resolução",
                min_value=10,
                max_value=50,
                value=25,
                step=5,
                key="resolucao_cavidade",
                help="Número de pontos por dimensão (valores menores = mais rápido)"
            )

        col3, col4 = st.columns(2)

        with col3:
            num_frames = st.slider(
                "Número de Frames",
                min_value=10,
                max_value=100,
                value=60,
                step=10,
                key="num_frames",
                help="Número de frames na animação"
            )

        with col4:
            duracao_frame = st.slider(
                "Duração do Frame (ms)",
                min_value=50,
                max_value=500,
                value=100,
                step=50,
                key="duracao_frame",
                help="Duração de cada frame em milissegundos"
            )

        profundidade_cavidade = st.number_input(
            "Profundidade da Cavidade (mm)",
            value=100.0,
            step=10.0,
            key="profundidade_cavidade",
            help="Profundidade da cavidade na direção Z"
        )

        if st.button("🎬 Gerar Animação 3D da Cavidade", use_container_width=True):
            if 'TEmn' not in state:
                st.markdown("""
                <div class="custom-warning">
                    ⚠️ <strong>Configuração não encontrada.</strong> Por favor, volte à etapa anterior.
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner("Gerando animação 3D da cavidade... Isso pode levar alguns instantes."):
                    try:
                        # Recuperar parâmetros do state
                        TEmn = state['TEmn']
                        largura_guia = TEmn.largura
                        altura_guia = TEmn.altura
                        frequencia_onda = TEmn.frequencia / 1e9
                        permissividade_meio = getattr(TEmn, 'mu', 1.0)
                        permeabilidade_meio = getattr(TEmn, 'epsilon', 1.0)

                        # Criar instância da CavityWall3D
                        cavity = CavityWall3D(
                            largura=largura_guia,
                            altura=altura_guia,
                            profundidade=profundidade_cavidade/1000,
                            frequencia=frequencia_onda * 1e9,
                            permissividade=permissividade_meio,
                            permeabilidade=permeabilidade_meio,
                            resolucao=resolucao_cavidade,
                            m=1,  # Modo m
                            n=0   # Modo n
                        )

                        # Gerar a animação
                        fig = cavity.animar_cavidade_plotly(
                            campo=campo_cavidade,
                            tipo_intensidade=tipo_intensidade,
                            direcao_vetor=direcao_vetor,
                            num_frames=num_frames,
                            duracao_frame=duracao_frame
                        )

                        # Salvar como HTML em arquivo temporário
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        html_filename = f'animacao_retangular_3d_{timestamp}.html'
                        html_path = os.path.join(tempfile.gettempdir(), html_filename)
                        fig.write_html(html_path)

                        # Ler o HTML gerado e salvar no session_state
                        with open(html_path, 'r', encoding='utf-8') as f:
                            html_string = f.read()

                        # Salvar no session_state para persistir entre reruns
                        st.session_state['html_retangular'] = html_string
                        st.session_state['html_path_retangular'] = html_path
                        st.session_state['html_filename_retangular'] = html_filename

                    except Exception as e:
                        st.markdown(f"""
                        <div class="custom-warning">
                            ⚠️ <strong>Erro ao gerar animação 3D da cavidade:</strong> {str(e)}
                        </div>
                        """, unsafe_allow_html=True)
                        import traceback
                        st.code(traceback.format_exc())

        # Exibir animação se existir no session_state
        if 'html_retangular' in st.session_state:
            # Exibir a animação incorporada com opção de maximizar
            col_title, col_maximize = st.columns([3, 1])
            with col_title:
                st.markdown("### Visualização Interativa 3D")
            with col_maximize:
                maximizar = st.checkbox("🔍 Maximizar", key="maximizar_retangular")

            # Ajustar altura baseado na opção de maximizar
            altura_viz = 1200 if maximizar else 700
            components.html(st.session_state['html_retangular'], height=altura_viz, scrolling=False)

            st.markdown("""
            <div class="custom-success">
                ✅ <strong>Animação 3D da cavidade gerada com sucesso!</strong> Use os controles para explorar.
            </div>
            """, unsafe_allow_html=True)

            # Botão para download do HTML
            col_download1, col_download2 = st.columns([1, 3])
            with col_download1:
                with open(st.session_state['html_path_retangular'], 'rb') as f:
                    st.download_button(
                        label="💾 Baixar HTML",
                        data=f.read(),
                        file_name=st.session_state['html_filename_retangular'],
                        mime='text/html',
                        use_container_width=True
                    )
            with col_download2:
                st.info(f"📄 Arquivo salvo temporariamente em: `{st.session_state['html_path_retangular']}`")

    with tab5:
        st.markdown("**Análise dos Resultados**")

        # Mostrar informações sobre os parâmetros calculados
        TEmn = state['TEmn']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #F18F01;">
                <h6 style="color: #F18F01; margin: 0 0 0.5rem 0;">📊 Parâmetros Calculados</h6>
            </div>
            """, unsafe_allow_html=True)

            st.write(f"• **Frequência:** {TEmn.frequencia/1e9:.2f} GHz")
            st.write(f"• **Largura:** {TEmn.largura:.2f} mm")
            st.write(f"• **Altura:** {TEmn.altura:.2f} mm")

        with col2:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #C73E1D;">
                <h6 style="color: #C73E1D; margin: 0 0 0.5rem 0;">📈 Propriedades do Campo</h6>
            </div>
            """, unsafe_allow_html=True)

            st.write(f"• **Campo:** {campo.capitalize()}")
            st.write(f"• **Componente:** {componente.upper()}")
            st.write(f"• **Plano:** {TEmn.plano.upper()}")

        # Seção de Relatório PDF
        st.markdown("---")
        st.markdown("**📄 Geração de Relatório Completo**")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            Gere um relatório PDF completo contendo:
            • Todos os parâmetros da simulação
            • Gráficos e visualizações geradas
            • Análise técnica dos resultados
            • Conclusões e dados técnicos
            """)

        with col2:
            if st.button("📄 Gerar Relatório PDF", use_container_width=True, type="primary"):
                if RELATORIO_DISPONIVEL:
                    gerar_relatorio_pdf("retangular")
                else:
                    st.error("📦 Para usar relatórios, instale: pip install reportlab plotly-kaleido")

    # Botões de navegação
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("⬅️ Voltar à Configuração"):
            st.session_state.step = 1
            st.rerun()

    with col2:
        if st.button("🏠 Dashboard Principal"):
            st.session_state.pagina_atual = "Dashboard"
            st.session_state.step = 0
            st.rerun()

    with col3:
        if st.button("🔁 Nova Simulação"):
            # Limpar o estado atual
            if 'TEmn' in state:
                del state['TEmn']
            st.session_state.step = 1
            st.rerun()

def resultados_retangular():
    st.markdown("""<h4 style="color: #2E86AB;">📈 Resultados e Análise</h4>""", unsafe_allow_html=True)
    st.info("Esta seção será implementada em futuras versões.")

def guia_cilindrica():
    st.markdown("""<div class="section-header"><h3>⭕ Modelo de Simulação - Guias Cilíndricas</h3></div>""", unsafe_allow_html=True)

    # Sistema de navegação por etapas
    if 'step' not in st.session_state:
        st.session_state.step = 1

    if st.session_state.step == 1:
        configuracao_parametros_cilindrica()
    elif st.session_state.step == 2:
        simulacao_cilindrica()
    elif st.session_state.step == 3:
        resultados_cilindrica()

def configuracao_parametros_cilindrica():
    st.markdown("""<h4 style="color: #A23B72;">⚙️ Configuração de Parâmetros</h4>""", unsafe_allow_html=True)

    # Biblioteca de dielétricos pré-configurados
    st.markdown('<h5 style="color: #A23B72; margin-top: 2rem;">📋 Dielétricos Pré-configurados</h5>', unsafe_allow_html=True)

    dieletricos = {
        "Personalizado": {"permissividade": 1.0, "permeabilidade": 1.0},
        "Ar": {"permissividade": 1.0, "permeabilidade": 1.0},
        "PTFE - Politetrafluoroetileno": {"permissividade": 2.25, "permeabilidade": 1.0},
        "Teflon": {"permissividade": 2.08, "permeabilidade": 1.0},
        "Porcelana": {"permissividade": 5.04, "permeabilidade": 1.0},
        "Nylon": {"permissividade": 2.28, "permeabilidade": 1.0}
    }

    material_selecionado = st.selectbox("Selecione o material dielétrico:", list(dieletricos.keys()),
                                       help="Material que preenche o interior da guia de onda")

    # Biblioteca de guias circulares pré-configuradas
    st.markdown('<h5 style="color: #A23B72; margin-top: 2rem;">📐 Guias Circulares Pré-configuradas</h5>', unsafe_allow_html=True)

    guias_circulares = {
        "Personalizada": {"banda": "Custom", "faixa_freq": "Custom", "raio": 23.0, "freq_min": 1.0, "freq_max": 50.0},
        "Guia 1 - Banda X": {"banda": "X", "faixa_freq": "8.5 - 11.6 GHz", "raio": 23.83, "freq_min": 8.5, "freq_max": 11.6},
        "Guia 2 - Banda Ku": {"banda": "Ku", "faixa_freq": "13.4 - 18.0 GHz", "raio": 15.08, "freq_min": 13.4, "freq_max": 18.0},
        "Guia 3 - Banda K": {"banda": "K", "faixa_freq": "20.0 - 24.5 GHz", "raio": 10.06, "freq_min": 20.0, "freq_max": 24.5},
        "Guia 4 - Banda Ka": {"banda": "Ka", "faixa_freq": "33.0 - 38.5 GHz", "raio": 6.35, "freq_min": 33.0, "freq_max": 38.5},
        "Guia 5 - Banda Q": {"banda": "Q", "faixa_freq": "38.5 - 43.0 GHz", "raio": 5.56, "freq_min": 38.5, "freq_max": 43.0}
    }

    guia_selecionada = st.selectbox("Selecione a guia circular:", list(guias_circulares.keys()),
                                   help="Escolha uma guia padrão ou configure manualmente")

    # Mostrar informações da guia selecionada com validação
    if guia_selecionada != "Personalizada":
        guia_info = guias_circulares[guia_selecionada]
        st.markdown(f"""
        <div class="custom-success">
            <strong>✅ {guia_selecionada}</strong> | <strong>Banda:</strong> {guia_info['banda']} |
            <strong>Faixa:</strong> {guia_info['faixa_freq']} | <strong>Raio:</strong> {guia_info['raio']} mm
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<h5 style="color: #A23B72; margin-top: 2rem;">⚙️ Parâmetros Detalhados</h5>', unsafe_allow_html=True)

    # Parâmetros da guia cilíndrica
    col1, col2 = st.columns(2)

    with col1:
        if guia_selecionada == "Personalizada":
            raio = st.number_input("Raio da Guia (mm) 📏", value=23.0, step=0.1, min_value=1.0,
                                 help="Raio interno da guia cilíndrica")
        else:
            raio = st.number_input("Raio da Guia (mm) 📏", value=guias_circulares[guia_selecionada]["raio"],
                                 step=0.1, min_value=1.0,
                                 help=f"Valor padrão para {guia_selecionada}")

    with col2:
        comprimento = st.number_input("Comprimento da Guia (mm) 📐", value=100.0, step=0.1, min_value=1.0,
                                    help="Comprimento da guia para análise")

    # Slider de frequência adaptativo
    guia_info = guias_circulares[guia_selecionada]
    freq_min = guia_info["freq_min"]
    freq_max = guia_info["freq_max"]

    if guia_selecionada == "Personalizada":
        frequencia = st.slider(
            "Frequência da Onda (GHz)",
            min_value=freq_min,
            max_value=freq_max,
            value=min(max(12.0, freq_min), freq_max),
            step=0.1,
            help="Frequência de operação da onda eletromagnética"
        )
    else:
        # Calcular valor padrão no centro da banda
        valor_central = (freq_min + freq_max) / 2
        frequencia = st.slider(
            f"Frequência da Onda - Banda {guia_info['banda']} (GHz)",
            min_value=freq_min,
            max_value=freq_max,
            value=valor_central,
            step=0.1,
            help=f"Faixa operacional da {guia_selecionada}: {guia_info['faixa_freq']}"
        )

        # Indicador visual da posição na banda
        posicao_na_banda = (frequencia - freq_min) / (freq_max - freq_min) * 100
        if posicao_na_banda < 25:
            status_cor = "🔵"
            status_texto = "Início da banda"
        elif posicao_na_banda < 75:
            status_cor = "🟢"
            status_texto = "Centro da banda (ótimo)"
        else:
            status_cor = "🟡"
            status_texto = "Final da banda"

        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 0.8rem; border-radius: 6px; margin: 0.5rem 0; border-left: 4px solid #A23B72;">
            {status_cor} <strong>{frequencia:.1f} GHz</strong> - {status_texto} ({posicao_na_banda:.1f}% da faixa)
        </div>
        """, unsafe_allow_html=True)

    # Parâmetros do meio com validação
    col1, col2 = st.columns(2)

    with col1:
        if material_selecionado == "Personalizado":
            permissividade = st.number_input("Permissividade Relativa (εᵣ)",
                                           value=1.0, step=0.1, min_value=0.1,
                                           help="Propriedade elétrica do material")
        else:
            material_props = dieletricos[material_selecionado]
            permissividade = st.number_input("Permissividade Relativa (εᵣ)",
                                           value=material_props["permissividade"], step=0.01, min_value=0.1,
                                           help=f"Valor característico do {material_selecionado}")

    with col2:
        if material_selecionado == "Personalizado":
            permeabilidade = st.number_input("Permeabilidade Relativa (μᵣ)",
                                           value=1.0, step=0.1, min_value=0.1,
                                           help="Propriedade magnética do material")
        else:
            permeabilidade = st.number_input("Permeabilidade Relativa (μᵣ)",
                                           value=material_props["permeabilidade"], step=0.01, min_value=0.1,
                                           help=f"Valor característico do {material_selecionado}")

    # Parâmetros dos modos
    col1, col2 = st.columns(2)

    with col1:
        modo_m = st.number_input("Modo m (0, 1 ou 2)", value=1, step=1, min_value=0, max_value=2,
                               help="Índice azimutal do modo (variação angular)")

    with col2:
        modo_n = st.number_input("Modo n (0, 1 ou 2)", value=1, step=1, min_value=0, max_value=2,
                               help="Índice radial do modo (variação radial)")

    # Resumo melhorado dos parâmetros selecionados
    st.markdown('<h5 style="color: #A23B72; margin-top: 2rem;">📊 Resumo da Configuração</h5>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #A23B72;">
            <h6 style="color: #A23B72; margin: 0 0 0.5rem 0;">🧪 Material Dielétrico</h6>
        </div>
        """, unsafe_allow_html=True)
        st.write(f"• **Material:** {material_selecionado}")
        st.write(f"• **Permissividade (εᵣ):** {permissividade}")
        st.write(f"• **Permeabilidade (μᵣ):** {permeabilidade}")

    with col2:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #2E86AB;">
            <h6 style="color: #2E86AB; margin: 0 0 0.5rem 0;">⭕ Guia Cilíndrica</h6>
        </div>
        """, unsafe_allow_html=True)
        if guia_selecionada != "Personalizada":
            guia_info = guias_circulares[guia_selecionada]
            st.write(f"• **Tipo:** {guia_selecionada}")
            st.write(f"• **Banda:** {guia_info['banda']} ({guia_info['faixa_freq']})")
            posicao_resumo = (frequencia - guia_info['freq_min']) / (guia_info['freq_max'] - guia_info['freq_min']) * 100
            st.write(f"• **Frequência:** {frequencia:.1f} GHz ({posicao_resumo:.0f}% da banda)")
        else:
            st.write(f"• **Tipo:** Configuração personalizada")
            st.write(f"• **Frequência:** {frequencia:.1f} GHz")
        st.write(f"• **Raio:** {raio:.2f} mm")
        st.write(f"• **Modo:** TE_{modo_n}{modo_m} / TM_{modo_n}{modo_m}")

    # Botões de ação
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("⚙️ Aplicar Parâmetros e Continuar", use_container_width=True, type="primary"):
            # Validação final antes de aplicar
            validacao_ok = True

            if raio <= 0:
                st.error("❌ Erro: O raio deve ser positivo")
                validacao_ok = False

            if permissividade <= 0 or permeabilidade <= 0:
                st.error("❌ Erro: Permissividade e permeabilidade devem ser positivas")
                validacao_ok = False

            if validacao_ok:
                try:
                    # Nota: A classe Modo_Cilindrico tem os parâmetros invertidos internamente
                    # permissividade -> self.mu, permeabilidade -> self.epsilon
                    cilindrico = Modo_Cilindrico(
                        raio=raio/1000,  # mm -> m
                        frequencia=frequencia * 1e9,  # Hz
                        permissividade=permissividade,
                        permeabilidade=permeabilidade,
                        m=modo_m,
                        n=modo_n,
                        z=0.25
                    )

                    X, Y, Rho, Phi = cilindrico.criar_meshgrid_cartesiano()
                    state['cilindro'] = cilindrico
                    state['X'] = X
                    state['Y'] = Y
                    state['Rho'] = Rho
                    state['Phi'] = Phi
                    state['modo_m'] = modo_m
                    state['modo_n'] = modo_n
                    # Armazenar valores originais dos parâmetros
                    state['permissividade_original'] = permissividade
                    state['permeabilidade_original'] = permeabilidade
                    st.session_state.step = 2
                    st.markdown("""
                    <div class="custom-success">
                        ✅ <strong>Parâmetros aplicados com sucesso!</strong> Prosseguindo para a simulação...
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
                except Exception as e:
                    st.markdown(f"""
                    <div class="custom-warning">
                        ⚠️ <strong>Erro na configuração:</strong> {str(e)}
                    </div>
                    """, unsafe_allow_html=True)

def simulacao_cilindrica():
    st.markdown("""<h4 style="color: #A23B72;">📊 Simulação e Visualização</h4>""", unsafe_allow_html=True)

    if 'cilindro' not in state:
        st.markdown("""
        <div class="custom-warning">
            ⚠️ <strong>Configuração não encontrada.</strong> Por favor, volte à etapa anterior.
        </div>
        """, unsafe_allow_html=True)
        if st.button("⬅️ Voltar à Configuração"):
            st.session_state.step = 1
            st.rerun()
        return

    # Criar abas para diferentes tipos de visualização
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Campo 3D Vetorial", "Animação de Fase", "Animação 3D Cavidade", "Coeficientes S", "Análise"])

    with tab1:
        cilindrico = state['cilindro']
        X, Y, Rho, Phi = state['X'], state['Y'], state['Rho'], state['Phi']

        st.markdown("**Visualização 3D Vetorial do Campo**")

        comprimento_3d = st.number_input("Comprimento para visualização 3D (m)", value=0.01, step=0.001, min_value=0.001)

        if st.button("🌍 Gerar Visualização 3D Vetorial", use_container_width=True):
            with st.spinner("Gerando visualização 3D vetorial..."):
                try:
                    cilindrico.pontos_por_dimensao = 8
                    cilindrico.num_planos = 13
                    X_3d, Y_3d, Rho_3d, Phi_3d, Z_3d = cilindrico.criar_meshgrid_cartesiano_com_z(comprimento=comprimento_3d)

                    fig = cilindrico.plot_vetores_3D(X_3d, Y_3d, Rho_3d, Phi_3d, Z_3d,
                                                    transversal='TE', campo='magnetico',
                                                    comprimento=comprimento_3d)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("""
                    <div class="custom-success">
                        ✅ <strong>Visualização 3D gerada com sucesso!</strong> Use o mouse para rotacionar e explorar.
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"""
                    <div class="custom-warning">
                        ⚠️ <strong>Erro na visualização:</strong> {str(e)}
                    </div>
                    """, unsafe_allow_html=True)

    with tab2:
        st.markdown("**Animação de Fase φ**")

        col1, col2 = st.columns(2)

        with col1:
            transversal_anim = st.selectbox("Modo (Animação)", ["TE", "TM"], key="transversal_anim")
            campo_anim = st.selectbox("Campo (Animação)", ["eletrico", "magnetico"], key="campo_anim")

        with col2:
            frames = st.slider("Número de Frames", min_value=10, max_value=100, value=60)
            interval = st.slider("Intervalo (ms)", min_value=50, max_value=500, value=85)

        z_fixo = st.number_input("Posição Z Fixa (m)", value=0.25, step=0.01)

        if st.button("🎬 Gerar Animação GIF", use_container_width=True):
            cilindrico = state['cilindro']
            X, Y, Rho, Phi, modo_m, modo_n = state['X'], state['Y'], state['Rho'], state['Phi'], state['modo_m'], state['modo_n']

            nome_arquivo = f"{transversal_anim}_{campo_anim}{modo_n}{modo_m}_fase_phi.gif"

            with st.spinner("Gerando GIF... Isso pode demorar um pouco."):
                try:
                    # Código de geração do GIF (mantido do original)
                    fig, ax = plt.subplots(figsize=(8, 8))

                    fase_phi_vals = np.linspace(0, 2*np.pi, frames)
                    images = []

                    for i, fase_phi in enumerate(fase_phi_vals):
                        ax.clear()

                        # Aplicar fase em φ
                        Phi_com_fase = Phi + fase_phi

                        # Calcular campos
                        if transversal_anim == 'TE':
                            if campo_anim == 'eletrico':
                                rho = cilindrico.TE_E_rho(rho=Rho, phi=Phi_com_fase)
                                phi = cilindrico.TE_E_phi(rho=Rho, phi=Phi_com_fase)
                            else:
                                rho = cilindrico.TE_H_rho(rho=Rho, phi=Phi_com_fase)
                                phi = cilindrico.TE_H_phi(rho=Rho, phi=Phi_com_fase)
                        else:  # TM
                            if campo_anim == 'eletrico':
                                rho = cilindrico.TM_E_rho(rho=Rho, phi=Phi_com_fase)
                                phi = cilindrico.TM_E_phi(rho=Rho, phi=Phi_com_fase)
                            else:
                                rho = cilindrico.TM_H_rho(rho=Rho, phi=Phi_com_fase)
                                phi = cilindrico.TM_H_phi(rho=Rho, phi=Phi_com_fase)

                        # Converter para cartesianas
                        e_x = rho * np.cos(Phi) - phi * np.sin(Phi)
                        e_y = rho * np.sin(Phi) + phi * np.cos(Phi)

                        # Filtrar pontos
                        mask = Rho <= cilindrico.raio
                        X_masked = X[mask]
                        Y_masked = Y[mask]
                        e_x = e_x[mask]
                        e_y = e_y[mask]

                        # Normalizar vetores
                        magnitude = np.sqrt(e_x**2 + e_y**2)
                        max_magnitude = np.max(magnitude) if np.max(magnitude) > 0 else 1e-10

                        magnitude_nonzero = np.where(magnitude > 1e-12, magnitude, 1e-12)
                        e_x_normalized = e_x / magnitude_nonzero
                        e_y_normalized = e_y / magnitude_nonzero

                        scale_factor = cilindrico.raio * 0.1
                        e_x_display = e_x_normalized * scale_factor
                        e_y_display = e_y_normalized * scale_factor

                        norm = plt.Normalize(vmin=0, vmax=max_magnitude)
                        cmap = plt.cm.viridis

                        quiver = ax.quiver(X_masked, Y_masked, e_x_display, e_y_display,
                                           magnitude, cmap=cmap, norm=norm,
                                           scale=1, scale_units='xy', angles='xy',
                                           pivot='middle', alpha=0.8)

                        if i == 0:
                            cbar = plt.colorbar(quiver, ax=ax, shrink=0.8, aspect=20)
                            cbar.set_label('Intensidade do Campo', rotation=270, labelpad=20)

                        circle = plt.Circle((0, 0), cilindrico.raio, color='red', fill=False, linestyle='--', linewidth=1.5)
                        ax.add_patch(plt.Circle((0, 0), cilindrico.raio, color='lightgray', alpha=0.5, zorder=0))
                        ax.add_artist(circle)

                        ax.set_xlabel("X (m)")
                        ax.set_ylabel("Y (m)")
                        ax.set_title(f"Campo {campo_anim.capitalize()} {transversal_anim}{modo_n}{modo_m} - Fase φ={fase_phi:.2f}rad (z={z_fixo}m)")
                        ax.axis('equal')
                        ax.set_xlim(-cilindrico.raio*1.2, cilindrico.raio*1.2)
                        ax.set_ylim(-cilindrico.raio*1.2, cilindrico.raio*1.2)

                        # Salvar frame
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                        buf.seek(0)
                        images.append(buf.getvalue())
                        buf.close()

                    plt.close(fig)

                    # Salvar como GIF
                    from PIL import Image
                    pil_images = [Image.open(io.BytesIO(img)) for img in images]
                    pil_images[0].save(
                        nome_arquivo,
                        save_all=True,
                        append_images=pil_images[1:],
                        duration=interval,
                        loop=0
                    )

                    # Exibir o GIF
                    with open(nome_arquivo, "rb") as file:
                        gif_data = file.read()

                    st.image(gif_data, caption=f"Animação: {nome_arquivo}")

                    # Link para download
                    b64_gif = base64.b64encode(gif_data).decode()
                    href = f'<a href="data:image/gif;base64,{b64_gif}" download="{nome_arquivo}">📥 Baixar {nome_arquivo}</a>'
                    st.markdown(href, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="custom-success">
                        ✅ <strong>Animação gerada com sucesso!</strong>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.markdown(f"""
                    <div class="custom-warning">
                        ⚠️ <strong>Erro ao gerar animação:</strong> {str(e)}
                    </div>
                    """, unsafe_allow_html=True)

    with tab3:
        st.markdown("**Animação 3D da Cavidade Cilíndrica - Visualização nas Paredes**")
        st.write("Esta visualização mostra a intensidade do campo nas superfícies da cavidade cilíndrica (tampas e superfície lateral).")

        col1, col2 = st.columns(2)

        with col1:
            modo_cavidade_cyl = st.selectbox("Modo (Cavidade Cilíndrica)", ["TE", "TM"], key="modo_cavidade_cyl")
            campo_cavidade_cyl = st.selectbox("Campo (Cavidade Cilíndrica)", ["magnetico", "eletrico"], key="campo_cavidade_cyl")

        with col2:
            tipo_intensidade_cyl = st.selectbox(
                "Tipo de Intensidade",
                ["total", "perpendicular", "direcional"],
                key="tipo_intensidade_cyl",
                help="Total: magnitude total | Perpendicular: componente perpendicular à superfície | Direcional: componente específica"
            )
            direcao_vetor_cyl = st.selectbox(
                "Direção do Vetor",
                ["rho", "phi", "z"],
                key="direcao_vetor_cyl",
                help="Direção da componente quando tipo_intensidade='direcional'"
            )

        col3, col4 = st.columns(2)

        with col3:
            resolucao_cavidade_cyl = st.slider(
                "Resolução",
                min_value=10,
                max_value=50,
                value=25,
                step=5,
                key="resolucao_cavidade_cyl",
                help="Número de pontos por dimensão (valores menores = mais rápido)"
            )
            num_frames_cyl = st.slider(
                "Número de Frames",
                min_value=10,
                max_value=100,
                value=60,
                step=10,
                key="num_frames_cyl",
                help="Número de frames na animação"
            )

        with col4:
            duracao_frame_cyl = st.slider(
                "Duração do Frame (ms)",
                min_value=50,
                max_value=500,
                value=100,
                step=50,
                key="duracao_frame_cyl",
                help="Duração de cada frame em milissegundos"
            )
            profundidade_cavidade_cyl = st.number_input(
                "Profundidade da Cavidade (mm)",
                value=100.0,
                step=10.0,
                key="profundidade_cavidade_cyl",
                help="Profundidade (comprimento) da cavidade cilíndrica na direção Z"
            )

        if st.button("🎬 Gerar Animação 3D da Cavidade Cilíndrica", use_container_width=True):
            if 'cilindro' not in state:
                st.markdown("""
                <div class="custom-warning">
                    ⚠️ <strong>Configuração não encontrada.</strong> Por favor, volte à etapa anterior.
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner("Gerando animação 3D da cavidade cilíndrica... Isso pode levar alguns instantes."):
                    try:
                        # Recuperar parâmetros do state
                        cilindrico = state['cilindro']
                        raio_mm = cilindrico.raio * 1000  # m -> mm
                        frequencia = cilindrico.frequencia
                        modo_m, modo_n = state['modo_m'], state['modo_n']
                        permissividade_val = state.get('permissividade_original', 1.0)
                        permeabilidade_val = state.get('permeabilidade_original', 1.0)

                        # Criar instância da CylindricalCavityWall3D
                        cavity_cyl = CylindricalCavityWall3D(
                            raio=raio_mm,
                            profundidade=profundidade_cavidade_cyl,
                            frequencia=frequencia,
                            permissividade=permissividade_val,
                            permeabilidade=permeabilidade_val,
                            resolucao=resolucao_cavidade_cyl,
                            m=modo_m,
                            n=modo_n
                        )

                        # Gerar a animação
                        fig = cavity_cyl.animar_cavidade_plotly(
                            modo=modo_cavidade_cyl,
                            campo=campo_cavidade_cyl,
                            tipo_intensidade=tipo_intensidade_cyl,
                            direcao_vetor=direcao_vetor_cyl,
                            num_frames=num_frames_cyl,
                            duracao_frame=duracao_frame_cyl
                        )

                        # Salvar como HTML em arquivo temporário
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        html_filename = f'animacao_cilindrica_3d_{timestamp}.html'
                        html_path = os.path.join(tempfile.gettempdir(), html_filename)
                        fig.write_html(html_path)

                        # Ler o HTML gerado e salvar no session_state
                        with open(html_path, 'r', encoding='utf-8') as f:
                            html_string = f.read()

                        # Salvar no session_state para persistir entre reruns
                        st.session_state['html_cilindrico'] = html_string
                        st.session_state['html_path_cilindrico'] = html_path
                        st.session_state['html_filename_cilindrico'] = html_filename

                    except Exception as e:
                        st.markdown(f"""
                        <div class="custom-warning">
                            ⚠️ <strong>Erro ao gerar animação 3D da cavidade cilíndrica:</strong> {str(e)}
                        </div>
                        """, unsafe_allow_html=True)
                        import traceback
                        st.code(traceback.format_exc())

        # Exibir animação se existir no session_state
        if 'html_cilindrico' in st.session_state:
            # Exibir a animação incorporada com opção de maximizar
            col_title, col_maximize = st.columns([3, 1])
            with col_title:
                st.markdown("### Visualização Interativa 3D")
            with col_maximize:
                maximizar = st.checkbox("🔍 Maximizar", key="maximizar_cilindrico")

            # Ajustar altura baseado na opção de maximizar
            altura_viz = 1200 if maximizar else 700
            components.html(st.session_state['html_cilindrico'], height=altura_viz, scrolling=False)

            st.markdown("""
            <div class="custom-success">
                ✅ <strong>Animação 3D da cavidade cilíndrica gerada com sucesso!</strong> Use os controles para explorar.
            </div>
            """, unsafe_allow_html=True)

            # Botão para download do HTML
            col_download1, col_download2 = st.columns([1, 3])
            with col_download1:
                with open(st.session_state['html_path_cilindrico'], 'rb') as f:
                    st.download_button(
                        label="💾 Baixar HTML",
                        data=f.read(),
                        file_name=st.session_state['html_filename_cilindrico'],
                        mime='text/html',
                        use_container_width=True
                    )
            with col_download2:
                st.info(f"📄 Arquivo salvo temporariamente em: `{st.session_state['html_path_cilindrico']}`")

    with tab4:
        st.markdown("**Coeficientes S e Equações**")
        st.write("As equações abaixo descrevem os campos elétricos e magnéticos na guia cilíndrica:")

        # Equações LaTeX
        st.latex(r"E_\rho(\rho, \phi, z) = -\frac{j \omega \mu}{k_c^2 \rho} \left[A \cos(n\phi) - B \sin(n\phi)\right] J_n(k_c \rho) e^{-j\beta z}")

        st.code('''
def TM_E_rho(self, rho, phi):
    const = -1j*self.beta_val/(self.k_c_val_tm)
    seno = self.seno_Nphi(phi)
    cosseno = self.cosseno_Nphi(phi)
    jv_n = self.jv_n(rho)
    return np.real(const*(self.A*seno + self.B*cosseno)*jv_n*self.exp_z_val)
        ''', language='python', line_numbers=True)

        st.latex(r"E_\phi(\rho, \phi, z) = \frac{j \omega \mu}{k_c} \left[A \sin(n\phi) + B \cos(n\phi)\right] J_n'(k_c \rho) e^{-j\beta z}")
        st.latex(r"H_\rho(\rho, \phi, z) = -\frac{j \beta}{k_c} \left[A \sin(n\phi) + B \cos(n\phi)\right] J_n'(k_c \rho) e^{-j\beta z}")
        st.latex(r"H_\phi(\rho, \phi, z) = -\frac{j \beta n}{k_c^2 \rho} \left[A \cos(n\phi) - B \sin(n\phi)\right] J_n(k_c \rho) e^{-j\beta z}")

    with tab5:
        st.markdown("**Análise dos Resultados**")

        cilindrico = state['cilindro']
        modo_m, modo_n = state['modo_m'], state['modo_n']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #F18F01;">
                <h6 style="color: #F18F01; margin: 0 0 0.5rem 0;">📊 Parâmetros Calculados</h6>
            </div>
            """, unsafe_allow_html=True)

            st.write(f"• **Frequência:** {cilindrico.frequencia/1e9:.2f} GHz")
            st.write(f"• **Raio:** {cilindrico.raio*1000:.2f} mm")
            st.write(f"• **Modo:** TE_{modo_n}{modo_m} / TM_{modo_n}{modo_m}")

        with col2:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #C73E1D;">
                <h6 style="color: #C73E1D; margin: 0 0 0.5rem 0;">📈 Propriedades</h6>
            </div>
            """, unsafe_allow_html=True)

            # Usar valores armazenados no estado ou tentar acessar os atributos corretos
            permissividade_val = state.get('permissividade_original', getattr(cilindrico, 'mu', 'N/A'))
            permeabilidade_val = state.get('permeabilidade_original', getattr(cilindrico, 'epsilon', 'N/A'))

            st.write(f"• **Permissividade:** {permissividade_val}")
            st.write(f"• **Permeabilidade:** {permeabilidade_val}")

        # Seção de Relatório PDF
        st.markdown("---")
        st.markdown("**📄 Geração de Relatório Completo**")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            Gere um relatório PDF completo contendo:
            • Todos os parâmetros da simulação cilíndrica
            • Gráficos 3D e animações capturadas
            • Análise de modos TEₘₙ e TMₘₙ
            • Equações e dados técnicos
            """)

        with col2:
            if st.button("📄 Gerar Relatório PDF", use_container_width=True, type="primary", key="relatorio_cilindrica"):
                if RELATORIO_DISPONIVEL:
                    gerar_relatorio_pdf("cilindrica")
                else:
                    st.error("📦 Para usar relatórios, instale: pip install reportlab plotly-kaleido")

    # Botões de navegação
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("⬅️ Voltar à Configuração"):
            st.session_state.step = 1
            st.rerun()

    with col2:
        if st.button("🏠 Dashboard Principal"):
            st.session_state.pagina_atual = "Dashboard"
            st.session_state.step = 0
            st.rerun()

    with col3:
        if st.button("🔁 Nova Simulação"):
            # Limpar o estado atual
            keys_to_remove = ['cilindro', 'X', 'Y', 'Rho', 'Phi', 'modo_m', 'modo_n', 'permissividade_original', 'permeabilidade_original']
            for key in keys_to_remove:
                if key in state:
                    del state[key]
            st.session_state.step = 1
            st.rerun()

def resultados_cilindrica():
    st.markdown("""<h4 style="color: #A23B72;">📈 Resultados e Análise</h4>""", unsafe_allow_html=True)
    st.info("Esta seção será implementada em futuras versões.")

def main():
    apply_custom_css()

    # Inicializar estado se não existir
    if 'pagina_atual' not in st.session_state:
        st.session_state.pagina_atual = "Dashboard"
        st.session_state.step = 0

    # Título da barra lateral
    st.sidebar.title("🧭 Navegação")

    # Botão Dashboard
    if st.sidebar.button("🏠 Dashboard Principal", use_container_width=True):
        st.session_state.pagina_atual = "Dashboard"
        st.session_state.step = 0
        st.rerun()

    st.sidebar.markdown("---")

    # Determinar índice atual do radio baseado na página atual
    menu_opcoes = ["📰 Guia Retangular", "⭕ Guia Cilíndrica"]

    # Mapear página atual para índice do radio
    if st.session_state.pagina_atual == "Guia Retangular":
        radio_index = 0
    elif st.session_state.pagina_atual == "Guia Cilíndrica":
        radio_index = 1
    else:
        radio_index = None

    # Radio button com estado persistente
    pagina_sidebar = st.sidebar.radio(
        "Selecione o tipo de guia:",
        menu_opcoes,
        index=radio_index,
        key="sidebar_navigation"
    )

    # Atualizar estado apenas se houve mudança
    if pagina_sidebar:
        nova_pagina = None
        if pagina_sidebar == "📰 Guia Retangular":
            nova_pagina = "Guia Retangular"
        elif pagina_sidebar == "⭕ Guia Cilíndrica":
            nova_pagina = "Guia Cilíndrica"

        # Só atualiza se realmente mudou
        if nova_pagina and nova_pagina != st.session_state.pagina_atual:
            st.session_state.pagina_atual = nova_pagina
            st.session_state.step = 1
            st.rerun()

    # Informações da página atual na sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📍 Página Atual:**")
    if st.session_state.pagina_atual == "Dashboard":
        st.sidebar.info("🏠 Dashboard Principal")
    elif st.session_state.pagina_atual == "Guia Retangular":
        st.sidebar.info(f"📰 Guia Retangular\n\n**Etapa:** {st.session_state.step}/3")
    elif st.session_state.pagina_atual == "Guia Cilíndrica":
        st.sidebar.info(f"⭕ Guia Cilíndrica\n\n**Etapa:** {st.session_state.step}/3")

    # Botão de reset na sidebar
    if st.session_state.pagina_atual != "Dashboard":
        st.sidebar.markdown("---")
        if st.sidebar.button("🔄 Resetar Simulação", use_container_width=True):
            # Limpar estados específicos
            keys_to_remove = ['TEmn', 'cilindro', 'X', 'Y', 'Rho', 'Phi', 'modo_m', 'modo_n',
                             'campo_componente', 'permissividade_original', 'permeabilidade_original']
            for key in keys_to_remove:
                if key in state:
                    del state[key]
            st.session_state.step = 1
            st.rerun()

    # Renderizar conteúdo principal
    if st.session_state.pagina_atual == "Dashboard":
        dashboard_principal()
    elif st.session_state.pagina_atual == "Guia Retangular":
        show_progress_indicator(st.session_state.step)
        guia_retangular()
    elif st.session_state.pagina_atual == "Guia Cilíndrica":
        show_progress_indicator(st.session_state.step)
        guia_cilindrica()

if __name__ == "__main__":
    main()