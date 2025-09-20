import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.TEmn_model import Modo_TEmn
from models.TMmn_model import Modo_TMmn
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
        'permissividade': getattr(TEmn, 'mu', 1.0),  # CORRIGIDO
        'permeabilidade': getattr(TEmn, 'epsilon', 1.0),  # CORRIGIDO
        'plano': TEmn.plano,
        'campo': campo,
        'componente': componente,
        'material': 'Material configurado',
        'imagens': {}
    }

    # Adicionar dados de matriz de espalhamento se disponíveis
    if 'scattering_data' in state:
        scattering_data = state['scattering_data']

        # Usar dados herdados da configuração principal
        dados['scattering_data'] = {
            'largura': TEmn.largura * 1000,  # m para mm
            'altura': TEmn.altura * 1000,    # m para mm
            'comprimento': scattering_data['config'].get('comprimento', 50.0),
            'freq_min': max(0.1, (TEmn.frequencia/1e9) - 3.0),
            'freq_max': (TEmn.frequencia/1e9) + 3.0,
            'permissividade': getattr(TEmn, 'mu', 1.0),
            'permeabilidade': getattr(TEmn, 'epsilon', 1.0),
            'Q_factor': scattering_data['config'].get('q_factor', 1000)
        }

        dados['scattering'] = {
            'disponivel': True,
            'config': scattering_data['config'],
            'S11': scattering_data['S11'],
            'S21': scattering_data['S21'],
            'frequencies': scattering_data['frequencies']
        }

        # Adicionar métricas calculadas
        s11_db = 20*np.log10(np.abs(scattering_data['S11']))
        s21_db = 20*np.log10(np.abs(scattering_data['S21']))

        dados['scattering']['metricas'] = {
            'melhor_casamento_db': float(np.min(s11_db)),
            'freq_melhor_casamento': float(scattering_data['frequencies'][np.argmin(s11_db)]),
            'melhor_transmissao_db': float(np.max(s21_db)),
            'freq_melhor_transmissao': float(scattering_data['frequencies'][np.argmax(s21_db)])
        }

        # Adicionar frequências de ressonância se disponíveis
        if 'scattering_object' in scattering_data:
            scattering_obj = scattering_data['scattering_object']
            freq_min_hz = dados['scattering_data']['freq_min'] * 1e9
            freq_max_hz = dados['scattering_data']['freq_max'] * 1e9

            dados['scattering_data']['frequencias_ressonancia'] = {}

            # TE modes na faixa
            for m, n in scattering_obj.modos_te:
                f_res = scattering_obj.calcular_freq_ressonancia_te(m, n)
                if freq_min_hz <= f_res <= freq_max_hz:
                    dados['scattering_data']['frequencias_ressonancia'][f'TE{m}{n}'] = f_res / 1e9

            # TM modes na faixa
            for m, n in scattering_obj.modos_tm:
                f_res = scattering_obj.calcular_freq_ressonancia_tm(m, n)
                if freq_min_hz <= f_res <= freq_max_hz:
                    dados['scattering_data']['frequencias_ressonancia'][f'TM{m}{n}'] = f_res / 1e9
    else:
        dados['scattering'] = {'disponivel': False}

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

        # Capturar gráfico de matriz de espalhamento se disponível
        if 'scattering_data' in state:
            with st.spinner("Capturando análise de matriz de espalhamento..."):
                try:
                    scattering_object = state['scattering_data']['scattering_object']
                    fig_scattering = scattering_object.plot_s_parameters_matplotlib()

                    if RELATORIO_DISPONIVEL:
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        temp_filename = file_manager.get_temp_path(f"matriz_espalhamento_{timestamp}.png")
                        fig_scattering.savefig(temp_filename, dpi=150, bbox_inches='tight',
                                             facecolor='white', edgecolor='none', format='png')

                        imagens['Matriz de Espalhamento - Parâmetros S'] = {'tipo': 'arquivo', 'caminho': str(temp_filename)}

                    plt.close(fig_scattering)

                except Exception as e_scatter:
                    imagens['Erro Matriz Espalhamento'] = f"Erro na captura da matriz S: {str(e_scatter)}"

        # Adicionar nota informativa
        imagens['Nota'] = "Gráficos 3D interativos e análises adicionais disponíveis na interface web"

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

                    TMmn = Modo_TMmn(
                        largura=largura_guia,
                        altura=altura_guia,
                        frequencia=frequencia_onda * 1e9,
                        permissividade=permissividade_meio,
                        permeabilidade=permeabilidade_meio,
                        plano=plano
                    )
                    TMmn.calcula_campos()
                    state['TMmn'] = TMmn
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

    # Seletor de modo
    st.markdown("**🔧 Escolha do Modo de Simulação**")
    modo_selecionado = st.selectbox(
        "Selecione o modo:",
        ["TE (Transverse Electric)", "TM (Transverse Magnetic)"],
        key="modo_simulacao"
    )

    # Determinar qual objeto usar baseado na seleção
    if "TE" in modo_selecionado:
        modo_obj = state['TEmn']
        modo_nome = "TE"
    else:
        modo_obj = state['TMmn']
        modo_nome = "TM"

    # Criar abas para diferentes tipos de visualização
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Campo 3D Interativo", "Campo 2D", "Campo Vetorial", "Análise", "Matriz de Espalhamento"])

    with tab1:
        campo, componente = state['campo_componente']

        st.markdown(f"**Visualização 3D Interativa do Campo - Modo {modo_nome}**")

        if st.button("🌍 Gerar Visualização 3D", use_container_width=True):
            with st.spinner("Gerando visualização 3D..."):
                # Obter os dados do campo
                modo_obj.calcula_campos()
                if campo == 'magnetico':
                    if componente == 'x':
                        imagem = modo_obj.Hx
                    elif componente == 'y':
                        imagem = modo_obj.Hy
                    elif componente == 'z':
                        imagem = modo_obj.Hz
                elif campo == 'eletrico':
                    if componente == 'x':
                        imagem = modo_obj.Ex
                    elif componente == 'y':
                        imagem = modo_obj.Ey
                    elif componente == 'z':
                        imagem = modo_obj.Ez

                # Criar o gráfico 3D interativo melhorado
                fig = go.Figure(data=[go.Surface(
                    z=imagem,
                    x=modo_obj.x[:, 0],
                    y=modo_obj.y[0, :],
                    colorscale='Viridis',
                    showscale=True
                )])

                fig.update_layout(
                    title=f"Modo {modo_nome} - Campo {campo.capitalize()} - Componente {componente.upper()}",
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
        campo, componente = state['campo_componente']

        st.markdown(f"**Visualização 2D da Intensidade do Campo - Modo {modo_nome}**")

        if st.button("🗺️ Gerar Mapa 2D", use_container_width=True):
            with st.spinner("Gerando mapa 2D..."):
                fig = modo_obj.plot3DField(campo=campo, componente=componente)
                st.pyplot(fig)
                plt.close(fig)

                st.markdown("""
                <div class="custom-success">
                    ✅ <strong>Mapa 2D gerado com sucesso!</strong> As cores representam a intensidade do campo.
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        campo, componente = state['campo_componente']

        st.markdown(f"**Visualização Vetorial do Campo - Modo {modo_nome}**")

        if st.button("➡️ Gerar Campo Vetorial", use_container_width=True):
            with st.spinner("Gerando campo vetorial..."):
                fig = modo_obj.plota_campo_vetorial(campo)
                st.pyplot(fig)
                plt.close(fig)

                st.markdown("""
                <div class="custom-success">
                    ✅ <strong>Campo vetorial gerado com sucesso!</strong> As setas indicam direção e intensidade.
                </div>
                """, unsafe_allow_html=True)

    with tab4:
        st.markdown(f"**Análise dos Resultados - Modo {modo_nome}**")

        # Mostrar informações sobre os parâmetros calculados

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #F18F01;">
                <h6 style="color: #F18F01; margin: 0 0 0.5rem 0;">📊 Parâmetros Calculados</h6>
            </div>
            """, unsafe_allow_html=True)

            st.write(f"• **Frequência:** {modo_obj.frequencia/1e9:.2f} GHz")
            st.write(f"• **Largura:** {modo_obj.largura:.2f} mm")
            st.write(f"• **Altura:** {modo_obj.altura:.2f} mm")

        with col2:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #C73E1D;">
                <h6 style="color: #C73E1D; margin: 0 0 0.5rem 0;">📈 Propriedades do Campo</h6>
            </div>
            """, unsafe_allow_html=True)

            st.write(f"• **Campo:** {campo.capitalize()}")
            st.write(f"• **Componente:** {componente.upper()}")
            st.write(f"• **Plano:** {modo_obj.plano.upper()}")
            st.write(f"• **Modo:** {modo_nome}")

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

    with tab5:
        st.markdown(f"**📊 Matriz de Espalhamento - Análise de Parâmetros S - Modo {modo_nome}**")

        # Informações sobre a análise
        st.markdown("""
        <div style="background: #f0f8ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #2E86AB; margin-bottom: 1rem;">
            <h6 style="color: #2E86AB; margin: 0 0 0.5rem 0;">ℹ️ Sobre a Matriz de Espalhamento</h6>
            <p style="margin: 0; color: #666;">
            A matriz de espalhamento (parâmetros S) descreve como a energia eletromagnética é
            refletida e transmitida em dispositivos de microondas. S₁₁ representa reflexão,
            S₂₁ representa transmissão.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Configurações herdadas da simulação principal
        st.markdown("**⚙️ Parâmetros Herdados da Configuração**")

        # Recuperar parâmetros dos dados de configuração armazenados
        largura_mm = float(modo_obj.largura * 1000)
        altura_mm = float(modo_obj.altura * 1000)
        freq_ghz = modo_obj.frequencia / 1e9
        permissividade = getattr(modo_obj, 'mu', 1.0)
        permeabilidade = getattr(modo_obj, 'epsilon', 1.0)

        # Calcular faixa de frequência automática baseada na frequência atual
        freq_min = max(0.1, freq_ghz - 3.0)
        freq_max = freq_ghz + 3.0

        # Mostrar os parâmetros herdados
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div style="background: #e8f5e8; padding: 0.8rem; border-radius: 6px; border-left: 4px solid #28a745;">
                <strong>📏 Largura:</strong> {largura_mm:.2f} mm<br/>
                <strong>📐 Altura:</strong> {altura_mm:.2f} mm
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="background: #e3f2fd; padding: 0.8rem; border-radius: 6px; border-left: 4px solid #2196f3;">
                <strong>📡 Frequência Central:</strong> {freq_ghz:.2f} GHz<br/>
                <strong>📊 Faixa:</strong> {freq_min:.1f} - {freq_max:.1f} GHz
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="background: #fff3e0; padding: 0.8rem; border-radius: 6px; border-left: 4px solid #ff9800;">
                <strong>⚡ Permissividade (εᵣ):</strong> {permissividade:.2f}<br/>
                <strong>🧲 Permeabilidade (μᵣ):</strong> {permeabilidade:.2f}
            </div>
            """, unsafe_allow_html=True)

        # Configurações adicionais personalizáveis
        st.markdown("**🔧 Configurações Adicionais**")

        col4, col5 = st.columns(2)

        with col4:
            comprimento_mm = st.number_input(
                "Comprimento da Cavidade (mm)",
                value=50.0,
                min_value=10.0,
                max_value=200.0,
                step=1.0,
                help="Dimensão c da cavidade (profundidade) - não definida na simulação principal"
            )

        with col5:
            q_factor = st.number_input(
                "Fator Q",
                value=1000,
                min_value=100,
                max_value=10000,
                step=100,
                help="Qualidade do ressonador (Q maior = menos perdas)"
            )

        # Validação
        if freq_min >= freq_max:
            st.error("❌ Frequência mínima deve ser menor que a máxima!")
            return

        # Botão para gerar análise
        if st.button("🔍 Gerar Análise de Matriz de Espalhamento", use_container_width=True, type="primary"):

            try:
                # Importar o modelo
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
                from Scattering_model import ScatteringMatrix

                with st.spinner("Calculando matriz de espalhamento..."):

                    # Criar instância do modelo usando parâmetros herdados
                    scattering = ScatteringMatrix(
                        largura=largura_mm,
                        altura=altura_mm,
                        comprimento=comprimento_mm,
                        permissividade=permissividade,
                        permeabilidade=permeabilidade,
                        freq_min=freq_min,
                        freq_max=freq_max,
                        num_pontos=500
                    )
                    scattering.Q_factor = q_factor

                    # Calcular parâmetros S
                    S11, S12, S21, S22 = scattering.calcular_matriz_s()
                    freq_ghz = scattering.frequencies / 1e9

                    # Armazenar no estado para o relatório
                    state['scattering_data'] = {
                        'scattering_object': scattering,
                        'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22,
                        'frequencies': freq_ghz,
                        'config': {
                            'largura': largura_mm,
                            'altura': altura_mm,
                            'comprimento': comprimento_mm,
                            'freq_min': freq_min,
                            'freq_max': freq_max,
                            'q_factor': q_factor
                        }
                    }

                    # Mostrar resultados principais
                    st.success("✅ Análise concluída!")

                    # Métricas importantes
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        s11_db = 20*np.log10(np.abs(S11))
                        min_s11 = np.min(s11_db)
                        min_freq = freq_ghz[np.argmin(s11_db)]
                        st.metric("Melhor Casamento", f"{min_s11:.1f} dB", f"@ {min_freq:.2f} GHz")

                    with col2:
                        s21_db = 20*np.log10(np.abs(S21))
                        max_s21 = np.max(s21_db)
                        max_freq = freq_ghz[np.argmax(s21_db)]
                        st.metric("Melhor Transmissão", f"{max_s21:.1f} dB", f"@ {max_freq:.2f} GHz")

                    with col3:
                        # Frequências de ressonância TE na faixa
                        resonances_te = []
                        for m, n in scattering.modos_te:
                            f_res = scattering.calcular_freq_ressonancia_te(m, n)
                            if freq_min*1e9 <= f_res <= freq_max*1e9:
                                resonances_te.append(f_res/1e9)
                        st.metric("Ressonâncias TE", len(resonances_te), "na faixa")

                    with col4:
                        # Frequências de ressonância TM na faixa
                        resonances_tm = []
                        for m, n in scattering.modos_tm:
                            f_res = scattering.calcular_freq_ressonancia_tm(m, n)
                            if freq_min*1e9 <= f_res <= freq_max*1e9:
                                resonances_tm.append(f_res/1e9)
                        st.metric("Ressonâncias TM", len(resonances_tm), "na faixa")

                    # Gráfico principal usando Plotly
                    try:
                        fig_plotly = scattering.plot_s_parameters_plotly()
                        st.plotly_chart(fig_plotly, use_container_width=True)

                        # Armazenar figura para relatório
                        state['scattering_data']['figure_plotly'] = fig_plotly

                    except Exception as e:
                        st.error(f"Erro ao gerar gráfico Plotly: {e}")

                        # Fallback para matplotlib
                        try:
                            fig_mpl = scattering.plot_s_parameters_matplotlib()
                            st.pyplot(fig_mpl, use_container_width=True)
                            plt.close(fig_mpl)

                            # Armazenar figura para relatório
                            state['scattering_data']['figure_matplotlib'] = fig_mpl

                        except Exception as e2:
                            st.error(f"Erro ao gerar gráfico matplotlib: {e2}")

                    # Tabela de frequências de ressonância
                    if resonances_te or resonances_tm:
                        st.markdown("**🎯 Frequências de Ressonância Identificadas**")

                        col1, col2 = st.columns(2)

                        with col1:
                            if resonances_te:
                                st.markdown("**Modos TE:**")
                                te_data = []
                                for i, (m, n) in enumerate(scattering.modos_te):
                                    f_res = scattering.calcular_freq_ressonancia_te(m, n)
                                    if freq_min*1e9 <= f_res <= freq_max*1e9:
                                        te_data.append({
                                            "Modo": f"TE{m}{n}",
                                            "Frequência": f"{f_res/1e9:.3f} GHz"
                                        })
                                if te_data:
                                    st.dataframe(te_data, use_container_width=True)

                        with col2:
                            if resonances_tm:
                                st.markdown("**Modos TM:**")
                                tm_data = []
                                for i, (m, n) in enumerate(scattering.modos_tm):
                                    f_res = scattering.calcular_freq_ressonancia_tm(m, n)
                                    if freq_min*1e9 <= f_res <= freq_max*1e9:
                                        tm_data.append({
                                            "Modo": f"TM{m}{n}",
                                            "Frequência": f"{f_res/1e9:.3f} GHz"
                                        })
                                if tm_data:
                                    st.dataframe(tm_data, use_container_width=True)

                    # Opção de download dos dados
                    if st.button("💾 Exportar Dados CSV", use_container_width=True):
                        try:
                            import pandas as pd
                            data = {
                                'Frequency_GHz': freq_ghz,
                                'S11_magnitude_dB': 20*np.log10(np.abs(S11)),
                                'S11_phase_deg': np.angle(S11)*180/np.pi,
                                'S21_magnitude_dB': 20*np.log10(np.abs(S21)),
                                'S21_phase_deg': np.angle(S21)*180/np.pi
                            }
                            df = pd.DataFrame(data)
                            csv = df.to_csv(index=False)

                            st.download_button(
                                label="📥 Download CSV",
                                data=csv,
                                file_name=f"parametros_s_{freq_min:.0f}-{freq_max:.0f}GHz.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        except ImportError:
                            st.error("Pandas não disponível para exportação")

            except ImportError:
                st.error("❌ Modelo de matriz de espalhamento não encontrado!")
                st.markdown("""
                **Solução:** Certifique-se de que o arquivo `Scattering_model.py` está em `src/models/`
                """)
            except Exception as e:
                st.error(f"❌ Erro durante análise: {str(e)}")

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
    tab1, tab2, tab3, tab4 = st.tabs(["Campo 3D Vetorial", "Animação de Fase", "Matriz de Espalhamento", "Análise"])

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
        st.markdown("**Análise da Matriz de Espalhamento - Guia Cilíndrica**")
        st.write("Análise completa dos parâmetros S (reflexão e transmissão) para guias de onda cilíndricas.")

        # Configurações da análise
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #007BFF;">
                <h6 style="color: #007BFF; margin: 0 0 0.5rem 0;">⚙️ Configurações da Análise</h6>
            </div>
            """, unsafe_allow_html=True)

            freq_min_cyl = st.number_input("Frequência Mínima (GHz)", value=6.0, min_value=1.0, max_value=50.0, step=0.5, key="freq_min_cyl")
            freq_max_cyl = st.number_input("Frequência Máxima (GHz)", value=18.0, min_value=1.0, max_value=50.0, step=0.5, key="freq_max_cyl")
            comprimento_guia = st.number_input("Comprimento da Guia (mm)", value=100.0, min_value=10.0, max_value=500.0, step=10.0, key="comp_guia_cyl")

        with col2:
            st.markdown("""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #28A745;">
                <h6 style="color: #28A745; margin: 0 0 0.5rem 0;">📊 Parâmetros da Simulação</h6>
            </div>
            """, unsafe_allow_html=True)

            cilindrico = state['cilindro']
            st.write(f"**Raio:** {cilindrico.raio*1000:.2f} mm")
            st.write(f"**Frequência Atual:** {cilindrico.frequencia/1e9:.2f} GHz")
            st.write(f"**Modo:** TE₀₁ (dominante)")
            st.write(f"**Material:** εᵣ={cilindrico.epsilon}, μᵣ={cilindrico.mu}")

        if st.button("📈 Calcular Matriz de Espalhamento", use_container_width=True, type="primary", key="calc_scattering_cyl"):
            with st.spinner("Calculando matriz de espalhamento cilíndrica..."):
                try:
                    # Importar o modelo de matriz de espalhamento cilíndrica
                    import sys
                    import os
                    sys.path.append(os.path.join(os.getcwd(), 'src'))
                    from models.Cylindrical_Scattering_model import CylindricalScatteringMatrix

                    # Criar instância da matriz de espalhamento
                    cyl_scattering = CylindricalScatteringMatrix(
                        raio=cilindrico.raio,
                        comprimento=comprimento_guia/1000.0,  # converter mm para m
                        permissividade=cilindrico.epsilon,
                        permeabilidade=cilindrico.mu,
                        freq_min=freq_min_cyl,
                        freq_max=freq_max_cyl,
                        num_pontos=300
                    )

                    # Calcular matriz S
                    S11, S12, S21, S22 = cyl_scattering.calcular_matriz_s()
                    frequencies_ghz = cyl_scattering.frequencies / 1e9

                    # Gerar gráfico
                    fig = cyl_scattering.plot_s_parameters_comparison()
                    st.plotly_chart(fig, use_container_width=True)

                    # Análise estatística
                    st.markdown("---")
                    st.markdown("**📊 Análise Estatística dos Parâmetros S**")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("**|S11| Médio**", f"{np.mean(np.abs(S11)):.3f}",
                                 delta=f"{np.std(np.abs(S11)):.3f} (σ)")

                    with col2:
                        st.metric("**|S21| Médio**", f"{np.mean(np.abs(S21)):.3f}",
                                 delta=f"{np.std(np.abs(S21)):.3f} (σ)")

                    with col3:
                        energia_media = np.mean(np.abs(S11)**2 + np.abs(S21)**2)
                        st.metric("**Conservação Energia**", f"{energia_media:.3f}",
                                 delta="✅" if energia_media <= 1.01 else "⚠️")

                    with col4:
                        freq_teste = cilindrico.frequencia
                        S11_atual = cyl_scattering.calcular_s11(freq_teste)
                        st.metric("**|S11| Atual**", f"{abs(S11_atual):.3f}",
                                 delta=f"{20*np.log10(abs(S11_atual)):.1f} dB")

                    # Frequências de corte
                    st.markdown("**🎯 Frequências de Corte dos Modos**")

                    modos_info = []
                    for n, m in [(0,1), (1,1), (0,2), (2,1), (3,1)]:
                        try:
                            f_c_te = cyl_scattering.calcular_freq_corte_te(n, m)
                            f_c_tm = cyl_scattering.calcular_freq_corte_tm(n, m)
                            modos_info.append({
                                'Modo TE': f"TE₍{n}{m}₎",
                                'Freq. Corte TE (GHz)': f"{f_c_te/1e9:.2f}",
                                'Modo TM': f"TM₍{n}{m}₎",
                                'Freq. Corte TM (GHz)': f"{f_c_tm/1e9:.2f}"
                            })
                        except:
                            continue

                    if modos_info:
                        import pandas as pd
                        df_modos = pd.DataFrame(modos_info)
                        st.dataframe(df_modos, use_container_width=True)

                    # Análise por frequência específica
                    st.markdown("**🔍 Análise em Frequência Específica**")
                    freq_analise = st.slider("Frequência para Análise (GHz)",
                                            float(freq_min_cyl), float(freq_max_cyl),
                                            float(cilindrico.frequencia/1e9), 0.1, key="freq_analise_cyl")

                    freq_analise_hz = freq_analise * 1e9
                    S11_freq = cyl_scattering.calcular_s11(freq_analise_hz)
                    S21_freq = cyl_scattering.calcular_s21(freq_analise_hz)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"""
                        **Reflexão (S11)**
                        - Magnitude: {abs(S11_freq):.3f}
                        - Fase: {np.angle(S11_freq)*180/np.pi:.1f}°
                        - dB: {20*np.log10(abs(S11_freq)):.1f} dB
                        """)

                    with col2:
                        st.markdown(f"""
                        **Transmissão (S21)**
                        - Magnitude: {abs(S21_freq):.3f}
                        - Fase: {np.angle(S21_freq)*180/np.pi:.1f}°
                        - dB: {20*np.log10(abs(S21_freq)):.1f} dB
                        """)

                    with col3:
                        energia_freq = abs(S11_freq)**2 + abs(S21_freq)**2
                        vswr = (1 + abs(S11_freq)) / (1 - abs(S11_freq)) if abs(S11_freq) < 1 else float('inf')
                        st.markdown(f"""
                        **Análise**
                        - Energia Total: {energia_freq:.3f}
                        - VSWR: {vswr:.2f}
                        - Status: {"✅ OK" if energia_freq <= 1.01 else "⚠️ Revisar"}
                        """)

                    # Carta de Smith interativa
                    st.markdown("**🎯 Carta de Smith - Trajetória S11**")

                    # Criar carta de Smith com Plotly
                    fig_smith = go.Figure()

                    # Círculo unitário
                    theta = np.linspace(0, 2*np.pi, 200)
                    fig_smith.add_trace(go.Scatter(
                        x=np.cos(theta), y=np.sin(theta),
                        mode='lines', name='Círculo Unitário',
                        line=dict(color='black', width=2),
                        showlegend=False
                    ))

                    # Círculos de resistência constante
                    r_values = [0.2, 0.5, 1.0, 2.0, 5.0]
                    for r in r_values:
                        center_x = r / (1 + r)
                        radius = 1 / (1 + r)
                        circle_theta = np.linspace(0, 2*np.pi, 100)
                        x_circle = center_x + radius * np.cos(circle_theta)
                        y_circle = radius * np.sin(circle_theta)

                        mask = x_circle**2 + y_circle**2 <= 1.001
                        fig_smith.add_trace(go.Scatter(
                            x=x_circle[mask], y=y_circle[mask],
                            mode='lines', name=f'R={r}',
                            line=dict(color='gray', width=0.5),
                            showlegend=False, hoverinfo='skip'
                        ))

                    # Trajetória S11
                    fig_smith.add_trace(go.Scatter(
                        x=np.real(S11), y=np.imag(S11),
                        mode='lines+markers', name='S11',
                        line=dict(color='blue', width=3),
                        marker=dict(size=3),
                        customdata=frequencies_ghz,
                        hovertemplate='S11<br>Real: %{x:.3f}<br>Imag: %{y:.3f}<br>Freq: %{customdata:.1f} GHz<extra></extra>'
                    ))

                    # Pontos inicial e final
                    fig_smith.add_trace(go.Scatter(
                        x=[np.real(S11)[0]], y=[np.imag(S11)[0]],
                        mode='markers', name='Início',
                        marker=dict(color='green', size=10, symbol='circle')
                    ))
                    fig_smith.add_trace(go.Scatter(
                        x=[np.real(S11)[-1]], y=[np.imag(S11)[-1]],
                        mode='markers', name='Fim',
                        marker=dict(color='red', size=10, symbol='circle')
                    ))

                    fig_smith.update_layout(
                        title="Carta de Smith - Parâmetro S11",
                        xaxis=dict(range=[-1.1, 1.1], constrain='domain'),
                        yaxis=dict(range=[-1.1, 1.1], scaleanchor="x", scaleratio=1),
                        width=600, height=600,
                        showlegend=True
                    )

                    st.plotly_chart(fig_smith, use_container_width=True)

                    # Comparação com modelo teórico
                    st.markdown("**⚖️ Validação Teórica**")

                    try:
                        # Comparar com Modo_Cilindrico
                        cyl_scattering.comparar_com_cilindrico(freq_analise_hz, 0, 1)

                        # Capturar a saída (simplified version)
                        te01_cutoff = cyl_scattering.calcular_freq_corte_te(0, 1)

                        validacao_info = f"""
                        **Validação dos Resultados:**
                        - Frequência de Corte TE₀₁: {te01_cutoff/1e9:.2f} GHz
                        - Modo Dominante: {"✅ TE₀₁" if te01_cutoff < freq_analise_hz else "⚠️ Evanescente"}
                        - Impedância Característica: Calculada segundo teoria de Bessel
                        - Conservação de Energia: {"✅ Respeitada" if energia_media <= 1.01 else "⚠️ Verificar"}
                        """

                        st.markdown(validacao_info)

                    except Exception as e:
                        st.warning(f"Validação teórica: {str(e)}")

                    st.markdown("""
                    <div class="custom-success">
                        ✅ <strong>Análise da Matriz de Espalhamento concluída!</strong>
                        <br>• Parâmetros S calculados com base na teoria de guias cilíndricos
                        <br>• Frequências de corte determinadas pelos zeros das funções de Bessel
                        <br>• Carta de Smith mostra comportamento da impedância vs frequência
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.markdown(f"""
                    <div class="custom-warning">
                        ⚠️ <strong>Erro na análise:</strong> {str(e)}
                        <br>Verifique se todos os módulos estão instalados corretamente.
                    </div>
                    """, unsafe_allow_html=True)
                    st.exception(e)

    with tab4:
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