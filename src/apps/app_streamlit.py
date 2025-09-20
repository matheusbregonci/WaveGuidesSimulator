import streamlit as st
from TEmn_model import Modo_TEmn
from Cilindrico_model import Modo_Cilindrico
# from streamlit_ace import st_ace
# from scattering_model import calcula_coeficientes_S, calcula_campo_TE10
import plotly.graph_objects as go
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64


@st.cache_resource
def get_state():
    return {}
state = get_state()


def guia_retangular():
    st.title("Modelo de simulação para campos em Guias Retangulares")

    # Criar abas
    tab1, tab2, tab3, tab4 = st.tabs(["Parâmetros", "Campo 3D", "Coeficientes S", "Visualização 2D"])

    with tab1:
        st.write("Parâmetros da guia retangular")

        # Biblioteca de dielétricos pré-configurados (mesmo da cilíndrica)
        st.subheader("📋 Dielétricos Pré-configurados")
        dieletricos = {
            "Personalizado": {"permissividade": 1.0, "permeabilidade": 1.0},
            "Ar": {"permissividade": 1.0, "permeabilidade": 1.0},
            "PTFE - Politetrafluoroetileno": {"permissividade": 2.25, "permeabilidade": 1.0},
            "Teflon": {"permissividade": 2.08, "permeabilidade": 1.0},
            "Porcelana": {"permissividade": 5.04, "permeabilidade": 1.0},
            "Nylon": {"permissividade": 2.28, "permeabilidade": 1.0}
        }

        material_selecionado = st.selectbox("Selecione o material dielétrico:", list(dieletricos.keys()))

        # Biblioteca de guias retangulares pré-configuradas
        st.subheader("📐 Guias Retangulares Pré-configuradas")
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

        guia_selecionada = st.selectbox("Selecione a guia retangular:", list(guias_retangulares.keys()))

        # Mostrar informações da guia selecionada
        if guia_selecionada != "Personalizada":
            guia_info = guias_retangulares[guia_selecionada]
            st.info(f"**{guia_info['nomenclatura']}** | **Banda:** {guia_info['banda']} | **Faixa:** {guia_info['faixa_freq']} | **Dimensões:** {guia_info['largura']}×{guia_info['altura']} mm")

        st.subheader("⚙️ Parâmetros Detalhados")

        # Parâmetros que dependem das seleções
        if guia_selecionada == "Personalizada":
            largura_guia = st.number_input("Largura da Guia (mm)", value=22.86, step=0.1)
            altura_guia = st.number_input("Altura da Guia (mm)", value=10.16, step=0.1)
        else:
            guia_info = guias_retangulares[guia_selecionada]
            largura_guia = st.number_input("Largura da Guia (mm)", value=guia_info["largura"], step=0.1)
            altura_guia = st.number_input("Altura da Guia (mm)", value=guia_info["altura"], step=0.1)

        # Slider de frequência adaptativo à guia selecionada
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
                help="Selecione qualquer frequência entre 1-50 GHz"
            )
        else:
            # Calcular valor padrão no centro da banda
            valor_central = (freq_min + freq_max) / 2
            frequencia_onda = st.slider(
                f"Frequência da Onda - Banda {guia_info['banda']} (GHz)",
                min_value=freq_min,
                max_value=freq_max,
                value=valor_central,
                step=0.1,
                help=f"Faixa operacional da {guia_info['nomenclatura']}: {guia_info['faixa_freq']}"
            )

            # Indicador visual da posição na banda
            posicao_na_banda = (frequencia_onda - freq_min) / (freq_max - freq_min) * 100
            if posicao_na_banda < 25:
                status_cor = "🔵"
                status_texto = "Início da banda"
            elif posicao_na_banda < 75:
                status_cor = "🟢"
                status_texto = "Centro da banda (ótimo)"
            else:
                status_cor = "🟡"
                status_texto = "Final da banda"

            st.caption(f"{status_cor} **{frequencia_onda:.1f} GHz** - {status_texto} ({posicao_na_banda:.1f}% da faixa)")

        # Parâmetros do meio
        if material_selecionado == "Personalizado":
            permissividade_meio = st.number_input("Permissividade Relativa", value=1.0, step=0.1)
            permeabilidade_meio = st.number_input("Permeabilidade Relativa", value=1.0, step=0.1)
        else:
            material_props = dieletricos[material_selecionado]
            permissividade_meio = st.number_input("Permissividade Relativa", value=material_props["permissividade"], step=0.01)
            permeabilidade_meio = st.number_input("Permeabilidade Relativa", value=material_props["permeabilidade"], step=0.01)

        # Escolha do plano
        plano_opcoes = ['xy', 'xz', 'yz']
        plano = st.selectbox("Plano", plano_opcoes)

        campo = st.selectbox("Campo", ["eletrico", "magnetico"])
        componente = st.selectbox("Componente", ['x', 'y', 'z'])

        # Resumo dos parâmetros selecionados
        st.subheader("📊 Resumo da Configuração")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Material Dielétrico:**")
            st.write(f"• {material_selecionado}")
            st.write(f"• εᵣ = {permissividade_meio}")
            st.write(f"• μᵣ = {permeabilidade_meio}")

        with col2:
            st.write("**Guia de Onda:**")
            if guia_selecionada != "Personalizada":
                guia_info = guias_retangulares[guia_selecionada]
                st.write(f"• {guia_info['nomenclatura']} - Banda {guia_info['banda']}")
                st.write(f"• {guia_info['faixa_freq']}")
                # Calcular posição da frequência na banda para o resumo
                posicao_resumo = (frequencia_onda - guia_info['freq_min']) / (guia_info['freq_max'] - guia_info['freq_min']) * 100
                st.write(f"• Frequência: {frequencia_onda:.1f} GHz ({posicao_resumo:.0f}% da banda)")
            else:
                st.write(f"• Configuração personalizada")
                st.write(f"• Frequência: {frequencia_onda:.1f} GHz")
            st.write(f"• Dimensões: {largura_guia}×{altura_guia} mm")
            st.write(f"• Plano: {plano.upper()}")

        state['campo_componente'] = [campo, componente]

        if st.button("Aplicar Parâmetros"):
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
            st.success("Parâmetros aplicados com sucesso!")

    with tab2:
        if st.button("Plotar Campo 3D"):
            if 'TEmn' not in state:
                st.warning("Por favor, aplique os parâmetros primeiro na aba Parâmetros")
            else:
                TEmn = state['TEmn']
                campo, componente = state['campo_componente']

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

                # Criar o gráfico 3D interativo com Plotly
                fig = go.Figure(data=[go.Surface(z=imagem, x=TEmn.x[:, 0], y=TEmn.y[0, :])])
                fig.update_layout(
                    title=f"Campo {campo} na componente {componente}",
                    scene=dict(
                        xaxis_title="Eixo X",
                        yaxis_title="Eixo Y",
                        zaxis_title="Intensidade"
                    )
                )

                # Exibir o gráfico no Streamlit
                st.plotly_chart(fig, use_container_width=False)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Plotar intensidade do campo 2D"):
                if 'TEmn' not in state:
                    st.warning("Por favor, aplique os parâmetros primeiro na aba Parâmetros")
                else:
                    # Obter os dados do campo
                    TEmn = state['TEmn']
                    campo, componente = state['campo_componente']

                    # Plotar o campo 3D usando TEmn.plot3DField()
                    fig = TEmn.plot3DField(campo=campo, componente=componente)

                    # Exibir o gráfico no Streamlit
                    st.pyplot(fig)
                    plt.close(fig)
        
        with col2:
            if st.button("Plotar Campo Vetorial"):
                if 'TEmn' not in state:
                    st.warning("Por favor, aplique os parâmetros primeiro na aba Parâmetros")
                else:
                    # Obter os dados do campo
                    TEmn = state['TEmn']
                    campo, componente = state['campo_componente']

                    # Plotar o campo vetorial
                    fig = TEmn.plota_campo_vetorial(campo)

                    # Exibir o gráfico no Streamlit
                    st.pyplot(fig)
                    plt.close(fig)


def guia_cilindrica():
    st.title("Modelo de simulação para campos em Guias Cilíndricas")
    st.write("Esta página está em desenvolvimento.")
    tab1, tab2, tab3, tab4 = st.tabs(["Parâmetros", "Campo 3D", "Coeficientes S", "Visualização 2D"])
    with tab1:
        st.write("Parâmetros da guia cilíndrica")

        # Biblioteca de dielétricos pré-configurados
        st.subheader("📋 Dielétricos Pré-configurados")
        dieletricos = {
            "Personalizado": {"permissividade": 1.0, "permeabilidade": 1.0},
            "Ar": {"permissividade": 1.0, "permeabilidade": 1.0},
            "PTFE - Politetrafluoroetileno": {"permissividade": 2.25, "permeabilidade": 1.0},
            "Teflon": {"permissividade": 2.08, "permeabilidade": 1.0},
            "Porcelana": {"permissividade": 5.04, "permeabilidade": 1.0},
            "Nylon": {"permissividade": 2.28, "permeabilidade": 1.0}
        }

        material_selecionado = st.selectbox("Selecione o material dielétrico:", list(dieletricos.keys()))

        # Biblioteca de guias circulares pré-configuradas
        st.subheader("📐 Guias Circulares Pré-configuradas")
        guias_circulares = {
            "Personalizada": {"banda": "Custom", "faixa_freq": "Custom", "raio": 23.0, "freq_min": 1.0, "freq_max": 50.0},
            "Guia 1 - Banda X": {"banda": "X", "faixa_freq": "8.5 - 11.6 GHz", "raio": 23.83, "freq_min": 8.5, "freq_max": 11.6},
            "Guia 2 - Banda Ku": {"banda": "Ku", "faixa_freq": "13.4 - 18.0 GHz", "raio": 15.08, "freq_min": 13.4, "freq_max": 18.0},
            "Guia 3 - Banda K": {"banda": "K", "faixa_freq": "20.0 - 24.5 GHz", "raio": 10.06, "freq_min": 20.0, "freq_max": 24.5},
            "Guia 4 - Banda Ka": {"banda": "Ka", "faixa_freq": "33.0 - 38.5 GHz", "raio": 6.35, "freq_min": 33.0, "freq_max": 38.5},
            "Guia 5 - Banda Q": {"banda": "Q", "faixa_freq": "38.5 - 43.0 GHz", "raio": 5.56, "freq_min": 38.5, "freq_max": 43.0}
        }

        guia_selecionada = st.selectbox("Selecione a guia circular:", list(guias_circulares.keys()))

        # Mostrar informações da guia selecionada
        if guia_selecionada != "Personalizada":
            guia_info = guias_circulares[guia_selecionada]
            st.info(f"**Banda:** {guia_info['banda']} | **Faixa de Frequência:** {guia_info['faixa_freq']} | **Raio:** {guia_info['raio']} mm")

        st.subheader("⚙️ Parâmetros Detalhados")

        # Parâmetros que dependem das seleções
        if guia_selecionada == "Personalizada":
            raio = st.number_input("Raio da Guia (mm)", value=23.0, step=0.1)
        else:
            raio = st.number_input("Raio da Guia (mm)", value=guias_circulares[guia_selecionada]["raio"], step=0.1)

        # Slider de frequência adaptativo à guia selecionada
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
                help="Selecione qualquer frequência entre 1-50 GHz"
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

            st.caption(f"{status_cor} **{frequencia:.1f} GHz** - {status_texto} ({posicao_na_banda:.1f}% da faixa)")

        if material_selecionado == "Personalizado":
            permissividade = st.number_input("Permissividade Relativa", value=1.0, step=0.1)
            permeabilidade = st.number_input("Permeabilidade Relativa", value=1.0, step=0.1)
        else:
            material_props = dieletricos[material_selecionado]
            permissividade = st.number_input("Permissividade Relativa", value=material_props["permissividade"], step=0.01)
            permeabilidade = st.number_input("Permeabilidade Relativa", value=material_props["permeabilidade"], step=0.01)

        comprimento = st.number_input("Comprimento da Guia (mm)", value=100.0, step=0.1)
        modo_m = st.number_input("Modo m (1, 2 ou 3)", value=1, step=1, min_value=0,max_value=2)
        modo_n = st.number_input("Modo n (0, 1, ou 2)", value=1, step=1, min_value=0,max_value=2)

        # Resumo dos parâmetros selecionados
        st.subheader("📊 Resumo da Configuração")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Material Dielétrico:**")
            st.write(f"• {material_selecionado}")
            st.write(f"• εᵣ = {permissividade}")
            st.write(f"• μᵣ = {permeabilidade}")

        with col2:
            st.write("**Guia de Onda:**")
            if guia_selecionada != "Personalizada":
                guia_info = guias_circulares[guia_selecionada]
                st.write(f"• {guia_selecionada}")
                st.write(f"• Banda {guia_info['banda']} ({guia_info['faixa_freq']})")
                # Calcular posição da frequência na banda para o resumo
                posicao_resumo = (frequencia - guia_info['freq_min']) / (guia_info['freq_max'] - guia_info['freq_min']) * 100
                st.write(f"• Frequência: {frequencia:.1f} GHz ({posicao_resumo:.0f}% da banda)")
            else:
                st.write(f"• Configuração personalizada")
                st.write(f"• Frequência: {frequencia:.1f} GHz")
            st.write(f"• Raio: {raio} mm")
            st.write(f"• Modo: TE₍{modo_n}{modo_m}₎ / TM₍{modo_n}{modo_m}₎")

        if st.button("Aplicar Parâmetros"):
            cilindrico = Modo_Cilindrico(raio=raio/1000,  # mm -> m
                                        frequencia=frequencia * 1e9,  # Hz
                                        permissividade=permissividade,
                                        permeabilidade=permeabilidade, 
                                        m=modo_m,
                                        n=modo_n,
                                        z=0.25)

            X, Y, Rho, Phi = cilindrico.criar_meshgrid_cartesiano()
            state['cilindro'] = cilindrico
            state['X'] = X
            state['Y'] = Y
            state['Rho'] = Rho
            state['Phi'] = Phi
            state['modo_m'] = modo_m
            state['modo_n'] = modo_n
            st.success("Parâmetros aplicados com sucesso!")

    with tab2:
        st.write("Campo 3D da guia cilíndrica")
        comprimento = st.number_input("Comprimento para visualização 3D (m)", value=0.01, step=0.001)
        if st.button("Plotar Campo 3D"):
            if 'cilindro' not in state:
                st.warning("Por favor, aplique os parâmetros primeiro na aba Parâmetros")
            else:
                cilindrico = state['cilindro']
                X, Y, Rho, Phi = state['X'], state['Y'], state['Rho'], state['Phi']
                
                
                cilindrico.pontos_por_dimensao = 8
                cilindrico.num_planos = 13
                X_3d, Y_3d, Rho_3d, Phi_3d, Z_3d = cilindrico.criar_meshgrid_cartesiano_com_z(comprimento=comprimento)
                
                fig = cilindrico.plot_vetores_3D(X_3d, Y_3d, Rho_3d, Phi_3d, Z_3d, 
                                                transversal='TE', campo='magnetico', 
                                                comprimento=comprimento)
                st.plotly_chart(fig, use_container_width=True)

        # Adicione o código para plotar o campo 3D aqui
    with tab3:
        st.write("Coeficientes S da guia cilíndrica")
        st.write("As equações abaixo descrevem os campos elétricos e magnéticos na guia cilíndrica:")
        # Adicionar as equações usando st.latex
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
        # code = st_ace(
        #     value="""print("Hello!")""",
        #     language='python',
        #     theme='tomorrow_night',
        #     tab_size= 4,
        #     font_size=16, height=200
        # )
        code = """print("Hello!")"""

        "*****"
        "## Output"

        html = f"""
        <html>
        <head>
            <link rel="stylesheet" href="https://pyscript.net/latest/pyscript.css" />
            <script defer src="https://pyscript.net/latest/pyscript.js"></script>
        </head>
        <body>
            <py-script>{code}</py-script>
        </body>
        </html>
        """

    with tab4:
        st.write("Visualização 2D da guia cilíndrica")
        

        st.subheader("Animação de Fase φ")
        transversal_anim = st.selectbox("Modo (Animação)", ["TE", "TM"], key="transversal_anim")
        campo_anim = st.selectbox("Campo (Animação)", ["eletrico", "magnetico"], key="campo_anim")
        
        frames = st.slider("Número de Frames", min_value=10, max_value=100, value=60)
        interval = st.slider("Intervalo (ms)", min_value=50, max_value=500, value=85)
        z_fixo = st.number_input("Posição Z Fixa (m)", value=0.25, step=0.01)
        
        if st.button("Gerar GIF"):
            if 'cilindro' not in state:
                st.warning("Por favor, aplique os parâmetros primeiro na aba Parâmetros")
            else:
                cilindrico = state['cilindro']
                X, Y, Rho, Phi, modo_m, modo_n = state['X'], state['Y'], state['Rho'], state['Phi'], state['modo_m'], state['modo_n']
                
                nome_arquivo = f"{transversal_anim}_{campo_anim}{modo_n}{modo_m}_fase_phi.gif"
                
                with st.spinner("Gerando GIF... Isso pode demorar um pouco."):
                    try:
                        # Criar a animação manualmente para controle total
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

                            # Normalizar vetores para tamanho consistente
                            magnitude = np.sqrt(e_x**2 + e_y**2)
                            max_magnitude = np.max(magnitude) if np.max(magnitude) > 0 else 1e-10

                            # Normalizar direção dos vetores (mantém direção, padroniza tamanho)
                            magnitude_nonzero = np.where(magnitude > 1e-12, magnitude, 1e-12)
                            e_x_normalized = e_x / magnitude_nonzero
                            e_y_normalized = e_y / magnitude_nonzero

                            # Escalar para tamanho visual agradável
                            scale_factor = cilindrico.raio * 0.1  # 10% do raio para tamanho consistente
                            e_x_display = e_x_normalized * scale_factor
                            e_y_display = e_y_normalized * scale_factor
                            
                            # Plotar com vetores normalizados mas cores baseadas na magnitude original
                            # Criar mapeamento de cores baseado na magnitude original
                            norm = plt.Normalize(vmin=0, vmax=max_magnitude)
                            cmap = plt.cm.viridis

                            quiver = ax.quiver(X_masked, Y_masked, e_x_display, e_y_display,
                                               magnitude, cmap=cmap, norm=norm,
                                               scale=1, scale_units='xy', angles='xy',
                                               pivot='middle', alpha=0.8)

                            # Adicionar colorbar apenas no primeiro frame
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
                        
                        # Salvar como GIF usando pillow
                        from PIL import Image
                        pil_images = [Image.open(io.BytesIO(img)) for img in images]
                        pil_images[0].save(
                            nome_arquivo,
                            save_all=True,
                            append_images=pil_images[1:],
                            duration=interval,
                            loop=0
                        )
                        
                        # Exibir o GIF na página
                        with open(nome_arquivo, "rb") as file:
                            gif_data = file.read()

                        # Mostrar o GIF na página
                        st.image(gif_data, caption=f"Animação: {nome_arquivo}")

                        # Criar link para download
                        b64_gif = base64.b64encode(gif_data).decode()
                        href = f'<a href="data:image/gif;base64,{b64_gif}" download="{nome_arquivo}">📥 Baixar {nome_arquivo}</a>'
                        st.markdown(href, unsafe_allow_html=True)

                        st.success(f"GIF gerado e exibido com sucesso: {nome_arquivo}")
                        
                    except Exception as e:
                        st.error(f"Erro ao gerar GIF: {e}")

# Menu de navegação
def main():
    st.sidebar.title("Navegação")
    pagina = st.sidebar.radio("Selecione a página:", ["Guia Retangular", "Guia Cilíndrica"])

    if pagina == "Guia Retangular":
        guia_retangular()
    elif pagina == "Guia Cilíndrica":
        guia_cilindrica()


if __name__ == "__main__":
    main()