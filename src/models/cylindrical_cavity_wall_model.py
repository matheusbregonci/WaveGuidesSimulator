import sys
import os
# Adicionar o diretório models ao path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Cilindrico_model import Modo_Cilindrico
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import imageio
import tempfile
import os
import plotly.express as px
from PIL import Image
import io

class CylindricalCavityWall3D:
    def __init__(self, raio=23.0, profundidade=100.0,
                 frequencia=12*10**9, permissividade=1, permeabilidade=1,
                 resolucao=25, m=1, n=0):
        """
        Classe para visualizar campos eletromagnéticos em cavidades cilíndricas 3D.

        Parameters:
        -----------
        raio : float
            Raio do cilindro em mm
        profundidade : float
            Profundidade (comprimento) do cilindro em mm
        frequencia : float
            Frequência de operação em Hz
        permissividade : float
            Permissividade relativa do meio
        permeabilidade : float
            Permeabilidade relativa do meio
        resolucao : int
            Número de pontos para discretização das superfícies
        m : int
            Índice de modo m (azimuthal)
        n : int
            Índice de modo n (radial)
        """
        self.raio = raio  # em mm
        self.profundidade = profundidade  # em mm
        self.frequencia = frequencia
        self.permissividade = permissividade
        self.permeabilidade = permeabilidade
        self.resolucao = resolucao
        self.m = m
        self.n = n

        # Criar instância do modelo cilíndrico (convertendo mm para metros)
        self.modelo = Modo_Cilindrico(
            raio=raio/1000,  # mm -> m
            frequencia=frequencia,
            permissividade=permissividade,
            permeabilidade=permeabilidade,
            n=n,
            m=m,
            z=0
        )

    def _calcular_campo_complexo(self, rho, phi, modo='TE', campo='magnetico'):
        """
        Calcula os campos complexos (sem aplicar np.real).

        Parameters:
        -----------
        rho, phi : arrays
            Coordenadas cilíndricas
        modo : str
            'TE' ou 'TM'
        campo : str
            'magnetico' ou 'eletrico'

        Returns:
        --------
        tuple : (componente_rho_complex, componente_phi_complex, componente_z_complex)
        """
        # Evitar divisão por zero no centro (rho=0)
        epsilon = 1e-10
        rho_safe = np.where(np.abs(rho) < epsilon, epsilon, rho)

        if modo == 'TE':
            if campo == 'eletrico':
                # Recriar cálculo sem np.real()
                const_rho = -1j * self.modelo.omega_0 * self.modelo.mu / (self.modelo.k_c_val**2 * rho_safe)
                E_rho_complex = const_rho * (self.modelo.A * np.cos(self.modelo.n * phi) -
                                            self.modelo.B * np.sin(self.modelo.n * phi)) * \
                               self.modelo.jv_n(rho) * self.modelo.exp_z_val

                const_phi = 1j * self.modelo.omega_0 * self.modelo.mu / self.modelo.k_c_val
                E_phi_complex = const_phi * (self.modelo.A * np.sin(self.modelo.n * phi) +
                                            self.modelo.B * np.cos(self.modelo.n * phi)) * \
                               self.modelo.jv_n_prime(rho) * self.modelo.exp_z_val

                E_z_complex = np.zeros_like(E_rho_complex, dtype=complex)
                return E_rho_complex, E_phi_complex, E_z_complex

            else:  # magnetico
                const_rho = -1j * self.modelo.beta_val / self.modelo.k_c_val
                H_rho_complex = const_rho * (self.modelo.A * np.cos(self.modelo.n * phi) +
                                            self.modelo.B * np.sin(self.modelo.n * phi)) * \
                               self.modelo.jv_n_prime(rho) * self.modelo.exp_z_val

                const_phi = -1j * self.modelo.beta_val * self.modelo.n / (self.modelo.k_c_val**2 * rho_safe)
                H_phi_complex = const_phi * (self.modelo.A * np.cos(self.modelo.n * phi) -
                                            self.modelo.B * np.sin(self.modelo.n * phi)) * \
                               self.modelo.jv_n(rho) * self.modelo.exp_z_val

                # Para TE_H_Z - Corrigido: A*cos(nφ) + B*sin(nφ)
                H_z_complex = (self.modelo.A * np.cos(self.modelo.n * phi) +
                              self.modelo.B * np.sin(self.modelo.n * phi)) * \
                             self.modelo.jv_n(rho) * self.modelo.exp_z_val

                return H_rho_complex, H_phi_complex, H_z_complex

        else:  # TM
            if campo == 'eletrico':
                const_rho = -1j * self.modelo.beta_val / self.modelo.k_c_val_tm
                E_rho_complex = const_rho * (self.modelo.A * np.sin(self.modelo.n * phi) +
                                            self.modelo.B * np.cos(self.modelo.n * phi)) * \
                               self.modelo.jv_n_prime(rho) * self.modelo.exp_z_val

                const_phi = -1j * self.modelo.beta_val * self.modelo.n / (self.modelo.k_c_val_tm**2 * rho_safe)
                E_phi_complex = const_phi * (self.modelo.A * np.cos(self.modelo.n * phi) -
                                            self.modelo.B * np.sin(self.modelo.n * phi)) * \
                               self.modelo.jv_n(rho) * self.modelo.exp_z_val

                # Para TM_E_Z - Corrigido: A*sin(nφ) + B*cos(nφ)
                E_z_complex = (self.modelo.A * np.sin(self.modelo.n * phi) +
                              self.modelo.B * np.cos(self.modelo.n * phi)) * \
                             self.modelo.jv_n(rho) * self.modelo.exp_z_val

                return E_rho_complex, E_phi_complex, E_z_complex

            else:  # magnetico
                const_rho = 1j * self.modelo.omega_0 * self.modelo.epsilon * self.modelo.n / \
                           ((self.modelo.k_c_val_tm**2) * rho_safe)
                H_rho_complex = const_rho * (self.modelo.A * np.cos(self.modelo.n * phi) -
                                            self.modelo.B * np.sin(self.modelo.n * phi)) * \
                               self.modelo.jv_n(rho) * self.modelo.exp_z_val

                const_phi = -1j * self.modelo.omega_0 * self.modelo.epsilon / self.modelo.k_c_val_tm
                H_phi_complex = const_phi * (self.modelo.A * np.sin(self.modelo.n * phi) +
                                            self.modelo.B * np.cos(self.modelo.n * phi)) * \
                               self.modelo.jv_n_prime(rho) * self.modelo.exp_z_val

                H_z_complex = np.zeros_like(H_rho_complex, dtype=complex)
                return H_rho_complex, H_phi_complex, H_z_complex

    def _calcular_campo_componentes(self, rho, phi, z, modo='TE', campo='magnetico', fase_temporal=0):
        """
        Calcula as componentes do campo em um ponto específico com variação temporal.

        Parameters:
        -----------
        rho : float or array
            Coordenada radial em metros
        phi : float or array
            Coordenada angular em radianos
        z : float
            Coordenada z em metros
        modo : str
            'TE' ou 'TM'
        campo : str
            'magnetico' ou 'eletrico'
        fase_temporal : float
            Fase temporal em radianos (ωt)

        Returns:
        --------
        tuple : (componente_rho, componente_phi, componente_z)
        """
        # Atualizar posição z no modelo
        self.modelo.update_z(z)

        # Calcular campos complexos
        comp_rho_complex, comp_phi_complex, comp_z_complex = self._calcular_campo_complexo(
            rho, phi, modo, campo
        )

        # Aplicar fase temporal como fator multiplicativo complexo
        fase_complexa = np.exp(1j * fase_temporal)

        # Aplicar fase temporal e tomar parte real
        comp_rho = np.real(comp_rho_complex * fase_complexa)
        comp_phi = np.real(comp_phi_complex * fase_complexa)
        comp_z = np.real(comp_z_complex * fase_complexa)

        return comp_rho, comp_phi, comp_z

    def _calcular_intensidade_superficie(self, rho, phi, z, modo='TE', campo='magnetico',
                                        tipo_intensidade='total', direcao_vetor='z', fase_temporal=0):
        """
        Calcula a intensidade do campo em uma superfície com fase temporal.

        Parameters:
        -----------
        rho, phi, z : arrays
            Coordenadas cilíndricas
        modo : str
            'TE' ou 'TM'
        campo : str
            'magnetico' ou 'eletrico'
        tipo_intensidade : str
            'total', 'perpendicular', ou 'direcional'
        direcao_vetor : str
            'rho', 'phi', ou 'z' (para tipo_intensidade='direcional')
        fase_temporal : float
            Fase temporal em radianos (ωt)

        Returns:
        --------
        array : Intensidade do campo na superfície
        """
        # Calcular componentes do campo com fase temporal
        comp_rho, comp_phi, comp_z = self._calcular_campo_componentes(
            rho, phi, z, modo, campo, fase_temporal
        )

        if tipo_intensidade == 'total':
            # Magnitude total do vetor campo
            return np.sqrt(comp_rho**2 + comp_phi**2 + comp_z**2)

        elif tipo_intensidade == 'direcional':
            # Componente específica
            if direcao_vetor == 'rho':
                return np.abs(comp_rho)
            elif direcao_vetor == 'phi':
                return np.abs(comp_phi)
            elif direcao_vetor == 'z':
                return np.abs(comp_z)

        elif tipo_intensidade == 'perpendicular':
            # Componente perpendicular à superfície (depende da superfície)
            # Para superfície lateral cilíndrica: componente radial
            # Para tampas: componente z
            # Esta lógica será tratada em cada método específico de superfície
            return np.abs(comp_rho)

        return np.sqrt(comp_rho**2 + comp_phi**2 + comp_z**2)

    def _calcular_intensidades_com_fase(self, modo='TE', campo='magnetico',
                                       tipo_intensidade='total', direcao_vetor='z', fase_temporal=0):
        """
        Calcula intensidades em todas as superfícies do cilindro para uma fase temporal específica.

        Parameters:
        -----------
        modo : str
            'TE' ou 'TM'
        campo : str
            'magnetico' ou 'eletrico'
        tipo_intensidade : str
            'total', 'perpendicular', ou 'direcional'
        direcao_vetor : str
            'rho', 'phi', ou 'z'
        fase_temporal : float
            Fase temporal em radianos

        Returns:
        --------
        dict : Dicionário com intensidades em cada superfície
        """
        intensidades = {}

        # Converter dimensões de mm para m
        raio_m = self.raio / 1000
        prof_m = self.profundidade / 1000

        # 1. Tampa frontal (disco circular em z=0)
        r_tampa = np.linspace(0, raio_m, self.resolucao)
        phi_tampa = np.linspace(0, 2*np.pi, self.resolucao)
        R_tampa, Phi_tampa = np.meshgrid(r_tampa, phi_tampa, indexing='ij')
        Z_tampa = 0

        # Calcular campos com fase temporal aplicada corretamente
        comp_rho_f, comp_phi_f, comp_z_f = self._calcular_campo_componentes(
            R_tampa, Phi_tampa, Z_tampa, modo, campo, fase_temporal
        )

        if tipo_intensidade == 'perpendicular':
            intensidades['tampa_frontal'] = np.abs(comp_z_f)
        elif tipo_intensidade == 'total':
            intensidades['tampa_frontal'] = np.sqrt(comp_rho_f**2 + comp_phi_f**2 + comp_z_f**2)
        elif tipo_intensidade == 'direcional':
            if direcao_vetor == 'rho':
                intensidades['tampa_frontal'] = np.abs(comp_rho_f)
            elif direcao_vetor == 'phi':
                intensidades['tampa_frontal'] = np.abs(comp_phi_f)
            else:
                intensidades['tampa_frontal'] = np.abs(comp_z_f)

        # 2. Tampa traseira (disco circular em z=profundidade)
        Z_tampa_t = prof_m
        comp_rho_t, comp_phi_t, comp_z_t = self._calcular_campo_componentes(
            R_tampa, Phi_tampa, Z_tampa_t, modo, campo, fase_temporal
        )

        if tipo_intensidade == 'perpendicular':
            intensidades['tampa_traseira'] = np.abs(comp_z_t)
        elif tipo_intensidade == 'total':
            intensidades['tampa_traseira'] = np.sqrt(comp_rho_t**2 + comp_phi_t**2 + comp_z_t**2)
        elif tipo_intensidade == 'direcional':
            if direcao_vetor == 'rho':
                intensidades['tampa_traseira'] = np.abs(comp_rho_t)
            elif direcao_vetor == 'phi':
                intensidades['tampa_traseira'] = np.abs(comp_phi_t)
            else:
                intensidades['tampa_traseira'] = np.abs(comp_z_t)

        # 3. Superfície lateral cilíndrica (r=raio, internamente)
        # Leitura interna: 98% do raio (evita problemas na borda exata)
        raio_interno = raio_m * 0.95
        phi_lateral = np.linspace(0, 2*np.pi, self.resolucao)
        z_lateral = np.linspace(0, prof_m, self.resolucao)
        Phi_lateral, Z_lateral = np.meshgrid(phi_lateral, z_lateral, indexing='ij')
        R_lateral = np.full_like(Phi_lateral, raio_interno)

        # Para cada z, calcular o campo com fase temporal correta
        intensidade_lateral = np.zeros_like(Phi_lateral)
        for i in range(self.resolucao):
            z_val = Z_lateral[0, i]
            comp_rho_l, comp_phi_l, comp_z_l = self._calcular_campo_componentes(
                R_lateral[:, i], Phi_lateral[:, i], z_val, modo, campo, fase_temporal
            )

            if tipo_intensidade == 'perpendicular':
                intensidade_lateral[:, i] = np.abs(comp_rho_l)
            elif tipo_intensidade == 'total':
                intensidade_lateral[:, i] = np.sqrt(comp_rho_l**2 + comp_phi_l**2 + comp_z_l**2)
            elif tipo_intensidade == 'direcional':
                if direcao_vetor == 'rho':
                    intensidade_lateral[:, i] = np.abs(comp_rho_l)
                elif direcao_vetor == 'phi':
                    intensidade_lateral[:, i] = np.abs(comp_phi_l)
                else:
                    intensidade_lateral[:, i] = np.abs(comp_z_l)

        intensidades['lateral'] = intensidade_lateral

        return intensidades

    def animar_cavidade_plotly(self, modo='TE', campo='magnetico', tipo_intensidade='total',
                              direcao_vetor='z', num_frames=60, duracao_frame=100):
        """
        Cria uma animação 3D interativa da cavidade cilíndrica usando Plotly.

        Parameters:
        -----------
        modo : str
            'TE' ou 'TM'
        campo : str
            'magnetico' ou 'eletrico'
        tipo_intensidade : str
            'total', 'perpendicular', ou 'direcional'
        direcao_vetor : str
            'rho', 'phi', ou 'z'
        num_frames : int
            Número de frames na animação
        duracao_frame : int
            Duração de cada frame em ms

        Returns:
        --------
        fig : plotly.graph_objects.Figure
            Figura Plotly com animação 3D
        """
        # Converter dimensões
        raio_m = self.raio / 1000
        prof_m = self.profundidade / 1000

        # Calcular intensidade máxima para normalização
        fases_teste = np.linspace(0, 2*np.pi, 10)
        max_intensidade_global = 0

        for fase in fases_teste:
            intensidades = self._calcular_intensidades_com_fase(modo, campo, tipo_intensidade,
                                                              direcao_vetor, fase)
            max_local = max([np.max(intensidade) for intensidade in intensidades.values()])
            max_intensidade_global = max(max_intensidade_global, max_local)

        # Preparar dados para animação
        frames_data = []

        for frame in range(num_frames):
            fase_temporal = 2 * np.pi * frame / num_frames
            intensidades = self._calcular_intensidades_com_fase(modo, campo, tipo_intensidade,
                                                              direcao_vetor, fase_temporal)

            frame_surfaces = []

            # Tampa frontal (z=0)
            r_tampa = np.linspace(0, raio_m, self.resolucao)
            phi_tampa = np.linspace(0, 2*np.pi, self.resolucao)
            R_tampa, Phi_tampa = np.meshgrid(r_tampa, phi_tampa, indexing='ij')

            # Converter para coordenadas cartesianas
            X_frontal = R_tampa * np.cos(Phi_tampa)
            Y_frontal = R_tampa * np.sin(Phi_tampa)
            Z_frontal = np.zeros_like(X_frontal)

            frame_surfaces.append(
                go.Surface(
                    x=X_frontal, y=Y_frontal, z=Z_frontal,
                    surfacecolor=intensidades['tampa_frontal'],
                    cmin=0, cmax=max_intensidade_global,
                    colorscale='Viridis',
                    showscale=False,
                    name='Tampa Frontal'
                )
            )

            # Tampa traseira (z=profundidade)
            Z_traseira = np.full_like(X_frontal, prof_m)

            frame_surfaces.append(
                go.Surface(
                    x=X_frontal, y=Y_frontal, z=Z_traseira,
                    surfacecolor=intensidades['tampa_traseira'],
                    cmin=0, cmax=max_intensidade_global,
                    colorscale='Viridis',
                    showscale=False,
                    name='Tampa Traseira'
                )
            )

            # Superfície lateral cilíndrica (visualizada no raio externo, dados do raio interno)
            phi_lateral = np.linspace(0, 2*np.pi, self.resolucao)
            z_lateral = np.linspace(0, prof_m, self.resolucao)
            Phi_lateral, Z_lateral = np.meshgrid(phi_lateral, z_lateral, indexing='ij')

            # Posição visual no raio externo
            X_lateral = raio_m * np.cos(Phi_lateral)
            Y_lateral = raio_m * np.sin(Phi_lateral)

            frame_surfaces.append(
                go.Surface(
                    x=X_lateral, y=Y_lateral, z=Z_lateral,
                    surfacecolor=intensidades['lateral'],
                    cmin=0, cmax=max_intensidade_global,
                    colorscale='Viridis',
                    showscale=frame == 0,
                    colorbar=dict(title=f"Intensidade {campo.capitalize()}", len=0.7) if frame == 0 else None,
                    name='Superfície Lateral'
                )
            )

            frames_data.append(frame_surfaces)

        # Criar figura com primeiro frame
        fig = go.Figure(data=frames_data[0])

        # Configurar layout
        fig.update_layout(
            title=f'Animação 3D Cilíndrica - Campo {campo.capitalize()} ({tipo_intensidade}) - Modo {modo}{self.n}{self.m}',
            paper_bgcolor='rgba(0,0,0,0)',  # Fundo transparente
            plot_bgcolor='rgba(0,0,0,0)',   # Fundo transparente
            autosize=True,  # Ocupa toda largura disponível
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8),  # Vista melhorada para ver as tampas
                    up=dict(x=0, y=0, z=1),  # Z aponta para cima
                    center=dict(x=0, y=0, z=0)
                ),
                aspectmode='data',
                bgcolor='rgba(0,0,0,0)',  # Fundo da cena 3D transparente
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False)
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            updatemenus=[{
                'type': 'buttons',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': duracao_frame, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }]
        )

        # Adicionar frames à animação
        frames = []
        for i, frame_data in enumerate(frames_data):
            frames.append(go.Frame(data=frame_data, name=f'frame{i}'))

        fig.frames = frames

        # Adicionar slider
        fig.update_layout(
            sliders=[{
                'steps': [
                    {
                        'args': [[f.name], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        'label': f'Frame {i}',
                        'method': 'animate'
                    }
                    for i, f in enumerate(fig.frames)
                ],
                'active': 0,
                'currentvalue': {'prefix': 'Frame: '},
                'len': 0.9,
                'x': 0.1,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }]
        )

        return fig

    def salvar_animacao_gif(self, nome_arquivo='animacao_cilindrica.gif',
                           modo='TE', campo='magnetico', tipo_intensidade='total',
                           direcao_vetor='z', num_frames=30, duracao_frame=100,
                           fps=10, width=800, height=600):
        """
        Salva a animação como arquivo GIF.

        Parameters:
        -----------
        nome_arquivo : str
            Nome do arquivo GIF a ser salvo
        modo : str
            'TE' ou 'TM'
        campo : str
            'magnetico' ou 'eletrico'
        tipo_intensidade : str
            'total', 'perpendicular', ou 'direcional'
        direcao_vetor : str
            'rho', 'phi', ou 'z'
        num_frames : int
            Número de frames na animação
        duracao_frame : int
            Duração de cada frame em ms
        fps : int
            Frames por segundo no GIF
        width : int
            Largura em pixels
        height : int
            Altura em pixels

        Returns:
        --------
        str : Caminho do arquivo salvo
        """
        try:
            import kaleido
        except ImportError:
            raise ImportError("kaleido não está instalado. Instale com: pip install kaleido")

        # Converter dimensões
        raio_m = self.raio / 1000
        prof_m = self.profundidade / 1000

        # Calcular intensidade máxima para normalização
        fases_teste = np.linspace(0, 2*np.pi, 10)
        max_intensidade_global = 0

        for fase in fases_teste:
            intensidades = self._calcular_intensidades_com_fase(modo, campo, tipo_intensidade,
                                                              direcao_vetor, fase)
            max_local = max([np.max(intensidade) for intensidade in intensidades.values()])
            max_intensidade_global = max(max_intensidade_global, max_local)

        # Criar diretório temporário para frames
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_files = []

            for frame_idx in range(num_frames):
                fase_temporal = 2 * np.pi * frame_idx / num_frames
                intensidades = self._calcular_intensidades_com_fase(modo, campo, tipo_intensidade,
                                                                  direcao_vetor, fase_temporal)

                # Criar superfícies para este frame
                frame_surfaces = []

                # Tampa frontal
                r_tampa = np.linspace(0, raio_m, self.resolucao)
                phi_tampa = np.linspace(0, 2*np.pi, self.resolucao)
                R_tampa, Phi_tampa = np.meshgrid(r_tampa, phi_tampa, indexing='ij')
                X_frontal = R_tampa * np.cos(Phi_tampa)
                Y_frontal = R_tampa * np.sin(Phi_tampa)
                Z_frontal = np.zeros_like(X_frontal)

                frame_surfaces.append(
                    go.Surface(
                        x=X_frontal, y=Y_frontal, z=Z_frontal,
                        surfacecolor=intensidades['tampa_frontal'],
                        cmin=0, cmax=max_intensidade_global,
                        colorscale='Viridis',
                        showscale=True if frame_idx == 0 else False,
                        colorbar=dict(title=f"Intensidade {campo.capitalize()}", len=0.7) if frame_idx == 0 else None,
                        name='Tampa Frontal'
                    )
                )

                # Tampa traseira
                Z_traseira = np.full_like(X_frontal, prof_m)
                frame_surfaces.append(
                    go.Surface(
                        x=X_frontal, y=Y_frontal, z=Z_traseira,
                        surfacecolor=intensidades['tampa_traseira'],
                        cmin=0, cmax=max_intensidade_global,
                        colorscale='Viridis',
                        showscale=False,
                        name='Tampa Traseira'
                    )
                )

                # Superfície lateral
                phi_lateral = np.linspace(0, 2*np.pi, self.resolucao)
                z_lateral = np.linspace(0, prof_m, self.resolucao)
                Phi_lateral, Z_lateral = np.meshgrid(phi_lateral, z_lateral, indexing='ij')
                X_lateral = raio_m * np.cos(Phi_lateral)
                Y_lateral = raio_m * np.sin(Phi_lateral)

                frame_surfaces.append(
                    go.Surface(
                        x=X_lateral, y=Y_lateral, z=Z_lateral,
                        surfacecolor=intensidades['lateral'],
                        cmin=0, cmax=max_intensidade_global,
                        colorscale='Viridis',
                        showscale=False,
                        name='Superfície Lateral'
                    )
                )

                # Criar figura para este frame
                fig = go.Figure(data=frame_surfaces)
                fig.update_layout(
                    title=f'Frame {frame_idx+1}/{num_frames} - Campo {campo.capitalize()} ({tipo_intensidade})',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    scene=dict(
                        xaxis_title='X (m)',
                        yaxis_title='Y (m)',
                        zaxis_title='Z (m)',
                        camera=dict(
                            eye=dict(x=1.8, y=1.2, z=0.3),
                            up=dict(x=0, y=1, z=0)
                        ),
                        aspectmode='data',
                        bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showbackground=False),
                        yaxis=dict(showbackground=False),
                        zaxis=dict(showbackground=False)
                    ),
                    margin=dict(l=0, r=0, t=40, b=0),
                    width=width,
                    height=height
                )

                # Salvar frame como imagem
                frame_file = os.path.join(tmpdir, f'frame_{frame_idx:03d}.png')
                fig.write_image(frame_file)
                frame_files.append(frame_file)
                print(f"Frame {frame_idx+1}/{num_frames} salvo")

            # Criar GIF a partir dos frames
            print("Criando GIF...")
            images = [imageio.imread(f) for f in frame_files]
            imageio.mimsave(nome_arquivo, images, fps=fps)

            print(f"✅ GIF salvo como: {nome_arquivo}")
            return nome_arquivo

    def criar_gif_plotly(self, nome_arquivo='animacao_cilindrica.gif',
                        modo='TE', campo='magnetico', tipo_intensidade='total',
                        direcao_vetor='z', num_frames=30, duracao_frame=200,
                        width=800, height=600, fps=5):
        """
        Cria um GIF a partir da animação Plotly da cavidade cilíndrica.

        Parameters:
        -----------
        nome_arquivo : str
            Nome do arquivo GIF a ser salvo
        modo : str
            'TE' ou 'TM'
        campo : str
            Tipo de campo ('magnetico' ou 'eletrico')
        tipo_intensidade : str
            Tipo de intensidade ('total', 'perpendicular', ou 'direcional')
        direcao_vetor : str
            Direção do vetor ('rho', 'phi', 'z')
        num_frames : int
            Número de frames na animação
        duracao_frame : int
            Duração de cada frame em ms (não usado para GIF, use fps)
        width : int
            Largura da imagem em pixels
        height : int
            Altura da imagem em pixels
        fps : int
            Frames por segundo do GIF

        Returns:
        --------
        str
            Caminho do arquivo GIF criado
        """
        try:
            import kaleido
        except ImportError:
            raise ImportError("kaleido não está instalado. Instale com: pip install kaleido")

        # Converter dimensões
        raio_m = self.raio / 1000
        prof_m = self.profundidade / 1000

        # Calcular intensidade máxima para normalização
        fases_teste = np.linspace(0, 2*np.pi, 10)
        max_intensidade_global = 0

        for fase in fases_teste:
            intensidades = self._calcular_intensidades_com_fase(modo, campo, tipo_intensidade,
                                                              direcao_vetor, fase)
            max_local = max([np.max(intensidade) for intensidade in intensidades.values()])
            max_intensidade_global = max(max_intensidade_global, max_local)

        # Lista para armazenar as imagens
        images = []

        print(f"Gerando {num_frames} frames para o GIF...")

        for frame in range(num_frames):
            fase_temporal = 2 * np.pi * frame / num_frames
            intensidades = self._calcular_intensidades_com_fase(modo, campo, tipo_intensidade,
                                                              direcao_vetor, fase_temporal)

            # Criar figura estática para este frame
            fig = go.Figure()

            # Tampa frontal (Z=0)
            r_tampa = np.linspace(0, raio_m, self.resolucao)
            phi_tampa = np.linspace(0, 2*np.pi, self.resolucao)
            R_tampa, Phi_tampa = np.meshgrid(r_tampa, phi_tampa, indexing='ij')
            X_frontal = R_tampa * np.cos(Phi_tampa)
            Y_frontal = R_tampa * np.sin(Phi_tampa)
            Z_frontal = np.zeros_like(X_frontal)

            fig.add_trace(go.Surface(
                x=X_frontal, y=Y_frontal, z=Z_frontal,
                surfacecolor=intensidades['tampa_frontal'],
                cmin=0, cmax=max_intensidade_global,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=f"Intensidade {campo.capitalize()}", len=0.7),
                name='Tampa Frontal'
            ))

            # Tampa traseira (Z=prof_m)
            Z_traseira = np.full_like(X_frontal, prof_m)
            fig.add_trace(go.Surface(
                x=X_frontal, y=Y_frontal, z=Z_traseira,
                surfacecolor=intensidades['tampa_traseira'],
                cmin=0, cmax=max_intensidade_global,
                colorscale='Viridis',
                showscale=False,
                name='Tampa Traseira'
            ))

            # Superfície lateral
            phi_lateral = np.linspace(0, 2*np.pi, self.resolucao)
            z_lateral = np.linspace(0, prof_m, self.resolucao)
            Phi_lateral, Z_lateral = np.meshgrid(phi_lateral, z_lateral, indexing='ij')
            X_lateral = raio_m * np.cos(Phi_lateral)
            Y_lateral = raio_m * np.sin(Phi_lateral)

            fig.add_trace(go.Surface(
                x=X_lateral, y=Y_lateral, z=Z_lateral,
                surfacecolor=intensidades['lateral'],
                cmin=0, cmax=max_intensidade_global,
                colorscale='Viridis',
                showscale=False,
                name='Superfície Lateral'
            ))

            # Configurar layout
            fig.update_layout(
                title=dict(
                    text=f'Campo {campo.capitalize()} ({tipo_intensidade}) - Modo {modo}{self.m}{self.n}<br>'
                         f'Fase: {fase_temporal:.2f} rad, Frame: {frame+1}/{num_frames}',
                    x=0.5,
                    y=0.95,
                    font=dict(size=14)
                ),
                scene=dict(
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)',
                    zaxis_title='Z (m)',
                    camera=dict(
                        eye=dict(x=1.3, y=-0.5, z=2),  # Vista mais elevada e balanceada
                        up=dict(x=0, y=-1, z=0),  # Z para cima
                        center=dict(x=0, y=0, z=0)
                    ),
                    aspectmode='data',
                    xaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230, 230)"),
                    yaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230, 230)"),
                    zaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230, 230)")
                ),
                width=width,
                height=height,
                margin=dict(l=20, r=20, t=60, b=20),
                scene_bgcolor='white'
            )

            # Converter para imagem
            img_bytes = fig.to_image(format="png", width=width, height=height)
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)

            # Progresso
            if (frame + 1) % 5 == 0 or frame == num_frames - 1:
                print(f"Progresso: {frame + 1}/{num_frames} frames")

        # Salvar como GIF
        print(f"Salvando GIF: {nome_arquivo}")

        # Calcular duração por frame para o GIF (em milissegundos)
        duracao_gif = 1000 // fps  # Converter FPS para duração em ms

        images[0].save(
            nome_arquivo,
            save_all=True,
            append_images=images[1:],
            duration=duracao_gif,
            loop=0  # Loop infinito
        )

        print(f"GIF salvo como: {nome_arquivo}")
        print(f"Tamanho: {width}x{height} pixels")
        print(f"Frames: {num_frames}")
        print(f"FPS: {fps}")

        return nome_arquivo

    def animar_cavidade_matplotlib(self, modo='TE', campo='magnetico',
                                   tipo_intensidade='total', direcao_vetor='z',
                                   num_frames=60, duracao_ciclo=2.0):
        """
        Cria uma animação matplotlib da cavidade cilíndrica 3D mostrando a evolução temporal do campo.

        Esta função retorna um objeto FuncAnimation que pode ser salvo como GIF usando:
        ani.save('arquivo.gif', writer='pillow', fps=30)

        Parameters:
        -----------
        modo : str
            'TE' ou 'TM'
        campo : str
            'magnetico' ou 'eletrico'
        tipo_intensidade : str
            'total', 'perpendicular', ou 'direcional'
        direcao_vetor : str
            'rho', 'phi', ou 'z'
        num_frames : int
            Número de frames na animação
        duracao_ciclo : float
            Duração do ciclo completo em segundos

        Returns:
        --------
        ani : matplotlib.animation.FuncAnimation
            Objeto de animação do matplotlib

        Example:
        --------
        >>> cavity = CylindricalCavityWall3D(raio=10.0, profundidade=50.0)
        >>> ani = cavity.animar_cavidade_matplotlib(modo='TE', campo='eletrico', num_frames=30)
        >>> ani.save('animacao.gif', writer='pillow', fps=15)
        """

        # Configurar figura
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Converter dimensões
        raio_m = self.raio / 1000
        prof_m = self.profundidade / 1000

        # Calcular intensidade máxima para normalização consistente
        fases_teste = np.linspace(0, 2*np.pi, 10)
        max_intensidade_global = 0

        for fase in fases_teste:
            intensidades = self._calcular_intensidades_com_fase(modo, campo, tipo_intensidade,
                                                              direcao_vetor, fase)
            max_local = max([np.max(intensidade) for intensidade in intensidades.values()])
            max_intensidade_global = max(max_intensidade_global, max_local)

        def init():
            ax.clear()
            return []

        def animate(frame):
            ax.clear()

            # Calcular fase temporal para este frame
            fase_temporal = 2 * np.pi * frame / num_frames

            # Calcular intensidades para esta fase
            intensidades = self._calcular_intensidades_com_fase(modo, campo, tipo_intensidade,
                                                              direcao_vetor, fase_temporal)

            # Tampa frontal (Z=0)
            r_tampa = np.linspace(0, raio_m, self.resolucao)
            phi_tampa = np.linspace(0, 2*np.pi, self.resolucao)
            R_tampa, Phi_tampa = np.meshgrid(r_tampa, phi_tampa, indexing='ij')
            X_frontal = R_tampa * np.cos(Phi_tampa)
            Y_frontal = R_tampa * np.sin(Phi_tampa)
            Z_frontal = np.zeros_like(X_frontal)

            # Normalizar cores
            colors_frontal = plt.cm.viridis(intensidades['tampa_frontal']/max_intensidade_global
                                          if max_intensidade_global > 0 else intensidades['tampa_frontal'])
            ax.plot_surface(X_frontal, Y_frontal, Z_frontal, facecolors=colors_frontal,
                          alpha=0.8, shade=False, antialiased=True)

            # Tampa traseira (Z=prof_m)
            Z_traseira = np.full_like(X_frontal, prof_m)
            colors_traseira = plt.cm.viridis(intensidades['tampa_traseira']/max_intensidade_global
                                           if max_intensidade_global > 0 else intensidades['tampa_traseira'])
            ax.plot_surface(X_frontal, Y_frontal, Z_traseira, facecolors=colors_traseira,
                          alpha=0.8, shade=False, antialiased=True)

            # Superfície lateral
            phi_lateral = np.linspace(0, 2*np.pi, self.resolucao)
            z_lateral = np.linspace(0, prof_m, self.resolucao)
            Phi_lateral, Z_lateral = np.meshgrid(phi_lateral, z_lateral, indexing='ij')
            X_lateral = raio_m * np.cos(Phi_lateral)
            Y_lateral = raio_m * np.sin(Phi_lateral)

            colors_lateral = plt.cm.viridis(intensidades['lateral']/max_intensidade_global
                                          if max_intensidade_global > 0 else intensidades['lateral'])
            ax.plot_surface(X_lateral, Y_lateral, Z_lateral, facecolors=colors_lateral,
                          alpha=0.8, shade=False, antialiased=True)

            # Configurar eixos
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Animação - Campo {campo.capitalize()} ({tipo_intensidade}) - Modo {modo}{self.m}{self.n}\n'
                        f'Fase: {fase_temporal:.2f} rad, Frame: {frame+1}/{num_frames}')

            # Configurar limites
            ax.set_xlim(-raio_m*1.1, raio_m*1.1)
            ax.set_ylim(-raio_m*1.1, raio_m*1.1)
            ax.set_zlim(-prof_m*0.1, prof_m*1.1)

            # Configurar aspecto proporcional
            ax.set_box_aspect([1, 1, prof_m/(raio_m*2)])

            # Configurar vista para melhor visualização das tampas
            ax.view_init(elev=25, azim=45)  # Elevação aumentada para ver as tampas
            ax.grid(True, alpha=0.3)

            return []

        # Criar animação
        intervalo = int(duracao_ciclo * 1000 / num_frames)  # intervalo em ms
        ani = animation.FuncAnimation(fig, animate, init_func=init,
                                     frames=num_frames, interval=intervalo,
                                     blit=False, repeat=True)

        return ani

    def gerar_gifs_varredura(self, output_dir='gifs_cilindricos',
                            modos=['TE', 'TM'],
                            campos=['magnetico', 'eletrico'],
                            lista_m=[1, 2, 3],
                            lista_n=[0, 1, 2],
                            tipo_intensidade='total',
                            direcao_vetor='rho',
                            num_frames=24,
                            width=800,
                            height=600,
                            fps=12,
                            pular_existentes=True):
        """
        Gera GIFs para todas as combinações de parâmetros especificadas.

        Parameters:
        -----------
        output_dir : str
            Diretório onde os GIFs serão salvos
        modos : list
            Lista de modos a varrer (ex: ['TE', 'TM'])
        campos : list
            Lista de campos a varrer (ex: ['magnetico', 'eletrico'])
        lista_m : list
            Lista de valores m a varrer (ex: [1, 2, 3])
        lista_n : list
            Lista de valores n a varrer (ex: [0, 1, 2])
        tipo_intensidade : str
            Tipo de intensidade ('total', 'perpendicular', 'direcional')
        direcao_vetor : str
            Direção do vetor ('rho', 'phi', 'z')
        num_frames : int
            Número de frames por GIF
        width : int
            Largura em pixels
        height : int
            Altura em pixels
        fps : int
            Frames por segundo
        pular_existentes : bool
            Se True, pula GIFs que já existem

        Returns:
        --------
        dict
            Dicionário com estatísticas da varredura
        """
        import os
        import time

        # Criar diretório de saída se não existir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 Diretório criado: {output_dir}")

        # Contar total de combinações
        total_combinacoes = len(modos) * len(campos) * len(lista_m) * len(lista_n)
        print(f"\n{'='*60}")
        print(f"🎬 INICIANDO VARREDURA DE GIFS")
        print(f"{'='*60}")
        print(f"Total de combinações: {total_combinacoes}")
        print(f"Modos: {modos}")
        print(f"Campos: {campos}")
        print(f"M: {lista_m}")
        print(f"N: {lista_n}")
        print(f"{'='*60}\n")

        # Estatísticas
        stats = {
            'total': total_combinacoes,
            'gerados': 0,
            'pulados': 0,
            'erros': 0,
            'tempo_total': 0,
            'arquivos': []
        }

        contador = 0
        tempo_inicio_total = time.time()

        # Loop sobre todas as combinações
        for modo in modos:
            for campo in campos:
                for m in lista_m:
                    for n in lista_n:
                        contador += 1

                        # Nome do arquivo
                        nome_arquivo = f'cyl_{modo}{n}{m}_{campo}_{tipo_intensidade}.gif'
                        caminho_completo = os.path.join(output_dir, nome_arquivo)

                        # Verificar se já existe
                        if pular_existentes and os.path.exists(caminho_completo):
                            print(f"[{contador}/{total_combinacoes}] ⏭️  Pulando (já existe): {nome_arquivo}")
                            stats['pulados'] += 1
                            continue

                        # Criar nova instância com os parâmetros atuais
                        try:
                            print(f"\n[{contador}/{total_combinacoes}] 🎬 Gerando: {nome_arquivo}")
                            print(f"   Modo: {modo}{n}{m} | Campo: {campo}")

                            tempo_inicio = time.time()

                            # Criar nova instância com m e n atualizados
                            cavity_temp = CylindricalCavityWall3D(
                                raio=self.raio,
                                profundidade=self.profundidade,
                                frequencia=self.frequencia,
                                permissividade=self.permissividade,
                                permeabilidade=self.permeabilidade,
                                resolucao=self.resolucao,
                                m=m,
                                n=n
                            )

                            # Gerar GIF
                            cavity_temp.criar_gif_plotly(
                                nome_arquivo=caminho_completo,
                                modo=modo,
                                campo=campo,
                                tipo_intensidade=tipo_intensidade,
                                direcao_vetor=direcao_vetor,
                                num_frames=num_frames,
                                width=width,
                                height=height,
                                fps=fps
                            )

                            tempo_decorrido = time.time() - tempo_inicio
                            stats['gerados'] += 1
                            stats['arquivos'].append(caminho_completo)

                            print(f"   ✅ Concluído em {tempo_decorrido:.1f}s")

                        except Exception as e:
                            print(f"   ❌ Erro: {str(e)}")
                            stats['erros'] += 1
                            continue

        # Tempo total
        stats['tempo_total'] = time.time() - tempo_inicio_total

        # Relatório final
        print(f"\n{'='*60}")
        print(f"📊 RELATÓRIO FINAL")
        print(f"{'='*60}")
        print(f"✅ GIFs gerados: {stats['gerados']}")
        print(f"⏭️  GIFs pulados: {stats['pulados']}")
        print(f"❌ Erros: {stats['erros']}")
        print(f"⏱️  Tempo total: {stats['tempo_total']/60:.1f} minutos")
        if stats['gerados'] > 0:
            print(f"⏱️  Tempo médio por GIF: {stats['tempo_total']/stats['gerados']:.1f}s")
        print(f"📁 Diretório: {output_dir}")
        print(f"{'='*60}\n")

        return stats


# Exemplo de uso
if __name__ == "__main__":

    # Criar instância base
    cavity = CylindricalCavityWall3D(
        raio=10.0,  # mm
        profundidade=50.0,  # mm
        frequencia=15e9,  # Hz
        permissividade=2.3,
        permeabilidade=1.0,
        resolucao=60,
        m=1,  # Será substituído na varredura
        n=0   # Será substituído na varredura
    )

    # ========================================================================
    # OPÇÃO 1: VARREDURA COMPLETA - Gera GIFs para todas as combinações
    # ========================================================================
    print("\n🎬 VARREDURA COMPLETA DE GIFS")
    print("Isso vai gerar GIFs para todas as combinações de:")
    print("  - Modos: TE, TM")
    print("  - Campos: magnético, elétrico")
    print("  - M: 1, 2, 3")
    print("  - N: 0, 1, 2")
    print("Total: 2 × 2 × 3 × 3 = 36 GIFs\n")

    stats = cavity.gerar_gifs_varredura(
        output_dir='gifs_cilindricos',
        modos=['TE', 'TM'],
        campos=['magnetico', 'eletrico'],
        lista_m=[1, 2, 3],
        lista_n=[0, 1, 2],
        tipo_intensidade='total',
        direcao_vetor='rho',
        num_frames=24,
        width=800,
        height=600,
        fps=12,
        pular_existentes=True  # Pula GIFs já existentes
    )

    # ========================================================================
    # OPÇÃO 2: GIF ÚNICO - Cria apenas um GIF específico
    # ========================================================================
    # modo = 'TE'
    # campo = 'magnetico'
    # m = 1
    # n = 2
    #
    # cavity_single = CylindricalCavityWall3D(
    #     raio=10.0,
    #     profundidade=50.0,
    #     frequencia=15e9,
    #     permissividade=2.3,
    #     permeabilidade=1.0,
    #     resolucao=60,
    #     m=m,
    #     n=n
    # )
    #
    # cavity_single.criar_gif_plotly(
    #     nome_arquivo=f'animacao_cilindrica_{modo}{n}{m}_{campo}.gif',
    #     modo=modo,
    #     campo=campo,
    #     tipo_intensidade='total',
    #     direcao_vetor='rho',
    #     num_frames=24,
    #     width=800,
    #     height=600,
    #     fps=12
    # )

    # ========================================================================
    # OPÇÃO 3: ANIMAÇÃO PLOTLY INTERATIVA (HTML)
    # ========================================================================
    # fig = cavity.animar_cavidade_plotly(
    #     modo='TE',
    #     campo='eletrico',
    #     tipo_intensidade='total',
    #     direcao_vetor='rho',
    #     num_frames=60,
    #     duracao_frame=100
    # )
    # fig.write_html('animacao_cilindrica_3d.html')
    # print("✅ Animação HTML salva como: animacao_cilindrica_3d.html")

    # ========================================================================
    # OPÇÃO 4: ANIMAÇÃO MATPLOTLIB (GIF mais leve)
    # ========================================================================
    # ani = cavity.animar_cavidade_matplotlib(
    #     modo='TE',
    #     campo='eletrico',
    #     tipo_intensidade='total',
    #     num_frames=30,
    #     duracao_ciclo=2.0
    # )
    # ani.save('animacao_cilindrica_matplotlib.gif', writer='pillow', fps=15)
    # print("✅ Animação matplotlib salva como: animacao_cilindrica_matplotlib.gif")
