from TEmn_model import Modo_TEmn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches
import matplotlib.animation as animation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from PIL import Image
import io
import os
import tempfile

class CavityWall3D:
    def __init__(self, largura=22.86, altura=10.16, profundidade=None,
                 frequencia=12*10**9, permissividade=1, permeabilidade=1,
                 resolucao=25, m=1, n=0):
        """
        Classe para visualização 3D da cavidade com intensidade de campo nas paredes.

        Parameters:
        -----------
        largura : float
            Largura da cavidade em mm
        altura : float
            Altura da cavidade em mm
        profundidade : float, optional
            Profundidade da cavidade em mm. Se None, usa o valor padrão do Modo_TEmn
        frequencia : float
            Frequência em Hz
        permissividade : float
            Permissividade relativa
        permeabilidade : float
            Permeabilidade relativa
        resolucao : int, optional
            Número de pontos por dimensão (padrão: 25 para performance)
        m : int, optional
            Número de modos na direção x (largura) - padrão: 1
        n : int, optional
            Número de modos na direção y (altura) - padrão: 0
        """
        self.largura = largura
        self.altura = altura
        self.frequencia = frequencia
        self.permissividade = permissividade
        self.permeabilidade = permeabilidade
        self.m = m
        self.n = n

        # Criar instâncias do Modo_TEmn para cada plano
        self.modo_xy = Modo_TEmn(largura, altura, frequencia, permissividade, permeabilidade, 'xy')
        self.modo_xz = Modo_TEmn(largura, altura, frequencia, permissividade, permeabilidade, 'xz')
        self.modo_yz = Modo_TEmn(largura, altura, frequencia, permissividade, permeabilidade, 'yz')

        # Configurar resolução e modos para todos os planos
        for modo in [self.modo_xy, self.modo_xz, self.modo_yz]:
            modo.pontos_por_dimensao = resolucao
            modo.m = m
            modo.n = n

        # Usar profundidade especificada ou do Modo_TEmn
        if profundidade is not None:
            self.profundidade = profundidade
            # Atualizar profundidade nos modos
            self.modo_xy.profundidade = profundidade/1000  # converter para metros
            self.modo_xz.profundidade = profundidade/1000
            self.modo_yz.profundidade = profundidade/1000
        else:
            self.profundidade = self.modo_xy.profundidade * 1000  # converter para mm

        # Recalcular meshgrids com nova resolução e calcular campos
        self.modo_xy.escolha_plano()
        self.modo_xz.escolha_plano()
        self.modo_yz.escolha_plano()

        # Recalcular funções trigonométricas e exponenciais
        self.modo_xy.cos_mx = self.modo_xy.cosseno_x()
        self.modo_xy.cos_ny = self.modo_xy.cosseno_y()
        self.modo_xy.sen_mx = self.modo_xy.seno_x()
        self.modo_xy.sen_ny = self.modo_xy.seno_y()
        self.modo_xy.expz = self.modo_xy.exp_z()

        self.modo_xz.cos_mx = self.modo_xz.cosseno_x()
        self.modo_xz.cos_ny = self.modo_xz.cosseno_y()
        self.modo_xz.sen_mx = self.modo_xz.seno_x()
        self.modo_xz.sen_ny = self.modo_xz.seno_y()
        self.modo_xz.expz = self.modo_xz.exp_z()

        self.modo_yz.cos_mx = self.modo_yz.cosseno_x()
        self.modo_yz.cos_ny = self.modo_yz.cosseno_y()
        self.modo_yz.sen_mx = self.modo_yz.seno_x()
        self.modo_yz.sen_ny = self.modo_yz.seno_y()
        self.modo_yz.expz = self.modo_yz.exp_z()

        # Calcular campos para todos os planos
        self.modo_xy.calcula_campos()
        self.modo_xz.calcula_campos()
        self.modo_yz.calcula_campos()

    def debug_dimensoes(self):
        """Debug para verificar as dimensões dos meshgrids."""
        print("=== DEBUG DIMENSÕES ===")
        print(f"Resolução configurada: {self.modo_xy.pontos_por_dimensao}")
        print(f"Modo XY - X.shape: {self.modo_xy.x.shape}, Y.shape: {self.modo_xy.y.shape}, Z.shape: {self.modo_xy.z.shape}")
        print(f"Modo XZ - X.shape: {self.modo_xz.x.shape}, Y.shape: {self.modo_xz.y.shape}, Z.shape: {self.modo_xz.z.shape}")
        print(f"Modo YZ - X.shape: {self.modo_yz.x.shape}, Y.shape: {self.modo_yz.y.shape}, Z.shape: {self.modo_yz.z.shape}")
        print(f"Largura: {self.largura}mm, Altura: {self.altura}mm, Profundidade: {self.profundidade}mm")
        print("=======================")

    def plota_cavidade_3d_com_intensidade(self, campo='magnetico', tipo_intensidade='total', direcao_vetor='x'):
        """
        Plota um modelo 3D da cavidade com a intensidade do campo nas paredes.

        Parameters:
        -----------
        campo : str
            Tipo de campo ('magnetico' ou 'eletrico')
        tipo_intensidade : str
            Tipo de intensidade ('total', 'perpendicular', ou 'direcional')
        direcao_vetor : str
            Direção do vetor ('x', 'y', 'z') quando tipo_intensidade='direcional'

        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figura 3D do matplotlib
        """
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Converter dimensões para metros para consistência
        largura_m = self.largura / 1000
        altura_m = self.altura / 1000
        profundidade_m = self.profundidade / 1000

        # Calcular intensidades para cada plano
        intensidades = self._calcular_intensidades_paredes(campo, tipo_intensidade, direcao_vetor)

        # Plotar cada parede com sua intensidade
        self._plotar_parede_xy_frontal(ax, intensidades['xy_frontal'], largura_m, altura_m, 0)
        self._plotar_parede_xy_traseira(ax, intensidades['xy_traseira'], largura_m, altura_m, profundidade_m)
        self._plotar_parede_xz_inferior(ax, intensidades['xz_inferior'], largura_m, profundidade_m, 0)
        self._plotar_parede_xz_superior(ax, intensidades['xz_superior'], largura_m, profundidade_m, altura_m)
        self._plotar_parede_yz_esquerda(ax, intensidades['yz_esquerda'], altura_m, profundidade_m, 0)
        self._plotar_parede_yz_direita(ax, intensidades['yz_direita'], altura_m, profundidade_m, largura_m)

        # Adicionar bordas da cavidade
        self._adicionar_bordas_cavidade(ax, largura_m, altura_m, profundidade_m)

        # Configurar eixos e labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_zlabel('Y (m)')
        ax.set_title(f'Cavidade 3D - Intensidade do Campo {campo.capitalize()}\n'
                    f'{self.largura:.1f}×{self.altura:.1f}×{self.profundidade:.1f} mm')

        # Configurar limites proporcionais
        ax.set_xlim(-largura_m*0.1, largura_m*1.1)
        ax.set_ylim(-profundidade_m*0.1, profundidade_m*1.1)
        ax.set_zlim(-altura_m*0.1, altura_m*1.1)

        # Configurar aspecto proporcional mais equilibrado
        # Limitar a diferença de escala para melhor visualização
        aspect_x = largura_m
        aspect_y = profundidade_m
        aspect_z = max(altura_m, largura_m * 0.4)  # Mínimo 40% da largura para Z

        max_aspect = max(aspect_x, aspect_y, aspect_z)
        ax.set_box_aspect([aspect_x/max_aspect, aspect_y/max_aspect, aspect_z/max_aspect])

        # Configurar visualização
        ax.view_init(elev=20, azim=45)
        ax.grid(True, alpha=0.3)

        return fig

    def _calcular_intensidades_paredes(self, campo, tipo_intensidade, direcao_vetor='x'):
        """
        Calcula as intensidades para cada parede da cavidade.

        Parameters:
        -----------
        campo : str
            Tipo de campo ('magnetico' ou 'eletrico')
        tipo_intensidade : str
            Tipo de intensidade ('total', 'perpendicular', ou 'direcional')
        direcao_vetor : str
            Direção específica ('x', 'y', 'z') quando tipo_intensidade='direcional'
        """
        intensidades = {}

        # Para as paredes frontais e traseiras (plano xy)
        if campo == 'magnetico':
            u_xy, v_xy, w_xy = self.modo_xy.Hx, self.modo_xy.Hy, self.modo_xy.Hz
        else:
            u_xy, v_xy, w_xy = self.modo_xy.Ex, self.modo_xy.Ey, self.modo_xy.Ez

        if tipo_intensidade == 'total':
            intensidades['xy_frontal'] = np.sqrt(u_xy**2 + v_xy**2 + w_xy**2)
            intensidades['xy_traseira'] = np.sqrt(u_xy**2 + v_xy**2 + w_xy**2)
        elif tipo_intensidade == 'perpendicular':
            intensidades['xy_frontal'] = np.abs(w_xy)
            intensidades['xy_traseira'] = np.abs(w_xy)
        elif tipo_intensidade == 'direcional':
            if direcao_vetor == 'x':
                intensidades['xy_frontal'] = np.abs(u_xy)
                intensidades['xy_traseira'] = np.abs(u_xy)
            elif direcao_vetor == 'y':
                intensidades['xy_frontal'] = np.abs(v_xy)
                intensidades['xy_traseira'] = np.abs(v_xy)
            elif direcao_vetor == 'z':
                intensidades['xy_frontal'] = np.abs(w_xy)
                intensidades['xy_traseira'] = np.abs(w_xy)

        # Para as paredes superior e inferior (plano xz)
        if campo == 'magnetico':
            u_xz, v_xz, w_xz = self.modo_xz.Hx, self.modo_xz.Hz, self.modo_xz.Hy
        else:
            u_xz, v_xz, w_xz = self.modo_xz.Ex, self.modo_xz.Ez, self.modo_xz.Ey

        if tipo_intensidade == 'total':
            intensidades['xz_inferior'] = np.sqrt(u_xz**2 + v_xz**2 + w_xz**2)
            intensidades['xz_superior'] = np.sqrt(u_xz**2 + v_xz**2 + w_xz**2)
        elif tipo_intensidade == 'perpendicular':
            intensidades['xz_inferior'] = np.abs(w_xz)
            intensidades['xz_superior'] = np.abs(w_xz)
        elif tipo_intensidade == 'direcional':
            if direcao_vetor == 'x':
                intensidades['xz_inferior'] = np.abs(u_xz)
                intensidades['xz_superior'] = np.abs(u_xz)
            elif direcao_vetor == 'y':
                intensidades['xz_inferior'] = np.abs(w_xz)
                intensidades['xz_superior'] = np.abs(w_xz)
            elif direcao_vetor == 'z':
                intensidades['xz_inferior'] = np.abs(v_xz)
                intensidades['xz_superior'] = np.abs(v_xz)

        # Para as paredes laterais (plano yz)
        if campo == 'magnetico':
            u_yz, v_yz, w_yz = self.modo_yz.Hy, self.modo_yz.Hz, self.modo_yz.Hx
        else:
            u_yz, v_yz, w_yz = self.modo_yz.Ey, self.modo_yz.Ez, self.modo_yz.Ex

        if tipo_intensidade == 'total':
            intensidades['yz_esquerda'] = np.sqrt(u_yz**2 + v_yz**2 + w_yz**2)
            intensidades['yz_direita'] = np.sqrt(u_yz**2 + v_yz**2 + w_yz**2)
        elif tipo_intensidade == 'perpendicular':
            intensidades['yz_esquerda'] = np.abs(w_yz)
            intensidades['yz_direita'] = np.abs(w_yz)
        elif tipo_intensidade == 'direcional':
            if direcao_vetor == 'x':
                intensidades['yz_esquerda'] = np.abs(w_yz)
                intensidades['yz_direita'] = np.abs(w_yz)
            elif direcao_vetor == 'y':
                intensidades['yz_esquerda'] = np.abs(u_yz)
                intensidades['yz_direita'] = np.abs(u_yz)
            elif direcao_vetor == 'z':
                intensidades['yz_esquerda'] = np.abs(v_yz)
                intensidades['yz_direita'] = np.abs(v_yz)

        return intensidades

    def _plotar_parede_xy_frontal(self, ax, intensidade, largura, altura, z_pos):
        """Plota a parede frontal (xy, z=0)."""
        x = np.linspace(0, largura, intensidade.shape[0])
        y = np.linspace(0, altura, intensidade.shape[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = np.full_like(X, z_pos)

        ax.plot_surface(X, Z, Y, facecolors=plt.cm.viridis(intensidade/intensidade.max()),
                       alpha=0.8, shade=False)

    def _plotar_parede_xy_traseira(self, ax, intensidade, largura, altura, z_pos):
        """Plota a parede traseira (xy, z=profundidade)."""
        x = np.linspace(0, largura, intensidade.shape[0])
        y = np.linspace(0, altura, intensidade.shape[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = np.full_like(X, z_pos)

        ax.plot_surface(X, Z, Y, facecolors=plt.cm.viridis(intensidade/intensidade.max()),
                       alpha=0.8, shade=False)

    def _plotar_parede_xz_inferior(self, ax, intensidade, largura, profundidade, y_pos):
        """Plota a parede inferior (xz, y=0)."""
        x = np.linspace(0, largura, intensidade.shape[0])
        z = np.linspace(0, profundidade, intensidade.shape[1])
        X, Z = np.meshgrid(x, z, indexing='ij')
        Y = np.full_like(X, y_pos)

        ax.plot_surface(X, Z, Y, facecolors=plt.cm.viridis(intensidade/intensidade.max()),
                       alpha=0.8, shade=False)

    def _plotar_parede_xz_superior(self, ax, intensidade, largura, profundidade, y_pos):
        """Plota a parede superior (xz, y=altura)."""
        x = np.linspace(0, largura, intensidade.shape[0])
        z = np.linspace(0, profundidade, intensidade.shape[1])
        X, Z = np.meshgrid(x, z, indexing='ij')
        Y = np.full_like(X, y_pos)

        ax.plot_surface(X, Z, Y, facecolors=plt.cm.viridis(intensidade/intensidade.max()),
                       alpha=0.8, shade=False)

    def _plotar_parede_yz_esquerda(self, ax, intensidade, altura, profundidade, x_pos):
        """Plota a parede esquerda (yz, x=0)."""
        y = np.linspace(0, altura, intensidade.shape[0])
        z = np.linspace(0, profundidade, intensidade.shape[1])
        Y, Z = np.meshgrid(y, z, indexing='ij')
        X = np.full_like(Y, x_pos)

        ax.plot_surface(X, Z, Y, facecolors=plt.cm.viridis(intensidade/intensidade.max()),
                       alpha=0.8, shade=False)

    def _plotar_parede_yz_direita(self, ax, intensidade, altura, profundidade, x_pos):
        """Plota a parede direita (yz, x=largura)."""
        y = np.linspace(0, altura, intensidade.shape[0])
        z = np.linspace(0, profundidade, intensidade.shape[1])
        Y, Z = np.meshgrid(y, z, indexing='ij')
        X = np.full_like(Y, x_pos)

        ax.plot_surface(X, Z, Y, facecolors=plt.cm.viridis(intensidade/intensidade.max()),
                       alpha=0.8, shade=False)

    def _adicionar_bordas_cavidade(self, ax, largura, altura, profundidade):
        """Adiciona as bordas wireframe da cavidade."""
        # Vértices da cavidade
        vertices = [
            [0, 0, 0], [largura, 0, 0], [largura, profundidade, 0], [0, profundidade, 0],  # base
            [0, 0, altura], [largura, 0, altura], [largura, profundidade, altura], [0, profundidade, altura]  # topo
        ]

        # Arestas da cavidade
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # base
            [4, 5], [5, 6], [6, 7], [7, 4],  # topo
            [0, 4], [1, 5], [2, 6], [3, 7]   # verticais
        ]

        for edge in edges:
            points = np.array([vertices[edge[0]], vertices[edge[1]]])
            ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'k-', linewidth=2, alpha=0.8)

    def plota_cavidade_3d_com_colorbar(self, campo='magnetico', tipo_intensidade='total', direcao_vetor='x'):
        """
        Plota um modelo 3D da cavidade com colorbar e legendas detalhadas.

        Parameters:
        -----------
        campo : str
            Tipo de campo ('magnetico' ou 'eletrico')
        tipo_intensidade : str
            Tipo de intensidade ('total', 'perpendicular', ou 'direcional')
        direcao_vetor : str
            Direção do vetor ('x', 'y', 'z') quando tipo_intensidade='direcional'

        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figura 3D do matplotlib
        """
        fig = plt.figure(figsize=(18, 14))
        ax = fig.add_subplot(111, projection='3d')

        # Converter dimensões para metros para consistência
        largura_m = self.largura / 1000
        altura_m = self.altura / 1000
        profundidade_m = self.profundidade / 1000

        # Calcular intensidades para cada plano
        intensidades = self._calcular_intensidades_paredes(campo, tipo_intensidade, direcao_vetor)

        # Encontrar valor máximo global para normalização
        max_intensidade = max([np.max(intensidade) for intensidade in intensidades.values()])

        # Plotar cada parede com sua intensidade normalizada
        self._plotar_parede_xy_frontal_colorbar(ax, intensidades['xy_frontal'], max_intensidade,
                                               largura_m, altura_m, 0)
        self._plotar_parede_xy_traseira_colorbar(ax, intensidades['xy_traseira'], max_intensidade,
                                                largura_m, altura_m, profundidade_m)
        self._plotar_parede_xz_inferior_colorbar(ax, intensidades['xz_inferior'], max_intensidade,
                                                largura_m, profundidade_m, 0)
        self._plotar_parede_xz_superior_colorbar(ax, intensidades['xz_superior'], max_intensidade,
                                                largura_m, profundidade_m, altura_m)
        self._plotar_parede_yz_esquerda_colorbar(ax, intensidades['yz_esquerda'], max_intensidade,
                                                altura_m, profundidade_m, 0)
        surface = self._plotar_parede_yz_direita_colorbar(ax, intensidades['yz_direita'], max_intensidade,
                                                         altura_m, profundidade_m, largura_m)

        # Adicionar colorbar usando a última superfície
        cbar = fig.colorbar(surface, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label(f'Intensidade do Campo {campo.capitalize()}', rotation=270, labelpad=20)

        # Adicionar bordas da cavidade
        self._adicionar_bordas_cavidade(ax, largura_m, altura_m, profundidade_m)

        # Adicionar texto informativo
        info_text = (f'Modo TEmn\n'
                    f'Frequência: {self.frequencia/1e9:.1f} GHz\n'
                    f'Dimensões: {self.largura:.1f}×{self.altura:.1f}×{self.profundidade:.1f} mm')

        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Configurar eixos e labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_zlabel('Y (m)')
        ax.set_title(f'Cavidade 3D - Intensidade {tipo_intensidade.capitalize()} do Campo {campo.capitalize()}\n'
                    f'Visualização das Paredes da Guia de Onda')

        # Configurar limites proporcionais
        ax.set_xlim(-largura_m*0.1, largura_m*1.1)
        ax.set_ylim(-profundidade_m*0.1, profundidade_m*1.1)
        ax.set_zlim(-altura_m*0.1, altura_m*1.1)

        # Configurar aspecto proporcional mais equilibrado
        # Limitar a diferença de escala para melhor visualização
        # aspect_x = largura_m
        # aspect_y = profundidade_m
        # aspect_z = max(altura_m, largura_m * 0.4)  # Mínimo 40% da largura para Z

        # max_aspect = max(aspect_x, aspect_y, aspect_z)
        # ax.set_box_aspect([aspect_x/max_aspect, aspect_y/max_aspect, aspect_z/max_aspect])

        # Configurar visualização
        ax.view_init(elev=20, azim=45)
        ax.grid(True, alpha=0.3)

        return fig

    def _plotar_parede_xy_frontal_colorbar(self, ax, intensidade, max_global, largura, altura, z_pos):
        """Plota a parede frontal com normalização global."""
        x = np.linspace(0, largura, intensidade.shape[0])
        y = np.linspace(0, altura, intensidade.shape[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = np.full_like(X, z_pos)

        return ax.plot_surface(X, Z, Y, facecolors=plt.cm.viridis(intensidade/max_global),
                              alpha=0.8, shade=False)

    def _plotar_parede_xy_traseira_colorbar(self, ax, intensidade, max_global, largura, altura, z_pos):
        """Plota a parede traseira com normalização global."""
        x = np.linspace(0, largura, intensidade.shape[0])
        y = np.linspace(0, altura, intensidade.shape[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = np.full_like(X, z_pos)

        return ax.plot_surface(X, Z, Y, facecolors=plt.cm.viridis(intensidade/max_global),
                              alpha=0.8, shade=False)

    def _plotar_parede_xz_inferior_colorbar(self, ax, intensidade, max_global, largura, profundidade, y_pos):
        """Plota a parede inferior com normalização global."""
        x = np.linspace(0, largura, intensidade.shape[0])
        z = np.linspace(0, profundidade, intensidade.shape[1])
        X, Z = np.meshgrid(x, z, indexing='ij')
        Y = np.full_like(X, y_pos)

        return ax.plot_surface(X, Z, Y, facecolors=plt.cm.viridis(intensidade/max_global),
                              alpha=0.8, shade=False)

    def _plotar_parede_xz_superior_colorbar(self, ax, intensidade, max_global, largura, profundidade, y_pos):
        """Plota a parede superior com normalização global."""
        x = np.linspace(0, largura, intensidade.shape[0])
        z = np.linspace(0, profundidade, intensidade.shape[1])
        X, Z = np.meshgrid(x, z, indexing='ij')
        Y = np.full_like(X, y_pos)

        return ax.plot_surface(X, Z, Y, facecolors=plt.cm.viridis(intensidade/max_global),
                              alpha=0.8, shade=False)

    def _plotar_parede_yz_esquerda_colorbar(self, ax, intensidade, max_global, altura, profundidade, x_pos):
        """Plota a parede esquerda com normalização global."""
        y = np.linspace(0, altura, intensidade.shape[0])
        z = np.linspace(0, profundidade, intensidade.shape[1])
        Y, Z = np.meshgrid(y, z, indexing='ij')
        X = np.full_like(Y, x_pos)

        return ax.plot_surface(X, Z, Y, facecolors=plt.cm.viridis(intensidade/max_global),
                              alpha=0.8, shade=False)

    def _plotar_parede_yz_direita_colorbar(self, ax, intensidade, max_global, altura, profundidade, x_pos):
        """Plota a parede direita com normalização global."""
        y = np.linspace(0, altura, intensidade.shape[0])
        z = np.linspace(0, profundidade, intensidade.shape[1])
        Y, Z = np.meshgrid(y, z, indexing='ij')
        X = np.full_like(Y, x_pos)

        # Criar superfície para retornar (usada para colorbar)
        surface = ax.plot_surface(X, Z, Y, facecolors=plt.cm.viridis(intensidade/max_global),
                                 alpha=0.8, shade=False)

        # Mapear cores para a superfície (necessário para colorbar)
        surface.set_array(intensidade.ravel())
        surface.set_clim(vmin=0, vmax=max_global)

        return surface

    def _calcular_campos_com_fase(self, fase_temporal=0):
        """
        Calcula os campos com uma fase temporal específica.

        Parameters:
        -----------
        fase_temporal : float
            Fase temporal em radianos (ωt)
        """
        # Calcular campos complexos para cada modo
        for modo in [self.modo_xy, self.modo_xz, self.modo_yz]:
            # Aplicar fase temporal aos campos complexos
            fase_complexa = np.exp(1j * fase_temporal)

            # Campos magnéticos com fase temporal
            modo.Hx_temp = np.real(modo.H_x() * fase_complexa)
            modo.Hy_temp = np.real(modo.H_y() * fase_complexa)
            modo.Hz_temp = np.real(modo.H_z() * fase_complexa)

            # Campos elétricos com fase temporal
            modo.Ex_temp = np.real(modo.E_x() * fase_complexa)
            modo.Ey_temp = np.real(modo.E_y() * fase_complexa)
            modo.Ez_temp = np.zeros_like(modo.Ex_temp)

    def _calcular_intensidades_com_fase(self, campo, tipo_intensidade, direcao_vetor, fase_temporal):
        """
        Calcula intensidades com fase temporal específica.
        """
        # Calcular campos com fase temporal
        self._calcular_campos_com_fase(fase_temporal)

        intensidades = {}

        # Para as paredes frontais e traseiras (plano xy)
        if campo == 'magnetico':
            u_xy = self.modo_xy.Hx_temp
            v_xy = self.modo_xy.Hy_temp
            w_xy = self.modo_xy.Hz_temp
        else:
            u_xy = self.modo_xy.Ex_temp
            v_xy = self.modo_xy.Ey_temp
            w_xy = self.modo_xy.Ez_temp

        if tipo_intensidade == 'total':
            intensidades['xy_frontal'] = np.sqrt(u_xy**2 + v_xy**2 + w_xy**2)
            intensidades['xy_traseira'] = np.sqrt(u_xy**2 + v_xy**2 + w_xy**2)
        elif tipo_intensidade == 'perpendicular':
            intensidades['xy_frontal'] = np.abs(w_xy)
            intensidades['xy_traseira'] = np.abs(w_xy)
        elif tipo_intensidade == 'direcional':
            if direcao_vetor == 'x':
                intensidades['xy_frontal'] = np.abs(u_xy)
                intensidades['xy_traseira'] = np.abs(u_xy)
            elif direcao_vetor == 'y':
                intensidades['xy_frontal'] = np.abs(v_xy)
                intensidades['xy_traseira'] = np.abs(v_xy)
            elif direcao_vetor == 'z':
                intensidades['xy_frontal'] = np.abs(w_xy)
                intensidades['xy_traseira'] = np.abs(w_xy)

        # Para as paredes superior e inferior (plano xz)
        if campo == 'magnetico':
            u_xz = self.modo_xz.Hx_temp
            v_xz = self.modo_xz.Hz_temp
            w_xz = self.modo_xz.Hy_temp
        else:
            u_xz = self.modo_xz.Ex_temp
            v_xz = self.modo_xz.Ez_temp
            w_xz = self.modo_xz.Ey_temp

        if tipo_intensidade == 'total':
            intensidades['xz_inferior'] = np.sqrt(u_xz**2 + v_xz**2 + w_xz**2)
            intensidades['xz_superior'] = np.sqrt(u_xz**2 + v_xz**2 + w_xz**2)
        elif tipo_intensidade == 'perpendicular':
            intensidades['xz_inferior'] = np.abs(w_xz)
            intensidades['xz_superior'] = np.abs(w_xz)
        elif tipo_intensidade == 'direcional':
            if direcao_vetor == 'x':
                intensidades['xz_inferior'] = np.abs(u_xz)
                intensidades['xz_superior'] = np.abs(u_xz)
            elif direcao_vetor == 'y':
                intensidades['xz_inferior'] = np.abs(w_xz)
                intensidades['xz_superior'] = np.abs(w_xz)
            elif direcao_vetor == 'z':
                intensidades['xz_inferior'] = np.abs(v_xz)
                intensidades['xz_superior'] = np.abs(v_xz)

        # Para as paredes laterais (plano yz)
        if campo == 'magnetico':
            u_yz = self.modo_yz.Hy_temp
            v_yz = self.modo_yz.Hz_temp
            w_yz = self.modo_yz.Hx_temp
        else:
            u_yz = self.modo_yz.Ey_temp
            v_yz = self.modo_yz.Ez_temp
            w_yz = self.modo_yz.Ex_temp

        if tipo_intensidade == 'total':
            intensidades['yz_esquerda'] = np.sqrt(u_yz**2 + v_yz**2 + w_yz**2)
            intensidades['yz_direita'] = np.sqrt(u_yz**2 + v_yz**2 + w_yz**2)
        elif tipo_intensidade == 'perpendicular':
            intensidades['yz_esquerda'] = np.abs(w_yz)
            intensidades['yz_direita'] = np.abs(w_yz)
        elif tipo_intensidade == 'direcional':
            if direcao_vetor == 'x':
                intensidades['yz_esquerda'] = np.abs(w_yz)
                intensidades['yz_direita'] = np.abs(w_yz)
            elif direcao_vetor == 'y':
                intensidades['yz_esquerda'] = np.abs(u_yz)
                intensidades['yz_direita'] = np.abs(u_yz)
            elif direcao_vetor == 'z':
                intensidades['yz_esquerda'] = np.abs(v_yz)
                intensidades['yz_direita'] = np.abs(v_yz)

        return intensidades

    def animar_cavidade_3d(self, campo='magnetico', tipo_intensidade='direcional',
                          direcao_vetor='z', num_frames=60, duracao_ciclo=2.0):
        """
        Cria uma animação da cavidade 3D mostrando a evolução temporal do campo.

        Parameters:
        -----------
        campo : str
            Tipo de campo ('magnetico' ou 'eletrico')
        tipo_intensidade : str
            Tipo de intensidade ('total', 'perpendicular', ou 'direcional')
        direcao_vetor : str
            Direção do vetor ('x', 'y', 'z') quando tipo_intensidade='direcional'
        num_frames : int
            Número de frames na animação
        duracao_ciclo : float
            Duração do ciclo em segundos

        Returns:
        --------
        ani : matplotlib.animation.FuncAnimation
            Objeto de animação do matplotlib
        """

        # Configurar figura
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')

        # Converter dimensões
        largura_m = self.largura / 1000
        altura_m = self.altura / 1000
        profundidade_m = self.profundidade / 1000

        # Calcular intensidade máxima para normalização consistente
        fases_teste = np.linspace(0, 2*np.pi, 20)
        max_intensidade_global = 0

        for fase in fases_teste:
            intensidades = self._calcular_intensidades_com_fase(campo, tipo_intensidade,
                                                              direcao_vetor, fase)
            max_local = max([np.max(intensidade) for intensidade in intensidades.values()])
            max_intensidade_global = max(max_intensidade_global, max_local)

        # Lista para armazenar superfícies
        superficies = []

        def init():
            ax.clear()
            return []

        def animate(frame):
            ax.clear()

            # Calcular fase temporal para este frame
            fase_temporal = 2 * np.pi * frame / num_frames

            # Calcular intensidades para esta fase
            intensidades = self._calcular_intensidades_com_fase(campo, tipo_intensidade,
                                                              direcao_vetor, fase_temporal)

            # Plotar cada parede
            self._plotar_parede_xy_frontal_animacao(ax, intensidades['xy_frontal'],
                                                   max_intensidade_global, largura_m, altura_m, 0)
            self._plotar_parede_xy_traseira_animacao(ax, intensidades['xy_traseira'],
                                                    max_intensidade_global, largura_m, altura_m, profundidade_m)
            self._plotar_parede_xz_inferior_animacao(ax, intensidades['xz_inferior'],
                                                    max_intensidade_global, largura_m, profundidade_m, 0)
            self._plotar_parede_xz_superior_animacao(ax, intensidades['xz_superior'],
                                                    max_intensidade_global, largura_m, profundidade_m, altura_m)
            self._plotar_parede_yz_esquerda_animacao(ax, intensidades['yz_esquerda'],
                                                    max_intensidade_global, altura_m, profundidade_m, 0)
            self._plotar_parede_yz_direita_animacao(ax, intensidades['yz_direita'],
                                                   max_intensidade_global, altura_m, profundidade_m, largura_m)

            # Adicionar bordas
            self._adicionar_bordas_cavidade(ax, largura_m, altura_m, profundidade_m)

            # Configurar eixos
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
            ax.set_zlabel('Y (m)')
            ax.set_title(f'Animação - Campo {campo.capitalize()} ({tipo_intensidade}) - Modo TE{self.m}{self.n}\n'
                        f'Fase: {fase_temporal:.2f} rad, Frame: {frame+1}/{num_frames}')

            # Configurar limites e aspecto
            ax.set_xlim(-largura_m*0.1, largura_m*1.1)
            ax.set_ylim(-profundidade_m*0.1, profundidade_m*1.1)
            ax.set_zlim(-altura_m*0.1, altura_m*1.1)

            # Configurar aspecto proporcional equilibrado
            aspect_x = largura_m
            aspect_y = profundidade_m
            aspect_z = altura_m  # Usar dimensão real da altura

            max_aspect = max(aspect_x, aspect_y, aspect_z)
            ax.set_box_aspect([aspect_x/max_aspect, aspect_y/max_aspect, aspect_z/max_aspect])

            ax.view_init(elev=20, azim=45)
            ax.grid(True, alpha=0.3)

            return []

        # Criar animação
        intervalo = int(duracao_ciclo * 1000 / num_frames)  # intervalo em ms
        ani = animation.FuncAnimation(fig, animate, init_func=init,
                                     frames=num_frames, interval=intervalo,
                                     blit=False, repeat=True)

        return ani

    def _plotar_parede_xy_frontal_animacao(self, ax, intensidade, max_global, largura, altura, z_pos):
        """Plota parede frontal para animação."""
        x = np.linspace(0, largura, intensidade.shape[0])
        y = np.linspace(0, altura, intensidade.shape[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = np.full_like(X, z_pos)

        colors = plt.cm.viridis(intensidade/max_global if max_global > 0 else intensidade)
        return ax.plot_surface(X, Z, Y, facecolors=colors, alpha=0.8, shade=False)

    def _plotar_parede_xy_traseira_animacao(self, ax, intensidade, max_global, largura, altura, z_pos):
        """Plota parede traseira para animação."""
        x = np.linspace(0, largura, intensidade.shape[0])
        y = np.linspace(0, altura, intensidade.shape[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = np.full_like(X, z_pos)

        colors = plt.cm.viridis(intensidade/max_global if max_global > 0 else intensidade)
        return ax.plot_surface(X, Z, Y, facecolors=colors, alpha=0.8, shade=False)

    def _plotar_parede_xz_inferior_animacao(self, ax, intensidade, max_global, largura, profundidade, y_pos):
        """Plota parede inferior para animação."""
        x = np.linspace(0, largura, intensidade.shape[0])
        z = np.linspace(0, profundidade, intensidade.shape[1])
        X, Z = np.meshgrid(x, z, indexing='ij')
        Y = np.full_like(X, y_pos)

        colors = plt.cm.viridis(intensidade/max_global if max_global > 0 else intensidade)
        return ax.plot_surface(X, Z, Y, facecolors=colors, alpha=0.8, shade=False)

    def _plotar_parede_xz_superior_animacao(self, ax, intensidade, max_global, largura, profundidade, y_pos):
        """Plota parede superior para animação."""
        x = np.linspace(0, largura, intensidade.shape[0])
        z = np.linspace(0, profundidade, intensidade.shape[1])
        X, Z = np.meshgrid(x, z, indexing='ij')
        Y = np.full_like(X, y_pos)

        colors = plt.cm.viridis(intensidade/max_global if max_global > 0 else intensidade)
        return ax.plot_surface(X, Z, Y, facecolors=colors, alpha=0.8, shade=False)

    def _plotar_parede_yz_esquerda_animacao(self, ax, intensidade, max_global, altura, profundidade, x_pos):
        """Plota parede esquerda para animação."""
        y = np.linspace(0, altura, intensidade.shape[0])
        z = np.linspace(0, profundidade, intensidade.shape[1])
        Y, Z = np.meshgrid(y, z, indexing='ij')
        X = np.full_like(Y, x_pos)

        colors = plt.cm.viridis(intensidade/max_global if max_global > 0 else intensidade)
        return ax.plot_surface(X, Z, Y, facecolors=colors, alpha=0.8, shade=False)

    def _plotar_parede_yz_direita_animacao(self, ax, intensidade, max_global, altura, profundidade, x_pos):
        """Plota parede direita para animação."""
        y = np.linspace(0, altura, intensidade.shape[0])
        z = np.linspace(0, profundidade, intensidade.shape[1])
        Y, Z = np.meshgrid(y, z, indexing='ij')
        X = np.full_like(Y, x_pos)

        colors = plt.cm.viridis(intensidade/max_global if max_global > 0 else intensidade)
        return ax.plot_surface(X, Z, Y, facecolors=colors, alpha=0.8, shade=False)

    def animar_cavidade_pyplot(self, campo='magnetico', tipo_intensidade='direcional',
                              direcao_vetor='z', num_frames=60, duracao_ciclo=2.0):
        """
        Cria uma animação 2D mais leve usando pyplot com subplots das 6 paredes.

        Parameters:
        -----------
        campo : str
            Tipo de campo ('magnetico' ou 'eletrico')
        tipo_intensidade : str
            Tipo de intensidade ('total', 'perpendicular', ou 'direcional')
        direcao_vetor : str
            Direção do vetor ('x', 'y', 'z') quando tipo_intensidade='direcional'
        num_frames : int
            Número de frames na animação
        duracao_ciclo : float
            Duração do ciclo em segundos

        Returns:
        --------
        ani : matplotlib.animation.FuncAnimation
            Objeto de animação do matplotlib
        """

        # Configurar figura com subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Animação - Campo {campo.capitalize()} ({tipo_intensidade}) - Modo TE{self.m}{self.n} - Direção {direcao_vetor.upper()}')

        # Organizar os subplots
        ax_xy_front = axes[0, 0]    # Parede frontal (XY)
        ax_xy_back = axes[0, 1]     # Parede traseira (XY)
        ax_xz_bottom = axes[0, 2]   # Parede inferior (XZ)
        ax_xz_top = axes[1, 0]      # Parede superior (XZ)
        ax_yz_left = axes[1, 1]     # Parede esquerda (YZ)
        ax_yz_right = axes[1, 2]    # Parede direita (YZ)

        # Configurar titles
        ax_xy_front.set_title('Parede Frontal (XY, Z=0)')
        ax_xy_back.set_title('Parede Traseira (XY, Z=Prof)')
        ax_xz_bottom.set_title('Parede Inferior (XZ, Y=0)')
        ax_xz_top.set_title('Parede Superior (XZ, Y=Alt)')
        ax_yz_left.set_title('Parede Esquerda (YZ, X=0)')
        ax_yz_right.set_title('Parede Direita (YZ, X=Larg)')

        # Converter dimensões
        largura_m = self.largura / 1000
        altura_m = self.altura / 1000
        profundidade_m = self.profundidade / 1000

        # Calcular intensidade máxima para normalização consistente
        fases_teste = np.linspace(0, 2*np.pi, 10)
        max_intensidade_global = 0

        for fase in fases_teste:
            intensidades = self._calcular_intensidades_com_fase(campo, tipo_intensidade,
                                                              direcao_vetor, fase)
            max_local = max([np.max(intensidade) for intensidade in intensidades.values()])
            max_intensidade_global = max(max_intensidade_global, max_local)

        # Armazenar objetos de imagem para atualização
        im_objects = {}

        def init():
            # Inicializar com primeira fase
            intensidades = self._calcular_intensidades_com_fase(campo, tipo_intensidade,
                                                              direcao_vetor, 0)

            # Configurar cada subplot
            # Parede frontal (XY)
            x_xy = np.linspace(0, largura_m, intensidades['xy_frontal'].shape[0])
            y_xy = np.linspace(0, altura_m, intensidades['xy_frontal'].shape[1])
            extent_xy = [0, largura_m, 0, altura_m]
            im_objects['xy_front'] = ax_xy_front.imshow(intensidades['xy_frontal'].T,
                                                       extent=extent_xy, origin='lower',
                                                       cmap='viridis', vmin=0, vmax=max_intensidade_global)
            ax_xy_front.set_xlabel('X (m)')
            ax_xy_front.set_ylabel('Y (m)')

            # Parede traseira (XY)
            im_objects['xy_back'] = ax_xy_back.imshow(intensidades['xy_traseira'].T,
                                                     extent=extent_xy, origin='lower',
                                                     cmap='viridis', vmin=0, vmax=max_intensidade_global)
            ax_xy_back.set_xlabel('X (m)')
            ax_xy_back.set_ylabel('Y (m)')

            # Parede inferior (XZ)
            x_xz = np.linspace(0, largura_m, intensidades['xz_inferior'].shape[0])
            z_xz = np.linspace(0, profundidade_m, intensidades['xz_inferior'].shape[1])
            extent_xz = [0, largura_m, 0, profundidade_m]
            im_objects['xz_bottom'] = ax_xz_bottom.imshow(intensidades['xz_inferior'].T,
                                                         extent=extent_xz, origin='lower',
                                                         cmap='viridis', vmin=0, vmax=max_intensidade_global)
            ax_xz_bottom.set_xlabel('X (m)')
            ax_xz_bottom.set_ylabel('Z (m)')

            # Parede superior (XZ)
            im_objects['xz_top'] = ax_xz_top.imshow(intensidades['xz_superior'].T,
                                                   extent=extent_xz, origin='lower',
                                                   cmap='viridis', vmin=0, vmax=max_intensidade_global)
            ax_xz_top.set_xlabel('X (m)')
            ax_xz_top.set_ylabel('Z (m)')

            # Parede esquerda (YZ)
            y_yz = np.linspace(0, altura_m, intensidades['yz_esquerda'].shape[0])
            z_yz = np.linspace(0, profundidade_m, intensidades['yz_esquerda'].shape[1])
            extent_yz = [0, altura_m, 0, profundidade_m]
            im_objects['yz_left'] = ax_yz_left.imshow(intensidades['yz_esquerda'].T,
                                                     extent=extent_yz, origin='lower',
                                                     cmap='viridis', vmin=0, vmax=max_intensidade_global)
            ax_yz_left.set_xlabel('Y (m)')
            ax_yz_left.set_ylabel('Z (m)')

            # Parede direita (YZ)
            im_objects['yz_right'] = ax_yz_right.imshow(intensidades['yz_direita'].T,
                                                       extent=extent_yz, origin='lower',
                                                       cmap='viridis', vmin=0, vmax=max_intensidade_global)
            ax_yz_right.set_xlabel('Y (m)')
            ax_yz_right.set_ylabel('Z (m)')

            # Adicionar colorbar compartilhada
            cbar = fig.colorbar(im_objects['xy_front'], ax=axes.ravel().tolist(), shrink=0.6)
            cbar.set_label(f'Intensidade do Campo {campo.capitalize()}')

            plt.tight_layout()
            return list(im_objects.values())

        def animate(frame):
            # Calcular fase temporal para este frame
            fase_temporal = 2 * np.pi * frame / num_frames

            # Calcular intensidades para esta fase
            intensidades = self._calcular_intensidades_com_fase(campo, tipo_intensidade,
                                                              direcao_vetor, fase_temporal)

            # Atualizar cada imagem
            im_objects['xy_front'].set_array(intensidades['xy_frontal'].T)
            im_objects['xy_back'].set_array(intensidades['xy_traseira'].T)
            im_objects['xz_bottom'].set_array(intensidades['xz_inferior'].T)
            im_objects['xz_top'].set_array(intensidades['xz_superior'].T)
            im_objects['yz_left'].set_array(intensidades['yz_esquerda'].T)
            im_objects['yz_right'].set_array(intensidades['yz_direita'].T)

            # Atualizar título com informações do frame
            fig.suptitle(f'Animação - Campo {campo.capitalize()} ({tipo_intensidade}) - Modo TE{self.m}{self.n} - Direção {direcao_vetor.upper()}\n'
                        f'Fase: {fase_temporal:.2f} rad, Frame: {frame+1}/{num_frames}')

            return list(im_objects.values())

        # Criar animação
        intervalo = int(duracao_ciclo * 1000 / num_frames)  # intervalo em ms
        ani = animation.FuncAnimation(fig, animate, init_func=init,
                                     frames=num_frames, interval=intervalo,
                                     blit=True, repeat=True)

        return ani

    def animar_multiplos_modos(self, modos_lista, campo='magnetico', tipo_intensidade='direcional',
                              direcao_vetor='z', num_frames=60, duracao_ciclo=3.0):
        """
        Cria uma animação comparando múltiplos modos TE_mn em uma grade.

        Parameters:
        -----------
        modos_lista : list of tuples
            Lista de tuplas (m, n) representando os modos a serem comparados
            Ex: [(1,0), (2,0), (1,1), (2,1)]
        campo : str
            Tipo de campo ('magnetico' ou 'eletrico')
        tipo_intensidade : str
            Tipo de intensidade ('total', 'perpendicular', ou 'direcional')
        direcao_vetor : str
            Direção do vetor ('x', 'y', 'z') quando tipo_intensidade='direcional'
        num_frames : int
            Número de frames na animação
        duracao_ciclo : float
            Duração do ciclo em segundos

        Returns:
        --------
        ani : matplotlib.animation.FuncAnimation
            Objeto de animação do matplotlib
        """

        # Criar instâncias para cada modo
        cavidades = {}
        for m, n in modos_lista:
            cavidade = CavityWall3D(
                largura=self.largura, altura=self.altura, profundidade=self.profundidade,
                frequencia=self.frequencia, permissividade=self.permissividade,
                permeabilidade=self.permeabilidade, resolucao=self.modo_xy.pontos_por_dimensao,
                m=m, n=n
            )
            cavidades[(m, n)] = cavidade

        # Calcular layout da grade
        n_modos = len(modos_lista)
        if n_modos <= 2:
            rows, cols = 1, n_modos
        elif n_modos <= 4:
            rows, cols = 2, 2
        elif n_modos <= 6:
            rows, cols = 2, 3
        elif n_modos <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = 4, 4

        # Configurar figura
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_modos == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        fig.suptitle(f'Comparação de Modos TE_mn - Campo {campo.capitalize()} ({tipo_intensidade})', fontsize=16)

        # Calcular intensidade máxima global para normalização
        max_intensidade_global = 0
        fases_teste = np.linspace(0, 2*np.pi, 10)

        for fase in fases_teste:
            for cavidade in cavidades.values():
                intensidades = cavidade._calcular_intensidades_com_fase(campo, tipo_intensidade,
                                                                      direcao_vetor, fase)
                max_local = max([np.max(intensidade) for intensidade in intensidades.values()])
                max_intensidade_global = max(max_intensidade_global, max_local)

        # Armazenar objetos de imagem para cada modo
        im_objects = {}

        def init():
            for i, (m, n) in enumerate(modos_lista):
                if i >= len(axes):
                    break

                ax = axes[i]
                cavidade = cavidades[(m, n)]

                # Usar apenas uma parede representativa (XY frontal)
                intensidades = cavidade._calcular_intensidades_com_fase(campo, tipo_intensidade,
                                                                      direcao_vetor, 0)

                largura_m = cavidade.largura / 1000
                altura_m = cavidade.altura / 1000
                extent = [0, largura_m, 0, altura_m]

                im = ax.imshow(intensidades['xy_frontal'].T, extent=extent, origin='lower',
                              cmap='viridis', vmin=0, vmax=max_intensidade_global)

                im_objects[(m, n)] = im

                ax.set_title(f'TE{m}{n}')
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')

            # Ocultar eixos não utilizados
            for i in range(len(modos_lista), len(axes)):
                axes[i].set_visible(False)

            # Adicionar colorbar compartilhada
            if len(modos_lista) > 0:
                cbar = fig.colorbar(list(im_objects.values())[0], ax=axes[:len(modos_lista)],
                                   shrink=0.6, aspect=20, pad=0.02)
                cbar.set_label(f'Intensidade do Campo {campo.capitalize()}')

            plt.tight_layout()
            return list(im_objects.values())

        def animate(frame):
            # Calcular fase temporal para este frame
            fase_temporal = 2 * np.pi * frame / num_frames

            for (m, n), im in im_objects.items():
                cavidade = cavidades[(m, n)]
                intensidades = cavidade._calcular_intensidades_com_fase(campo, tipo_intensidade,
                                                                      direcao_vetor, fase_temporal)
                im.set_array(intensidades['xy_frontal'].T)

            # Atualizar título com informações do frame
            fig.suptitle(f'Comparação de Modos TE_mn - Campo {campo.capitalize()} ({tipo_intensidade})\n'
                        f'Direção: {direcao_vetor.upper()}, Fase: {fase_temporal:.2f} rad, Frame: {frame+1}/{num_frames}',
                        fontsize=16)

            return list(im_objects.values())

        # Criar animação
        intervalo = int(duracao_ciclo * 1000 / num_frames)
        ani = animation.FuncAnimation(fig, animate, init_func=init,
                                     frames=num_frames, interval=intervalo,
                                     blit=True, repeat=True)

        return ani

    def animar_cavidade_plotly(self, campo='magnetico', tipo_intensidade='direcional',
                              direcao_vetor='z', num_frames=60, duracao_frame=100):
        """
        Cria uma animação 3D interativa usando Plotly.

        Parameters:
        -----------
        campo : str
            Tipo de campo ('magnetico' ou 'eletrico')
        tipo_intensidade : str
            Tipo de intensidade ('total', 'perpendicular', ou 'direcional')
        direcao_vetor : str
            Direção do vetor ('x', 'y', 'z') quando tipo_intensidade='direcional'
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
        largura_m = self.largura / 1000
        altura_m = self.altura / 1000
        profundidade_m = self.profundidade / 1000

        # Calcular intensidade máxima para normalização
        fases_teste = np.linspace(0, 2*np.pi, 10)
        max_intensidade_global = 0

        for fase in fases_teste:
            intensidades = self._calcular_intensidades_com_fase(campo, tipo_intensidade,
                                                              direcao_vetor, fase)
            max_local = max([np.max(intensidade) for intensidade in intensidades.values()])
            max_intensidade_global = max(max_intensidade_global, max_local)

        # Preparar dados para animação
        frames_data = []

        for frame in range(num_frames):
            fase_temporal = 2 * np.pi * frame / num_frames
            intensidades = self._calcular_intensidades_com_fase(campo, tipo_intensidade,
                                                              direcao_vetor, fase_temporal)

            frame_surfaces = []

            # Parede frontal (XY, Z=0)
            x_xy = np.linspace(0, largura_m, intensidades['xy_frontal'].shape[0])
            y_xy = np.linspace(0, altura_m, intensidades['xy_frontal'].shape[1])
            X_xy, Y_xy = np.meshgrid(x_xy, y_xy, indexing='ij')
            Z_xy = np.zeros_like(X_xy)

            frame_surfaces.append(
                go.Surface(
                    x=X_xy, y=Z_xy, z=Y_xy,
                    surfacecolor=intensidades['xy_frontal'],
                    cmin=0, cmax=max_intensidade_global,
                    colorscale='Viridis',
                    showscale=False,
                    name='Parede Frontal'
                )
            )

            # Parede traseira (XY, Z=profundidade)
            Z_xy_back = np.full_like(X_xy, profundidade_m)
            frame_surfaces.append(
                go.Surface(
                    x=X_xy, y=Z_xy_back, z=Y_xy,
                    surfacecolor=intensidades['xy_traseira'],
                    cmin=0, cmax=max_intensidade_global,
                    colorscale='Viridis',
                    showscale=False,
                    name='Parede Traseira'
                )
            )

            # Parede inferior (XZ, Y=0)
            x_xz = np.linspace(0, largura_m, intensidades['xz_inferior'].shape[0])
            z_xz = np.linspace(0, profundidade_m, intensidades['xz_inferior'].shape[1])
            X_xz, Z_xz = np.meshgrid(x_xz, z_xz, indexing='ij')
            Y_xz = np.zeros_like(X_xz)

            frame_surfaces.append(
                go.Surface(
                    x=X_xz, y=Z_xz, z=Y_xz,
                    surfacecolor=intensidades['xz_inferior'],
                    cmin=0, cmax=max_intensidade_global,
                    colorscale='Viridis',
                    showscale=False,
                    name='Parede Inferior'
                )
            )

            # Parede superior (XZ, Y=altura)
            Y_xz_top = np.full_like(X_xz, altura_m)
            frame_surfaces.append(
                go.Surface(
                    x=X_xz, y=Z_xz, z=Y_xz_top,
                    surfacecolor=intensidades['xz_superior'],
                    cmin=0, cmax=max_intensidade_global,
                    colorscale='Viridis',
                    showscale=False,
                    name='Parede Superior'
                )
            )

            # Parede esquerda (YZ, X=0)
            y_yz = np.linspace(0, altura_m, intensidades['yz_esquerda'].shape[0])
            z_yz = np.linspace(0, profundidade_m, intensidades['yz_esquerda'].shape[1])
            Y_yz, Z_yz = np.meshgrid(y_yz, z_yz, indexing='ij')
            X_yz = np.zeros_like(Y_yz)

            frame_surfaces.append(
                go.Surface(
                    x=X_yz, y=Z_yz, z=Y_yz,
                    surfacecolor=intensidades['yz_esquerda'],
                    cmin=0, cmax=max_intensidade_global,
                    colorscale='Viridis',
                    showscale=False,
                    name='Parede Esquerda'
                )
            )

            # Parede direita (YZ, X=largura)
            X_yz_right = np.full_like(Y_yz, largura_m)
            frame_surfaces.append(
                go.Surface(
                    x=X_yz_right, y=Z_yz, z=Y_yz,
                    surfacecolor=intensidades['yz_direita'],
                    cmin=0, cmax=max_intensidade_global,
                    colorscale='Viridis',
                    showscale=frame == 0,  # Mostrar colorbar apenas no primeiro frame
                    colorbar=dict(title=f"Intensidade {campo.capitalize()}", len=0.7) if frame == 0 else None,
                    name='Parede Direita'
                )
            )

            frames_data.append(frame_surfaces)

        # Criar figura com primeiro frame
        fig = go.Figure(data=frames_data[0])

        # Configurar layout
        fig.update_layout(
            title=f'Animação 3D - Campo {campo.capitalize()} ({tipo_intensidade}) - Modo TE{self.m}{self.n}',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Z (m)',
                zaxis_title='Y (m)',
                camera=dict(
                    eye=dict(x=0.06, y=0.08, z=0.04)  # Valores menores = mais zoom
                ),
                aspectmode='manual',
                aspectratio=dict(
                    x=largura_m,
                    y=altura_m,
                    z=profundidade_m
                ),
                xaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230,230)"),
                yaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230,230)"),
                zaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230,230)")
            ),
            margin=dict(l=20, r=20, t=60, b=20),
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
            fase_temporal = 2 * np.pi * i / num_frames
            frames.append(
                go.Frame(
                    data=frame_data,
                    name=str(i),
                    layout=go.Layout(
                        title=f'Animação 3D - Campo {campo.capitalize()} ({tipo_intensidade}) - Modo TE{self.m}{self.n}<br>'
                              f'Fase: {fase_temporal:.2f} rad, Frame: {i+1}/{num_frames}'
                    )
                )
            )

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

    def salvar_animacao_plotly_html(self, nome_arquivo='animacao_3d_plotly.html', **kwargs):
        """
        Salva a animação Plotly como arquivo HTML interativo.

        Parameters:
        -----------
        nome_arquivo : str
            Nome do arquivo HTML a ser salvo
        **kwargs : dict
            Argumentos para animar_cavidade_plotly()
        """
        fig = self.animar_cavidade_plotly(**kwargs)
        fig.write_html(nome_arquivo)
        print(f"Animação salva como: {nome_arquivo}")
        return fig

    def criar_gif_plotly(self, nome_arquivo='animacao_plotly.gif',
                        campo='magnetico', tipo_intensidade='direcional',
                        direcao_vetor='z', num_frames=30, duracao_frame=200,
                        width=800, height=600, fps=5):
        """
        Cria um GIF a partir da animação Plotly.

        Parameters:
        -----------
        nome_arquivo : str
            Nome do arquivo GIF a ser salvo
        campo : str
            Tipo de campo ('magnetico' ou 'eletrico')
        tipo_intensidade : str
            Tipo de intensidade ('total', 'perpendicular', ou 'direcional')
        direcao_vetor : str
            Direção do vetor ('x', 'y', 'z')
        num_frames : int
            Número de frames na animação
        duracao_frame : int
            Duração de cada frame em ms (para HTML)
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

        # Converter dimensões
        largura_m = self.largura / 1000
        altura_m = self.altura / 1000
        profundidade_m = self.profundidade / 1000

        # Calcular intensidade máxima para normalização
        fases_teste = np.linspace(0, 2*np.pi, 10)
        max_intensidade_global = 0

        for fase in fases_teste:
            intensidades = self._calcular_intensidades_com_fase(campo, tipo_intensidade,
                                                              direcao_vetor, fase)
            max_local = max([np.max(intensidade) for intensidade in intensidades.values()])
            max_intensidade_global = max(max_intensidade_global, max_local)

        # Lista para armazenar as imagens
        images = []

        print(f"Gerando {num_frames} frames para o GIF...")

        for frame in range(num_frames):
            fase_temporal = 2 * np.pi * frame / num_frames
            intensidades = self._calcular_intensidades_com_fase(campo, tipo_intensidade,
                                                              direcao_vetor, fase_temporal)

            # Criar figura estática para este frame
            fig = go.Figure()

            # Adicionar todas as 6 paredes
            # Parede frontal (XY, Z=0)
            x_xy = np.linspace(0, largura_m, intensidades['xy_frontal'].shape[0])
            y_xy = np.linspace(0, altura_m, intensidades['xy_frontal'].shape[1])
            X_xy, Y_xy = np.meshgrid(x_xy, y_xy, indexing='ij')
            Z_xy = np.zeros_like(X_xy)

            fig.add_trace(go.Surface(
                x=X_xy, y=Z_xy, z=Y_xy,
                surfacecolor=intensidades['xy_frontal'],
                cmin=0, cmax=max_intensidade_global,
                colorscale='Viridis',
                showscale=False,  # Mostrar colorbar apenas no primeiro frame
                colorbar=dict(title=f"Intensidade {campo.capitalize()}", len=0.7) if frame == 0 else None,
                name='Frontal'
            ))

            # Parede traseira (XY, Z=profundidade)
            Z_xy_back = np.full_like(X_xy, profundidade_m)
            fig.add_trace(go.Surface(
                x=X_xy, y=Z_xy_back, z=Y_xy,
                surfacecolor=intensidades['xy_traseira'],
                cmin=0, cmax=max_intensidade_global,
                colorscale='Viridis',
                showscale=False,
                name='Traseira'
            ))

            # Parede inferior (XZ, Y=0)
            x_xz = np.linspace(0, largura_m, intensidades['xz_inferior'].shape[0])
            z_xz = np.linspace(0, profundidade_m, intensidades['xz_inferior'].shape[1])
            X_xz, Z_xz = np.meshgrid(x_xz, z_xz, indexing='ij')
            Y_xz = np.zeros_like(X_xz)

            fig.add_trace(go.Surface(
                x=X_xz, y=Z_xz, z=Y_xz,
                surfacecolor=intensidades['xz_inferior'],
                cmin=0, cmax=max_intensidade_global,
                colorscale='Viridis',
                showscale=False,
                name='Inferior'
            ))

            # Parede superior (XZ, Y=altura)
            Y_xz_top = np.full_like(X_xz, altura_m)
            fig.add_trace(go.Surface(
                x=X_xz, y=Z_xz, z=Y_xz_top,
                surfacecolor=intensidades['xz_superior'],
                cmin=0, cmax=max_intensidade_global,
                colorscale='Viridis',
                showscale=False,
                name='Superior'
            ))

            # Parede esquerda (YZ, X=0)
            y_yz = np.linspace(0, altura_m, intensidades['yz_esquerda'].shape[0])
            z_yz = np.linspace(0, profundidade_m, intensidades['yz_esquerda'].shape[1])
            Y_yz, Z_yz = np.meshgrid(y_yz, z_yz, indexing='ij')
            X_yz = np.zeros_like(Y_yz)

            fig.add_trace(go.Surface(
                x=X_yz, y=Z_yz, z=Y_yz,
                surfacecolor=intensidades['yz_esquerda'],
                cmin=0, cmax=max_intensidade_global,
                colorscale='Viridis',
                showscale=False,
                name='Esquerda'
            ))

            # Parede direita (YZ, X=largura)
            X_yz_right = np.full_like(Y_yz, largura_m)
            fig.add_trace(go.Surface(
                x=X_yz_right, y=Z_yz, z=Y_yz,
                surfacecolor=intensidades['yz_direita'],
                cmin=0, cmax=max_intensidade_global,
                colorscale='Viridis',
                showscale=False,
                name='Direita'
            ))

            # Configurar layout
            fig.update_layout(
                title=dict(
                    text=f'Campo {campo.capitalize()} ({tipo_intensidade}) - Modo TE{self.m}{self.n}<br>'
                         f'Fase: {fase_temporal:.2f} rad, Frame: {frame+1}/{num_frames}',
                    x=0.5,
                    y=0.95,
                    font=dict(size=14)
                ),
                scene=dict(
                    xaxis_title='X (m)',
                    yaxis_title='Z (m)',
                    zaxis_title='Y (m)',
                    camera=dict(
                        eye=dict(x=0.06, y=0.08, z=0.04)  # Valores menores = mais zoom
                    ),
                    aspectmode='manual',
                    aspectratio=dict(
                        x=largura_m,
                        y=profundidade_m,
                        z=altura_m
                    ),
                    xaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230,230)"),
                    yaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230,230)"),
                    zaxis=dict(showbackground=True, backgroundcolor="rgb(230, 230,230)")
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


if __name__ == "__main__":
    # Criar instância da cavidade 3D com modo específico
    cavidade = CavityWall3D(largura=22.86, altura=10.16, profundidade=100,
                           frequencia=12*10**9, permissividade=1, permeabilidade=1,
                           resolucao=40, m=1, n=2)  # Modo TE11

    # Debug das dimensões
    # cavidade.debug_dimensoes()

    # Exemplos de uso com diferentes modos:

    # 1. Imagens estáticas (comentadas para focar na animação)
    # fig1 = cavidade.plota_cavidade_3d_com_intensidade(campo='magnetico', tipo_intensidade='total')
    # fig2 = cavidade.plota_cavidade_3d_com_intensidade(campo='magnetico', tipo_intensidade='perpendicular')
    # fig3 = cavidade.plota_cavidade_3d_com_intensidade(campo='magnetico', tipo_intensidade='direcional', direcao_vetor='z')

    # 2. ANIMAÇÃO 3D MATPLOTLIB (mais pesada - comentada)
    # ani_3d = cavidade.animar_cavidade_3d(campo='eletrico', tipo_intensidade='direcional', direcao_vetor='y', num_frames=20, duracao_ciclo=20.0)

    # 3. ANIMAÇÃO 2D (LEVE) - Evolução temporal com pyplot (comentada)
    # ani = cavidade.animar_cavidade_pyplot(campo='magnetico', tipo_intensidade='direcional', direcao_vetor='x', num_frames=60, duracao_ciclo=3.0)

    # 4. ANIMAÇÃO 3D PLOTLY - INTERATIVA E EFICIENTE! (comentada)
    # fig_plotly = cavidade.animar_cavidade_plotly(
    #     campo='magnetico',
    #     tipo_intensidade='direcional',
    #     direcao_vetor='z',
    #     num_frames=30,          # Menos frames para Plotly
    #     duracao_frame=200       # 200ms por frame
    # )
    # fig_plotly.write_html('animacao_3d_interativa.html')
    # fig_plotly.show()

    # 5. CRIAR GIF DA ANIMAÇÃO PLOTLY - MELHOR QUALIDADE!
    gif_path = cavidade.criar_gif_plotly(
        nome_arquivo='animacao_plotly_3d.gif',
        campo='magnetico',
        tipo_intensidade='direcional',
        direcao_vetor='z',
        num_frames=60,          # Frames para GIF
        width=1000,             # Largura em pixels
        height=800,             # Altura em pixels
        fps=20                  # 60 frames por segundo
    )

    print(f"GIF criado: {gif_path}")

    # 6. ANIMAÇÃO COMPARANDO MÚLTIPLOS MODOS (comentada)
    # modos_para_comparar = [(1,0), (2,0), (1,1), (2,1)]  # TE10, TE20, TE11, TE21
    # ani_multiplos = cavidade.animar_multiplos_modos(modos_lista=modos_para_comparar, campo='magnetico', tipo_intensidade='direcional', direcao_vetor='z', num_frames=60, duracao_ciclo=4.0)

    # Opções para diferentes configurações de GIF:
    # cavidade.criar_gif_plotly('gif_baixa_res.gif', num_frames=15, width=600, height=400, fps=3)  # Arquivo menor
    # cavidade.criar_gif_plotly('gif_alta_res.gif', num_frames=30, width=1200, height=900, fps=6)  # Alta qualidade