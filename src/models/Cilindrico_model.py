import numpy as np
import cmath
import matplotlib.pyplot as plt
from scipy.special import jv, jvp  # Bessel functions and their derivatives
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
class Modo_Cilindrico():
    def __init__(self, 
                 raio=0.05, 
                 frequencia=12e9, 
                 permissividade=1, 
                 permeabilidade=1, 
                 n=0, m=1, z=0):
        self.m = m  # Ordem do modo -> O m é de 1 a 3 na tabela, mas internamente é de 0 a 2
        self.n = n  # Número do modo
        self.raio = raio  # Raio do cilindro (m)
        self.frequencia = frequencia  # Frequência (Hz)
        self.mu =  permeabilidade # Permissividade relativa 
        self.epsilon = permissividade # Permeabilidade relativa
        self.A = 1  # Amplitude
        self.B = 1  # Amplitude
        self.pi = np.pi
        self.num_planos = 15  # Número de planos Z para o meshgrid 3D
        self.pontos_por_dimensao = 30  # Número de pontos por dimensão para o meshgrid
        self.light_speed = 299792458  # Velocidade da luz (m/s)
        self.vacuo = False  # Flag para indicar se o meio é vácuo
        self.omega_0 = self.omega()  # Frequência angular
        self.k_val = self.k()  # Número de onda
        self.k_c_val = self.k_c()  # Número de onda de corte
        self.k_c_val_tm = self.k_c(tm=True)  # Número de onda de corte para TM
        self.beta_val = self.beta()  # Constante de fase
        self.z = z
        self.exp_z_val = self.exp_z(z)  # Renomeado para exp_z_val

    def obter_pnm(self, n, coluna):
        # Tabela de valores de P_nm
        pnm_tabela = {
        #   n   m=1,   m=2,    m=3
            0: [2.405, 5.520, 8.654],  # n = 0
            1: [3.832, 7.016, 10.174],  # n = 1
            2: [5.135, 8.417, 11.620]   # n = 2
        }
        
        # Verifica se o valor de n está na tabela
        if n not in pnm_tabela:
            raise ValueError(f"Valor de n={n} não está na tabela.")

        # Verifica se a coluna é válida (0, 1 ou 2)
        if coluna < 0 or coluna > 2:
            raise ValueError(f"Coluna inválida: {coluna}. Deve ser 1, 2 ou 3.")

        # Retorna o valor correspondente
        print(f"Obtendo P'_{n}{coluna+1} = {pnm_tabela[n][coluna]}")  
        return pnm_tabela[n][coluna]

    def obter_pnm_prime(self, n, coluna):
        # Tabela de valores de P'_nm
        pnm_tabela = {
        #   n   m=1,   m=2,    m=3
            0: [1.841, 7.016, 10.174],  # n = 0
            1: [3.832, 5.331, 8.536],   # n = 1
            2: [3.054, 6.706, 9.970]    # n = 2
        }
        
        # Verifica se o valor de n está na tabela
        if n not in pnm_tabela:
            raise ValueError(f"Valor de n={n} não está na tabela.")
        
        # Verifica se a coluna é válida (1, 2 ou 3)
        if coluna < 0 or coluna > 2:
            raise ValueError(f"Coluna inválida: {coluna}. Deve ser 1, 2 ou 3.")
        

        # Retorna o valor correspondente
        print(f"Obtendo P'_{n}{coluna+1} = {pnm_tabela[n][coluna]}")  # Linha de debug
        return pnm_tabela[n][coluna]    
    
    def cosseno_Nphi(self, phi):
        return np.cos(self.n * phi)
    
    def seno_Nphi(self, phi):
        return np.sin(self.n * phi)
    
    def exp_z(self, z):
        return np.exp(-1j * self.beta_val * z)

    def update_z(self, z):
        """Atualiza o valor de z e recalcula exp_z"""
        self.z = z
        self.exp_z_val = self.exp_z(z)

    def jv_n(self, rho):
        return jv(self.n, self.k_c_val * rho)

    def jv_n_prime(self,rho):
        return jvp(self.n, self.k_c_val * rho)


    def omega(self):
        return self.frequencia*2*self.pi

    def beta(self): # Constante de fase
        return cmath.sqrt(self.k_val**2-self.k_c_val**2)
    
    def k(self):
        if self.vacuo == True:
            return self.omega_0*np.sqrt(1/self.light_speed**2) # Mu0 * Epsilon0 = 1/c^2 No vacuo
        else:
            return self.omega_0*np.sqrt(self.mu*self.epsilon)

    def k_c(self,tm = False):
        if tm == True:
            return self.obter_pnm(self.n, self.m-1) / self.raio
        else:
            return self.obter_pnm_prime(self.n, self.m-1) / self.raio

    def TM_E_Z(self, rho, phi):
        seno = self.seno_Nphi(phi)
        cosseno = self.cosseno_Nphi(phi)
        jv_n = self.jv_n(rho)
        return np.real((self.A*seno+self.B*cosseno)*jv_n*self.exp_z_val)    

    def TE_H_Z(self, rho, phi):
        seno = self.seno_Nphi(phi)
        cosseno = self.cosseno_Nphi(phi)
        jv_n = self.jv_n(rho)
        return np.real((self.A*cosseno+self.B*seno)*jv_n*self.exp_z_val)

    def TM_E_rho(self, rho, phi):
        const = -1j*self.beta_val/(self.k_c_val_tm)
        seno = self.seno_Nphi(phi)
        cosseno = self.cosseno_Nphi(phi)
        jv_n = self.jv_n_prime(rho)
        return np.real(const*(self.A*seno + self.B*cosseno)*jv_n*self.exp_z_val)

    def TM_E_phi(self, rho, phi):
        const = -1j*self.beta_val*self.n/(self.k_c_val_tm**2 *rho)
        seno = self.seno_Nphi(phi)
        cosseno = self.cosseno_Nphi(phi)
        jv_n = self.jv_n(rho)
        return np.real(const*(self.A*cosseno - self.B*seno)*jv_n*self.exp_z_val)
    
    def TM_H_rho(self, rho, phi):
        const = 1j*self.omega_0*self.epsilon*self.n/((self.k_c_val_tm**2)*rho)
        seno = self.seno_Nphi(phi)
        cosseno = self.cosseno_Nphi(phi)
        jv_n = self.jv_n(rho)
        return np.real(const*(self.A*cosseno-self.B*seno)*jv_n*self.exp_z_val)

    def TM_H_phi(self, rho, phi):
        const = -1j*self.omega_0*self.epsilon/self.k_c_val_tm
        seno = self.seno_Nphi(phi)
        cosseno = self.cosseno_Nphi(phi)
        jv_n = self.jv_n_prime(rho)
        return np.real(const*(self.A*seno + self.B*cosseno)*jv_n*self.exp_z_val)

    def TE_E_rho(self, rho, phi):
        const = -1j*self.omega_0*self.mu/(self.k_c_val**2 *rho)
        seno = self.seno_Nphi(phi)
        cosseno = self.cosseno_Nphi(phi)
        jv_n = self.jv_n(rho)
        return np.real(const*(self.A*cosseno-self.B*seno)*jv_n*self.exp_z_val)
    
    def TE_E_phi(self, rho, phi):
        const = 1j*self.omega_0*self.mu/(self.k_c_val)
        seno = self.seno_Nphi(phi)
        cosseno = self.cosseno_Nphi(phi)
        jv_n = self.jv_n_prime(rho)
        return np.real(const*(self.A*seno+self.B*cosseno)*jv_n*self.exp_z_val)
    
    def TE_H_rho(self, rho, phi):
        const = -1j*self.beta_val/(self.k_c_val)
        seno = self.seno_Nphi(phi)
        cosseno = self.cosseno_Nphi(phi)
        jv_n = self.jv_n_prime(rho)
        return np.real(const*(self.A*cosseno+self.B*seno)*jv_n*self.exp_z_val)
    
    def TE_H_phi(self, rho, phi):
        const = -1j*self.beta_val*self.n/(self.k_c_val**2 *rho)
        seno = self.seno_Nphi(phi)
        cosseno = self.cosseno_Nphi(phi)
        jv_n = self.jv_n(rho)
        return np.real(const*(self.A*cosseno-self.B*seno)*jv_n*self.exp_z_val)
    
    def criar_meshgrid_cartesiano(self):
        # Define os limites para X e Y
        x = np.linspace(-self.raio, self.raio, self.pontos_por_dimensao)  # Espaçamento uniforme em X
        y = np.linspace(-self.raio, self.raio, self.pontos_por_dimensao)  # Espaçamento uniforme em Y

        # Cria o meshgrid em coordenadas cartesianas
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Converte para coordenadas cilíndricas
        Rho = np.sqrt(X**2 + Y**2)  # Distância radial
        Phi = np.arctan2(Y, X)      # Ângulo em radianos

        # Filtra os pontos fora do raio do cilindro
        mask = Rho <= self.raio
        X = X[mask]
        Y = Y[mask]
        Rho = Rho[mask]
        Phi = Phi[mask]

        return X, Y, Rho, Phi

    def criar_meshgrid_cartesiano_com_z(self, comprimento=0.1):
        # Ajusta o número de pontos por dimensão com base no comprimento e num_planos
        self.pontos_por_dimensao = 4

        # Define os limites para X e Y

        x = np.arange(-self.raio, self.raio, self.raio/self.pontos_por_dimensao)  # Espaçamento uniforme em X
        y = np.arange(-self.raio, self.raio, self.raio/self.pontos_por_dimensao)  # Espaçamento uniforme em Y

        # Cria o meshgrid em coordenadas cartesianas
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Converte para coordenadas cilíndricas
        Rho = np.sqrt(X**2 + Y**2)  # Distância radial
        Phi = np.arctan2(Y, X)      # Ângulo em radianos

        # Filtra os pontos fora do raio do cilindro
        mask = Rho <= self.raio
        X = X[mask]
        Y = Y[mask]
        Rho = Rho[mask]
        Phi = Phi[mask]

        # Cria os valores de Z igualmente espaçados
        Z = np.arange(0, comprimento, self.raio/self.pontos_por_dimensao)  # Espaçamento uniforme em Z

        return X, Y, Rho, Phi, Z

    def plot_vetores_de_campo(self, X, Y, Rho, Phi, transversal='TE', campo='eletrico'):
        """
        Plota os vetores de campo no plano XY com controle deslizante para z.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.25)  # Fazer espaço para o slider

        # Criar eixos para o slider
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        z_slider = Slider(
            ax=ax_slider,
            label='z (m)',
            valmin=0,
            valmax=10,
            valinit=self.z,
        )
        
        def update(val):
            ax.clear()
            self.update_z(z_slider.val)
            
            # Calcula os vetores de campo
            if transversal == 'TE':
                if campo == 'eletrico':
                    rho = self.TE_E_rho(rho=Rho, phi=Phi)
                    phi = self.TE_E_phi(rho=Rho, phi=Phi)
                elif campo == 'magnetico':
                    rho = self.TE_H_rho(rho=Rho, phi=Phi)
                    phi = self.TE_H_phi(rho=Rho, phi=Phi)
            elif transversal == 'TM':
                if campo == 'eletrico':
                    rho = self.TM_E_rho(rho=Rho, phi=Phi)
                    phi = self.TM_E_phi(rho=Rho, phi=Phi)
                elif campo == 'magnetico':
                    rho = self.TM_H_rho(rho=Rho, phi=Phi)
                    phi = self.TM_H_phi(rho=Rho, phi=Phi)

            # Converte para coordenadas cartesianas
            e_x = rho * np.cos(Phi) - phi * np.sin(Phi)
            e_y = rho * np.sin(Phi) + phi * np.cos(Phi)

            # Filtra pontos dentro do círculo
            mask = Rho <= self.raio
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
            scale_factor = self.raio * 0.1  # 10% do raio para tamanho consistente
            e_x_display = e_x_normalized * scale_factor
            e_y_display = e_y_normalized * scale_factor

            # Plota os vetores com colorbar
            norm = plt.Normalize(vmin=0, vmax=max_magnitude)
            cmap = plt.cm.viridis

            quiver = ax.quiver(X_masked, Y_masked, e_x_display, e_y_display,
                              magnitude, cmap=cmap, norm=norm,
                              scale=1, scale_units='xy', angles='xy',
                              pivot='middle', alpha=0.8)

            # Adicionar colorbar se ainda não existe
            if not hasattr(ax, '_colorbar_slider'):
                cbar = plt.colorbar(quiver, ax=ax, shrink=0.8, aspect=20)
                cbar.set_label('Intensidade do Campo', rotation=270, labelpad=20)
                ax._colorbar_slider = cbar

            circle = plt.Circle((0, 0), self.raio, color='red', fill=False, linestyle='--', linewidth=1.5)
            ax.add_patch(plt.Circle((0, 0), self.raio, color='lightgray', alpha=0.5, zorder=0))
            ax.add_artist(circle)
            
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(f"Vetores de Campo {campo.capitalize()} no Plano XY (z={z_slider.val:.3f}m)")
            ax.axis('equal')
            fig.canvas.draw_idle()

        z_slider.on_changed(update)
        update(self.z)  # Plot inicial
        plt.show()

    def plot_vetores_de_campo_animado(self, X, Y, Rho, Phi, transversal='TE', campo='eletrico', frames=100, interval=50):
        """
        Cria uma animação dos vetores de campo variando z.
        
        Args:
            frames: número de frames da animação
            interval: intervalo entre frames em milissegundos
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        z_min, z_max = 0.1, 5  # Limites de z em metros
        z_vals = np.linspace(z_min, z_max, frames)
        
        def animate(frame):
            ax.clear()
            z = z_vals[frame]
            self.update_z(z)
            
            if transversal == 'TE':
                if campo == 'eletrico':
                    rho = self.TE_E_rho(rho=Rho, phi=Phi)
                    phi = self.TE_E_phi(rho=Rho, phi=Phi)
                elif campo == 'magnetico':
                    rho = self.TE_H_rho(rho=Rho, phi=Phi)
                    phi = self.TE_H_phi(rho=Rho, phi=Phi)
            elif transversal == 'TM':
                if campo == 'eletrico':
                    rho = self.TM_E_rho(rho=Rho, phi=Phi)
                    phi = self.TM_E_phi(rho=Rho, phi=Phi)
                elif campo == 'magnetico':
                    rho = self.TM_H_rho(rho=Rho, phi=Phi)
                    phi = self.TM_H_phi(rho=Rho, phi=Phi)

            # Converte para coordenadas cartesianas
            e_x = rho * np.cos(Phi) - phi * np.sin(Phi)
            e_y = rho * np.sin(Phi) + phi * np.cos(Phi)

            # Filtra pontos dentro do círculo
            mask = Rho <= self.raio
            X_masked = X[mask]
            Y_masked = Y[mask]
            e_x = e_x[mask]
            e_y = e_y[mask]

            # Plota os vetores
            ax.quiver(X_masked, Y_masked, e_x, e_y, color='blue', scale=100000, pivot='middle')
            circle = plt.Circle((0, 0), self.raio, color='red', fill=False, linestyle='--', linewidth=1.5)
            ax.add_patch(plt.Circle((0, 0), self.raio, color='lightgray', alpha=0.5, zorder=0))
            ax.add_artist(circle)
            
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(f"Vetores de Campo {campo.capitalize()} no Plano XY (z={round(z,2)}m)")
            ax.axis('equal')
            ax.set_xlim(-self.raio*1.2, self.raio*1.2)
            ax.set_ylim(-self.raio*1.2, self.raio*1.2)

        anim = FuncAnimation(fig, animate, frames=frames, interval=interval, repeat=True)
        plt.show()
        return anim

    def plot_vetores_de_campo_fase_animado(self, X, Y, Rho, Phi, transversal='TE', campo='eletrico', frames=60, interval=100, z_fixo=0.1):
        """
        Cria uma animação dos vetores de campo variando a fase em φ.
        
        Args:
            frames: número de frames da animação
            interval: intervalo entre frames em milissegundos
            z_fixo: posição Z fixa para visualização
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Fixa z e varia a fase em φ de 0 a 2π
        self.update_z(z_fixo)
        fase_phi_vals = np.linspace(0, 2*np.pi, frames)
        
        def animate(frame):
            ax.clear()
            fase_phi = fase_phi_vals[frame]
            
            # Aplica a fase adicional em φ
            Phi_com_fase = Phi + fase_phi
            
            if transversal == 'TE':
                if campo == 'eletrico':
                    rho = self.TE_E_rho(rho=Rho, phi=Phi_com_fase)
                    phi = self.TE_E_phi(rho=Rho, phi=Phi_com_fase)
                elif campo == 'magnetico':
                    rho = self.TE_H_rho(rho=Rho, phi=Phi_com_fase)
                    phi = self.TE_H_phi(rho=Rho, phi=Phi_com_fase)
            elif transversal == 'TM':
                if campo == 'eletrico':
                    rho = self.TM_E_rho(rho=Rho, phi=Phi_com_fase)
                    phi = self.TM_E_phi(rho=Rho, phi=Phi_com_fase)
                elif campo == 'magnetico':
                    rho = self.TM_H_rho(rho=Rho, phi=Phi_com_fase)
                    phi = self.TM_H_phi(rho=Rho, phi=Phi_com_fase)

            # Converte para coordenadas cartesianas
            e_x = rho * np.cos(Phi) - phi * np.sin(Phi)
            e_y = rho * np.sin(Phi) + phi * np.cos(Phi)

            # Filtra pontos dentro do círculo
            mask = Rho <= self.raio
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
            scale_factor = self.raio * 0.1  # 10% do raio para tamanho consistente
            e_x_display = e_x_normalized * scale_factor
            e_y_display = e_y_normalized * scale_factor

            # Plota os vetores
            # Criar mapeamento de cores baseado na magnitude original
            norm = plt.Normalize(vmin=0, vmax=max_magnitude)
            cmap = plt.cm.viridis

            quiver = ax.quiver(X_masked, Y_masked, e_x_display, e_y_display,
                              magnitude, cmap=cmap, norm=norm,
                              scale=1, scale_units='xy', angles='xy',
                              pivot='middle', alpha=0.8)

            # Adicionar colorbar
            if not hasattr(ax, '_colorbar_added'):
                cbar = plt.colorbar(quiver, ax=ax, shrink=0.8, aspect=20)
                cbar.set_label('Intensidade do Campo', rotation=270, labelpad=20)
                ax._colorbar_added = True

            circle = plt.Circle((0, 0), self.raio, color='red', fill=False, linestyle='--', linewidth=1.5)
            ax.add_patch(plt.Circle((0, 0), self.raio, color='lightgray', alpha=0.5, zorder=0))
            ax.add_artist(circle)

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(f"Campo {campo.capitalize()} {transversal}{self.n}{self.m} - Fase φ={fase_phi:.2f}rad (z={z_fixo}m)")
            ax.axis('equal')
            ax.set_xlim(-self.raio*1.2, self.raio*1.2)
            ax.set_ylim(-self.raio*1.2, self.raio*1.2)

        anim = FuncAnimation(fig, animate, frames=frames, interval=interval, repeat=True)
        return anim
    
    def salvar_animacao_gif(self, X, Y, Rho, Phi, transversal='TE', campo='eletrico', 
                           frames=60, interval=100, z_fixo=0.1, nome_arquivo='campo_animado.gif'):
        """
        Cria e salva uma animação GIF dos vetores de campo variando a fase em φ.
        """
        anim = self.plot_vetores_de_campo_fase_animado(X, Y, Rho, Phi, transversal, campo, frames, interval, z_fixo)
        
        # Salva como GIF
        try:
            anim.save(nome_arquivo, writer='pillow', fps=1000//interval)
            print(f"Animação salva como: {nome_arquivo}")
        except Exception as e:
            print(f"Erro ao salvar GIF: {e}")
            print("Tentando salvar como MP4...")
            try:
                anim.save(nome_arquivo.replace('.gif', '.mp4'), writer='ffmpeg', fps=1000//interval)
                print(f"Animação salva como: {nome_arquivo.replace('.gif', '.mp4')}")
            except Exception as e2:
                print(f"Erro ao salvar MP4: {e2}")
        
        plt.show()
        return anim

    def plot_vetores_3D(self, X, Y, Rho, Phi, Z, transversal='TE', campo='eletrico', comprimento=0.1):
        """
        Plota os vetores de campo em 3D dentro de um cilindro com a superfície colorida
        pela média do módulo dos vetores em cada plano XY, utilizando Plotly.
        
        Args:
            X, Y, Rho, Phi, Z: Coordenadas do meshgrid
            transversal: 'TE' ou 'TM'
            campo: 'eletrico' ou 'magnetico'
            num_planos: número de planos XY ao longo do eixo Z
        """

        # Definir planos Z
        z_vals = Z  # Utiliza os valores de Z fornecidos
        vectors = []

        # Criar linhas de campo (Streamtube)
        u_field = np.zeros_like(X)  # Inicializar componentes do campo
        v_field = np.zeros_like(Y)
        w_field = np.zeros_like(X)  # Componente Z

        Rho = np.where(Rho == 0, 1e-12, Rho)

        # Processar vetores em cada plano Z
        for z in z_vals:
            self.update_z(z)
            
            if transversal == 'TE':
                if campo == 'eletrico':
                    rho = self.TE_E_rho(rho=Rho, phi=Phi)
                    phi = self.TE_E_phi(rho=Rho, phi=Phi)
                elif campo == 'magnetico':
                    rho = self.TE_H_rho(rho=Rho, phi=Phi)
                    phi = self.TE_H_phi(rho=Rho, phi=Phi)
                    e_z = self.TE_H_Z(rho=Rho, phi=Phi)
            elif transversal == 'TM':
                if campo == 'eletrico':
                    rho = self.TM_E_rho(rho=Rho, phi=Phi)
                    phi = self.TM_E_phi(rho=Rho, phi=Phi)
                    e_z = self.TM_E_Z(rho=Rho, phi=Phi)
                elif campo == 'magnetico':
                    rho = self.TM_H_rho(rho=Rho, phi=Phi)
                    phi = self.TM_H_phi(rho=Rho, phi=Phi)

            # Converter para coordenadas cartesianas
            e_x = rho * np.cos(Phi) - phi * np.sin(Phi)
            e_y = rho * np.sin(Phi) + phi * np.cos(Phi)
            
            # Filtrar pontos dentro do círculo
            mask = Rho <= self.raio
            X_masked = X[mask]
            Y_masked = Y[mask]
            e_x = e_x[mask]
            e_y = e_y[mask]
            e_z = e_z[mask]
            # Criar array Z para os vetores
            Z_masked = np.full_like(X_masked, z)
            print(f"e_z_vector em z=({z}): {np.mean(e_z)}")

            # Adicionar vetores ao gráfico
            for i in range(len(X_masked)):
                vectors.append(
                    go.Cone(
                        x=[X_masked[i]],
                        y=[Y_masked[i]],
                        z=[Z_masked[i]],
                        u=[e_x[i]],
                        v=[e_y[i]],
                        w=[e_z[i]],
                        sizemode="absolute",
                        sizeref=self.raio / 8,
                        colorscale="blackbody",
                        showscale=False
                    )
                )
            
        # Criar superfície cilíndrica como anéis empilhados
        theta = np.linspace(0, 2 * np.pi, 100)  # Divisões angulares
        z_cylinder = np.linspace(0, comprimento, len(z_vals))  # Alturas dos anéis
        theta, z_cylinder = np.meshgrid(theta, z_vals)  # Malha de coordenadas
        x_cylinder = self.raio * np.cos(theta)  # Coordenadas X dos anéis
        y_cylinder = self.raio * np.sin(theta)  # Coordenadas Y dos anéis

        # Calcular a intensidade média do campo e_z para cada altura z
        intensidade_media = []
        for z in z_vals:
            self.update_z(z)  # Atualizar o valor de z
            e_z = self.TE_H_Z(rho=self.raio, phi=theta[0])  # Exemplo para TE_H_Z
            intensidade_media.append(np.mean(e_z**2))  # Intensidade média em e_z ao quadrado

        # Normalizar os valores de intensidade média para destacar a variação
        intensidade_media = np.array(intensidade_media)
        intensidade_media_normalizada = (intensidade_media - np.min(intensidade_media)) / (np.max(intensidade_media) - np.min(intensidade_media))

        # Expandir a lista de intensidades médias normalizadas para corresponder ao formato da superfície
        intensidade_surface = np.tile(intensidade_media_normalizada, (theta.shape[1], 1)).T

        # Criar a superfície cilíndrica com as cores baseadas na intensidade média
        surface = go.Surface(
            x=x_cylinder,
            y=y_cylinder,
            z=z_cylinder,
            surfacecolor=intensidade_surface,  # Usar a intensidade média como cor
            opacity=0.8,
            colorscale="Inferno",
            showscale=True  # Exibir a escala de cores
        )

        # Criar figura 3D
        fig = go.Figure(data=vectors + [surface])

        # Configurar layout
        fig.update_layout(
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectratio=dict(x=1, y=1, z=comprimento / self.raio),  # Ajustar proporção do eixo Z
            ),
            title="Anéis Empilhados com Intensidade Média do Campo",
        )

        fig.write_html("plot_aneis_empilhados.html")
        return fig

if __name__ == "__main__":
    # # Criar a instância e o meshgrid como antes
    Modo_Cilindrico = Modo_Cilindrico(raio=0.0015, frequencia=15e9, permissividade=1, permeabilidade=1, n=0, m=2, z=0.25)
    X, Y, Rho, Phi = Modo_Cilindrico.criar_meshgrid_cartesiano()

    # # Exemplos de uso das novas funções de animação por fase:

    # 1. Animação interativa (visualização em tempo real)
    # Modo_Cilindrico.plot_vetores_de_campo_fase_animado(X, Y, Rho, Phi, transversal='TE', campo='eletrico', frames=60, interval=100, z_fixo=1)

    # # 2. Salvar como GIF
    # Modo_Cilindrico.salvar_animacao_gif(X, Y, Rho, Phi, transversal='TM', campo='eletrico',
    #                                    frames=40, interval=120, z_fixo=0.25, nome_arquivo='TM_eletrico_fase.gif')

    # # # 3. Animação do campo magnético TM
    # Modo_Cilindrico.salvar_animacao_gif(X, Y, Rho, Phi, transversal='TM', campo='magnetico',
    #                                    frames=60, interval=85, z_fixo=0.25, nome_arquivo='TM_magnetico_fase.gif')

    # # 3. Animação do campo magnético TM
    Modo_Cilindrico.salvar_animacao_gif(X, Y, Rho, Phi, transversal='TM', campo='magnetico',
                                       frames=30, interval=85, z_fixo=0.25, nome_arquivo='TE_magnetico_fase.gif')

    # # 3. Animação do campo magnético TM
    Modo_Cilindrico.salvar_animacao_gif(X, Y, Rho, Phi, transversal='TM', campo='eletrico',
                                       frames=40, interval=120, z_fixo=0.25, nome_arquivo='TE_eletrico_fase.gif')

    # # Funções antigas ainda funcionam:
    # Modo_Cilindrico.plot_vetores_de_campo_animado(X, Y, Rho, Phi, transversal='TM', campo='magnetico', frames=50, interval=200)
    # Modo_Cilindrico.plot_vetores_de_campo(X, Y, Rho, Phi, transversal='TE', campo='magnetico')

# Modo_Cilindrico.pontos_por_dimensao = 8  # Número de pontos por dimensão para o meshgrid
# Modo_Cilindrico.num_planos = 13  # Número de planos Z
# X, Y, Rho, Phi, Z = Modo_Cilindrico.criar_meshgrid_cartesiano_com_z(comprimento=0.01)
# Modo_Cilindrico.plot_vetores_3D(X, Y, Rho, Phi, Z, transversal='TE', campo='magnetico',comprimento=0.01)