import numpy as np
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

class Modo_TEmn():
    def __init__(self, largura = 22.86, 
                 altura = 10.16, 
                 frequencia = 12*10**9, 
                 permissividade = 1, 
                 permeabilidade = 1 , 
                 plano = 'xy'):
        self.plano = plano
        
        self.pi = np.pi

        self.A = 1 # Amplitude
        self.frequencia = frequencia # (Hz)

        self.vacuo = False
        self.m = 1
        self.n = 0

        # Dados da página 36 da dissertação
        self.largura = largura/1000 # = a (m)
        self.altura = altura/1000 # = b (m)
        self.profundidade = 0.11 # profundidade é sempre fixa?

        self.mu = permissividade # Permissividade Relativa
        self.epsilon = permeabilidade # Permeabilidade relativa
        
        self.light_speed = 299792458 # m/s
        self.pontos_por_dimensao = 50

        self.omega = self.omega() # Frequência angular
        self.k = self.k() # Número de Onda:
        self.k_c = self.k_c() # Número de Onda de corte, que depende de corte e da geometria
        self.beta = self.beta() 

        self.escolha_plano()
        self.cos_mx = self.cosseno_x()
        self.cos_ny = self.cosseno_y()
        self.sen_mx = self.seno_x()
        self.sen_ny = self.seno_y()
        self.expz = self.exp_z()

    def criar_meshgrid_xy(self):
        x = np.linspace(0, self.largura, self.pontos_por_dimensao)
        y = np.linspace(0, self.altura, self.pontos_por_dimensao)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = np.ones_like(X)
        return X, Y, Z

    def criar_meshgrid_xz(self):
        x = np.linspace(0, self.largura, self.pontos_por_dimensao)
        z = np.linspace(0, self.profundidade, self.pontos_por_dimensao)
        X, Z = np.meshgrid(x, z, indexing='ij')
        Y = np.ones_like(X)
        return X, Y, Z

    def criar_meshgrid_yz(self):
        y = np.linspace(0, self.altura, self.pontos_por_dimensao)
        z = np.linspace(0, self.profundidade, self.pontos_por_dimensao)
        Y, Z = np.meshgrid(y, z, indexing='ij')
        X = np.ones_like(Y)
        return X, Y, Z


    def escolha_plano(self):
        if self.plano == 'xy':
            x, y, z = self.criar_meshgrid_xy()
        elif self.plano == 'xz':
            x, y, z = self.criar_meshgrid_xz()
        elif self.plano == 'yz':
            x, y, z = self.criar_meshgrid_yz()
        
        self.x = x
        self.y = y
        self.z = z

    def omega(self):
        return self.frequencia*2*self.pi

    def k(self):
        if self.vacuo == True:
            return self.omega*np.sqrt(1/self.light_speed**2) # Mu0 * Epsilon0 = 1/c^2 No vacuo
        else:
            return self.omega*np.sqrt(self.mu*self.epsilon)


    def k_c(self):
        return (self.m*self.pi/self.largura)**2+(self.n*self.pi/self.altura)**2

    def beta(self): # Constante de fase
        return cmath.sqrt(self.k**2-self.k_c)

    def cosseno_x(self):
        return np.cos(self.m*self.pi*self.x/self.largura)
    
    def cosseno_y(self):
        return np.cos(self.n*self.pi*self.y/self.altura)

    def seno_x(self):
        return np.sin(self.m*self.pi*self.x/self.largura)

    def seno_y(self):
        return np.sin(self.n*self.pi*self.y/self.altura)

    def exp_z(self):
        return np.exp(-1j*self.beta*self.z)

    def H_z(self):
        cos_mx = self.cosseno_x()
        cos_ny = self.cosseno_y()
        expz = self.exp_z()

        return self.A*cos_mx*cos_ny*expz

    def H_x(self):
        const = 1j*self.beta*self.m*self.pi/(self.k_c**2*self.largura)
        return const*self.A*self.sen_mx*self.cos_ny*self.expz

    def H_y(self):
        const = 1j*self.beta*self.n*self.pi/(self.k_c**2*self.altura)
        return const*self.A*self.cos_mx*self.sen_ny*self.expz
    
    def E_x(self):
        const = 1j*self.omega*self.mu*self.n*self.pi/(self.k_c**2 *self.altura)
        return const*self.A*self.cos_mx*self.cos_ny*self.expz
    
    def E_y(self):
        const = -1j*self.omega*self.mu*self.m*self.pi/(self.k_c**2 *self.largura)
        return const*self.A*self.sen_mx*self.cos_ny*self.expz

    def calcula_campos(self):
        self.Hx = np.real(self.H_x())
        self.Hy = np.real(self.H_y())
        self.Hz = np.real(self.H_z())
        self.Ex = np.real(self.E_x())
        self.Ey = np.real(self.E_y())
        self.Ez = np.zeros_like(self.Ex)

        return self.Hx, self.Hy, self.Hz, self.Ex, self.Ey, self.Ez
    
    def coordenadas(self):
        return self.x, self.y, self.z
    
    def plota_campo_vetorial(self, campo='magnetico'):

        if self.plano == 'xy':
            abscissas = self.x
            ordenadas = self.y
            if campo == 'magnetico':
                u = self.Hx
                v = self.Hy
            elif campo == 'eletrico':
                u = self.Ex
                v = self.Ey

        elif self.plano == 'xz':
            abscissas = self.x
            ordenadas = self.z
            if campo == 'magnetico':
                u = self.Hx
                v = self.Hz
            elif campo == 'eletrico':
                u = self.Ex
                v = self.Ez

        elif self.plano == 'yz':
            abscissas = self.z
            ordenadas = self.y
            if campo == 'magnetico':
                u = self.Hz
                v = self.Hy
            elif campo == 'eletrico':
                u = self.Ez
                v = self.Ey

        # Calcular magnitude para normalização e colormap (mesmo padrão das guias cilíndricas)
        magnitude = np.sqrt(u**2 + v**2)
        max_magnitude = np.max(magnitude) if np.max(magnitude) > 0 else 1e-10

        # Normalizar vetores para tamanho consistente
        magnitude_nonzero = np.where(magnitude > 1e-12, magnitude, 1e-12)
        u_normalized = u / magnitude_nonzero
        v_normalized = v / magnitude_nonzero

        # Escalar para tamanho visual agradável (baseado nas dimensões da guia)
        scale_factor = min(self.largura, self.altura) * 0.05
        u_display = u_normalized * scale_factor
        v_display = v_normalized * scale_factor

        # Configurar plot com colormap viridis (mesmo das guias cilíndricas)
        fig, ax = plt.subplots(figsize=(8, 6))

        norm = plt.Normalize(vmin=0, vmax=max_magnitude)
        cmap = plt.cm.viridis

        quiver = ax.quiver(abscissas, ordenadas, u_display, v_display,
                          magnitude, cmap=cmap, norm=norm,
                          scale=1, scale_units='xy', angles='xy',
                          pivot='middle', alpha=0.8)

        # Adicionar colorbar
        cbar = plt.colorbar(quiver, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Intensidade do Campo', rotation=270, labelpad=20)

        # Adicionar bordas da guia retangular
        rect = plt.Rectangle((0, 0), self.largura, self.altura,
                           fill=False, edgecolor='red', linewidth=2, linestyle='--')
        ax.add_patch(rect)
        ax.add_patch(plt.Rectangle((0, 0), self.largura, self.altura,
                                 fill=True, facecolor='lightgray', alpha=0.3, zorder=0))

        ax.set_title(f'Campo {campo.capitalize()} no plano {self.plano.upper()}')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlim(-self.largura*0.1, self.largura*1.1)
        ax.set_ylim(-self.altura*0.1, self.altura*1.1)
        ax.set_aspect('equal')

        return fig

    def plot3DField(self, campo = 'magnetico', componente = 'x'):
        x = self.x
        y = self.y

        if campo == 'magnetico' and componente == 'x':
            imagem = self.Hx
        elif campo == 'magnetico' and componente == 'y':
            imagem = self.Hy
        elif campo == 'magnetico' and componente == 'z':
            imagem = self.Hz
        elif campo == 'eletrico' and componente == 'x':
            imagem = self.Ex
        elif campo == 'eletrico' and componente == 'y':
            imagem = self.Ey
        elif campo == 'eletrico' and componente == 'z':
            imagem = self.Ez

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Usar colormap viridis consistente com guias cilíndricas
        ground = np.min(imagem)
        ceiling = np.max(imagem)

        # Escalar a intensidade do campo para proporções adequadas
        max_physical_dim = max(self.largura, self.altura)
        z_scale_factor = max_physical_dim * 0.5
        z_range = abs(ceiling - ground)

        if z_range > 0:
            # Escalar os valores da imagem
            imagem_scaled = (imagem - ground) * z_scale_factor / z_range + ground * z_scale_factor / z_range
            ground_scaled = ground * z_scale_factor / z_range
        else:
            imagem_scaled = imagem
            ground_scaled = ground

        # Plotar a superfície do campo com valores escalonados
        surface = ax.plot_surface(x, y, imagem_scaled, cmap='viridis', alpha=0.8,
                                linewidth=0, antialiased=True)

        # Adicionar contorno no plano base com mesma cor
        ax.contourf(x, y, imagem, zdir='z', offset=ground_scaled, cmap='viridis', alpha=0.6)

        # Adicionar paredes da guia de onda
        wall_color = 'lightgray'
        wall_alpha = 0.4
        wall_edgecolor = 'darkgray'

        # Usar as coordenadas Z escalonadas para as paredes
        # Calcular escalonamento Z primeiro
        max_physical_dim = max(self.largura, self.altura)
        z_scale_factor = max_physical_dim * 0.5
        z_range = abs(ceiling - ground)

        if z_range > 0:
            z_scaled_ground = ground * z_scale_factor / z_range
            z_scaled_ceiling = ceiling * z_scale_factor / z_range
        else:
            z_scaled_ground = ground
            z_scaled_ceiling = ceiling

        # Coordenadas das paredes
        x_wall = [0, self.largura, self.largura, 0, 0]
        y_wall = [0, 0, self.altura, self.altura, 0]
        z_wall_bottom = [z_scaled_ground] * 5
        z_wall_top = [z_scaled_ceiling] * 5

        # Parede inferior (base)
        xx_base, yy_base = np.meshgrid([0, self.largura], [0, self.altura])
        zz_base = np.full_like(xx_base, z_scaled_ground)
        ax.plot_surface(xx_base, yy_base, zz_base, color=wall_color, alpha=wall_alpha,
                       edgecolor=wall_edgecolor, linewidth=0.5)

        # Parede superior (teto)
        zz_top = np.full_like(xx_base, z_scaled_ceiling)
        ax.plot_surface(xx_base, yy_base, zz_top, color=wall_color, alpha=wall_alpha/2,
                       edgecolor=wall_edgecolor, linewidth=0.5)

        # Paredes laterais
        # Parede x=0 (esquerda)
        yy_wall, zz_wall = np.meshgrid([0, self.altura], [z_scaled_ground, z_scaled_ceiling])
        xx_wall = np.zeros_like(yy_wall)
        ax.plot_surface(xx_wall, yy_wall, zz_wall, color=wall_color, alpha=wall_alpha,
                       edgecolor=wall_edgecolor, linewidth=0.5)

        # Parede x=largura (direita)
        xx_wall = np.full_like(yy_wall, self.largura)
        ax.plot_surface(xx_wall, yy_wall, zz_wall, color=wall_color, alpha=wall_alpha,
                       edgecolor=wall_edgecolor, linewidth=0.5)

        # Parede y=0 (frente)
        xx_wall, zz_wall = np.meshgrid([0, self.largura], [z_scaled_ground, z_scaled_ceiling])
        yy_wall = np.zeros_like(xx_wall)
        ax.plot_surface(xx_wall, yy_wall, zz_wall, color=wall_color, alpha=wall_alpha,
                       edgecolor=wall_edgecolor, linewidth=0.5)

        # Parede y=altura (fundo)
        yy_wall = np.full_like(xx_wall, self.altura)
        ax.plot_surface(xx_wall, yy_wall, zz_wall, color=wall_color, alpha=wall_alpha,
                       edgecolor=wall_edgecolor, linewidth=0.5)

        # Adicionar bordas da guia (wireframe)
        ax.plot(x_wall, y_wall, z_wall_bottom, color='red', linewidth=2, alpha=0.8)
        ax.plot(x_wall, y_wall, z_wall_top, color='red', linewidth=2, alpha=0.8)

        # Bordas verticais
        for i in range(4):
            ax.plot([x_wall[i], x_wall[i]], [y_wall[i], y_wall[i]],
                   [z_scaled_ground, z_scaled_ceiling], color='red', linewidth=2, alpha=0.8)

        # Configurar colorbar
        cbar = fig.colorbar(surface, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label('Intensidade do Campo', rotation=270, labelpad=20)

        # Adiciona rótulos e título
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Intensidade')
        ax.set_title(f'Campo {campo.capitalize()} - Componente {componente.upper()}\nGuia Retangular {self.largura*1000:.1f}×{self.altura*1000:.1f} mm')

        # Configurar limites
        ax.set_xlim(-self.largura*0.1, self.largura*1.1)
        ax.set_ylim(-self.altura*0.1, self.altura*1.1)
        ax.set_zlim(ground*1.1, ceiling*1.1)

        # Configurar aspecto proporcional baseado nas dimensões físicas
        # Usar a maior dimensão da guia como referência
        max_physical_dim = max(self.largura, self.altura)

        # Escalar a altura do campo (Z) para ser proporcional às dimensões físicas
        # Usar aproximadamente 50% da maior dimensão como altura máxima do campo
        z_scale_factor = max_physical_dim * 0.5
        z_range = abs(ceiling - ground)

        if z_range > 0:
            # Normalizar e escalar a intensidade do campo
            z_scaled_ground = ground * z_scale_factor / z_range
            z_scaled_ceiling = ceiling * z_scale_factor / z_range
        else:
            z_scaled_ground = ground
            z_scaled_ceiling = ceiling

        # Centralizar em cada eixo
        x_middle = self.largura/2
        y_middle = self.altura/2
        z_middle = (z_scaled_ceiling + z_scaled_ground)/2

        # Definir limites proporcionais
        x_range = self.largura * 1.2
        y_range = self.altura * 1.2
        z_range_scaled = abs(z_scaled_ceiling - z_scaled_ground) * 1.2

        ax.set_xlim(x_middle - x_range/2, x_middle + x_range/2)
        ax.set_ylim(y_middle - y_range/2, y_middle + y_range/2)
        ax.set_zlim(z_middle - z_range_scaled/2, z_middle + z_range_scaled/2)

        # Configurar aspecto proporcional às dimensões reais
        # Razão de aspecto baseada nas dimensões físicas
        aspect_ratio = [self.largura, self.altura, max_physical_dim * 0.5]
        ax.set_box_aspect(aspect_ratio)

        # Configurar grid
        ax.grid(True, alpha=0.2)

        # Ajustar visualização para melhor perspectiva
        ax.view_init(elev=20, azim=45)

        return fig

if __name__ == "__main__":
    TEmn = Modo_TEmn(largura = 22.86,
                     altura = 10.16,
                     frequencia = 12*10**9,
                     permissividade = 1,
                     permeabilidade = 1 ,
                     plano = 'xy')

    TEmn.calcula_campos()
    # TEmn.plota_campo_vetorial('magnetico')
    TEmn.plot3DField(campo = 'magnetico', componente = 'x')
    plt.show()