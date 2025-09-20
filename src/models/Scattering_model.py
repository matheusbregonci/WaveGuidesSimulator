"""
Modelo de Matriz de Espalhamento para Cavidade Retangular
Análise de parâmetros S (S11, S12, S21, S22) em função da frequência

Baseado na teoria de cavidades ressonantes e análise de microondas
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import jv, jvp
import cmath

class ScatteringMatrix:
    """
    Classe para calcular e visualizar a matriz de espalhamento de uma cavidade retangular.

    A matriz de espalhamento relaciona as ondas incidentes e refletidas nas portas:
    [b1]   [S11 S12] [a1]
    [b2] = [S21 S22] [a2]

    onde:
    - a1, a2: ondas incidentes nas portas 1 e 2
    - b1, b2: ondas refletidas/transmitidas nas portas 1 e 2
    """

    def __init__(self,
                 largura=22.86,      # mm - dimensão a
                 altura=10.16,       # mm - dimensão b
                 comprimento=50.0,   # mm - dimensão c (profundidade)
                 permissividade=1.0, # εr
                 permeabilidade=1.0, # μr
                 freq_min=8.0,       # GHz
                 freq_max=18.0,      # GHz
                 num_pontos=1000):   # Número de pontos de frequência

        # Dimensões da cavidade (converter mm para m)
        self.a = largura / 1000.0
        self.b = altura / 1000.0
        self.c = comprimento / 1000.0

        # Propriedades do material
        self.epsilon_r = permissividade
        self.mu_r = permeabilidade

        # Constantes físicas
        self.c0 = 299792458  # velocidade da luz no vácuo (m/s)
        self.mu0 = 4*np.pi*1e-7  # permeabilidade do vácuo
        self.epsilon0 = 1/(self.mu0 * self.c0**2)  # permissividade do vácuo

        # Faixa de frequência
        self.freq_min = freq_min * 1e9  # Hz
        self.freq_max = freq_max * 1e9  # Hz
        self.num_pontos = num_pontos
        self.frequencies = np.linspace(self.freq_min, self.freq_max, num_pontos)

        # Modos da cavidade (TE e TM)
        self.modos_te = [(1,0), (2,0), (0,1), (1,1), (2,1), (3,0), (0,2)]
        self.modos_tm = [(1,1), (2,1), (1,2), (3,1), (2,2), (4,1), (1,3)]

        # Fator de qualidade e perdas
        self.Q_factor = 1000  # Fator de qualidade da cavidade

    def calcular_freq_ressonancia_te(self, m, n):
        """Calcula frequência de ressonância para modo TE_mn"""
        k_c = np.pi * np.sqrt((m/self.a)**2 + (n/self.b)**2)
        f_res = k_c * self.c0 / (2*np.pi*np.sqrt(self.epsilon_r * self.mu_r))
        return f_res

    def calcular_freq_ressonancia_tm(self, m, n):
        """Calcula frequência de ressonância para modo TM_mn"""
        if m == 0 or n == 0:
            return np.inf  # TM00, TM0n, TMm0 não existem
        k_c = np.pi * np.sqrt((m/self.a)**2 + (n/self.b)**2)
        f_res = k_c * self.c0 / (2*np.pi*np.sqrt(self.epsilon_r * self.mu_r))
        return f_res

    def calcular_impedancia_modo(self, freq, modo_tipo, m, n):
        """Calcula impedância característica do modo"""
        omega = 2*np.pi*freq
        k = omega * np.sqrt(self.epsilon_r * self.mu_r) / self.c0

        if modo_tipo == 'TE':
            if m == 0 and n == 0:
                return np.inf
            k_c = np.pi * np.sqrt((m/self.a)**2 + (n/self.b)**2)
            gamma = np.sqrt(k_c**2 - k**2 + 0j)
            Z = omega * self.mu0 * self.mu_r / gamma
        else:  # TM
            if m == 0 or n == 0:
                return np.inf
            k_c = np.pi * np.sqrt((m/self.a)**2 + (n/self.b)**2)
            gamma = np.sqrt(k_c**2 - k**2 + 0j)
            Z = gamma / (omega * self.epsilon0 * self.epsilon_r)

        return Z

    def calcular_s11(self, freq):
        """Calcula parâmetro S11 (reflexão na porta 1) - modelo realístico"""

        # Modelo de cavidade ressonante com múltiplos modos
        s11_total = 0 + 0j
        num_modos_ativos = 0

        # Contribuição de todos os modos próximos à frequência
        for m, n in self.modos_te:
            f_res = self.calcular_freq_ressonancia_te(m, n)
            if self.freq_min <= f_res <= self.freq_max:
                # Modelo de ressonador acoplado
                Q_loaded = self.Q_factor / 2  # Q carregado (acoplado)
                delta_f = (freq - f_res) / f_res

                # Parâmetro s normalizado
                s = 2 * Q_loaded * delta_f + 1j

                # Coeficiente de reflexão do modo
                k_coupling = 0.3  # Fator de acoplamento moderado
                s11_modo = (s - 1j / k_coupling) / (s + 1j / k_coupling)

                # Peso do modo baseado na proximidade da frequência
                peso = np.exp(-abs(delta_f) * 5)
                s11_total += s11_modo * peso * 0.2
                num_modos_ativos += peso

        # Contribuição dos modos TM (menor)
        for m, n in self.modos_tm:
            f_res = self.calcular_freq_ressonancia_tm(m, n)
            if self.freq_min <= f_res <= self.freq_max:
                Q_loaded = self.Q_factor / 2
                delta_f = (freq - f_res) / f_res
                s = 2 * Q_loaded * delta_f + 1j

                k_coupling = 0.2  # Acoplamento menor para TM
                s11_modo = (s - 1j / k_coupling) / (s + 1j / k_coupling)

                peso = np.exp(-abs(delta_f) * 5)
                s11_total += s11_modo * peso * 0.1
                num_modos_ativos += peso

        # Normalização e linha de base
        if num_modos_ativos > 0:
            s11_total = s11_total / (1 + num_modos_ativos * 0.1)

        # Adicionar linha de base realística
        baseline = 0.05 * np.exp(1j * 2 * np.pi * freq * self.c / self.c0)
        s11_final = s11_total + baseline

        # Garantir |S11| <= 0.9 para sistema bem casado
        if abs(s11_final) > 0.9:
            s11_final = s11_final / abs(s11_final) * 0.9

        return s11_final

    def calcular_s21(self, freq):
        """Calcula parâmetro S21 (transmissão da porta 1 para porta 2) - modelo realístico"""

        # Calcular S11 primeiro
        s11 = self.calcular_s11(freq)

        # Modelo de cavidade ressonante com perdas
        s21_total = 0 + 0j
        num_modos_ativos = 0

        # Contribuição dos modos ressonantes
        for m, n in self.modos_te:
            f_res = self.calcular_freq_ressonancia_te(m, n)
            if self.freq_min <= f_res <= self.freq_max:
                Q_loaded = self.Q_factor / 2
                delta_f = (freq - f_res) / f_res

                # Transmissão através do modo ressonante
                s = 2 * Q_loaded * delta_f + 1j
                k_coupling = 0.3

                # S21 baseado no modelo de ressonador acoplado
                s21_modo = (2j / k_coupling) / (s + 1j / k_coupling)

                peso = np.exp(-abs(delta_f) * 5)
                s21_total += s21_modo * peso * 0.3
                num_modos_ativos += peso

        # Contribuição dos modos TM
        for m, n in self.modos_tm:
            f_res = self.calcular_freq_ressonancia_tm(m, n)
            if self.freq_min <= f_res <= self.freq_max:
                Q_loaded = self.Q_factor / 2
                delta_f = (freq - f_res) / f_res
                s = 2 * Q_loaded * delta_f + 1j

                k_coupling = 0.2
                s21_modo = (2j / k_coupling) / (s + 1j / k_coupling)

                peso = np.exp(-abs(delta_f) * 5)
                s21_total += s21_modo * peso * 0.2
                num_modos_ativos += peso

        # Transmissão de fundo (propagação direta)
        fase_propagacao = 2 * np.pi * freq * self.c / self.c0
        transmissao_fundo = 0.7 * np.exp(1j * fase_propagacao)

        # Combinar ressonâncias com fundo
        if num_modos_ativos > 0:
            s21_final = transmissao_fundo + s21_total / (1 + num_modos_ativos * 0.1)
        else:
            s21_final = transmissao_fundo

        # Aplicar conservação de energia: |S11|² + |S21|² ≤ 1
        energia_total = abs(s11)**2 + abs(s21_final)**2
        if energia_total > 1.0:
            fator_correcao = 0.95 / np.sqrt(energia_total)
            s21_final = s21_final * fator_correcao

        return s21_final

    def calcular_matriz_s(self):
        """Calcula a matriz de espalhamento completa para todas as frequências"""
        S11 = np.zeros(len(self.frequencies), dtype=complex)
        S12 = np.zeros(len(self.frequencies), dtype=complex)
        S21 = np.zeros(len(self.frequencies), dtype=complex)
        S22 = np.zeros(len(self.frequencies), dtype=complex)

        for i, freq in enumerate(self.frequencies):
            S11[i] = self.calcular_s11(freq)
            S21[i] = self.calcular_s21(freq)

            # Por reciprocidade: S12 = S21
            S12[i] = S21[i]

            # Por simetria (assumindo portas idênticas): S22 = S11
            S22[i] = S11[i]

        return S11, S12, S21, S22

    def plot_s_parameters_matplotlib(self):
        """Cria gráficos dos parâmetros S usando matplotlib"""
        S11, S12, S21, S22 = self.calcular_matriz_s()
        freq_ghz = self.frequencies / 1e9

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Parâmetros S da Cavidade Retangular', fontsize=16, fontweight='bold')

        # S11 - Magnitude e Fase
        ax1.plot(freq_ghz, 20*np.log10(np.abs(S11)), 'b-', linewidth=2, label='|S11| (dB)')
        ax1_phase = ax1.twinx()
        ax1_phase.plot(freq_ghz, np.angle(S11)*180/np.pi, 'r--', linewidth=1, label='∠S11 (°)')
        ax1.set_xlabel('Frequência (GHz)')
        ax1.set_ylabel('|S11| (dB)', color='b')
        ax1_phase.set_ylabel('Fase (°)', color='r')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('S11 - Reflexão Porta 1')

        # S21 - Magnitude e Fase
        ax2.plot(freq_ghz, 20*np.log10(np.abs(S21)), 'g-', linewidth=2, label='|S21| (dB)')
        ax2_phase = ax2.twinx()
        ax2_phase.plot(freq_ghz, np.angle(S21)*180/np.pi, 'm--', linewidth=1, label='∠S21 (°)')
        ax2.set_xlabel('Frequência (GHz)')
        ax2.set_ylabel('|S21| (dB)', color='g')
        ax2_phase.set_ylabel('Fase (°)', color='m')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('S21 - Transmissão 1→2')

        # Carta de Smith completa para S11
        self._plot_smith_chart(ax3, S11)

        # Frequências de ressonância
        freq_res_te = []
        modos_te_labels = []
        for m, n in self.modos_te:
            f_res = self.calcular_freq_ressonancia_te(m, n)
            if self.freq_min <= f_res <= self.freq_max:
                freq_res_te.append(f_res/1e9)
                modos_te_labels.append(f'TE{m}{n}')

        freq_res_tm = []
        modos_tm_labels = []
        for m, n in self.modos_tm:
            f_res = self.calcular_freq_ressonancia_tm(m, n)
            if self.freq_min <= f_res <= self.freq_max:
                freq_res_tm.append(f_res/1e9)
                modos_tm_labels.append(f'TM{m}{n}')

        ax4.scatter(freq_res_te, [1]*len(freq_res_te), c='blue', s=100, alpha=0.7, label='Modos TE')
        ax4.scatter(freq_res_tm, [2]*len(freq_res_tm), c='red', s=100, alpha=0.7, label='Modos TM')

        for i, (freq, label) in enumerate(zip(freq_res_te, modos_te_labels)):
            ax4.annotate(label, (freq, 1), xytext=(5, 5), textcoords='offset points', fontsize=8)

        for i, (freq, label) in enumerate(zip(freq_res_tm, modos_tm_labels)):
            ax4.annotate(label, (freq, 2), xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax4.set_xlabel('Frequência (GHz)')
        ax4.set_ylabel('Tipo de Modo')
        ax4.set_yticks([1, 2])
        ax4.set_yticklabels(['TE', 'TM'])
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_title('Frequências de Ressonância')
        ax4.set_xlim(freq_ghz[0], freq_ghz[-1])

        plt.tight_layout()
        return fig

    def _plot_smith_chart(self, ax, S_param):
        """Plota uma carta de Smith completa com círculos de resistência e reatância constantes"""

        # Círculo unitário (|Γ| = 1)
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1.5, alpha=0.8)

        # Círculos de resistência constante (r = 0.2, 0.5, 1, 2, 5)
        r_values = [0.2, 0.5, 1.0, 2.0, 5.0]
        for r in r_values:
            # Centro e raio do círculo de resistência constante
            center_x = r / (1 + r)
            center_y = 0
            radius = 1 / (1 + r)

            circle_theta = np.linspace(0, 2*np.pi, 100)
            x_circle = center_x + radius * np.cos(circle_theta)
            y_circle = center_y + radius * np.sin(circle_theta)

            # Plotar apenas a parte dentro do círculo unitário
            mask = x_circle**2 + y_circle**2 <= 1.001  # Pequena tolerância
            if np.any(mask):
                ax.plot(x_circle[mask], y_circle[mask], 'gray', linewidth=0.8, alpha=0.6)

                # Label para resistência
                if r <= 1:
                    label_x = center_x + radius * 0.7
                else:
                    label_x = center_x - radius * 0.3
                ax.text(label_x, 0.05, f'{r}', fontsize=8, ha='center', va='bottom', color='gray')

        # Círculos de reatância constante (x = ±0.2, ±0.5, ±1, ±2, ±5)
        x_values = [0.2, 0.5, 1.0, 2.0, 5.0]
        for x in x_values:
            for sign in [1, -1]:
                x_val = sign * x
                # Centro e raio do círculo de reatância constante
                center_x = 1
                center_y = 1 / x_val
                radius = abs(1 / x_val)

                circle_theta = np.linspace(0, 2*np.pi, 100)
                x_circle = center_x + radius * np.cos(circle_theta)
                y_circle = center_y + radius * np.sin(circle_theta)

                # Plotar apenas a parte dentro do círculo unitário
                mask = x_circle**2 + y_circle**2 <= 1.001
                if np.any(mask):
                    ax.plot(x_circle[mask], y_circle[mask], 'lightblue', linewidth=0.8, alpha=0.6)

        # Eixos principais
        ax.axhline(y=0, color='k', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='k', linewidth=1, alpha=0.5)

        # Plotar o parâmetro S
        ax.plot(np.real(S_param), np.imag(S_param), 'b-', linewidth=2.5, label='S11')

        # Marcar pontos inicial e final
        ax.plot(np.real(S_param)[0], np.imag(S_param)[0], 'go', markersize=6, label='Início')
        ax.plot(np.real(S_param)[-1], np.imag(S_param)[-1], 'ro', markersize=6, label='Fim')

        # Configurações do gráfico
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Carta de Smith - S11', fontsize=12, fontweight='bold')
        ax.set_xlabel('Resistência Normalizada')
        ax.set_ylabel('Reatância Normalizada')
        ax.legend(loc='upper right', fontsize=8)

        # Labels dos pontos cardinais
        ax.text(1.05, 0, '∞', fontsize=10, ha='center', va='center', fontweight='bold')
        ax.text(-1.05, 0, '0', fontsize=10, ha='center', va='center', fontweight='bold')
        ax.text(0, 1.05, '+j∞', fontsize=10, ha='center', va='center', fontweight='bold')
        ax.text(0, -1.05, '-j∞', fontsize=10, ha='center', va='center', fontweight='bold')

    def plot_s_parameters_plotly(self):
        """Cria gráficos interativos dos parâmetros S usando Plotly"""
        S11, S12, S21, S22 = self.calcular_matriz_s()
        freq_ghz = self.frequencies / 1e9

        # Criar subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('S11 - Reflexão', 'S21 - Transmissão',
                          'Carta de Smith (S11)', 'Frequências de Ressonância'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # S11 - Magnitude
        fig.add_trace(
            go.Scatter(x=freq_ghz, y=20*np.log10(np.abs(S11)),
                      name='|S11| (dB)', line=dict(color='blue', width=2)),
            row=1, col=1, secondary_y=False
        )

        # S11 - Fase
        fig.add_trace(
            go.Scatter(x=freq_ghz, y=np.angle(S11)*180/np.pi,
                      name='∠S11 (°)', line=dict(color='red', dash='dash')),
            row=1, col=1, secondary_y=True
        )

        # S21 - Magnitude
        fig.add_trace(
            go.Scatter(x=freq_ghz, y=20*np.log10(np.abs(S21)),
                      name='|S21| (dB)', line=dict(color='green', width=2)),
            row=1, col=2, secondary_y=False
        )

        # S21 - Fase
        fig.add_trace(
            go.Scatter(x=freq_ghz, y=np.angle(S21)*180/np.pi,
                      name='∠S21 (°)', line=dict(color='magenta', dash='dash')),
            row=1, col=2, secondary_y=True
        )

        # Carta de Smith completa
        self._add_smith_chart_plotly(fig, S11, row=2, col=1)

        # Frequências de ressonância
        freq_res_te = []
        modos_te_labels = []
        for m, n in self.modos_te:
            f_res = self.calcular_freq_ressonancia_te(m, n)
            if self.freq_min <= f_res <= self.freq_max:
                freq_res_te.append(f_res/1e9)
                modos_te_labels.append(f'TE{m}{n}')

        freq_res_tm = []
        modos_tm_labels = []
        for m, n in self.modos_tm:
            f_res = self.calcular_freq_ressonancia_tm(m, n)
            if self.freq_min <= f_res <= self.freq_max:
                freq_res_tm.append(f_res/1e9)
                modos_tm_labels.append(f'TM{m}{n}')

        if freq_res_te:
            fig.add_trace(
                go.Scatter(x=freq_res_te, y=[1]*len(freq_res_te),
                          mode='markers+text', name='Modos TE',
                          text=modos_te_labels, textposition='top center',
                          marker=dict(color='blue', size=10)),
                row=2, col=2
            )

        if freq_res_tm:
            fig.add_trace(
                go.Scatter(x=freq_res_tm, y=[2]*len(freq_res_tm),
                          mode='markers+text', name='Modos TM',
                          text=modos_tm_labels, textposition='top center',
                          marker=dict(color='red', size=10)),
                row=2, col=2
            )

        # Configurar eixos
        fig.update_xaxes(title_text="Frequência (GHz)", row=1, col=1)
        fig.update_xaxes(title_text="Frequência (GHz)", row=1, col=2)
        fig.update_xaxes(title_text="Parte Real", row=2, col=1)
        fig.update_xaxes(title_text="Frequência (GHz)", row=2, col=2)

        fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Fase (°)", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Magnitude (dB)", row=1, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Fase (°)", row=1, col=2, secondary_y=True)
        fig.update_yaxes(title_text="Parte Imaginária", row=2, col=1)
        fig.update_yaxes(title_text="Tipo de Modo", row=2, col=2,
                        tickvals=[1, 2], ticktext=['TE', 'TM'])

        # Layout
        fig.update_layout(
            title='Análise de Parâmetros S - Cavidade Retangular',
            height=800,
            showlegend=True
        )

        return fig

    def _add_smith_chart_plotly(self, fig, S_param, row, col):
        """Adiciona uma carta de Smith completa ao gráfico Plotly"""

        # Círculo unitário (|Γ| = 1)
        theta = np.linspace(0, 2*np.pi, 200)
        fig.add_trace(
            go.Scatter(x=np.cos(theta), y=np.sin(theta),
                      mode='lines', name='Círculo Unitário',
                      line=dict(color='black', width=1.5),
                      showlegend=False),
            row=row, col=col
        )

        # Círculos de resistência constante
        r_values = [0.2, 0.5, 1.0, 2.0, 5.0]
        for r in r_values:
            center_x = r / (1 + r)
            center_y = 0
            radius = 1 / (1 + r)

            circle_theta = np.linspace(0, 2*np.pi, 100)
            x_circle = center_x + radius * np.cos(circle_theta)
            y_circle = center_y + radius * np.sin(circle_theta)

            # Filtrar pontos dentro do círculo unitário
            mask = x_circle**2 + y_circle**2 <= 1.001
            if np.any(mask):
                fig.add_trace(
                    go.Scatter(x=x_circle[mask], y=y_circle[mask],
                              mode='lines', name=f'R={r}',
                              line=dict(color='gray', width=0.8),
                              showlegend=False, hoverinfo='skip'),
                    row=row, col=col
                )

        # Círculos de reatância constante
        x_values = [0.2, 0.5, 1.0, 2.0, 5.0]
        for x in x_values:
            for sign in [1, -1]:
                x_val = sign * x
                center_x = 1
                center_y = 1 / x_val
                radius = abs(1 / x_val)

                circle_theta = np.linspace(0, 2*np.pi, 100)
                x_circle = center_x + radius * np.cos(circle_theta)
                y_circle = center_y + radius * np.sin(circle_theta)

                mask = x_circle**2 + y_circle**2 <= 1.001
                if np.any(mask):
                    fig.add_trace(
                        go.Scatter(x=x_circle[mask], y=y_circle[mask],
                                  mode='lines', name=f'X={x_val:.1f}',
                                  line=dict(color='lightblue', width=0.8),
                                  showlegend=False, hoverinfo='skip'),
                        row=row, col=col
                    )

        # Eixos principais
        fig.add_trace(
            go.Scatter(x=[-1.1, 1.1], y=[0, 0],
                      mode='lines', name='Eixo Real',
                      line=dict(color='black', width=1),
                      showlegend=False, hoverinfo='skip'),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=[0, 0], y=[-1.1, 1.1],
                      mode='lines', name='Eixo Imaginário',
                      line=dict(color='black', width=1),
                      showlegend=False, hoverinfo='skip'),
            row=row, col=col
        )

        # Plotar S11
        fig.add_trace(
            go.Scatter(x=np.real(S_param), y=np.imag(S_param),
                      mode='lines+markers', name='S11',
                      line=dict(color='blue', width=2.5),
                      marker=dict(size=3)),
            row=row, col=col
        )

        # Pontos inicial e final
        fig.add_trace(
            go.Scatter(x=[np.real(S_param)[0]], y=[np.imag(S_param)[0]],
                      mode='markers', name='Início',
                      marker=dict(color='green', size=8, symbol='circle')),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=[np.real(S_param)[-1]], y=[np.imag(S_param)[-1]],
                      mode='markers', name='Fim',
                      marker=dict(color='red', size=8, symbol='circle')),
            row=row, col=col
        )

    def exportar_dados_csv(self, filename="s_parameters.csv"):
        """Exporta os dados dos parâmetros S para arquivo CSV"""
        S11, S12, S21, S22 = self.calcular_matriz_s()
        freq_ghz = self.frequencies / 1e9

        try:
            import pandas as pd

            data = {
                'Frequency_GHz': freq_ghz,
                'S11_magnitude_dB': 20*np.log10(np.abs(S11)),
                'S11_phase_deg': np.angle(S11)*180/np.pi,
                'S21_magnitude_dB': 20*np.log10(np.abs(S21)),
                'S21_phase_deg': np.angle(S21)*180/np.pi,
                'S11_real': np.real(S11),
                'S11_imag': np.imag(S11),
                'S21_real': np.real(S21),
                'S21_imag': np.imag(S21)
            }

            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            print(f"Dados exportados para: {filename}")
        except ImportError:
            # Fallback sem pandas
            import csv
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Frequency_GHz', 'S11_magnitude_dB', 'S11_phase_deg',
                               'S21_magnitude_dB', 'S21_phase_deg'])
                for i in range(len(freq_ghz)):
                    writer.writerow([freq_ghz[i],
                                   20*np.log10(np.abs(S11[i])),
                                   np.angle(S11[i])*180/np.pi,
                                   20*np.log10(np.abs(S21[i])),
                                   np.angle(S21[i])*180/np.pi])
            print(f"Dados exportados para: {filename}")

# Exemplo de uso
if __name__ == "__main__":
    # Criar instância do modelo
    scattering = ScatteringMatrix(
        largura=22.86,      # WR-75 waveguide
        altura=10.16,
        comprimento=50.0,
        permissividade=1.0,
        permeabilidade=1.0,
        freq_min=8.0,
        freq_max=18.0,
        num_pontos=1000
    )

    # Gerar gráficos
    print("Gerando gráficos matplotlib...")
    fig_mpl = scattering.plot_s_parameters_matplotlib()
    plt.show()

    print("Gerando gráficos Plotly...")
    fig_plotly = scattering.plot_s_parameters_plotly()
    fig_plotly.show()

    # Exportar dados
    print("Exportando dados...")
    scattering.exportar_dados_csv("s_parameters_cavity.csv")