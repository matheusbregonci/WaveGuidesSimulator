"""
Modelo de Matriz de Espalhamento para Guias de Onda Retangulares
Implementação baseada na teoria apresentada no documento LaTeX

Baseado na teoria de guias retangulares e análise de microondas
Compatível com as configurações de simulação do TEmn_model.py
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cmath
from .TEmn_model import Modo_TEmn

class RectangularScatteringMatrix:
    """
    Classe para calcular e visualizar a matriz de espalhamento de guias retangulares.

    Utiliza os parâmetros de configuração do Modo_TEmn para garantir compatibilidade.
    """

    def __init__(self,
                 largura=22.86,        # mm - dimensão a
                 altura=10.16,         # mm - dimensão b
                 comprimento=50.0,     # mm - dimensão c (profundidade)
                 permissividade=1.0,   # εr
                 permeabilidade=1.0,   # μr
                 freq_min=8.0,         # GHz
                 freq_max=18.0,        # GHz
                 num_pontos=1000):     # Número de pontos de frequência

        # Dimensões da guia (compatível com TEmn_model)
        self.largura_mm = largura
        self.altura_mm = altura
        self.a = largura / 1000.0  # converter mm para m
        self.b = altura / 1000.0   # converter mm para m
        self.c = comprimento / 1000.0  # comprimento em m

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

        # Modos TE dominantes (baseado na teoria de guias retangulares)
        self.modos_te = [(1,0), (2,0), (0,1), (1,1), (2,1), (3,0), (0,2), (1,2), (3,1)]

        # Para guias retangulares, TM11 é o primeiro modo TM
        self.modos_tm = [(1,1), (2,1), (1,2), (3,1), (2,2), (4,1), (1,3), (3,2)]

        # Fator de qualidade e perdas
        self.Q_factor = 800  # Fator Q típico para guias retangulares

        # Instância do modelo TEmn para compatibilidade
        self.temn_instance = None

    def criar_instancia_temn(self, frequencia):
        """Cria uma instância do Modo_TEmn para uma frequência específica"""
        return Modo_TEmn(
            largura=self.largura_mm,
            altura=self.altura_mm,
            frequencia=frequencia,
            permissividade=self.epsilon_r,
            permeabilidade=self.mu_r,
            plano='xy'
        )

    def calcular_freq_corte_te(self, m, n):
        """
        Calcula frequência de corte para modo TE_mn
        Baseado na teoria de guias retangulares
        """
        if m == 0 and n == 0:
            return np.inf  # TE00 não existe

        # Número de onda de corte
        k_c = np.pi * np.sqrt((m/self.a)**2 + (n/self.b)**2)

        # Frequência de corte
        f_c = k_c * self.c0 / (2*np.pi*np.sqrt(self.epsilon_r * self.mu_r))
        return f_c

    def calcular_freq_corte_tm(self, m, n):
        """
        Calcula frequência de corte para modo TM_mn
        TM_mn só existe para m ≥ 1 e n ≥ 1
        """
        if m == 0 or n == 0:
            return np.inf  # TM_m0 ou TM_0n não existem

        # Mesmo cálculo que TE para guias retangulares
        k_c = np.pi * np.sqrt((m/self.a)**2 + (n/self.b)**2)
        f_c = k_c * self.c0 / (2*np.pi*np.sqrt(self.epsilon_r * self.mu_r))
        return f_c

    def calcular_impedancia_te(self, freq, m, n):
        """Calcula impedância característica do modo TE_mn"""
        omega = 2*np.pi*freq
        k = omega * np.sqrt(self.epsilon_r * self.mu_r) / self.c0

        if m == 0 and n == 0:
            return np.inf

        k_c = np.pi * np.sqrt((m/self.a)**2 + (n/self.b)**2)

        # Impedância TE
        if k_c**2 > k**2:
            # Modo evanescente
            gamma = np.sqrt(k_c**2 - k**2)
            Z_te = 1j * omega * self.mu0 * self.mu_r / gamma
        else:
            # Modo propagante
            beta = np.sqrt(k**2 - k_c**2)
            Z_te = omega * self.mu0 * self.mu_r / beta

        return Z_te

    def calcular_impedancia_tm(self, freq, m, n):
        """Calcula impedância característica do modo TM_mn"""
        if m == 0 or n == 0:
            return np.inf

        omega = 2*np.pi*freq
        k = omega * np.sqrt(self.epsilon_r * self.mu_r) / self.c0
        k_c = np.pi * np.sqrt((m/self.a)**2 + (n/self.b)**2)

        # Impedância TM
        if k_c**2 > k**2:
            # Modo evanescente
            gamma = np.sqrt(k_c**2 - k**2)
            Z_tm = gamma / (1j * omega * self.epsilon0 * self.epsilon_r)
        else:
            # Modo propagante
            beta = np.sqrt(k**2 - k_c**2)
            Z_tm = beta / (omega * self.epsilon0 * self.epsilon_r)

        return Z_tm

    def calcular_s11(self, freq):
        """
        Calcula parâmetro S11 (reflexão) para guia retangular
        Baseado no modelo de múltiplos modos ressonantes
        """
        # Impedância de referência (geralmente 50Ω ou baseada no modo dominante)
        Z0 = 50.0  # Ohms

        s11_total = 0 + 0j
        peso_total = 0

        # Contribuição dos modos TE
        for m, n in self.modos_te:
            if m == 0 and n == 0:
                continue

            f_c = self.calcular_freq_corte_te(m, n)

            # Considerar apenas modos próximos à faixa de frequência
            if f_c < self.freq_max * 2:  # Margem para efeitos de modos de alta ordem
                Z_te = self.calcular_impedancia_te(freq, m, n)

                # Coeficiente de reflexão do modo
                if np.isfinite(Z_te) and abs(Z_te) > 1e-12:
                    Gamma_te = (Z_te - Z0) / (Z_te + Z0)
                else:
                    Gamma_te = 1.0  # Reflexão total para modos não propagantes

                # Peso baseado na proximidade da frequência de corte
                if f_c > 0:
                    peso = np.exp(-abs(freq - f_c) / f_c * 2)
                    s11_total += Gamma_te * peso * 0.3  # Peso menor para TE
                    peso_total += peso

        # Contribuição dos modos TM
        for m, n in self.modos_tm:
            f_c = self.calcular_freq_corte_tm(m, n)

            if f_c < self.freq_max * 2:
                Z_tm = self.calcular_impedancia_tm(freq, m, n)

                if np.isfinite(Z_tm) and abs(Z_tm) > 1e-12:
                    Gamma_tm = (Z_tm - Z0) / (Z_tm + Z0)
                else:
                    Gamma_tm = 1.0

                if f_c > 0:
                    peso = np.exp(-abs(freq - f_c) / f_c * 2)
                    s11_total += Gamma_tm * peso * 0.2  # Peso menor para TM
                    peso_total += peso

        # Normalização
        if peso_total > 0:
            s11_total = s11_total / (1 + peso_total * 0.1)

        # Linha de base para reflexão residual
        baseline = 0.02 * np.exp(1j * 2 * np.pi * freq * self.c / self.c0)
        s11_final = s11_total + baseline

        # Garantir |S11| ≤ 1
        if abs(s11_final) > 0.95:
            s11_final = s11_final / abs(s11_final) * 0.95

        return s11_final

    def calcular_s21(self, freq):
        """
        Calcula parâmetro S21 (transmissão) para guia retangular
        Considera propagação através da guia com perdas
        """
        s11 = self.calcular_s11(freq)

        # Modo dominante TE10 para guias retangulares padrão
        f_c_dominante = self.calcular_freq_corte_te(1, 0)

        if freq > f_c_dominante:
            # Modo propagante - calcular constante de propagação
            omega = 2*np.pi*freq
            k = omega * np.sqrt(self.epsilon_r * self.mu_r) / self.c0
            k_c = np.pi / self.a  # TE10
            beta = np.sqrt(k**2 - k_c**2)

            # Fator de perdas
            alpha = 0.01 * freq / 1e9  # Perdas crescem com frequência

            # Transmissão através da guia
            s21_base = np.exp(-1j * beta * self.c) * np.exp(-alpha * self.c)

            # Reduzir transmissão se houver forte reflexão
            s21_base *= (1 - abs(s11)**2)**0.5

        else:
            # Modo evanescente - forte atenuação
            omega = 2*np.pi*freq
            k = omega * np.sqrt(self.epsilon_r * self.mu_r) / self.c0
            k_c = np.pi / self.a
            gamma = np.sqrt(k_c**2 - k**2)

            s21_base = np.exp(-gamma * self.c) * 0.1  # Transmissão muito baixa

        # Contribuição de ressonâncias (menor que em cavidades)
        s21_resonance = 0 + 0j
        for m, n in self.modos_te[:3]:  # Apenas alguns modos principais
            f_c = self.calcular_freq_corte_te(m, n)
            if f_c > 0 and self.freq_min <= f_c <= self.freq_max:
                delta_f = (freq - f_c) / f_c
                resonance_factor = 0.05 / (1 + (delta_f * 10)**2)
                s21_resonance += resonance_factor * np.exp(1j * np.pi * delta_f)

        s21_final = s21_base + s21_resonance

        # Conservação de energia: |S11|² + |S21|² ≤ 1
        energia_total = abs(s11)**2 + abs(s21_final)**2
        if energia_total > 1.0:
            fator_correcao = 0.98 / np.sqrt(energia_total)
            s21_final *= fator_correcao

        return s21_final

    def calcular_matriz_s(self):
        """Calcula a matriz de espalhamento completa"""
        S11 = np.zeros(len(self.frequencies), dtype=complex)
        S12 = np.zeros(len(self.frequencies), dtype=complex)
        S21 = np.zeros(len(self.frequencies), dtype=complex)
        S22 = np.zeros(len(self.frequencies), dtype=complex)

        for i, freq in enumerate(self.frequencies):
            S11[i] = self.calcular_s11(freq)
            S21[i] = self.calcular_s21(freq)

            # Por reciprocidade: S12 = S21
            S12[i] = S21[i]

            # Por simetria (guia uniforme): S22 = S11
            S22[i] = S11[i]

        return S11, S12, S21, S22

    def plot_s_parameters_comparison(self):
        """
        Cria gráficos comparando os parâmetros S com uma instância do TEmn_model
        """
        S11, S12, S21, S22 = self.calcular_matriz_s()
        freq_ghz = self.frequencies / 1e9

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Parâmetros S - Guia Retangular {self.largura_mm:.1f}×{self.altura_mm:.1f} mm',
                     fontsize=16, fontweight='bold')

        # S11 - Magnitude e Fase
        ax1.plot(freq_ghz, 20*np.log10(np.abs(S11)), 'b-', linewidth=2, label='|S11| (dB)')
        ax1_phase = ax1.twinx()
        ax1_phase.plot(freq_ghz, np.angle(S11)*180/np.pi, 'r--', linewidth=1, label='∠S11 (°)')
        ax1.set_xlabel('Frequência (GHz)')
        ax1.set_ylabel('|S11| (dB)', color='b')
        ax1_phase.set_ylabel('Fase (°)', color='r')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('S11 - Reflexão')

        # S21 - Magnitude e Fase
        ax2.plot(freq_ghz, 20*np.log10(np.abs(S21)), 'g-', linewidth=2, label='|S21| (dB)')
        ax2_phase = ax2.twinx()
        ax2_phase.plot(freq_ghz, np.angle(S21)*180/np.pi, 'm--', linewidth=1, label='∠S21 (°)')
        ax2.set_xlabel('Frequência (GHz)')
        ax2.set_ylabel('|S21| (dB)', color='g')
        ax2_phase.set_ylabel('Fase (°)', color='m')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('S21 - Transmissão')

        # Carta de Smith para S11
        self._plot_smith_chart(ax3, S11)

        # Frequências de corte dos modos
        freq_corte_te = []
        labels_te = []
        for m, n in self.modos_te[:6]:  # Primeiros 6 modos
            if not (m == 0 and n == 0):
                f_c = self.calcular_freq_corte_te(m, n)
                if self.freq_min <= f_c <= self.freq_max:
                    freq_corte_te.append(f_c/1e9)
                    labels_te.append(f'TE{m}{n}')

        freq_corte_tm = []
        labels_tm = []
        for m, n in self.modos_tm[:4]:  # Primeiros 4 modos TM
            f_c = self.calcular_freq_corte_tm(m, n)
            if self.freq_min <= f_c <= self.freq_max:
                freq_corte_tm.append(f_c/1e9)
                labels_tm.append(f'TM{m}{n}')

        if freq_corte_te:
            ax4.scatter(freq_corte_te, [1]*len(freq_corte_te), c='blue', s=100, alpha=0.7, label='Modos TE')
            for freq, label in zip(freq_corte_te, labels_te):
                ax4.annotate(label, (freq, 1), xytext=(5, 5), textcoords='offset points', fontsize=8)

        if freq_corte_tm:
            ax4.scatter(freq_corte_tm, [2]*len(freq_corte_tm), c='red', s=100, alpha=0.7, label='Modos TM')
            for freq, label in zip(freq_corte_tm, labels_tm):
                ax4.annotate(label, (freq, 2), xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax4.set_xlabel('Frequência (GHz)')
        ax4.set_ylabel('Tipo de Modo')
        ax4.set_yticks([1, 2])
        ax4.set_yticklabels(['TE', 'TM'])
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_title('Frequências de Corte dos Modos')
        ax4.set_xlim(freq_ghz[0], freq_ghz[-1])

        plt.tight_layout()
        return fig

    def _plot_smith_chart(self, ax, S_param):
        """Plota carta de Smith (mesmo código da classe ScatteringMatrix)"""
        # Círculo unitário
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1.5, alpha=0.8)

        # Círculos de resistência constante
        r_values = [0.2, 0.5, 1.0, 2.0, 5.0]
        for r in r_values:
            center_x = r / (1 + r)
            center_y = 0
            radius = 1 / (1 + r)

            circle_theta = np.linspace(0, 2*np.pi, 100)
            x_circle = center_x + radius * np.cos(circle_theta)
            y_circle = center_y + radius * np.sin(circle_theta)

            mask = x_circle**2 + y_circle**2 <= 1.001
            if np.any(mask):
                ax.plot(x_circle[mask], y_circle[mask], 'gray', linewidth=0.8, alpha=0.6)

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
                    ax.plot(x_circle[mask], y_circle[mask], 'lightblue', linewidth=0.8, alpha=0.6)

        # Eixos principais
        ax.axhline(y=0, color='k', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='k', linewidth=1, alpha=0.5)

        # Plotar S11
        ax.plot(np.real(S_param), np.imag(S_param), 'b-', linewidth=2.5, label='S11')
        ax.plot(np.real(S_param)[0], np.imag(S_param)[0], 'go', markersize=6, label='Início')
        ax.plot(np.real(S_param)[-1], np.imag(S_param)[-1], 'ro', markersize=6, label='Fim')

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Carta de Smith - S11')
        ax.legend(loc='upper right', fontsize=8)

    def comparar_com_temn(self, frequencia_teste=12e9):
        """
        Compara os resultados com uma instância do TEmn_model
        para validação da implementação
        """
        print(f"\n=== Comparação com TEmn_model ===")
        print(f"Frequência de teste: {frequencia_teste/1e9:.1f} GHz")
        print(f"Dimensões da guia: {self.largura_mm:.1f} × {self.altura_mm:.1f} mm\n")

        # Criar instância TEmn
        temn = self.criar_instancia_temn(frequencia_teste)

        # Calcular parâmetros básicos
        print("Parametros basicos do TEmn_model:")
        print(f"  omega = {temn.omega:.2e} rad/s")
        print(f"  k = {temn.k:.2f} rad/m")
        print(f"  k_c = {temn.k_c:.2f} rad/m")
        print(f"  beta = {abs(temn.beta):.2f} rad/m")

        # Frequências de corte calculadas aqui
        print(f"\nFrequências de corte (este modelo):")
        for m, n in [(1,0), (2,0), (0,1), (1,1)]:
            if not (m == 0 and n == 0):
                f_c = self.calcular_freq_corte_te(m, n)
                print(f"  TE{m}{n}: {f_c/1e9:.2f} GHz")

        # Calcular S-parameters nesta frequência
        S11 = self.calcular_s11(frequencia_teste)
        S21 = self.calcular_s21(frequencia_teste)

        print(f"\nParâmetros S calculados:")
        print(f"  |S11| = {abs(S11):.3f} ({20*np.log10(abs(S11)):.1f} dB)")
        print(f"  ∠S11 = {np.angle(S11)*180/np.pi:.1f}°")
        print(f"  |S21| = {abs(S21):.3f} ({20*np.log10(abs(S21)):.1f} dB)")
        print(f"  ∠S21 = {np.angle(S21)*180/np.pi:.1f}°")

        # Verificar conservação de energia
        energia = abs(S11)**2 + abs(S21)**2
        print(f"  Conservação de energia: |S11|² + |S21|² = {energia:.3f}")

    def exportar_resultados(self, filename="rectangular_s_parameters.csv"):
        """Exporta os resultados para arquivo CSV"""
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
            print(f"Resultados exportados para: {filename}")
        except ImportError:
            print("Pandas não disponível. Use numpy para salvar os dados.")


# Exemplo de uso e teste
if __name__ == "__main__":
    # Criar instância com parâmetros típicos de guia WR-75
    rectangular_scattering = RectangularScatteringMatrix(
        largura=22.86,      # WR-75
        altura=10.16,
        comprimento=50.0,
        permissividade=1.0,
        permeabilidade=1.0,
        freq_min=8.0,
        freq_max=18.0,
        num_pontos=500
    )

    print("=== Teste do Modelo de Matriz de Espalhamento Retangular ===")

    # Comparar com TEmn_model
    rectangular_scattering.comparar_com_temn(12e9)

    # Gerar gráficos
    print("\nGerando gráficos...")
    fig = rectangular_scattering.plot_s_parameters_comparison()
    plt.show()

    # Exportar dados
    print("\nExportando dados...")
    rectangular_scattering.exportar_resultados()