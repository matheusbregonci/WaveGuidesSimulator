"""
Modelo de Matriz de Espalhamento para Guias de Onda Cilíndricas
Implementação baseada na teoria apresentada no documento LaTeX

Baseado na teoria de guias cilíndricos e análise de microondas
Compatível com as configurações de simulação do Cilindrico_model.py
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cmath
from scipy.special import jv, jvp
from .Cilindrico_model import Modo_Cilindrico

class CylindricalScatteringMatrix:
    """
    Classe para calcular e visualizar a matriz de espalhamento de guias cilíndricos.

    Utiliza os parâmetros de configuração do Modo_Cilindrico para garantir compatibilidade.
    Implementa a teoria completa de reflexão e transmissão em guias cilíndricos.
    """

    def __init__(self,
                 raio=0.05,            # m - raio do cilindro
                 comprimento=0.1,      # m - comprimento do guia
                 permissividade=1.0,   # εr
                 permeabilidade=1.0,   # μr
                 freq_min=8.0,         # GHz
                 freq_max=18.0,        # GHz
                 num_pontos=1000):     # Número de pontos de frequência

        # Parâmetros geométricos (compatível com Cilindrico_model)
        self.raio = raio  # m
        self.comprimento = comprimento  # m

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

        # Modos cilíndricos principais (baseado nas tabelas de Bessel)
        self.modos_te = [(0,1), (1,1), (2,1), (0,2), (3,1), (1,2), (4,1), (2,2), (0,3)]
        self.modos_tm = [(0,1), (1,1), (2,1), (0,2), (3,1), (1,2), (4,1), (2,2)]

        # Tabelas de zeros das funções de Bessel (do documento LaTeX)
        self.p_nm_table = {
            # Zeros de J_n(x) = 0 para modos TM
            0: [2.405, 5.520, 8.654],  # n = 0
            1: [3.832, 7.016, 10.174], # n = 1
            2: [5.135, 8.417, 11.620]  # n = 2
        }

        self.p_nm_prime_table = {
            # Zeros de J'_n(x) = 0 para modos TE
            0: [1.841, 7.016, 10.174], # n = 0
            1: [3.832, 5.331, 8.536],  # n = 1
            2: [3.054, 6.706, 9.970]   # n = 2
        }

        # Fator de qualidade
        self.Q_factor = 1200  # Típico para guias cilíndricos

        # Instância do modelo cilíndrico para compatibilidade
        self.cilindrico_instance = None

    def criar_instancia_cilindrico(self, frequencia, n=0, m=1):
        """Cria uma instância do Modo_Cilindrico para uma frequência específica"""
        return Modo_Cilindrico(
            raio=self.raio,
            frequencia=frequencia,
            permissividade=self.epsilon_r,
            permeabilidade=self.mu_r,
            n=n, m=m, z=0
        )

    def obter_zero_bessel(self, n, m, modo_tipo='TE'):
        """
        Obtém o m-ésimo zero da função de Bessel para modo TE ou TM
        Baseado nas tabelas do documento LaTeX
        """
        try:
            if modo_tipo == 'TE':
                return self.p_nm_prime_table[n][m-1]  # J'_n
            else:  # TM
                return self.p_nm_table[n][m-1]  # J_n
        except (KeyError, IndexError):
            # Aproximação para modos de alta ordem não tabelados
            if modo_tipo == 'TE':
                return (m + n/2 - 1/4) * np.pi
            else:
                return (m + n/2 - 1/4) * np.pi

    def calcular_freq_corte_te(self, n, m):
        """
        Calcula frequência de corte para modo TE_nm
        f_c = k_c * c / (2π√(εᵣμᵣ))
        onde k_c = p'_nm / a
        """
        if n == 0 and m == 1:
            # TE01 é o modo dominante para cilindros
            p_prime = self.obter_zero_bessel(n, m, 'TE')
        else:
            p_prime = self.obter_zero_bessel(n, m, 'TE')

        k_c = p_prime / self.raio
        f_c = k_c * self.c0 / (2*np.pi*np.sqrt(self.epsilon_r * self.mu_r))
        return f_c

    def calcular_freq_corte_tm(self, n, m):
        """
        Calcula frequência de corte para modo TM_nm
        k_c = p_nm / a
        """
        p = self.obter_zero_bessel(n, m, 'TM')
        k_c = p / self.raio
        f_c = k_c * self.c0 / (2*np.pi*np.sqrt(self.epsilon_r * self.mu_r))
        return f_c

    def calcular_impedancia_te(self, freq, n, m):
        """
        Calcula impedância característica do modo TE_nm
        Z_TE = ωμ/β = η/√(1-(fc/f)²)
        """
        omega = 2*np.pi*freq
        k = omega * np.sqrt(self.epsilon_r * self.mu_r) / self.c0
        f_c = self.calcular_freq_corte_te(n, m)

        if freq <= f_c:
            # Modo evanescente
            k_c = 2*np.pi*f_c*np.sqrt(self.epsilon_r * self.mu_r) / self.c0
            gamma = np.sqrt(k_c**2 - k**2)
            Z_te = 1j * omega * self.mu0 * self.mu_r / gamma
        else:
            # Modo propagante
            eta = np.sqrt(self.mu0 * self.mu_r / (self.epsilon0 * self.epsilon_r))
            Z_te = eta / np.sqrt(1 - (f_c/freq)**2)

        return Z_te

    def calcular_impedancia_tm(self, freq, n, m):
        """
        Calcula impedância característica do modo TM_nm
        Z_TM = β/(ωε) = η√(1-(fc/f)²)
        """
        omega = 2*np.pi*freq
        k = omega * np.sqrt(self.epsilon_r * self.mu_r) / self.c0
        f_c = self.calcular_freq_corte_tm(n, m)

        if freq <= f_c:
            # Modo evanescente
            k_c = 2*np.pi*f_c*np.sqrt(self.epsilon_r * self.mu_r) / self.c0
            gamma = np.sqrt(k_c**2 - k**2)
            Z_tm = gamma / (1j * omega * self.epsilon0 * self.epsilon_r)
        else:
            # Modo propagante
            eta = np.sqrt(self.mu0 * self.mu_r / (self.epsilon0 * self.epsilon_r))
            Z_tm = eta * np.sqrt(1 - (f_c/freq)**2)

        return Z_tm

    def calcular_s11(self, freq):
        """
        Calcula parâmetro S11 (reflexão) para guia cilíndrico
        Baseado no modelo de múltiplos modos ressonantes cilíndricos
        """
        # Impedância de referência
        Z0 = 50.0  # Ohms

        s11_total = 0 + 0j
        peso_total = 0

        # Contribuição dos modos TE
        for n, m in self.modos_te:
            f_c = self.calcular_freq_corte_te(n, m)

            # Considerar apenas modos relevantes
            if f_c < self.freq_max * 1.5:
                Z_te = self.calcular_impedancia_te(freq, n, m)

                if np.isfinite(Z_te) and abs(Z_te) > 1e-12:
                    Gamma_te = (Z_te - Z0) / (Z_te + Z0)
                else:
                    Gamma_te = 1.0  # Reflexão total

                # Peso baseado na importância do modo
                if n == 0 and m == 1:
                    # TE01 é o modo dominante
                    peso_base = 0.5
                else:
                    peso_base = 0.2

                if f_c > 0:
                    peso = peso_base * np.exp(-abs(freq - f_c) / f_c * 3)
                    s11_total += Gamma_te * peso
                    peso_total += peso

        # Contribuição dos modos TM (menor para cilindros)
        for n, m in self.modos_tm[:4]:  # Apenas primeiros modos TM
            f_c = self.calcular_freq_corte_tm(n, m)

            if f_c < self.freq_max * 1.5:
                Z_tm = self.calcular_impedancia_tm(freq, n, m)

                if np.isfinite(Z_tm) and abs(Z_tm) > 1e-12:
                    Gamma_tm = (Z_tm - Z0) / (Z_tm + Z0)
                else:
                    Gamma_tm = 1.0

                if f_c > 0:
                    peso = 0.1 * np.exp(-abs(freq - f_c) / f_c * 3)
                    s11_total += Gamma_tm * peso
                    peso_total += peso

        # Normalização
        if peso_total > 0:
            s11_total = s11_total / (1 + peso_total * 0.05)

        # Linha de base para reflexão residual
        baseline = 0.03 * np.exp(1j * 2 * np.pi * freq * self.comprimento / self.c0)
        s11_final = s11_total + baseline

        # Garantir |S11| ≤ 1
        if abs(s11_final) > 0.98:
            s11_final = s11_final / abs(s11_final) * 0.98

        return s11_final

    def calcular_s21(self, freq):
        """
        Calcula parâmetro S21 (transmissão) para guia cilíndrico
        Baseado na propagação do modo dominante TE01
        """
        s11 = self.calcular_s11(freq)

        # Modo dominante TE01
        f_c_dominante = self.calcular_freq_corte_te(0, 1)

        if freq > f_c_dominante:
            # Modo propagante
            omega = 2*np.pi*freq
            k = omega * np.sqrt(self.epsilon_r * self.mu_r) / self.c0
            k_c = 2*np.pi*f_c_dominante*np.sqrt(self.epsilon_r * self.mu_r) / self.c0
            beta = np.sqrt(k**2 - k_c**2)

            # Perdas por atenuação (menores em cilindros que retangulares)
            alpha = 0.005 * freq / 1e9

            # Transmissão através do guia
            s21_base = np.exp(-1j * beta * self.comprimento) * np.exp(-alpha * self.comprimento)

            # Fator de transmissão baseado na reflexão
            s21_base *= (1 - abs(s11)**2)**0.5

        else:
            # Modo evanescente - atenuação exponencial
            omega = 2*np.pi*freq
            k = omega * np.sqrt(self.epsilon_r * self.mu_r) / self.c0
            k_c = 2*np.pi*f_c_dominante*np.sqrt(self.epsilon_r * self.mu_r) / self.c0
            gamma = np.sqrt(k_c**2 - k**2)

            # Forte atenuação para modos evanescentes
            s21_base = np.exp(-gamma * self.comprimento) * 0.05

        # Efeitos de ressonância (menores em guias cilíndricos)
        s21_resonance = 0 + 0j
        for n, m in [(0,1), (1,1), (0,2)]:  # Apenas modos principais
            f_c = self.calcular_freq_corte_te(n, m)
            if f_c > 0 and self.freq_min <= f_c <= self.freq_max:
                delta_f = (freq - f_c) / f_c
                resonance_factor = 0.03 / (1 + (delta_f * 8)**2)
                s21_resonance += resonance_factor * np.exp(1j * np.pi * delta_f / 2)

        s21_final = s21_base + s21_resonance

        # Conservação de energia
        energia_total = abs(s11)**2 + abs(s21_final)**2
        if energia_total > 1.0:
            fator_correcao = 0.99 / np.sqrt(energia_total)
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
        Cria gráficos dos parâmetros S para guia cilíndrico
        """
        S11, S12, S21, S22 = self.calcular_matriz_s()
        freq_ghz = self.frequencies / 1e9

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Parâmetros S - Guia Cilíndrico (raio={self.raio*1000:.1f} mm)',
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
        for n, m in self.modos_te[:6]:
            f_c = self.calcular_freq_corte_te(n, m)
            if self.freq_min <= f_c <= self.freq_max:
                freq_corte_te.append(f_c/1e9)
                labels_te.append(f'TE{n}{m}')

        freq_corte_tm = []
        labels_tm = []
        for n, m in self.modos_tm[:4]:
            f_c = self.calcular_freq_corte_tm(n, m)
            if self.freq_min <= f_c <= self.freq_max:
                freq_corte_tm.append(f_c/1e9)
                labels_tm.append(f'TM{n}{m}')

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
        """Plota carta de Smith (adaptada para guias cilíndricos)"""
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
        ax.set_title('Carta de Smith - S11 (Cilíndrico)')
        ax.legend(loc='upper right', fontsize=8)

    def comparar_com_cilindrico(self, frequencia_teste=12e9, n=0, m=1):
        """
        Compara os resultados com uma instância do Modo_Cilindrico
        """
        print(f"\n=== Comparacao com Modo_Cilindrico ===")
        print(f"Frequencia de teste: {frequencia_teste/1e9:.1f} GHz")
        print(f"Raio do guia: {self.raio*1000:.1f} mm")
        print(f"Modo: TE{n}{m}\n")

        # Criar instância Cilindrico
        cilindrico = self.criar_instancia_cilindrico(frequencia_teste, n, m)

        # Parâmetros básicos
        print("Parametros basicos do Modo_Cilindrico:")
        print(f"  omega = {cilindrico.omega_0:.2e} rad/s")
        print(f"  k = {cilindrico.k_val:.2f} rad/m")
        print(f"  k_c = {cilindrico.k_c_val:.2f} rad/m")
        print(f"  beta = {abs(cilindrico.beta_val):.2f} rad/m")

        # Frequências de corte calculadas aqui
        print(f"\nFrequencias de corte (este modelo):")
        for n_test, m_test in [(0,1), (1,1), (0,2), (2,1)]:
            f_c = self.calcular_freq_corte_te(n_test, m_test)
            print(f"  TE{n_test}{m_test}: {f_c/1e9:.2f} GHz")

        # Calcular S-parameters
        S11 = self.calcular_s11(frequencia_teste)
        S21 = self.calcular_s21(frequencia_teste)

        print(f"\nParametros S calculados:")
        print(f"  |S11| = {abs(S11):.3f} ({20*np.log10(abs(S11)):.1f} dB)")
        print(f"  Fase S11 = {np.angle(S11)*180/np.pi:.1f} graus")
        print(f"  |S21| = {abs(S21):.3f} ({20*np.log10(abs(S21)):.1f} dB)")
        print(f"  Fase S21 = {np.angle(S21)*180/np.pi:.1f} graus")

        # Verificar conservação de energia
        energia = abs(S11)**2 + abs(S21)**2
        print(f"  Conservacao de energia: |S11|² + |S21|² = {energia:.3f}")

    def exportar_resultados(self, filename="cylindrical_s_parameters.csv"):
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
            print("Pandas nao disponivel. Use numpy para salvar os dados.")


# Exemplo de uso
if __name__ == "__main__":
    # Teste com parâmetros típicos
    cylindrical_scattering = CylindricalScatteringMatrix(
        raio=0.015,        # 15 mm de raio
        comprimento=0.1,   # 10 cm de comprimento
        permissividade=1.0,
        permeabilidade=1.0,
        freq_min=8.0,
        freq_max=18.0,
        num_pontos=500
    )

    print("=== Teste do Modelo de Matriz de Espalhamento Cilindrico ===")

    # Comparar com Modo_Cilindrico
    cylindrical_scattering.comparar_com_cilindrico(12e9, 0, 1)

    # Gerar gráficos
    print("\nGerando graficos...")
    fig = cylindrical_scattering.plot_s_parameters_comparison()
    plt.show()

    # Exportar dados
    print("\nExportando dados...")
    cylindrical_scattering.exportar_resultados()