"""
Teste da implementação da Matriz de Espalhamento para Guias Retangulares
Validação com configurações existentes do simulador
"""

import sys
import os
sys.path.append('src')

try:
    from models.Rectangular_Scattering_model import RectangularScatteringMatrix
    from models.TEmn_model import Modo_TEmn
    import matplotlib.pyplot as plt
    import numpy as np

    print("="*60)
    print("TESTE DA MATRIZ DE ESPALHAMENTO PARA GUIAS RETANGULARES")
    print("="*60)

    # Configurações de teste baseadas no código existente
    configuracoes_teste = [
        {
            'nome': 'WR-75 Padrão',
            'largura': 22.86,    # mm
            'altura': 10.16,     # mm
            'comprimento': 50.0, # mm
            'freq_min': 8.0,     # GHz
            'freq_max': 18.0,    # GHz
            'freq_teste': 12.0   # GHz
        },
        {
            'nome': 'Guia Customizada',
            'largura': 25.0,     # mm
            'altura': 12.0,      # mm
            'comprimento': 30.0, # mm
            'freq_min': 6.0,     # GHz
            'freq_max': 15.0,    # GHz
            'freq_teste': 10.0   # GHz
        }
    ]

    for i, config in enumerate(configuracoes_teste):
        print(f"\n--- Teste {i+1}: {config['nome']} ---")

        # Criar instância da matriz de espalhamento
        rect_scattering = RectangularScatteringMatrix(
            largura=config['largura'],
            altura=config['altura'],
            comprimento=config['comprimento'],
            freq_min=config['freq_min'],
            freq_max=config['freq_max'],
            num_pontos=200
        )

        print(f"Dimensões: {config['largura']:.1f} × {config['altura']:.1f} × {config['comprimento']:.1f} mm")
        print(f"Faixa de frequência: {config['freq_min']:.1f} - {config['freq_max']:.1f} GHz")

        # Testar frequências de corte
        print("\nFrequências de corte calculadas:")
        modos_principais = [(1,0), (2,0), (0,1), (1,1), (2,1)]
        for m, n in modos_principais:
            if not (m == 0 and n == 0):
                f_c_te = rect_scattering.calcular_freq_corte_te(m, n)
                print(f"  TE{m}{n}: {f_c_te/1e9:.2f} GHz")

        # Calcular matriz S completa
        print("\nCalculando matriz de espalhamento...")
        S11, S12, S21, S22 = rect_scattering.calcular_matriz_s()

        # Estatísticas dos parâmetros S
        print(f"\nEstatísticas dos parâmetros S:")
        print(f"  |S11| médio: {np.mean(np.abs(S11)):.3f}")
        print(f"  |S11| máximo: {np.max(np.abs(S11)):.3f}")
        print(f"  |S21| médio: {np.mean(np.abs(S21)):.3f}")
        print(f"  |S21| mínimo: {np.min(np.abs(S21)):.3f}")

        # Verificar conservação de energia
        energia = np.abs(S11)**2 + np.abs(S21)**2
        print(f"  Conservação de energia (média): {np.mean(energia):.3f}")
        print(f"  Conservação de energia (máximo): {np.max(energia):.3f}")

        if np.max(energia) > 1.01:
            print("  AVISO: Violacao da conservacao de energia detectada!")
        else:
            print("  Conservacao de energia OK")

        # Testar comparação com TEmn_model
        freq_teste = config['freq_teste'] * 1e9
        print(f"\nComparacao com TEmn_model em {config['freq_teste']:.1f} GHz:")

        try:
            rect_scattering.comparar_com_temn(freq_teste)
            print("  Comparacao com TEmn_model executada com sucesso")
        except Exception as e:
            print(f"  Erro na comparacao: {e}")

        # Gerar gráfico
        print(f"\nGerando grafico para {config['nome']}...")
        try:
            fig = rect_scattering.plot_s_parameters_comparison()

            # Salvar gráfico
            nome_arquivo = f"s_parameters_{config['nome'].replace(' ', '_').lower()}.png"
            plt.savefig(nome_arquivo, dpi=150, bbox_inches='tight')
            print(f"  Grafico salvo: {nome_arquivo}")
            plt.close(fig)

        except Exception as e:
            print(f"  Erro ao gerar grafico: {e}")

    # Teste de validação teórica
    print(f"\n--- Teste de Validacao Teorica ---")

    # Teste 1: Verificar modo dominante TE10
    rect_test = RectangularScatteringMatrix(largura=22.86, altura=10.16)
    f_c_te10 = rect_test.calcular_freq_corte_te(1, 0)
    f_c_te01 = rect_test.calcular_freq_corte_te(0, 1)

    print(f"TE10 (modo dominante): {f_c_te10/1e9:.2f} GHz")
    print(f"TE01: {f_c_te01/1e9:.2f} GHz")

    if f_c_te10 < f_c_te01:
        print("OK: TE10 e corretamente o modo dominante")
    else:
        print("ERRO: TE10 deveria ser o modo dominante")

    # Teste 2: Verificar que TM modes têm frequência de corte mais alta
    f_c_tm11 = rect_test.calcular_freq_corte_tm(1, 1)
    print(f"TM11 (primeiro TM): {f_c_tm11/1e9:.2f} GHz")

    if f_c_tm11 > f_c_te10:
        print("OK: TM11 tem frequencia de corte maior que TE10")
    else:
        print("ERRO: TM11 deveria ter frequencia de corte maior que TE10")

    # Teste 3: Verificar comportamento em frequências baixas vs altas
    freq_baixa = 5e9   # Abaixo do corte
    freq_alta = 15e9   # Acima do corte

    S11_baixa = rect_test.calcular_s11(freq_baixa)
    S21_baixa = rect_test.calcular_s21(freq_baixa)
    S11_alta = rect_test.calcular_s11(freq_alta)
    S21_alta = rect_test.calcular_s21(freq_alta)

    print(f"\nComportamento por frequencia:")
    print(f"  {freq_baixa/1e9:.1f} GHz: |S11|={abs(S11_baixa):.3f}, |S21|={abs(S21_baixa):.3f}")
    print(f"  {freq_alta/1e9:.1f} GHz: |S11|={abs(S11_alta):.3f}, |S21|={abs(S21_alta):.3f}")

    if abs(S11_baixa) > abs(S11_alta) and abs(S21_alta) > abs(S21_baixa):
        print("OK: Comportamento correto - mais reflexao em baixa freq, mais transmissao em alta freq")
    else:
        print("AVISO: Comportamento pode precisar de ajustes")

    print(f"\n{'='*60}")
    print("TESTE CONCLUIDO!")
    print("Verifique os arquivos de graficos gerados.")
    print("Para analise detalhada, examine os parametros S em diferentes frequencias.")
    print(f"{'='*60}")

except ImportError as e:
    print(f"Erro de importacao: {e}")
    print("Certifique-se de que todos os modulos estao disponiveis.")
    print("Estrutura esperada:")
    print("  src/")
    print("    models/")
    print("      Rectangular_Scattering_model.py")
    print("      TEmn_model.py")

except Exception as e:
    print(f"Erro durante execucao: {e}")
    import traceback
    traceback.print_exc()