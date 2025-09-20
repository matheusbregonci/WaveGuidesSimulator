"""
Teste da implementação da Matriz de Espalhamento para Guias Cilíndricas
Validação com configurações existentes do simulador
Baseado na teoria do documento LaTeX
"""

import sys
import os
sys.path.append('src')

try:
    from models.Cylindrical_Scattering_model import CylindricalScatteringMatrix
    from models.Cilindrico_model import Modo_Cilindrico
    import matplotlib.pyplot as plt
    import numpy as np

    print("="*65)
    print("TESTE DA MATRIZ DE ESPALHAMENTO PARA GUIAS CILINDRICAS")
    print("="*65)

    # Configurações de teste baseadas no código existente
    configuracoes_teste = [
        {
            'nome': 'Guia Cilindrica Pequena',
            'raio': 0.015,       # 15 mm
            'comprimento': 0.1,  # 10 cm
            'freq_min': 6.0,     # GHz
            'freq_max': 18.0,    # GHz
            'freq_teste': 12.0   # GHz
        },
        {
            'nome': 'Guia Cilindrica Media',
            'raio': 0.025,       # 25 mm
            'comprimento': 0.05, # 5 cm
            'freq_min': 4.0,     # GHz
            'freq_max': 15.0,    # GHz
            'freq_teste': 10.0   # GHz
        },
        {
            'nome': 'Guia Cilindrica Grande',
            'raio': 0.05,        # 50 mm (compatível com exemplo no Cilindrico_model)
            'comprimento': 0.2,  # 20 cm
            'freq_min': 2.0,     # GHz
            'freq_max': 8.0,     # GHz
            'freq_teste': 5.0    # GHz
        }
    ]

    for i, config in enumerate(configuracoes_teste):
        print(f"\n--- Teste {i+1}: {config['nome']} ---")

        # Criar instância da matriz de espalhamento cilíndrica
        cyl_scattering = CylindricalScatteringMatrix(
            raio=config['raio'],
            comprimento=config['comprimento'],
            freq_min=config['freq_min'],
            freq_max=config['freq_max'],
            num_pontos=200
        )

        print(f"Raio: {config['raio']*1000:.1f} mm")
        print(f"Comprimento: {config['comprimento']*1000:.1f} mm")
        print(f"Faixa de frequencia: {config['freq_min']:.1f} - {config['freq_max']:.1f} GHz")

        # Testar frequências de corte principais
        print("\nFrequencias de corte calculadas:")
        modos_principais = [(0,1), (1,1), (2,1), (0,2), (3,1)]
        for n, m in modos_principais:
            try:
                f_c_te = cyl_scattering.calcular_freq_corte_te(n, m)
                f_c_tm = cyl_scattering.calcular_freq_corte_tm(n, m)
                print(f"  TE{n}{m}: {f_c_te/1e9:.2f} GHz")
                print(f"  TM{n}{m}: {f_c_tm/1e9:.2f} GHz")
            except Exception as e:
                print(f"  Erro no modo {n}{m}: {e}")

        # Identificar modo dominante
        try:
            f_c_te01 = cyl_scattering.calcular_freq_corte_te(0, 1)
            f_c_tm01 = cyl_scattering.calcular_freq_corte_tm(0, 1)
            f_c_te11 = cyl_scattering.calcular_freq_corte_te(1, 1)

            print(f"\nAnalise do modo dominante:")
            print(f"  TE01: {f_c_te01/1e9:.2f} GHz")
            print(f"  TM01: {f_c_tm01/1e9:.2f} GHz")
            print(f"  TE11: {f_c_te11/1e9:.2f} GHz")

            dominante_freq = min(f_c_te01, f_c_tm01, f_c_te11)
            if dominante_freq == f_c_te01:
                print("  Modo dominante: TE01 (correto para guias circulares)")
            elif dominante_freq == f_c_tm01:
                print("  Modo dominante: TM01")
            else:
                print("  Modo dominante: TE11")

        except Exception as e:
            print(f"  Erro na analise do modo dominante: {e}")

        # Calcular matriz S completa
        print("\nCalculando matriz de espalhamento...")
        try:
            S11, S12, S21, S22 = cyl_scattering.calcular_matriz_s()

            # Estatísticas dos parâmetros S
            print(f"\nEstatisticas dos parametros S:")
            print(f"  |S11| medio: {np.mean(np.abs(S11)):.3f}")
            print(f"  |S11| maximo: {np.max(np.abs(S11)):.3f}")
            print(f"  |S21| medio: {np.mean(np.abs(S21)):.3f}")
            print(f"  |S21| minimo: {np.min(np.abs(S21)):.3f}")

            # Verificar conservação de energia
            energia = np.abs(S11)**2 + np.abs(S21)**2
            print(f"  Conservacao de energia (media): {np.mean(energia):.3f}")
            print(f"  Conservacao de energia (maximo): {np.max(energia):.3f}")

            if np.max(energia) > 1.01:
                print("  AVISO: Violacao da conservacao de energia detectada!")
            else:
                print("  Conservacao de energia OK")

        except Exception as e:
            print(f"  Erro no calculo da matriz S: {e}")

        # Testar comparação com Modo_Cilindrico
        freq_teste = config['freq_teste'] * 1e9
        print(f"\nComparacao com Modo_Cilindrico em {config['freq_teste']:.1f} GHz:")

        try:
            cyl_scattering.comparar_com_cilindrico(freq_teste, n=0, m=1)
            print("  Comparacao executada com sucesso")
        except Exception as e:
            print(f"  Erro na comparacao: {e}")

        # Gerar gráfico
        print(f"\nGerando grafico para {config['nome']}...")
        try:
            fig = cyl_scattering.plot_s_parameters_comparison()

            # Salvar gráfico
            nome_arquivo = f"s_parameters_cyl_{config['nome'].replace(' ', '_').lower()}.png"
            plt.savefig(nome_arquivo, dpi=150, bbox_inches='tight')
            print(f"  Grafico salvo: {nome_arquivo}")
            plt.close(fig)

        except Exception as e:
            print(f"  Erro ao gerar grafico: {e}")

    # Teste de validação teórica específica para cilindros
    print(f"\n--- Teste de Validacao Teorica (Cilindros) ---")

    # Teste 1: Verificar que TE01 é o modo dominante
    cyl_test = CylindricalScatteringMatrix(raio=0.02)  # 20 mm de raio
    try:
        f_c_te01 = cyl_test.calcular_freq_corte_te(0, 1)
        f_c_tm01 = cyl_test.calcular_freq_corte_tm(0, 1)
        f_c_te11 = cyl_test.calcular_freq_corte_te(1, 1)

        print(f"TE01: {f_c_te01/1e9:.2f} GHz")
        print(f"TM01: {f_c_tm01/1e9:.2f} GHz")
        print(f"TE11: {f_c_te11/1e9:.2f} GHz")

        if f_c_te01 < f_c_tm01 and f_c_te01 < f_c_te11:
            print("OK: TE01 e o modo dominante (correto para guias circulares)")
        else:
            print("AVISO: TE01 deveria ser o modo dominante")

    except Exception as e:
        print(f"Erro no teste do modo dominante: {e}")

    # Teste 2: Verificar impedâncias características
    print(f"\nTeste de impedancias caracteristicas:")
    freq_teste = 10e9  # 10 GHz
    try:
        Z_te01 = cyl_test.calcular_impedancia_te(freq_teste, 0, 1)
        Z_tm01 = cyl_test.calcular_impedancia_tm(freq_teste, 0, 1)

        print(f"Z_TE01 = {Z_te01:.2f} Ohms")
        print(f"Z_TM01 = {Z_tm01:.2f} Ohms")

        # Verificar se são finitas e positivas
        if np.isfinite(Z_te01) and np.real(Z_te01) > 0:
            print("OK: Impedancia TE01 fisicamente consistente")
        else:
            print("AVISO: Impedancia TE01 pode ter problemas")

        if np.isfinite(Z_tm01) and np.real(Z_tm01) > 0:
            print("OK: Impedancia TM01 fisicamente consistente")
        else:
            print("AVISO: Impedancia TM01 pode ter problemas")

    except Exception as e:
        print(f"Erro no teste de impedancias: {e}")

    # Teste 3: Comportamento por frequência
    print(f"\nTeste comportamental por frequencia:")
    try:
        freq_baixa = 3e9   # Abaixo do corte
        freq_alta = 15e9   # Acima do corte

        S11_baixa = cyl_test.calcular_s11(freq_baixa)
        S21_baixa = cyl_test.calcular_s21(freq_baixa)
        S11_alta = cyl_test.calcular_s11(freq_alta)
        S21_alta = cyl_test.calcular_s21(freq_alta)

        print(f"  {freq_baixa/1e9:.1f} GHz: |S11|={abs(S11_baixa):.3f}, |S21|={abs(S21_baixa):.3f}")
        print(f"  {freq_alta/1e9:.1f} GHz: |S11|={abs(S11_alta):.3f}, |S21|={abs(S21_alta):.3f}")

        if abs(S21_alta) > abs(S21_baixa):
            print("OK: Maior transmissao em frequencias mais altas")
        else:
            print("AVISO: Comportamento de transmissao inesperado")

    except Exception as e:
        print(f"Erro no teste comportamental: {e}")

    # Teste 4: Comparação com funções de Bessel do Cilindrico_model
    print(f"\nTeste de compatibilidade com funcoes de Bessel:")
    try:
        # Criar instância do Modo_Cilindrico para comparação
        modo_cil = Modo_Cilindrico(raio=0.02, frequencia=10e9, n=0, m=1)

        # Comparar zeros de Bessel
        p_01_teorico = cyl_test.obter_zero_bessel(0, 1, 'TE')  # p'_01
        print(f"Zero da funcao de Bessel J'_0 (m=1): {p_01_teorico:.3f}")
        print(f"Valor tabelado esperado: 1.841")

        if abs(p_01_teorico - 1.841) < 0.01:
            print("OK: Zeros de Bessel corretos")
        else:
            print("AVISO: Zeros de Bessel podem estar incorretos")

    except Exception as e:
        print(f"Erro no teste de compatibilidade: {e}")

    print(f"\n{'='*65}")
    print("TESTE CONCLUIDO!")
    print("Verifique os arquivos de graficos gerados.")
    print("Para analise detalhada:")
    print("1. Compare as frequencias de corte com valores teoricos")
    print("2. Verifique a conservacao de energia em todas as frequencias")
    print("3. Analise o comportamento da carta de Smith")
    print("4. Compare com medidas experimentais se disponiveis")
    print(f"{'='*65}")

except ImportError as e:
    print(f"Erro de importacao: {e}")
    print("Certifique-se de que todos os modulos estao disponiveis.")
    print("Estrutura esperada:")
    print("  src/")
    print("    models/")
    print("      Cylindrical_Scattering_model.py")
    print("      Cilindrico_model.py")

except Exception as e:
    print(f"Erro durante execucao: {e}")
    import traceback
    traceback.print_exc()