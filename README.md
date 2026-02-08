# Waveguide Simulator - Ambiente de Distribuição

## 📦 **Pacote Completo para Execução**

Este é o ambiente isolado do **Simulador de Guias de Onda Eletromagnéticas** pronto para distribuição e criação de executável.

## 🚀 **Opções de Execução**

### **Opção 1: Executar com Python (Recomendado)**
```bash
# Execute o arquivo batch (Windows)
install_and_run.bat

# Ou manualmente:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run_app.py
```

### **Opção 2: Criar Executável**
```bash
# Execute o build script (Windows)
build_executable.bat

# Ou manualmente:
pip install cx_Freeze
python setup.py build
```

## 📁 **Estrutura do Projeto**

```
WaveguideSimulator_App/
├── 📄 launch.py              # Launcher principal do Streamlit
├── 📄 run_app.py             # Ponto de entrada para executável
├── 📄 requirements.txt       # Dependências Python
├── 📄 setup.py              # Configuração para cx_Freeze
├── 🔧 install_and_run.bat   # Script automático de instalação
├── 🔧 build_executable.bat  # Script cx_Freeze (pode falhar)
├── 🔧 build_pyinstaller.bat # Script PyInstaller (mais confiável)
├── 🔧 create_portable.bat   # Versão portável (RECOMENDADO)
├── 📁 src/                  # Código fonte principal
│   ├── 📁 apps/            # Aplicação Streamlit
│   ├── 📁 models/          # Modelos eletromagnéticos
│   ├── 📁 reports/         # Sistema de relatórios PDF
│   └── 📁 utils/           # Utilitários
└── 📁 models_old/          # Modelos de fallback
```

## 🎯 **Funcionalidades**

### **Simulação de Guias de Onda**
- ✅ **Guias Retangulares** - Modos TE e TM
- ✅ **Guias Cilíndricas** - Modos TE₁₁, TM₁₁, etc.
- ✅ **Visualizações 2D/3D** - Campos elétricos e magnéticos
- ✅ **Animações** - Propagação de ondas
- ✅ **Relatórios PDF** - Análise técnica completa

### **Interface Web**
- 🌐 **Streamlit** - Interface moderna e intuitiva
- 📊 **Plotly/Matplotlib** - Gráficos interativos
- 📱 **Responsivo** - Funciona em desktop e mobile
- 🎨 **Visualizações** - Campos vetoriais, mapas de calor

## ⚙️ **Requisitos**

### **Sistema**
- 🖥️ **Windows** 10/11 (recomendado)
- 🐍 **Python** 3.8+
- 💾 **RAM** 4GB mínimo
- 🌐 **Navegador** Chrome/Firefox/Edge

### **Dependências**
- `streamlit` - Interface web
- `numpy` - Computação numérica
- `matplotlib` - Gráficos
- `plotly` - Visualizações interativas
- `scipy` - Funções especiais (Bessel)
- `reportlab` - Geração de PDF
- `Pillow` - Processamento de imagens

## 🔧 **Instruções de Uso**

### **1. Executar Diretamente**
```bash
# Duplo clique em:
install_and_run.bat
```

### **2. Criar Executável**
```bash
# Duplo clique em:
build_executable.bat
```

### **3. Distribuir**
- Para **Python**: Copie toda a pasta `WaveguideSimulator_App`
- Para **Executável**: Copie a pasta `build/exe.win-amd64-X.X`

## 📋 **Como Usar o Simulador**

1. **Iniciar**: Execute o app via batch ou Python
2. **Acessar**: Abra http://localhost:8501 no navegador
3. **Selecionar**: Escolha tipo de guia (Retangular/Cilíndrica)
4. **Configurar**: Defina parâmetros (dimensões, frequência, material)
5. **Simular**: Execute a simulação
6. **Visualizar**: Veja campos elétricos/magnéticos
7. **Gerar PDF**: Crie relatório técnico completo

## 🏷️ **Versão**

- **App**: Waveguide Simulator v2.0
- **Estrutura**: Distribuição Standalone
- **Compatibilidade**: Windows 10/11, Python 3.8+

## 📞 **Suporte**

Para dúvidas ou problemas:
1. Verifique se Python 3.8+ está instalado
2. Execute `install_and_run.bat` como administrador
3. Certifique-se de ter conexão à internet para instalar dependências

---
**Simulador de Guias de Onda Eletromagnéticas**
