# Perceptron Implementation

## Requisitos

- Python 3.7 ou superior
- Bibliotecas: 
  - NumPy
  - Matplotlib
  - Scikit-learn

## Instalação de Dependências

### Windows
```powershell
pip install numpy matplotlib scikit-learn
```

### Linux/macOS
```bash
pip3 install numpy matplotlib scikit-learn
```

## Executando o Programa

### Windows (PowerShell)

1. **Configurar Política de Execução**:
   Abra o PowerShell como Administrador e execute:
   ```powershell
   Set-ExecutionPolicy RemoteSigned
   ```

2. **Executar o Programa**:
   ```powershell
   .\perceptron.ps1 <NUMERO_MATRICULA>
   ```
   Exemplo:
   ```powershell
   .\perceptron.ps1 23110438
   ```

### Linux/macOS (Bash Script)

1. **Dar Permissão ao Script**:
   ```bash
   chmod +x perceptron.sh
   ```

2. **Executar o Programa**:
   ```bash
   ./perceptron.sh <NUMERO_MATRICULA>
   ```
   Exemplo:
   ```bash
   ./perceptron.sh 23110438
   ```

## Explicação do Programa

Este programa implementa um Perceptron simples para classificação binária:
- Gera dados sintéticos com base no número de matrícula
- Treina um classificador Perceptron
- Exibe a acurácia do modelo
- Plota a fronteira de decisão do classificador

## Opções de Linha de Comando

Parâmetros opcionais ao rodar diretamente com Python:
```
--samples: Número de amostras de dados (padrão: 200)
--noise: Desvio padrão dos clusters (padrão: 1.5)
--learning_rate: Taxa de aprendizado (padrão: 0.01)
--epochs: Número de épocas de treinamento (padrão: 1000)
```

Exemplo:
```bash
python perceptron.py --registration_number 23110438 --samples 300 --noise 2.0
```

## Solução de Problemas

- Certifique-se de que todas as bibliotecas estão instaladas
- Verifique se está usando Python 3
- No Windows, execute o PowerShell como Administrador para alterar políticas de execução
