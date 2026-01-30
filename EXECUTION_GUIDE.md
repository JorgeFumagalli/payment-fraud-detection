# 🚀 Guia Passo a Passo - Execução do Projeto

Este documento explica a **ordem correta de execução** do projeto Payment Fraud Detection, desde a instalação até a produção.

---

## 📋 Índice

1. [Instalação Inicial](#1-instalação-inicial)
2. [Preparação dos Dados](#2-preparação-dos-dados)
3. [Exploração Inicial (Opcional)](#3-exploração-inicial-opcional)
4. [Pipeline Completo de Treinamento](#4-pipeline-completo-de-treinamento)
5. [Análise Cold Start (Opcional)](#5-análise-cold-start-opcional)
6. [Fazer Predições](#6-fazer-predições)
7. [Deploy para Produção](#7-deploy-para-produção)

---

## 1. Instalação Inicial

### Passo 1.1: Clonar/Extrair o Projeto

```bash
# Se baixou o .tar.gz
tar -xzf payment-fraud-detection-final.tar.gz
cd payment-fraud-detection

# Ou se clonou do GitHub
git clone https://github.com/seu-usuario/payment-fraud-detection.git
cd payment-fraud-detection
```

### Passo 1.2: Criar Ambiente Virtual

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# No Linux/Mac:
source venv/bin/activate

# No Windows:
venv\Scripts\activate
```

### Passo 1.3: Instalar Dependências

```bash
# Atualizar pip
pip install --upgrade pip

# Instalar dependências do projeto
pip install -r requirements.txt

# (Opcional) Para desenvolvimento:
pip install -r requirements-dev.txt
```

### Passo 1.4: Verificar Instalação

```bash
# Verificar se o pacote foi instalado corretamente
python -c "import src; print('✅ Instalação OK!')"

# Verificar versões importantes
python -c "import sklearn, xgboost, pandas; print('✅ Dependências OK!')"
```

**✅ Checkpoint**: Se não houver erros, prossiga para o próximo passo.

---

## 2. Preparação dos Dados

### Passo 2.1: Organizar Seus Dados

Coloque seu arquivo de transações em `data/raw/`:

```bash
# Estrutura esperada do arquivo:
data/raw/df.xlsx  # ou .csv, .parquet
```

**Colunas Obrigatórias:**
```
- transaction_id      (str/int)
- transaction_date    (datetime)
- transaction_amount  (float)
- user_id            (str/int)
- merchant_id        (str/int)
- device_id          (str/int, pode ter NaN)
- has_cbk            (int: 0 ou 1)
```

### Passo 2.2: Validar Dados (Opcional mas Recomendado)

```bash
# Script de validação rápida
python -c "
from src.data_loader import load_data, validate_data

# Carregar dados
df = load_data('data/raw/df.xlsx')
print(f'✅ Dados carregados: {len(df)} transações')

# Validar
diag = validate_data(df)
print(f'✅ Taxa de fraude: {diag[\"fraud_rate\"]*100:.2f}%')
print(f'✅ Device faltando: {diag[\"missing_device_rate\"]*100:.1f}%')
"
```

**✅ Checkpoint**: Dados carregados sem erros? Continue!

---

## 3. Exploração Inicial (Opcional)

### Passo 3.1: Análise Exploratória Rápida

Se quiser entender melhor seus dados antes de treinar:

```bash
# Usando o script original (análise completa)
python notebooks/fraud_detection_original.py
```

**O que isso faz:**
- Gera gráficos de distribuição
- Analisa padrões temporais
- Cria features iniciais
- Salva visualizações em `outputs/eda/`

**Saídas:**
- `outputs/eda/eda_transaction_amount.png`
- `outputs/eda/temporal_patterns.png`

### Passo 3.2: Revisar Outputs da EDA

```bash
# Ver os gráficos gerados
ls -la outputs/eda/

# No Linux com interface gráfica:
xdg-open outputs/eda/eda_transaction_amount.png

# No Mac:
open outputs/eda/eda_transaction_amount.png

# No Windows:
start outputs/eda/eda_transaction_amount.png
```

---

## 4. Pipeline Completo de Treinamento

### Passo 4.1: Executar Pipeline Principal (Básico)

Este é o **comando mais importante** do projeto:

```bash
python src/main.py \
    --data data/raw/df.xlsx \
    --output outputs/
```

**O que acontece:**
1. ✅ Carrega e valida dados
2. ✅ Divide train/test (80/20)
3. ✅ Cria 25 features automaticamente
4. ✅ Treina 4 modelos (LogReg, RF, XGBoost, MLP)
5. ✅ Avalia performance
6. ✅ Gera visualizações (ROC, PR, confusion matrices)
7. ✅ Salva modelos treinados

**Tempo estimado:** 2-5 minutos (depende do tamanho dos dados)

### Passo 4.2: Executar com Otimização (Avançado)

Para melhores resultados (mais lento):

```bash
python src/main.py \
    --data data/raw/df.xlsx \
    --output outputs/ \
    --optimize
```

**O que muda:**
- Usa Bayesian Optimization para Random Forest
- Testa diferentes hiperparâmetros
- **Tempo:** 10-30 minutos

### Passo 4.3: Executar Sem Visualizações (Mais Rápido)

Se quiser apenas treinar modelos:

```bash
python src/main.py \
    --data data/raw/df.xlsx \
    --output outputs/ \
    --skip-viz
```

**Economia:** ~30% mais rápido

### Passo 4.4: Revisar Resultados

```bash
# Ver comparação de modelos
cat outputs/tables/model_comparison.csv

# Ver estrutura de arquivos gerados
ls -la outputs/
```

**Arquivos gerados:**

```
outputs/
├── eda/
│   ├── transaction_amount_distribution.png  📊 Histograma + boxplot
│   ├── fraud_rate_by_hour.png              📈 Fraude por hora
│   └── fraud_rate_by_day.png               📈 Fraude por dia da semana
├── tables/
│   └── model_comparison.csv                 📊 Métricas dos modelos
├── curves/
│   ├── roc_curves.png                       📈 Curvas ROC
│   ├── pr_curves.png                        📈 Precision-Recall
│   └── cost_sensitivity.png                 💰 Análise de custo
├── confusion_matrices/
│   ├── LogReg_cm.png                        🔢 Logistic Regression
│   ├── RF_cm.png                            🔢 Random Forest
│   ├── XGBoost_cm.png                       🔢 XGBoost
│   └── MLP_cm.png                           🔢 MLP Neural Network
├── shap/
│   ├── shap_bar_top10.png                   🔍 Feature importance
│   └── shap_beeswarm_top10.png              🔍 Distribuição SHAP
└── models/
    └── fraud_detection_pipeline.pkl         🤖 Modelo treinado

data/
└── processed/
    ├── train_features.csv                   💾 Features de treino
    └── test_features.csv                    💾 Features de teste
```

**✅ Checkpoint**: Pipeline executou sem erros? Modelos salvos? Continue!

---

## 5. Análise Cold Start (Opcional)

### Passo 5.1: Executar Pipeline Cold Start

Para analisar performance em usuários novos:

```bash
python src/cold_start.py \
    --data data/raw/df.xlsx \
    --output outputs/
```

**O que faz:**
1. Identifica transações "cold start" (novos usuários/merchants)
2. Treina modelo especializado
3. Compara performance: cold vs. non-cold
4. Gera relatórios segmentados

**Saídas:**
- `outputs/tables/segmented_coldstart_metrics.csv`
- `outputs/tables/cold_start_rows_scored.csv`
- `outputs/tables/non_cold_start_rows_scored.csv`

### Passo 5.2: Analisar Resultados

```bash
# Ver métricas segmentadas
cat outputs/tables/segmented_coldstart_metrics.csv

# Ver quais transações foram classificadas como cold start
head outputs/tables/cold_start_rows_scored.csv
```

---

## 6. Fazer Predições

### Passo 6.1: Predição em Lote (Batch)

Para prever fraude em novas transações:

**Método 1: Via Script Python**

```python
# Criar arquivo: predict.py
from src.models import FraudDetectionPipeline
from src.feature_engineering import FeatureEngineer
from src.data_loader import load_data, prepare_data
import pandas as pd

# 1. Carregar modelo treinado
pipeline = FraudDetectionPipeline.load('outputs/models/fraud_detection_pipeline.pkl')

# 2. Carregar novos dados
new_data = load_data('data/raw/new_transactions.xlsx')
new_data = prepare_data(new_data)

# 3. Engenharia de features (precisa do engineer treinado)
# Nota: Você precisa salvar o engineer junto com o pipeline
# Por enquanto, retreine ou use pipeline completo

# 4. Fazer predições
predictions = pipeline.predict(new_data)
probabilities = pipeline.predict_proba(new_data)

# 5. Adicionar resultados ao dataframe
new_data['fraud_prediction'] = predictions
new_data['fraud_probability'] = probabilities

# 6. Salvar resultados
new_data.to_csv('outputs/predictions.csv', index=False)

print(f"✅ Predições concluídas!")
print(f"   Frauds detectadas: {predictions.sum()} de {len(predictions)}")
print(f"   Taxa de fraude: {predictions.mean()*100:.2f}%")
```

Execute:
```bash
python predict.py
```

### Passo 6.2: Predição Interativa (Python REPL)

```bash
python
```

```python
from src.models import FraudDetectionPipeline
import pandas as pd

# Carregar modelo
pipeline = FraudDetectionPipeline.load('outputs/models/fraud_detection_pipeline.pkl')

# Criar transação de teste
transaction = pd.DataFrame({
    'transaction_amount': [1500.0],
    'hour': [23],
    'day_of_week': [5],
    'merchant_id_cbk_rate': [0.15],
    'device_id_cbk_rate': [0.05],
    'user_id_cbk_rate': [0.02],
    # ... adicionar todas as 25 features
})

# Prever
prob = pipeline.predict_proba(transaction)[0]
pred = pipeline.predict(transaction)[0]

print(f"Probabilidade de fraude: {prob*100:.2f}%")
print(f"Predição: {'FRAUDE' if pred else 'LEGÍTIMA'}")
```

---

## 7. Deploy para Produção

### Passo 7.1: Criar API REST (FastAPI)

```bash
# Instalar FastAPI
pip install fastapi uvicorn

# Criar arquivo de API (já existe exemplo no deployment_guide.md)
# Use o código em docs/deployment_guide.md seção "Option 2: Real-Time API"
```

### Passo 7.2: Executar API Localmente

```bash
# Rodar servidor de desenvolvimento
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### Passo 7.3: Testar API

```bash
# Em outro terminal, testar endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "test_001",
    "transaction_amount": 1500.0,
    "user_id": "user_123",
    "merchant_id": "merchant_456",
    "device_id": "device_789",
    "transaction_date": "2024-01-28T15:30:00"
  }'
```

### Passo 7.4: Deploy para Produção

Siga o guia completo em `docs/deployment_guide.md` para:
- Docker
- Kubernetes
- Cloud (AWS/GCP/Azure)
- Monitoring

---

## 📊 Ordem de Execução Resumida

### Primeira Vez (Setup Completo)

```bash
# 1. Setup inicial
cd payment-fraud-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Preparar dados
# Coloque df.xlsx em data/raw/

# 3. Treinar modelos
python src/main.py --data data/raw/df.xlsx --output outputs/

# 4. Revisar resultados
cat outputs/tables/model_comparison.csv
```

### Uso Diário (Predições)

```bash
# 1. Ativar ambiente
source venv/bin/activate

# 2. Fazer predições
python predict.py  # seu script customizado

# 3. Revisar resultados
cat outputs/predictions.csv
```

### Retreinamento Mensal

```bash
# 1. Ativar ambiente
source venv/bin/activate

# 2. Retreinar com dados atualizados
python src/main.py \
    --data data/raw/df_2024_02.xlsx \
    --output outputs/retrain/ \
    --optimize

# 3. Validar novo modelo
python scripts/validate_model.py  # criar esse script

# 4. Substituir modelo em produção
cp outputs/retrain/models/fraud_detection_pipeline.pkl \
   outputs/models/fraud_detection_pipeline.pkl
```

---

## 🐛 Troubleshooting

### Erro: "ModuleNotFoundError: No module named 'src'"

**Solução:**
```bash
pip install -e .
```

### Erro: "File not found: data/raw/df.xlsx"

**Solução:**
```bash
# Verificar caminho
ls -la data/raw/
# Ajustar comando
python src/main.py --data /caminho/completo/para/df.xlsx
```

### Erro: "Memory Error"

**Solução:**
```bash
# Reduzir complexidade do modelo em config.json
{
  "models": {
    "random_forest": {
      "n_estimators": 100  # reduzir de 300
    }
  }
}
```

### Performance Lenta

**Solução:**
```bash
# Usar menos otimização
python src/main.py --data data/raw/df.xlsx --skip-viz
```

### Visualizações Não Foram Geradas

**Solução:**
```bash
# 1. Verificar se as pastas existem
mkdir -p outputs/{eda,confusion_matrices,curves,shap,tables,models}

# 2. NÃO usar --skip-viz
python src/main.py --data data/raw/df.xlsx --output outputs/

# 3. Verificar dependências
pip install matplotlib seaborn shap

# 4. Ver guia completo
cat VISUALIZATION_TROUBLESHOOTING.md
```

---

## ✅ Checklist de Execução

Use este checklist para garantir que executou tudo corretamente:

- [ ] Ambiente virtual criado e ativado
- [ ] Dependências instaladas (`requirements.txt`)
- [ ] Dados em `data/raw/df.xlsx` (ou similar)
- [ ] Pipeline principal executado (`src/main.py`)
- [ ] Resultados gerados em `outputs/`
- [ ] Modelo salvo em `outputs/models/`
- [ ] Métricas revisadas (`model_comparison.csv`)
- [ ] (Opcional) Pipeline cold start executado
- [ ] (Opcional) API testada localmente
- [ ] (Opcional) Deploy para produção

---

## 🎯 Fluxo Visual

```
┌─────────────────┐
│  1. INSTALAÇÃO  │
│  pip install    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. PREPARAÇÃO  │
│  data/raw/      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. TREINAMENTO │
│  src/main.py    │◄─── Loop de otimização (opcional)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. VALIDAÇÃO   │
│  Ver outputs/   │
└────────┬────────┘
         │
         ├─────────────────┐
         ▼                 ▼
┌─────────────────┐  ┌─────────────────┐
│  5a. PREDIÇÕES  │  │  5b. COLD START │
│  predict.py     │  │  cold_start.py  │
└────────┬────────┘  └────────┬────────┘
         │                    │
         └──────────┬─────────┘
                    ▼
           ┌─────────────────┐
           │  6. PRODUÇÃO    │
           │  API / Docker   │
           └─────────────────┘
```

---

## 📞 Suporte

Se encontrar problemas:
1. Verifique os logs em `fraud_detection.log`
2. Consulte `QUICKSTART.md` para problemas comuns
3. Leia `docs/deployment_guide.md` para produção
4. Abra issue no GitHub

**Bom trabalho! 🚀**
