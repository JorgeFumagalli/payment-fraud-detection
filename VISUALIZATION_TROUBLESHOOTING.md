# 🖼️ Troubleshooting - Visualizações não Geradas

## Problema

Você executou `python src/main.py` mas as visualizações (SHAP, confusion matrices, EDA plots) não foram geradas.

## ✅ Soluções

### 1. Verificar se as Pastas Existem

```bash
# Criar pastas se não existirem
mkdir -p outputs/eda
mkdir -p outputs/confusion_matrices
mkdir -p outputs/curves
mkdir -p outputs/shap
mkdir -p outputs/tables
mkdir -p outputs/models
```

### 2. Executar SEM o Flag `--skip-viz`

```bash
# ❌ Errado (pula visualizações)
python src/main.py --data data/raw/df.xlsx --skip-viz

# ✅ Correto (gera visualizações)
python src/main.py --data data/raw/df.xlsx --output outputs/
```

### 3. Verificar Dependências

```bash
# Instalar bibliotecas de visualização
pip install matplotlib seaborn shap

# Verificar instalação
python -c "import matplotlib, seaborn, shap; print('✅ OK')"
```

### 4. Executar Manualmente (Passo a Passo)

Se o problema persistir, execute cada etapa separadamente:

#### A. Gerar EDA Plots

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar dados
df = pd.read_excel('data/raw/df.xlsx')

# 1. Transaction Amount Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['transaction_amount'], bins=50, kde=True, ax=axes[0])
axes[0].set_title('Transaction Amount Distribution')
sns.boxplot(x=df['transaction_amount'], ax=axes[1])
axes[1].set_title('Transaction Amount Boxplot')
plt.tight_layout()
plt.savefig('outputs/eda/transaction_amount.png', dpi=300)
plt.close()
print("✅ EDA plot saved")
```

#### B. Gerar Confusion Matrices

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar modelo treinado
from src.models import FraudDetectionPipeline
pipeline = FraudDetectionPipeline.load('outputs/models/fraud_detection_pipeline.pkl')

# Fazer predições (você precisa ter X_test e y_test)
for model_name in pipeline.models.keys():
    y_pred = pipeline.predict(X_test, model_name=model_name)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    safe_name = model_name.replace(' ', '_')
    plt.savefig(f'outputs/confusion_matrices/{safe_name}_cm.png', dpi=300)
    plt.close()
    print(f"✅ {model_name} confusion matrix saved")
```

#### C. Gerar SHAP Plots

```python
import shap
import matplotlib.pyplot as plt

# Carregar modelo
from src.models import FraudDetectionPipeline
pipeline = FraudDetectionPipeline.load('outputs/models/fraud_detection_pipeline.pkl')

# Usar Random Forest para SHAP
rf_model = pipeline.models['RF']

# Sample dos dados (max 300 para velocidade)
X_sample = X_test.iloc[:300]

# Criar explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_sample)

# Lidar com formato de output
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Plot 1: Bar plot
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_sample, plot_type='bar', 
                  max_display=10, show=False)
plt.title('Global Feature Importance (SHAP)')
plt.tight_layout()
plt.savefig('outputs/shap/shap_bar_top10.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ SHAP bar plot saved")

# Plot 2: Beeswarm
plt.figure(figsize=(12, 7))
shap.summary_plot(shap_values, X_sample, max_display=10, show=False)
plt.title('SHAP Beeswarm - Top 10 Features')
plt.tight_layout()
plt.savefig('outputs/shap/shap_beeswarm_top10.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ SHAP beeswarm plot saved")
```

### 5. Verificar Permissões de Escrita

```bash
# Linux/Mac
ls -la outputs/
chmod -R 755 outputs/

# Se ainda não funcionar, tente outro diretório
python src/main.py --data data/raw/df.xlsx --output /tmp/fraud_outputs/
```

### 6. Modo Headless (Servidores sem GUI)

Se estiver em um servidor sem interface gráfica:

```python
# Adicionar no início do script
import matplotlib
matplotlib.use('Agg')  # Backend sem display
import matplotlib.pyplot as plt
```

Ou definir variável de ambiente:

```bash
export MPLBACKEND=Agg
python src/main.py --data data/raw/df.xlsx
```

### 7. Verificar Logs

```bash
# Ver logs completos
cat fraud_detection.log

# Procurar por erros de visualização
grep -i "visualization\|plot\|figure" fraud_detection.log
```

## 📊 Checklist de Diagnóstico

Execute este checklist para identificar o problema:

```bash
# 1. Pastas existem?
ls -la outputs/eda outputs/confusion_matrices outputs/curves outputs/shap

# 2. Dependências instaladas?
python -c "import matplotlib, seaborn, shap; print('✅ Deps OK')"

# 3. Permissões OK?
touch outputs/test.txt && rm outputs/test.txt && echo "✅ Write OK"

# 4. Script executou até o fim?
tail -20 fraud_detection.log

# 5. Arquivos foram gerados?
find outputs/ -name "*.png" -mtime -1
```

## 🎯 Execução Garantida

Para garantir que todas as visualizações sejam geradas:

```bash
# 1. Limpar outputs antigos
rm -rf outputs/*

# 2. Recriar estrutura
mkdir -p outputs/{eda,confusion_matrices,curves,shap,tables,models}

# 3. Executar com logging verbose
python src/main.py \
    --data data/raw/df.xlsx \
    --output outputs/ \
    2>&1 | tee execution.log

# 4. Verificar resultados
find outputs/ -name "*.png" | wc -l
# Deve mostrar: ~10 arquivos (2 EDA + 4 CM + 3 curves + 2 SHAP)
```

## 📧 Ainda com Problemas?

Se nada funcionar:

1. Compartilhe o erro completo do `fraud_detection.log`
2. Execute `python --version` e `pip list`
3. Informe seu sistema operacional

---

**Última atualização**: Janeiro 2026
