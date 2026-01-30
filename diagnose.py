"""
Script de Diagnóstico - Identificar Problemas
==============================================

Este script testa cada componente individualmente para identificar
exatamente onde está falhando.
"""

import sys
from pathlib import Path

print("=" * 70)
print("DIAGNÓSTICO DETALHADO")
print("=" * 70)

# Teste 1: Importações
print("\n1. Testando importações...")
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("   ✅ Bibliotecas básicas: OK")
except ImportError as e:
    print(f"   ❌ Erro nas importações: {e}")
    sys.exit(1)

# Teste 2: Estrutura de pastas
print("\n2. Verificando estrutura de pastas...")
required_dirs = [
    'src',
    'data/raw',
    'data/processed',
    'outputs/eda',
    'outputs/confusion_matrices',
    'outputs/curves',
    'outputs/shap',
    'outputs/tables',
    'outputs/models'
]

for dir_path in required_dirs:
    path = Path(dir_path)
    if path.exists():
        print(f"   ✅ {dir_path}")
    else:
        print(f"   ❌ {dir_path} - NÃO EXISTE")
        path.mkdir(parents=True, exist_ok=True)
        print(f"      → Criada automaticamente")

# Teste 3: Importar módulos do projeto
print("\n3. Testando módulos do projeto...")
try:
    from src.utils import setup_logging, create_output_dirs
    print("   ✅ src.utils: OK")
except ImportError as e:
    print(f"   ❌ src.utils: {e}")
    sys.exit(1)

try:
    from src.data_loader import load_data, prepare_data
    print("   ✅ src.data_loader: OK")
except ImportError as e:
    print(f"   ❌ src.data_loader: {e}")
    sys.exit(1)

try:
    from src.feature_engineering import FeatureEngineer
    print("   ✅ src.feature_engineering: OK")
except ImportError as e:
    print(f"   ❌ src.feature_engineering: {e}")
    sys.exit(1)

try:
    from src.models import FraudDetectionPipeline
    print("   ✅ src.models: OK")
except ImportError as e:
    print(f"   ❌ src.models: {e}")
    sys.exit(1)

# Teste 4: Criar pastas de output
print("\n4. Testando criação de pastas...")
try:
    dirs = create_output_dirs('outputs')
    print(f"   ✅ Criadas {len(dirs)} pastas")
    for name, path in dirs.items():
        print(f"      - {name}: {path}")
except Exception as e:
    print(f"   ❌ Erro ao criar pastas: {e}")

# Teste 5: Verificar se há dados
print("\n5. Verificando dados...")
data_files = list(Path('data/raw').glob('*'))
if data_files:
    print(f"   ✅ Encontrados {len(data_files)} arquivo(s):")
    for f in data_files:
        if f.is_file():
            size = f.stat().st_size / 1024
            print(f"      - {f.name} ({size:.1f} KB)")
else:
    print("   ⚠️  Nenhum arquivo em data/raw/")
    print("      → Coloque seu arquivo df.xlsx em data/raw/")

# Teste 6: Testar geração de plot simples
print("\n6. Testando geração de plot...")
try:
    import matplotlib
    matplotlib.use('Agg')  # Backend sem display
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title('Teste')
    
    test_path = Path('outputs/test_plot.png')
    plt.savefig(test_path, dpi=100)
    plt.close()
    
    if test_path.exists():
        size = test_path.stat().st_size / 1024
        print(f"   ✅ Plot gerado com sucesso ({size:.1f} KB)")
        print(f"      Salvo em: {test_path}")
        # Limpar
        test_path.unlink()
    else:
        print("   ❌ Plot não foi salvo")
        
except Exception as e:
    print(f"   ❌ Erro ao gerar plot: {e}")
    import traceback
    traceback.print_exc()

# Teste 7: Verificar função generate_eda_plots
print("\n7. Testando função generate_eda_plots...")
try:
    # Importar a função
    import importlib.util
    spec = importlib.util.spec_from_file_location("main", "src/main.py")
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)
    
    if hasattr(main_module, 'generate_eda_plots'):
        print("   ✅ Função generate_eda_plots existe")
    else:
        print("   ❌ Função generate_eda_plots NÃO ENCONTRADA")
        print("      → Verifique se src/main.py foi atualizado corretamente")
        
except Exception as e:
    print(f"   ❌ Erro ao importar main.py: {e}")
    import traceback
    traceback.print_exc()

# Teste 8: Testar com dados fictícios
print("\n8. Testando com dados fictícios...")
try:
    # Criar dados de teste
    test_data = pd.DataFrame({
        'transaction_id': range(100),
        'transaction_date': pd.date_range('2024-01-01', periods=100, freq='H'),
        'transaction_amount': np.random.lognormal(6, 1, 100),
        'user_id': np.random.randint(1, 20, 100),
        'merchant_id': np.random.randint(1, 10, 100),
        'device_id': np.random.choice([1, 2, 3, None], 100),
        'has_cbk': np.random.choice([0, 1], 100, p=[0.88, 0.12])
    })
    
    # Adicionar colunas necessárias
    test_data['hour'] = test_data['transaction_date'].dt.hour
    test_data['day_of_week'] = test_data['transaction_date'].dt.dayofweek
    
    print(f"   ✅ Dados de teste criados: {test_data.shape}")
    
    # Tentar gerar EDA
    if hasattr(main_module, 'generate_eda_plots'):
        print("   Tentando gerar EDA plots...")
        main_module.generate_eda_plots(test_data, Path('outputs/eda'))
        
        # Verificar se foram criados
        eda_files = list(Path('outputs/eda').glob('*.png'))
        if eda_files:
            print(f"   ✅ EDA plots gerados: {len(eda_files)} arquivo(s)")
            for f in eda_files:
                print(f"      - {f.name}")
        else:
            print("   ❌ Nenhum plot EDA foi gerado")
    
except Exception as e:
    print(f"   ❌ Erro no teste com dados fictícios: {e}")
    import traceback
    traceback.print_exc()

# Resumo final
print("\n" + "=" * 70)
print("RESUMO DO DIAGNÓSTICO")
print("=" * 70)

issues_found = []

# Verificar outputs
eda_plots = list(Path('outputs/eda').glob('*.png'))
if not eda_plots:
    issues_found.append("❌ Nenhum plot EDA encontrado")
else:
    print(f"✅ EDA plots: {len(eda_plots)} arquivo(s)")

processed_data = list(Path('data/processed').glob('*.csv'))
if not processed_data:
    issues_found.append("❌ Nenhum dado processado encontrado")
else:
    print(f"✅ Dados processados: {len(processed_data)} arquivo(s)")

if issues_found:
    print("\n⚠️  Problemas encontrados:")
    for issue in issues_found:
        print(f"   {issue}")
else:
    print("\n✅ Tudo funcionando corretamente!")

print("\n" + "=" * 70)
print("\n💡 PRÓXIMOS PASSOS:")
print("\n1. Se não tem dados ainda:")
print("   Coloque seu arquivo df.xlsx em data/raw/")
print("\n2. Para executar o pipeline completo:")
print("   python src/main.py --data data/raw/df.xlsx")
print("\n3. Para verificar outputs:")
print("   python verify_outputs.py")
print("\n4. Para gerar apenas visualizações:")
print("   python generate_visualizations.py --data data/raw/df.xlsx")
print("=" * 70)
