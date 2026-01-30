#!/usr/bin/env python
"""
Verify Outputs - Check if all expected files were generated
============================================================

Este script verifica se todos os outputs esperados foram gerados
após executar o pipeline de treinamento.

Uso:
----
python verify_outputs.py
"""

from pathlib import Path
from typing import Dict, List, Tuple


def check_files(base_path: str = "outputs") -> Dict[str, Tuple[int, int, List[str]]]:
    """
    Verifica se todos os arquivos esperados foram gerados.
    
    Returns
    -------
    dict
        Para cada categoria: (encontrados, esperados, arquivos_faltando)
    """
    base = Path(base_path)
    
    expected_files = {
        "EDA Plots": [
            "eda/transaction_amount_distribution.png",
            "eda/fraud_rate_by_hour.png",
            "eda/fraud_rate_by_day.png",
        ],
        "Confusion Matrices": [
            "confusion_matrices/LogReg_cm.png",
            "confusion_matrices/RF_cm.png",
            "confusion_matrices/XGBoost_cm.png",
            "confusion_matrices/MLP_cm.png",
        ],
        "Performance Curves": [
            "curves/roc_curves.png",
            "curves/pr_curves.png",
            "curves/cost_sensitivity.png",
        ],
        "SHAP Analysis": [
            "shap/shap_bar_top10.png",
            "shap/shap_beeswarm_top10.png",
        ],
        "Model Files": [
            "models/fraud_detection_pipeline.pkl",
        ],
        "Tables": [
            "tables/model_comparison.csv",
        ],
        "Processed Data": [
            "../data/processed/train_features.csv",
            "../data/processed/test_features.csv",
        ],
    }
    
    results = {}
    
    for category, files in expected_files.items():
        found = []
        missing = []
        
        for file_path in files:
            full_path = base / file_path
            if full_path.exists():
                found.append(file_path)
            else:
                missing.append(file_path)
        
        results[category] = (len(found), len(files), missing)
    
    return results


def print_report(results: Dict[str, Tuple[int, int, List[str]]]) -> bool:
    """
    Imprime relatório de verificação.
    
    Returns
    -------
    bool
        True se tudo OK, False se houver arquivos faltando
    """
    print("=" * 70)
    print("OUTPUT VERIFICATION REPORT")
    print("=" * 70)
    
    all_ok = True
    total_found = 0
    total_expected = 0
    
    for category, (found, expected, missing) in results.items():
        total_found += found
        total_expected += expected
        
        status = "✅" if found == expected else "⚠️"
        print(f"\n{status} {category}: {found}/{expected}")
        
        if missing:
            all_ok = False
            print(f"   Missing files:")
            for file in missing:
                print(f"   - {file}")
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {total_found}/{total_expected} files found")
    
    if all_ok:
        print("✅ All expected outputs generated successfully!")
    else:
        print("⚠️  Some outputs are missing. See details above.")
    
    print("=" * 70)
    
    return all_ok


def print_suggestions():
    """Imprime sugestões para resolver problemas."""
    print("\n📋 TROUBLESHOOTING SUGGESTIONS:")
    print("\n1. Missing EDA plots:")
    print("   python generate_visualizations.py --data data/raw/df.xlsx")
    
    print("\n2. Missing confusion matrices or SHAP:")
    print("   python generate_visualizations.py --data data/raw/df.xlsx")
    
    print("\n3. Missing model file:")
    print("   python src/main.py --data data/raw/df.xlsx --output outputs/")
    
    print("\n4. Missing processed data:")
    print("   Check if feature engineering step completed successfully")
    
    print("\n5. For complete troubleshooting guide:")
    print("   cat VISUALIZATION_TROUBLESHOOTING.md")


def main():
    """Execute verification."""
    print("\n🔍 Checking outputs...")
    
    results = check_files()
    all_ok = print_report(results)
    
    if not all_ok:
        print_suggestions()
        return 1
    
    # List all generated PNG files
    print("\n📊 Generated visualization files:")
    png_files = sorted(Path("outputs").rglob("*.png"))
    for i, file in enumerate(png_files, 1):
        size = file.stat().st_size / 1024  # KB
        print(f"   {i:2d}. {file} ({size:.1f} KB)")
    
    print(f"\n✅ Total: {len(png_files)} PNG files")
    
    # Check processed data
    processed_path = Path("data/processed")
    if processed_path.exists():
        csv_files = list(processed_path.glob("*.csv"))
        if csv_files:
            print(f"\n💾 Processed data files:")
            for file in csv_files:
                size = file.stat().st_size / 1024  # KB
                rows = sum(1 for _ in open(file)) - 1  # exclude header
                print(f"   - {file.name}: {rows} rows ({size:.1f} KB)")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
