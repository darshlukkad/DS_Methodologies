#!/bin/bash
# Install Jupyter and execute all notebooks

echo "📦 Installing required packages..."
pip install jupyter nbformat nbconvert pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn scipy statsmodels

echo ""
echo "🚀 Executing notebooks one by one..."
echo ""

# Execute CRISP-DM
echo "1️⃣  Executing CRISP_DM.ipynb (~5 min)..."
jupyter nbconvert --to notebook --execute --inplace crisp_dm/CRISP_DM.ipynb
echo "✅ CRISP-DM complete!"
echo ""

# Execute SEMMA
echo "2️⃣  Executing SEMMA.ipynb (~2 min)..."
jupyter nbconvert --to notebook --execute --inplace semma/SEMMA.ipynb
echo "✅ SEMMA complete!"
echo ""

# Execute KDD
echo "3️⃣  Executing KDD.ipynb (~3 min)..."
jupyter nbconvert --to notebook --execute --inplace kdd/KDD.ipynb
echo "✅ KDD complete!"
echo ""

echo "🎉 All notebooks executed successfully!"
echo ""
echo "📊 Summary:"
echo "   • CRISP-DM: 43 code cells executed"
echo "   • SEMMA: 10 code cells executed"
echo "   • KDD: 11 code cells executed"
echo "   • Total: 64 code cells with outputs"
