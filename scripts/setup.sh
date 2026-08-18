#!/bin/bash
set -e

echo "=== NIDS Adversarial Evasion Hardening - Setup Script ==="
echo ""

PYTHON_VERSION=$(python3 --version 2>&1)
echo "Python version: $PYTHON_VERSION"
echo ""

echo "[1/4] Upgrading pip..."
python3 -m pip install --upgrade pip

echo ""
echo "[2/4] Installing PyTorch (CPU-only)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "[3/4] Installing ML and utility packages..."
pip install scikit-learn numpy matplotlib pandas pytest flask

echo ""
echo "[4/4] Attempting to install Adversarial Robustness Toolbox (optional)..."
pip install adversarial-robustness-toolbox 2>/dev/null && echo "ART installed successfully" || echo "ART installation skipped (not required for core functionality)"

echo ""
echo "=== Checking installations ==="

python3 -c "
import torch
print('PyTorch: {} (CPU: {})'.format(torch.__version__, not torch.cuda.is_available()))

import sklearn
print('Scikit-Learn: {}'.format(sklearn.__version__))

import numpy
print('NumPy: {}'.format(numpy.__version__))

import matplotlib
print('Matplotlib: {}'.format(matplotlib.__version__))

import pandas
print('Pandas: {}'.format(pandas.__version__))

try:
    import art
    print('ART: {}'.format(art.__version__))
except ImportError:
    print('ART: not installed (optional)')

import flask
print('Flask: {}'.format(flask.__version__))
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick start:"
echo "  Train baseline:       python -m src.core.model_trainer --synthetic --epochs 20"
echo "  Run FGSM attack:      python -m src.core.adversarial_generator --model-path models/ids_baseline.pt --attack fgsm"
echo "  Adversarial training: python -m src.modules.adversarial_training --model-path models/ids_baseline.pt --epochs 15"
echo "  Run tests:            python -m pytest tests/ -v"
