#!/bin/bash
# Скрипт установки зависимостей и проверки окружения

set -e

echo "=============================================="
echo "Shaman MVP - Environment Setup"
echo "=============================================="

# Проверяем Python
echo -n "Checking Python... "
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "FAILED"
    echo "Error: Python not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1 | cut -d' ' -f2)
echo "OK ($PYTHON_VERSION)"

# Проверяем pip
echo -n "Checking pip... "
if ! $PYTHON -m pip --version &> /dev/null; then
    echo "FAILED"
    echo "Error: pip not found"
    exit 1
fi
echo "OK"

# Создаём виртуальное окружение если его нет
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
fi

# Активируем окружение
echo "Activating virtual environment..."
source venv/bin/activate

# Обновляем pip
echo "Upgrading pip..."
pip install --upgrade pip

# Устанавливаем зависимости
echo "Installing dependencies..."
pip install -r requirements.txt

# Проверяем CUDA
echo -n "Checking CUDA... "
$PYTHON -c "import torch; print('OK' if torch.cuda.is_available() else 'NOT AVAILABLE (will use CPU)')"

# Проверяем импорты
echo "Checking imports..."
$PYTHON -c "
import torch
import transformers
import sentence_transformers
import numpy
import scipy
print('All imports OK')
"

# Проверяем TDA
echo -n "Checking TDA (ripser)... "
$PYTHON -c "import ripser; print('OK')" 2>/dev/null || echo "NOT INSTALLED (optional)"

echo ""
echo "=============================================="
echo "Setup complete!"
echo ""
echo "To run the experiment:"
echo "  source venv/bin/activate"
echo "  python run_experiment.py --quick  # quick test"
echo "  python run_experiment.py          # full run"
echo "=============================================="
