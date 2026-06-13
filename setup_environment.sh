#!/bin/bash
# setup_environment.sh
# Sets up a virtual environment for the MLFFN-Empathy project

echo "🚀 Setting up MLFFN-Empathy Virtual Environment"
echo "=============================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if venv module is available gyfgh
if ! python3 -m venv --help &> /dev/null; then
    echo "❌ venv module not available. Installing python3-venv..."
    sudo apt-get update
    sudo apt-get install -y python3-venv
fi

# Create virtual environment
echo "📦 Creating virtual environment 'mlffn_env'..."
python3 -m venv mlffn_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source mlffn_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing required packages..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data
mkdir -p results
mkdir -p embeddings

echo ""
echo "✅ Virtual environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source mlffn_env/bin/activate"
echo ""
echo "To deactivate when done, run:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "1. Download word embeddings (see README.md)"
echo "2. Set environment variables:"
echo "   export VECTORS=/path/to/your/embeddings"
echo "3. Run the experiments:"
echo "   bash run_experiments.sh"
