#!/bin/bash
# One-click training script for CTF Writeup Generator

echo "🚀 CTF Writeup Generator Training Script"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "ctf_env" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv ctf_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source ctf_env/Scripts/activate
else
    source ctf_env/bin/activate
fi

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Check if dataset exists
if [ ! -f "ctf_writeups_dataset.json" ]; then
    echo "📊 Collecting training data..."
    python src/data_collector.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Data collection failed!"
        exit 1
    fi
else
    echo "✅ Training dataset found"
fi

# Start training
echo "🎯 Starting model training..."
echo "This will take 2-4 hours on GPU or 8-12 hours on CPU"
python src/trainer.py

if [ $? -eq 0 ]; then
    echo "🎉 Training completed successfully!"
    echo "Model saved to: ./trained-ctf-model"
    echo ""
    echo "You can now generate writeups using:"
    echo "python src/generator.py"
else
    echo "❌ Training failed!"
    exit 1
fi
