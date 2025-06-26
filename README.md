# CTF Writeup Generator 🚀

A fine-tuned AI model that generates detailed CTF (Capture The Flag) writeups from challenge descriptions. Built on microsoft/DialoGPT-medium and trained on real CTF writeups from top security competitions.

## 🎯 Features

- **Automatic Data Collection**: Scrapes 300-800 high-quality CTF writeups from GitHub repositories
- **Smart Fine-tuning**: Uses microsoft/DialoGPT-medium for optimal performance
- **Multi-category Support**: Web, Binary Exploitation, Cryptography, Forensics, Reverse Engineering
- **Step-by-step Solutions**: Generates detailed, educational writeups
- **Easy to Use**: Simple API for generating writeups from challenge descriptions

## 🚀 Quick Start

### 1. Setup Environment

```bash
git clone https://github.com/yourusername/ctf-writeup-generator.git
cd ctf-writeup-generator

# Create virtual environment
python -m venv ctf_env
source ctf_env/bin/activate  # Windows: ctf_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Collect Training Data

```bash
python src/data_collector.py
```
This will collect 300-800 CTF writeups (takes 10-15 minutes).

### 3. Train the Model

```bash
python src/trainer.py
```
Training time: 2-4 hours (GPU) or 8-12 hours (CPU).

### 4. Generate Writeups

```python
from src.generator import CTFGenerator

generator = CTFGenerator("./trained-ctf-model")
writeup = generator.generate(
    title="SQL Injection Login Bypass",
    category="Web",
    difficulty="Easy", 
    description="Bypass the login form using SQL injection"
)
print(writeup)
```

## 📁 Repository Structure

```
├── src/
│   ├── data_collector.py    # Collect training data
│   ├── trainer.py          # Fine-tune the model
│   └── generator.py        # Generate writeups
├── examples/
│   ├── sample_challenges.json
│   └── example_usage.py
├── scripts/
│   ├── train.sh           # One-click training
│   └── setup.sh           # Environment setup
└── requirements.txt
```

## 🛠️ Requirements

- Python 3.8+
- GPU with 8GB+ VRAM (recommended) or 16GB+ RAM
- 50GB+ free disk space
- Stable internet connection

## 📊 Example Output

```markdown
# SQL Injection Login Bypass - CTF Writeup

## Challenge Overview
**Category:** Web
**Difficulty:** Easy
**Points:** 100

## Solution

### Step 1: Initial Analysis
First, let's examine the login form to understand how it processes user input...

### Step 2: Testing for SQL Injection
We'll test basic SQL injection payloads like `' OR '1'='1`...

### Step 3: Bypassing Authentication
Using the payload `admin'--` we can bypass the password check...

## Flag
```
FLAG{sql_injection_master}
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## ⚠️ Disclaimer

This tool is for educational purposes only. Use responsibly and only on authorized systems.
