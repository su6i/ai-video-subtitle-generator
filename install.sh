#!/bin/bash

echo "🎬 Subtitle Generator Installer"
echo "=============================="

# 1. Environment Setup
if [[ ! -d "venv" ]]; then
    echo "📦 Creating Virtual Environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# 2. Dependencies
echo "📥 Installing Dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies Installed."

# 3. System Deps Check
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  ffmpeg not found! Please install it (e.g., 'brew install ffmpeg' or 'sudo apt install ffmpeg')."
fi

# 4. Configuration
echo ""
echo "🔧 Configuration (.env)"
echo "-----------------------"
read -p "Enter DEEPSEEK_API_KEY (Optional for better translation): " key

cat <<EOF > .env
DEEPSEEK_API_KEY=$key
EOF

echo ""
echo "✅ Configuration saved to .env"
echo "------------------------------"
echo "🎉 Setup Complete! Run: './run_app.sh'"
