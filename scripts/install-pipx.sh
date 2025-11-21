#!/usr/bin/env bash
#
# Install LLM Chat Knowledge Converter using pipx
# Creates a truly portable installation in XDG-compliant directories
#

set -e

echo "🚀 Installing LLM Chat Knowledge Converter with pipx"
echo "===================================================="
echo

# Check if pipx is installed
if ! command -v pipx &> /dev/null; then
    echo "❌ pipx is not installed."
    echo
    echo "Install pipx first:"
    echo "  Ubuntu/Debian:  sudo apt install pipx"
    echo "  Fedora:         sudo dnf install pipx"
    echo "  macOS:          brew install pipx"
    echo "  Or via pip:     python3 -m pip install --user pipx"
    echo
    echo "Then ensure PATH is set up:"
    echo "  pipx ensurepath"
    exit 1
fi

# Get the project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📦 Installing from: $PROJECT_ROOT"
echo

# Install with pipx
pipx install "$PROJECT_ROOT"

# Download NLTK data to XDG location
echo
echo "📚 Downloading NLTK data to XDG data directory..."
python3 << 'EOF'
from config import setup_nltk_data_path
import nltk

# Configure NLTK to use XDG path
data_dir = setup_nltk_data_path()
print(f"   Installing to: {data_dir}")

# Download required data
nltk.download('punkt', download_dir=str(data_dir), quiet=True)
nltk.download('punkt_tab', download_dir=str(data_dir), quiet=True)
nltk.download('stopwords', download_dir=str(data_dir), quiet=True)
print("   ✓ NLTK data installed")
EOF

echo
echo "===================================================="
echo "✅ Installation complete!"
echo "===================================================="
echo
echo "Tools installed:"
echo "  • llmchat-convert  - Convert chat exports"
echo "  • llmchat-search   - Search converted vaults"
echo
echo "XDG-compliant directories created:"
echo "  • ~/.local/bin/                   (executables)"
echo "  • ~/.local/pipx/venvs/            (isolated environment)"
echo "  • ~/.config/llmchat-converter/    (configuration)"
echo "  • ~/.cache/llmchat-converter/     (embeddings cache)"
echo "  • ~/.local/share/llmchat-converter/ (NLTK data)"
echo
echo "🎉 You can now delete the cloned repository!"
echo
echo "Usage:"
echo "  llmchat-convert claude ~/Downloads/export.zip"
echo "  llmchat-search output/my-vault \"python testing\""
echo
echo "Update:"
echo "  cd $PROJECT_ROOT && git pull && pipx upgrade llmchat-knowledge-converter"
echo
echo "Uninstall:"
echo "  pipx uninstall llmchat-knowledge-converter"
echo "  rm -rf ~/.config/llmchat-converter ~/.cache/llmchat-converter"
echo
