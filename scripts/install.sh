#!/usr/bin/env bash
#
# Installation script for LLM Chat Knowledge Converter
# Installs to ~/.local/bin for system-wide CLI access
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INSTALL_DIR="${HOME}/.local/bin"
CONFIG_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/llmchat-converter"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "🚀 Installing LLM Chat Knowledge Converter"
echo "=========================================="
echo

# Create install directory
echo "📁 Creating installation directories..."
mkdir -p "${INSTALL_DIR}"
mkdir -p "${CONFIG_DIR}"

# Create virtual environment
if [ ! -d "${VENV_DIR}" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv "${VENV_DIR}"
else
    echo "✓ Virtual environment already exists"
fi

# Activate venv and install dependencies
echo "📦 Installing dependencies..."
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip -q
pip install -r "${PROJECT_ROOT}/requirements.txt" -q

# Download NLTK data
echo "📚 Downloading NLTK data..."
python3 << 'EOF'
import nltk
import sys
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("✓ NLTK data downloaded")
except Exception as e:
    print(f"⚠️  Warning: Could not download NLTK data: {e}", file=sys.stderr)
EOF

# Create wrapper scripts
echo "🔧 Creating CLI wrapper scripts..."

cat > "${INSTALL_DIR}/llmchat-convert" << EOF
#!/usr/bin/env bash
source "${VENV_DIR}/bin/activate"
python "${PROJECT_ROOT}/src/convert.py" "\$@"
EOF

cat > "${INSTALL_DIR}/llmchat-search" << EOF
#!/usr/bin/env bash
source "${VENV_DIR}/bin/activate"
python "${PROJECT_ROOT}/src/search_chats.py" "\$@"
EOF

chmod +x "${INSTALL_DIR}/llmchat-convert"
chmod +x "${INSTALL_DIR}/llmchat-search"

# Create config file with defaults
if [ ! -f "${CONFIG_DIR}/config.json" ]; then
    echo "⚙️  Creating default configuration..."
    cat > "${CONFIG_DIR}/config.json" << 'EOF'
{
  "default_output_dir": "output",
  "generate_embeddings": true,
  "tag_color_scheme": "rainbow",
  "file_color_scheme": "ocean",
  "last_opened_vault": null
}
EOF
fi

echo
echo "=========================================="
echo "✅ Installation complete!"
echo "=========================================="
echo
echo "CLI tools installed to: ${INSTALL_DIR}"
echo "  - llmchat-convert"
echo "  - llmchat-search"
echo
echo "Configuration directory: ${CONFIG_DIR}"
echo
echo "Make sure ${INSTALL_DIR} is in your PATH."
echo "Add this to your ~/.bashrc or ~/.zshrc if needed:"
echo "  export PATH=\"\${HOME}/.local/bin:\${PATH}\""
echo
echo "Usage:"
echo "  llmchat-convert claude /path/to/export.zip"
echo "  llmchat-search output/vault-name \"your query\""
echo
