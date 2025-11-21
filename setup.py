"""
Setup configuration for LLM Chat Knowledge Converter
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="llmchat-knowledge-converter",
    version="2.0.0",
    author="LLM Chat Knowledge Converter Team",
    description="Convert Claude and ChatGPT exports to searchable knowledge bases with semantic search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llmchat-knowledge-converter",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "ijson>=3.2.0",
        "nltk>=3.8.0",
        "nomic>=3.0.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "llmchat-convert=convert:main",
            "llmchat-search=search_chats:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Text Processing",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="claude chatgpt export converter knowledge-base obsidian search embeddings",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/llmchat-knowledge-converter/issues",
        "Source": "https://github.com/yourusername/llmchat-knowledge-converter",
    },
)
