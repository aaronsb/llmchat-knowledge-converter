"""
XDG-compliant configuration management for LLM Chat Knowledge Converter.

Handles paths, user preferences, and persistent state following the
XDG Base Directory specification.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Manages XDG-compliant configuration and paths."""

    APP_NAME = "llmchat-converter"

    def __init__(self):
        """Initialize configuration with XDG paths."""
        # XDG Base Directories
        self.config_home = Path(os.environ.get(
            'XDG_CONFIG_HOME',
            Path.home() / '.config'
        ))
        self.cache_home = Path(os.environ.get(
            'XDG_CACHE_HOME',
            Path.home() / '.cache'
        ))
        self.data_home = Path(os.environ.get(
            'XDG_DATA_HOME',
            Path.home() / '.local' / 'share'
        ))

        # Application directories
        self.config_dir = self.config_home / self.APP_NAME
        self.cache_dir = self.cache_home / self.APP_NAME
        self.data_dir = self.data_home / self.APP_NAME

        # Specific paths
        self.config_file = self.config_dir / 'config.json'
        self.embedding_cache_dir = self.cache_dir / 'embeddings'
        self.nltk_data_dir = self.data_dir / 'nltk_data'

        # Ensure directories exist
        self._ensure_dirs()

        # Load or create config
        self.settings = self._load_config()

    def _ensure_dirs(self):
        """Create XDG directories if they don't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create defaults."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Warning: Could not load config: {e}")
                return self._default_config()
        else:
            config = self._default_config()
            self.save()
            return config

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'default_output_dir': 'output',
            'generate_embeddings': True,
            'embedding_model': 'nomic-embed-text-v1.5',
            'tag_color_scheme': 'rainbow',
            'file_color_scheme': 'ocean',
            'tag_water_level': 30,
            'file_water_level': 30,
            'last_opened_vault': None,
            'search_history': [],
            'max_search_history': 50,
        }

    def save(self):
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"⚠️  Warning: Could not save config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.settings.get(key, default)

    def set(self, key: str, value: Any):
        """Set a configuration value and save."""
        self.settings[key] = value
        self.save()

    def remember_vault(self, vault_path: str):
        """Remember the last opened vault."""
        self.set('last_opened_vault', str(vault_path))

    def add_search_history(self, query: str, vault: str):
        """Add a search to history."""
        history = self.settings.get('search_history', [])
        history.insert(0, {
            'query': query,
            'vault': vault,
            'timestamp': __import__('datetime').datetime.utcnow().isoformat()
        })
        # Limit history size
        max_size = self.settings.get('max_search_history', 50)
        history = history[:max_size]
        self.set('search_history', history)

    def clear_cache(self):
        """Clear all cache directories."""
        import shutil
        for item in self.cache_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print(f"✅ Cache cleared: {self.cache_dir}")

    def print_info(self):
        """Print configuration information."""
        print("📁 Configuration Paths:")
        print(f"   Config:  {self.config_dir}")
        print(f"   Cache:   {self.cache_dir}")
        print(f"   Data:    {self.data_dir}")
        print()
        print("⚙️  Current Settings:")
        for key, value in self.settings.items():
            if key != 'search_history':  # Don't print full history
                print(f"   {key}: {value}")


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def setup_nltk_data_path():
    """Configure NLTK to use XDG data directory."""
    import nltk
    config = get_config()

    # Add our XDG data dir to NLTK's search path
    if str(config.nltk_data_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(config.nltk_data_dir))

    return config.nltk_data_dir


if __name__ == '__main__':
    """Display configuration information."""
    config = get_config()
    config.print_info()
