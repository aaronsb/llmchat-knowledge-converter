"""
Embedding generation for semantic search.

Handles conversation title/description embedding using Nomic,
with caching and batch processing capabilities.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import hashlib
import numpy as np

try:
    from nomic import embed
    NOMIC_AVAILABLE = True
except ImportError:
    NOMIC_AVAILABLE = False


class EmbeddingGenerator:
    """Generates and caches embeddings for conversations."""

    def __init__(self, cache_dir: Optional[str] = None, model: str = 'nomic-embed-text-v1.5'):
        """
        Initialize embedding generator.

        Args:
            cache_dir: Directory to cache embeddings (default: XDG cache)
            model: Nomic model name
        """
        self.model = model

        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            # Use XDG cache directory via config
            try:
                from config import get_config
                self.cache_dir = get_config().embedding_cache_dir
            except ImportError:
                # Fallback if config module not available
                xdg_cache = os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
                self.cache_dir = Path(xdg_cache) / 'llmchat-converter' / 'embeddings'

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f'{model.replace("/", "_")}_cache.json'
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, List[float]]:
        """Load embedding cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Warning: Could not load embedding cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        """Save embedding cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"⚠️  Warning: Could not save embedding cache: {e}")

    def _make_cache_key(self, text: str) -> str:
        """Create a cache key from text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def generate_single(self, text: str, task_type: str = 'search_document') -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            task_type: Nomic task type ('search_document', 'search_query', etc.)

        Returns:
            Embedding as numpy array
        """
        if not NOMIC_AVAILABLE:
            raise ImportError(
                "Nomic package not installed. Install with: pip install nomic"
            )

        # Check cache
        cache_key = self._make_cache_key(text)
        if cache_key in self.cache:
            return np.array(self.cache[cache_key])

        # Generate embedding
        result = embed.text(
            texts=[text],
            model=self.model,
            task_type=task_type
        )

        embedding = np.array(result['embeddings'][0])

        # Cache result
        self.cache[cache_key] = embedding.tolist()
        self._save_cache()

        return embedding

    def generate_batch(self, texts: List[str], task_type: str = 'search_document',
                      batch_size: int = 100, show_progress: bool = True) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            task_type: Nomic task type
            batch_size: Number of texts per API call
            show_progress: Whether to show progress indicators

        Returns:
            List of embeddings as numpy arrays
        """
        if not NOMIC_AVAILABLE:
            raise ImportError(
                "Nomic package not installed. Install with: pip install nomic"
            )

        embeddings = []
        texts_to_generate = []
        text_indices = []

        # Check which texts need generation
        for i, text in enumerate(texts):
            cache_key = self._make_cache_key(text)
            if cache_key in self.cache:
                embeddings.append(np.array(self.cache[cache_key]))
            else:
                texts_to_generate.append(text)
                text_indices.append(i)
                embeddings.append(None)  # Placeholder

        if not texts_to_generate:
            if show_progress:
                print("✅ All embeddings found in cache")
            return embeddings

        # Generate in batches
        if show_progress:
            print(f"🔮 Generating {len(texts_to_generate)} embeddings...")

        generated = []
        for i in range(0, len(texts_to_generate), batch_size):
            batch = texts_to_generate[i:i + batch_size]

            if show_progress:
                progress = min(i + batch_size, len(texts_to_generate))
                print(f"   Processing {progress}/{len(texts_to_generate)}...", end='\r')

            result = embed.text(
                texts=batch,
                model=self.model,
                task_type=task_type
            )

            for text, embedding_list in zip(batch, result['embeddings']):
                embedding = np.array(embedding_list)
                generated.append(embedding)

                # Cache result
                cache_key = self._make_cache_key(text)
                self.cache[cache_key] = embedding.tolist()

        if show_progress:
            print(f"   Processing {len(texts_to_generate)}/{len(texts_to_generate)}... Done!")

        # Save cache
        self._save_cache()

        # Insert generated embeddings at correct positions
        for idx, embedding in zip(text_indices, generated):
            embeddings[idx] = embedding

        return embeddings

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the embedding cache."""
        return {
            'cached_embeddings': len(self.cache),
            'cache_file': str(self.cache_file),
            'cache_exists': self.cache_file.exists(),
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()


def generate_conversation_embedding(title: str, keywords: List[str],
                                    first_message: Optional[str] = None) -> str:
    """
    Create a text representation of a conversation for embedding.

    Combines title, keywords, and optionally the first message into
    a searchable text that represents the conversation's content.

    Args:
        title: Conversation title/name
        keywords: List of extracted keywords
        first_message: Optional first message preview

    Returns:
        Combined text for embedding
    """
    parts = [title]

    if keywords:
        # Add top keywords
        parts.append("Keywords: " + ", ".join(keywords[:10]))

    if first_message:
        # Add truncated first message (for context)
        preview = first_message[:500] if len(first_message) > 500 else first_message
        parts.append("Context: " + preview)

    return " | ".join(parts)


def test_nomic_connection() -> Tuple[bool, str]:
    """
    Test connection to Nomic API.

    Returns:
        Tuple of (success, message)
    """
    if not NOMIC_AVAILABLE:
        return False, "Nomic package not installed"

    try:
        result = embed.text(
            texts=["test"],
            model='nomic-embed-text-v1.5',
            task_type='search_document'
        )
        if result and 'embeddings' in result and len(result['embeddings']) > 0:
            return True, "Connection successful"
        return False, "Unexpected response format"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


if __name__ == '__main__':
    """Test embedding generation."""
    print("Testing Nomic embedding generation...\n")

    # Test connection
    success, message = test_nomic_connection()
    print(f"Connection test: {message}")

    if success:
        # Test embedding generation
        generator = EmbeddingGenerator()

        test_text = "How to implement binary search in Python"
        print(f"\nGenerating embedding for: '{test_text}'")

        embedding = generator.generate_single(test_text)
        print(f"✅ Generated embedding of dimension {embedding.shape[0]}")

        # Test batch
        test_batch = [
            "Machine learning basics",
            "Web development with React",
            "Database optimization techniques"
        ]
        print(f"\nGenerating batch of {len(test_batch)} embeddings...")
        embeddings = generator.generate_batch(test_batch)
        print(f"✅ Generated {len(embeddings)} embeddings")

        # Cache stats
        stats = generator.get_cache_stats()
        print(f"\nCache stats: {stats['cached_embeddings']} embeddings cached")
