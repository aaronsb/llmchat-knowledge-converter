#!/usr/bin/env python3
"""
Semantic search utility for LLM chat conversations.

Searches converted chat vaults using text search, keyword matching,
and semantic similarity with embeddings.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from nomic import embed
    NOMIC_AVAILABLE = True
except ImportError:
    NOMIC_AVAILABLE = False

from database import ConversationDatabase


class ConversationSearcher:
    """Search interface for conversation databases."""

    def __init__(self, vault_path: str):
        """
        Initialize searcher with a vault path.

        Args:
            vault_path: Path to the converted chat vault
        """
        self.vault_path = Path(vault_path)
        self.db_path = self.vault_path / "conversations.db"

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self.db_path}\n"
                f"Make sure you've converted the chat archive first."
            )

        self.db = ConversationDatabase(str(self.db_path))

    def search_text(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Full-text search across message content.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching conversations
        """
        return self.db.search_text(query, limit)

    def search_keywords(self, keywords: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search by extracted keywords.

        Args:
            keywords: List of keywords to match
            limit: Maximum results

        Returns:
            List of matching conversations
        """
        return self.db.search_by_keywords(keywords, limit)

    def search_semantic(self, query: str, limit: int = 20) -> List[tuple]:
        """
        Semantic search using embeddings.

        Args:
            query: Natural language query
            limit: Maximum results

        Returns:
            List of (conversation, similarity_score) tuples
        """
        if not NOMIC_AVAILABLE:
            raise ImportError(
                "Nomic package not installed. Install with: pip install nomic"
            )

        # Generate embedding for query
        result = embed.text(
            texts=[query],
            model='nomic-embed-text-v1.5',
            task_type='search_query'
        )
        query_embedding = np.array(result['embeddings'][0])

        # Search by similarity
        return self.db.search_by_embedding(query_embedding, limit)

    def display_results(self, results: List[Dict[str, Any]], show_snippets: bool = False):
        """
        Display search results in a formatted way.

        Args:
            results: List of conversation dictionaries
            show_snippets: Whether to show text snippets
        """
        if not results:
            print("\n🔍 No results found.\n")
            return

        print(f"\n📚 Found {len(results)} matching conversation(s):\n")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            # Handle both regular results and semantic search results
            if isinstance(result, tuple):
                conv, score = result
                score_display = f" (similarity: {score:.3f})"
            else:
                conv = result
                score_display = ""

            # Build full path to conversation
            full_path = self.vault_path / "conversations" / conv['relative_path']

            print(f"\n{i}. {conv['name']}{score_display}")
            print(f"   📅 {conv['created_at'][:10]}")
            print(f"   💬 {conv['message_count']} messages")
            print(f"   📍 {full_path}")

            # Show keywords if available
            keywords = self.db.get_conversation_keywords(conv['id'])
            if keywords:
                keyword_str = ', '.join([f"#{k}" for k, _ in keywords[:5]])
                print(f"   🏷️  {keyword_str}")

            # Show snippet if available and requested
            if show_snippets and 'snippet' in conv:
                print(f"   📝 {conv['snippet']}")

        print("\n" + "=" * 80 + "\n")

    def display_statistics(self):
        """Display database statistics."""
        stats = self.db.get_statistics()

        print("\n📊 Database Statistics:")
        print("=" * 80)
        print(f"  Total conversations: {stats['total_conversations']}")
        print(f"  Total messages: {stats['total_messages']}")
        print(f"  Unique keywords: {stats['total_keywords']}")
        print(f"  Conversations with embeddings: {stats['conversations_with_embeddings']}")

        if stats['conversations_by_source']:
            print("\n  By source:")
            for source, count in stats['conversations_by_source'].items():
                print(f"    {source}: {count}")

        if stats['total_projects'] > 0:
            print(f"\n  Claude projects: {stats['total_projects']}")

        print("=" * 80 + "\n")

    def close(self):
        """Close database connection."""
        self.db.close()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Search converted LLM chat conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text search
  %(prog)s output/my-vault "machine learning optimization"

  # Keyword search
  %(prog)s output/my-vault --keywords python testing automation

  # Semantic search (requires embeddings)
  %(prog)s output/my-vault --semantic "how to build REST APIs"

  # Show statistics
  %(prog)s output/my-vault --stats

  # Limit results
  %(prog)s output/my-vault "debugging" --limit 10
        """
    )

    parser.add_argument(
        'vault_path',
        help='Path to the converted chat vault'
    )

    parser.add_argument(
        'query',
        nargs='?',
        help='Search query (not needed for --stats)'
    )

    parser.add_argument(
        '--keywords', '-k',
        nargs='+',
        help='Search by keywords instead of full-text'
    )

    parser.add_argument(
        '--semantic', '-s',
        action='store_true',
        help='Use semantic search with embeddings'
    )

    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=20,
        help='Maximum number of results (default: 20)'
    )

    parser.add_argument(
        '--snippets',
        action='store_true',
        help='Show text snippets in results'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show database statistics'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.stats and not args.query:
        parser.error("query is required unless using --stats")

    try:
        searcher = ConversationSearcher(args.vault_path)

        # Show statistics
        if args.stats:
            searcher.display_statistics()
            if not args.query:
                return 0

        # Perform search
        if args.semantic:
            if not NOMIC_AVAILABLE:
                print("❌ Error: Semantic search requires the 'nomic' package.", file=sys.stderr)
                print("   Install with: pip install nomic", file=sys.stderr)
                return 1

            results = searcher.search_semantic(args.query, args.limit)
        elif args.keywords:
            results = searcher.search_keywords(args.keywords, args.limit)
        else:
            results = searcher.search_text(args.query, args.limit)

        searcher.display_results(results, show_snippets=args.snippets)
        searcher.close()
        return 0

    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
