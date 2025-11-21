"""
SQLite database management for LLM chat conversations.

Provides schema definition, CRUD operations, and search functionality
for conversation indexing and semantic search.
"""

import sqlite3
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np


class ConversationDatabase:
    """Manages SQLite database for conversation indexing and search."""

    def __init__(self, db_path: str):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self):
        """Create database schema if it doesn't exist."""
        cursor = self.conn.cursor()

        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                relative_path TEXT NOT NULL,
                message_count INTEGER DEFAULT 0,
                has_markdown BOOLEAN DEFAULT 0,
                source TEXT NOT NULL,
                UNIQUE(uuid)
            )
        """)

        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                message_uuid TEXT,
                sender TEXT NOT NULL,
                content TEXT,
                created_at TEXT,
                index_in_conversation INTEGER NOT NULL,
                has_code BOOLEAN DEFAULT 0,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)

        # Keywords table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT UNIQUE NOT NULL
            )
        """)

        # Conversation-Keywords junction table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_keywords (
                conversation_id INTEGER NOT NULL,
                keyword_id INTEGER NOT NULL,
                score REAL DEFAULT 0.0,
                PRIMARY KEY (conversation_id, keyword_id),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                FOREIGN KEY (keyword_id) REFERENCES keywords(id) ON DELETE CASCADE
            )
        """)

        # Embeddings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                model_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)

        # Projects table (Claude-specific)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                relative_path TEXT NOT NULL
            )
        """)

        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_created ON conversations(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_source ON conversations(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_path ON conversations(relative_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_sender ON messages(sender)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_keywords_word ON keywords(keyword)")

        # Full-text search virtual table for message content
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                conversation_id UNINDEXED,
                sender UNINDEXED,
                content,
                content=messages,
                content_rowid=id
            )
        """)

        # Triggers to keep FTS table in sync
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, conversation_id, sender, content)
                VALUES (new.id, new.conversation_id, new.sender, new.content);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                DELETE FROM messages_fts WHERE rowid = old.id;
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                DELETE FROM messages_fts WHERE rowid = old.id;
                INSERT INTO messages_fts(rowid, conversation_id, sender, content)
                VALUES (new.id, new.conversation_id, new.sender, new.content);
            END
        """)

        self.conn.commit()

    def add_conversation(self, uuid: str, name: str, created_at: str,
                        relative_path: str, source: str, updated_at: Optional[str] = None,
                        message_count: int = 0, has_markdown: bool = False) -> int:
        """
        Add a conversation to the database.

        Args:
            uuid: Unique identifier for the conversation
            name: Conversation title/name
            created_at: ISO format timestamp
            relative_path: Path relative to vault root
            source: 'claude' or 'chatgpt'
            updated_at: Optional update timestamp
            message_count: Number of messages
            has_markdown: Whether conversation contains markdown

        Returns:
            Database ID of the inserted conversation
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO conversations
            (uuid, name, created_at, updated_at, relative_path, message_count, has_markdown, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (uuid, name, created_at, updated_at, relative_path, message_count, has_markdown, source))
        self.conn.commit()
        return cursor.lastrowid

    def add_message(self, conversation_id: int, sender: str, content: str,
                   index_in_conversation: int, message_uuid: Optional[str] = None,
                   created_at: Optional[str] = None, has_code: bool = False) -> int:
        """
        Add a message to the database.

        Args:
            conversation_id: Foreign key to conversations table
            sender: 'human' or 'assistant'
            content: Message text content
            index_in_conversation: Sequential index within conversation
            message_uuid: Optional unique message identifier
            created_at: Optional ISO format timestamp
            has_code: Whether message contains code blocks

        Returns:
            Database ID of the inserted message
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO messages
            (conversation_id, message_uuid, sender, content, created_at, index_in_conversation, has_code)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (conversation_id, message_uuid, sender, content, created_at, index_in_conversation, has_code))
        self.conn.commit()
        return cursor.lastrowid

    def add_keywords(self, conversation_id: int, keywords: List[Tuple[str, float]]):
        """
        Add keywords for a conversation.

        Args:
            conversation_id: Foreign key to conversations table
            keywords: List of (keyword, score) tuples
        """
        cursor = self.conn.cursor()

        for keyword, score in keywords:
            # Insert keyword if it doesn't exist
            cursor.execute("INSERT OR IGNORE INTO keywords (keyword) VALUES (?)", (keyword,))

            # Get keyword ID
            cursor.execute("SELECT id FROM keywords WHERE keyword = ?", (keyword,))
            keyword_id = cursor.fetchone()[0]

            # Link keyword to conversation
            cursor.execute("""
                INSERT OR REPLACE INTO conversation_keywords (conversation_id, keyword_id, score)
                VALUES (?, ?, ?)
            """, (conversation_id, keyword_id, score))

        self.conn.commit()

    def add_embedding(self, conversation_id: int, embedding: np.ndarray, model_name: str):
        """
        Add an embedding vector for a conversation.

        Args:
            conversation_id: Foreign key to conversations table
            embedding: Numpy array of embedding values
            model_name: Name of the embedding model used
        """
        cursor = self.conn.cursor()
        embedding_blob = pickle.dumps(embedding)
        created_at = datetime.utcnow().isoformat()

        cursor.execute("""
            INSERT INTO embeddings (conversation_id, embedding, model_name, created_at)
            VALUES (?, ?, ?, ?)
        """, (conversation_id, embedding_blob, model_name, created_at))
        self.conn.commit()

    def add_project(self, uuid: str, name: str, created_at: str,
                   relative_path: str, description: Optional[str] = None) -> int:
        """
        Add a Claude project to the database.

        Args:
            uuid: Unique identifier for the project
            name: Project name
            created_at: ISO format timestamp
            relative_path: Path relative to vault root
            description: Optional project description

        Returns:
            Database ID of the inserted project
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO projects (uuid, name, description, created_at, relative_path)
            VALUES (?, ?, ?, ?, ?)
        """, (uuid, name, description, created_at, relative_path))
        self.conn.commit()
        return cursor.lastrowid

    def search_text(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Full-text search across message content.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of conversation dictionaries with matching messages
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT c.*,
                   snippet(messages_fts, 2, '<mark>', '</mark>', '...', 64) as snippet
            FROM conversations c
            JOIN messages_fts mfts ON c.id = mfts.conversation_id
            WHERE messages_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))

        return [dict(row) for row in cursor.fetchall()]

    def search_by_keywords(self, keywords: List[str], limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search conversations by keywords.

        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results

        Returns:
            List of conversation dictionaries
        """
        cursor = self.conn.cursor()
        placeholders = ','.join('?' * len(keywords))
        cursor.execute(f"""
            SELECT c.*, SUM(ck.score) as total_score
            FROM conversations c
            JOIN conversation_keywords ck ON c.id = ck.conversation_id
            JOIN keywords k ON ck.keyword_id = k.id
            WHERE k.keyword IN ({placeholders})
            GROUP BY c.id
            ORDER BY total_score DESC
            LIMIT ?
        """, (*keywords, limit))

        return [dict(row) for row in cursor.fetchall()]

    def search_by_embedding(self, query_embedding: np.ndarray, limit: int = 20) -> List[Tuple[Dict[str, Any], float]]:
        """
        Semantic search using cosine similarity with embeddings.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results

        Returns:
            List of (conversation_dict, similarity_score) tuples
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT c.*, e.embedding
            FROM conversations c
            JOIN embeddings e ON c.id = e.conversation_id
        """)

        results = []
        for row in cursor.fetchall():
            conv_dict = dict(row)
            stored_embedding = pickle.loads(conv_dict.pop('embedding'))

            # Cosine similarity
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            results.append((conv_dict, float(similarity)))

        # Sort by similarity and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def get_conversation_by_uuid(self, uuid: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by its UUID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM conversations WHERE uuid = ?", (uuid,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_conversation_messages(self, conversation_id: int) -> List[Dict[str, Any]]:
        """Get all messages for a conversation."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM messages
            WHERE conversation_id = ?
            ORDER BY index_in_conversation
        """, (conversation_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_conversation_keywords(self, conversation_id: int) -> List[Tuple[str, float]]:
        """Get keywords for a conversation with their scores."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT k.keyword, ck.score
            FROM keywords k
            JOIN conversation_keywords ck ON k.id = ck.keyword_id
            WHERE ck.conversation_id = ?
            ORDER BY ck.score DESC
        """, (conversation_id,))
        return [(row[0], row[1]) for row in cursor.fetchall()]

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self.conn.cursor()

        stats = {}

        # Conversation counts
        cursor.execute("SELECT source, COUNT(*) FROM conversations GROUP BY source")
        stats['conversations_by_source'] = dict(cursor.fetchall())

        cursor.execute("SELECT COUNT(*) FROM conversations")
        stats['total_conversations'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM messages")
        stats['total_messages'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM keywords")
        stats['total_keywords'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM embeddings")
        stats['conversations_with_embeddings'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM projects")
        stats['total_projects'] = cursor.fetchone()[0]

        return stats

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
