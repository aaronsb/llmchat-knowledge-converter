# ADR-001: Semantic Search Architecture & Standalone Distribution

**Status:** In Progress
**Date:** 2025-11-20
**Authors:** Project Team
**Deciders:** Project Maintainers

## Context

The current LLM Chat Knowledge Converter has several limitations that hinder usability and scalability:

### Current Limitations

1. **Manual Multi-Step Workflow**: Users must manually extract zip files, place them in `input/`, run scripts, and then move output directories
2. **No Search Capability**: Once converted, there's no way to search conversations beyond manual file browsing or Obsidian's built-in search
3. **Poor Indexing**: Only JSON files with limited query capabilities
4. **Repo-Dependent**: Users must run scripts from within the git repository, making it difficult to use as a general-purpose tool
5. **Volume Problem**: Power users (like those who talk to Claude daily) generate thousands of conversations, making manual organization impractical

### User Pain Points

> "I talk a lot with Claude (which is great!), but I want to ingest SOME of my chats into another tool, but not ALL things. Right now I can't easily find which conversations are relevant."

> "The current workflow is a weird multi-step ritual of 'drop your files here' then run the script and choose some things. I just want to point it at a zip file."

### Goals

1. **Simplification**: Single-command conversion from zip to searchable vault
2. **Discoverability**: Powerful search to find relevant conversations among thousands
3. **Portability**: Install once, use anywhere - not tied to git repo
4. **Future-Proof**: Architecture that supports both CLI and GUI interfaces
5. **User Experience**: Remember preferences, cache models, smart defaults

## Decision

We will refactor the application into a **modular, installable toolkit** with three core components:

### 1. **Conversion Engine** (`llmchat-convert`)
- Direct zip file input with automatic extraction
- Unified CLI for Claude and ChatGPT exports
- SQLite database generation with full-text search indexes
- Semantic embeddings using Nomic for conversation titles
- Obsidian-compatible markdown output (maintains existing structure)

### 2. **Search Engine** (`llmchat-search`)
- Multiple search modes:
  - **Text search**: Fast full-text search using SQLite FTS5
  - **Keyword search**: Tag-based filtering
  - **Semantic search**: Similarity search using embeddings
- Rich result display with snippets and metadata
- Database statistics and analytics

### 3. **Installation & Distribution** (Future: GUI wrapper)
- XDG-compliant configuration management
- One-time setup script for dependencies and model caching
- Installable to `~/.local/bin` for system-wide access
- Future: Python GUI combining conversion + search in one interface

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User's Workflow                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Download export.zip from Claude/ChatGPT                  │
│  2. llmchat-convert claude export.zip                        │
│  3. llmchat-search output/vault-name "query"                 │
│                                                               │
│  [Future: GUI combines steps 2-3 with drag-drop]            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Component Architecture                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │  CLI Tools   │      │   GUI Tool   │  (Future)           │
│  ├──────────────┤      ├──────────────┤                     │
│  │ • convert    │      │ • Drag/Drop  │                     │
│  │ • search     │      │ • Search UI  │                     │
│  └──────┬───────┘      └──────┬───────┘                     │
│         │                     │                              │
│         └─────────┬───────────┘                              │
│                   │                                          │
│         ┌─────────▼──────────┐                               │
│         │   Core Libraries    │                               │
│         ├────────────────────┤                               │
│         │ • database.py      │  SQLite + FTS5               │
│         │ • converter_base   │  Shared conversion logic     │
│         │ • embeddings.py    │  Nomic integration           │
│         │ • config.py        │  XDG config management       │
│         └────────────────────┘                               │
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Data Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Input: export.zip                                           │
│     │                                                         │
│     ├─► Temporary Extraction                                 │
│     │                                                         │
│     ├─► Conversion Pipeline:                                 │
│     │   1. Parse conversations.json (streaming)              │
│     │   2. Extract keywords (TF-IDF)                         │
│     │   3. Generate embeddings (Nomic)                       │
│     │   4. Create markdown files                             │
│     │   5. Populate SQLite database                          │
│     │                                                         │
│     └─► Output: vault/                                       │
│          ├── conversations/         (markdown, organized)    │
│          ├── conversations.db       (SQLite + FTS5)          │
│          │   ├── conversations      (metadata, paths)        │
│          │   ├── messages           (content, FTS indexed)   │
│          │   ├── keywords           (tags, scores)           │
│          │   ├── embeddings         (semantic vectors)       │
│          │   └── messages_fts       (FTS5 virtual table)     │
│          └── .obsidian/             (graph config)           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Database Schema

```sql
-- Core conversation metadata
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    uuid TEXT UNIQUE,
    name TEXT,                    -- Title/description
    created_at TEXT,              -- ISO timestamp
    relative_path TEXT,           -- conversations/2024/11-November/20/title_abc123/
    message_count INTEGER,
    has_markdown BOOLEAN,
    source TEXT                   -- 'claude' or 'chatgpt'
);

-- Full message content for FTS
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER,
    sender TEXT,                  -- 'human' or 'assistant'
    content TEXT,                 -- Full message text
    index_in_conversation INTEGER,
    has_code BOOLEAN,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

-- Extracted keywords (TF-IDF)
CREATE TABLE keywords (
    id INTEGER PRIMARY KEY,
    keyword TEXT UNIQUE
);

CREATE TABLE conversation_keywords (
    conversation_id INTEGER,
    keyword_id INTEGER,
    score REAL,                   -- TF-IDF score
    PRIMARY KEY (conversation_id, keyword_id)
);

-- Semantic embeddings (Nomic)
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER,
    embedding BLOB,               -- Serialized numpy array
    model_name TEXT,              -- 'nomic-embed-text-v1.5'
    created_at TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

-- Full-text search (SQLite FTS5)
CREATE VIRTUAL TABLE messages_fts USING fts5(
    conversation_id UNINDEXED,
    sender UNINDEXED,
    content                        -- Indexed for full-text search
);
```

### XDG Configuration Structure

Following XDG Base Directory specification:

```
~/.config/llmchat-converter/
├── config.json              # User preferences
│   ├── last_opened_vault
│   ├── default_output_dir
│   ├── tag_color_scheme
│   ├── embedding_model
│   └── search_preferences
└── models/                  # Cached embedding models
    └── nomic-embed-text-v1.5/
```

```
~/.cache/llmchat-converter/
└── temp/                    # Temporary extraction directories
```

## Implementation Phases

### Phase 1: Core Refactoring (Current)
**Status:** In Progress

- [x] Design SQLite schema with FTS5
- [x] Create `database.py` module
- [x] Create `search_chats.py` CLI tool
- [x] Design `convert.py` unified CLI
- [ ] Create `embeddings.py` module for Nomic integration
- [ ] Update `convert_enhanced.py` to populate database
- [ ] Update `convert_chatgpt.py` to populate database
- [ ] Remove JSON index generation (replaced by SQLite)
- [ ] Update `requirements.txt` with new dependencies

**Dependencies to Add:**
- `nomic` - Embedding generation
- `numpy` - Vector operations
- Remove: `scikit-learn` (unused)

### Phase 2: Installation & Distribution
**Status:** Planned

- [ ] Create `config.py` for XDG-compliant configuration
- [ ] Create `/scripts/install.sh` for one-time setup
  - Creates virtual environment
  - Installs dependencies
  - Downloads/caches Nomic model
  - Creates wrapper scripts in `~/.local/bin`
  - Initializes XDG config directory
- [ ] Create wrapper scripts:
  - `llmchat-convert` → `python /path/to/venv/src/convert.py`
  - `llmchat-search` → `python /path/to/venv/src/search_chats.py`
- [ ] Test installation on clean system
- [ ] Update README with installation instructions

### Phase 3: Enhanced Features
**Status:** Future

- [ ] Preference management
  - Remember last opened vault
  - Search history
  - Custom tag color schemes
  - Embedding model selection
- [ ] Vault management commands
  - `llmchat-convert --list-vaults` (from config)
  - `llmchat-search --open <vault>` (remember as default)
  - `llmchat-search --stats <vault>` (analytics)
- [ ] Advanced search features
  - Date range filtering
  - Conversation source filtering
  - Combined search modes (text + semantic)
  - Export search results
- [ ] Incremental updates
  - Re-convert only new conversations
  - Merge into existing database
  - Preserve user customizations

### Phase 4: GUI Application
**Status:** Future

**Concept:** Unified desktop application combining all functionality

**Features:**
- Drag-and-drop zip file conversion
- Visual vault browser
- Integrated search interface with live results
- Tag cloud visualization
- Conversation timeline view
- Export capabilities (PDF, HTML, filtered subsets)

**Technology Options:**
- PyQt6 / PySide6 (native performance)
- Electron + Python backend (web technologies)
- Tkinter (lightweight, stdlib)

**GUI Architecture:**
```
┌────────────────────────────────────────┐
│        LLM Chat Knowledge Tool         │
├────────────────────────────────────────┤
│                                        │
│  ┌──────────────────────────────────┐ │
│  │  Convert Tab                     │ │
│  │  • Drag zip file here            │ │
│  │  • Select provider (auto-detect) │ │
│  │  • Choose output location        │ │
│  │  • [Convert] button              │ │
│  └──────────────────────────────────┘ │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │  Search Tab                      │ │
│  │  • Vault selector (dropdown)     │ │
│  │  • Search box with mode toggle   │ │
│  │  • Results list with preview     │ │
│  │  • [Open in Obsidian] button     │ │
│  └──────────────────────────────────┘ │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │  Manage Tab                      │ │
│  │  • Vault list                    │ │
│  │  • Statistics                    │ │
│  │  • Settings/Preferences          │ │
│  └──────────────────────────────────┘ │
└────────────────────────────────────────┘
```

## Technical Decisions

### Why SQLite?
- **Portable**: Single file, no server needed
- **Fast**: Optimized for read-heavy workloads
- **FTS5**: Built-in full-text search engine
- **ACID**: Reliable data integrity
- **Embeddable**: Works in both CLI and GUI

### Why Nomic Embeddings?
- **Quality**: State-of-the-art text embeddings
- **Efficiency**: Optimized for semantic search
- **Cacheable**: Model can be stored locally
- **API**: Easy integration
- **Alternative**: Could support local models (sentence-transformers) later

### Why XDG Compliance?
- **Standards**: Follows Linux/Unix conventions
- **Clean**: Doesn't pollute home directory
- **Portable**: Easy to backup/sync configuration
- **Multi-Platform**: XDG works on Linux/macOS, adaptable for Windows

### CLI vs GUI Priority?
**Decision: CLI First**

**Rationale:**
- Faster to implement and test
- More scriptable and automatable
- Appeals to technical users (early adopters)
- Easier to debug and maintain
- GUI can wrap CLI tools later

**GUI as Enhancement:**
- Expands user base to non-technical users
- Better for visual exploration
- Can be added without breaking CLI workflows

## Consequences

### Benefits

**For Users:**
- ✅ One command to convert archives
- ✅ Fast, powerful search across thousands of conversations
- ✅ Install once, use anywhere (not repo-dependent)
- ✅ Smart defaults with customization options
- ✅ Backward compatible with Obsidian workflows

**For Developers:**
- ✅ Clean separation of concerns (database, conversion, search)
- ✅ Testable modules
- ✅ Easy to extend with new providers
- ✅ GUI-ready architecture

### Tradeoffs

**Increased Complexity:**
- More dependencies (Nomic, SQLite extensions)
- Larger installation footprint
- Model caching requires disk space

**Migration Path:**
- Existing users need to re-convert vaults
- Old JSON indexes won't work with new search tool
- Need clear migration guide

**Nomic Dependency:**
- Requires API key (free tier available)
- Network dependency for first-time model download
- Could add local embedding model fallback

### Risks & Mitigations

**Risk: Nomic API changes**
- Mitigation: Abstract embedding interface, easy to swap providers

**Risk: Large database size**
- Mitigation: FTS5 is efficient; benchmarking needed; optional features

**Risk: Installation complexity**
- Mitigation: Single install script; thorough testing; clear docs

**Risk: Breaking changes for existing users**
- Mitigation: Version compatibility checks; migration tooling; changelog

## Alternatives Considered

### Alternative 1: Keep JSON Indexes
**Rejected:** Limited query capabilities, poor performance at scale

### Alternative 2: Use Elasticsearch/Meilisearch
**Rejected:** Overkill for single-user tool, requires separate server

### Alternative 3: Cloud-Based Search (SaaS)
**Rejected:** Privacy concerns, requires internet, ongoing costs

### Alternative 4: Build Embeddings with Sentence-Transformers
**Considered:** More privacy-friendly, but:
- Larger dependencies
- Slower inference
- Could add as alternative in Phase 3

### Alternative 5: Electron App First
**Rejected:** CLI provides faster MVP, easier testing, GUI can come later

## Success Metrics

**Phase 1 (Core Refactoring):**
- [ ] Convert 10,000+ conversation archive in < 5 minutes
- [ ] Search returns results in < 100ms
- [ ] Database size reasonable (< 2x markdown size)
- [ ] Zero data loss from current conversion format

**Phase 2 (Installation):**
- [ ] Install script works on Ubuntu, Arch, macOS
- [ ] User can run conversion within 5 minutes of git clone
- [ ] Tools available system-wide after install

**Phase 3 (Features):**
- [ ] Semantic search accuracy > 80% relevance
- [ ] Configuration persists across sessions
- [ ] Support incremental vault updates

**Phase 4 (GUI):**
- [ ] Non-technical users can convert and search without CLI
- [ ] GUI provides value beyond CLI (visualization, etc.)

## References

- [SQLite FTS5 Documentation](https://www.sqlite.org/fts5.html)
- [Nomic Embeddings](https://docs.nomic.ai/reference/endpoints/nomic-embed-text)
- [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)
- [Obsidian Graph Configuration](https://help.obsidian.md/Plugins/Graph+view)

## Revision History

- 2025-11-20: Initial draft (Phase 1 in progress)
- _Future revisions will be tracked here_

---

**Next Steps:**
1. Complete Phase 1 implementation
2. Test with real archive data
3. Create installation scripts
4. Update documentation
5. Gather user feedback before Phase 2
