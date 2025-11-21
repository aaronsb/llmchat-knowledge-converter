#!/usr/bin/env python3
"""
LLM Chat Knowledge Converter - Unified CLI

Converts chat exports from Claude and ChatGPT into searchable knowledge bases
with Obsidian-compatible markdown and SQLite indexing.
"""

import argparse
import sys
import zipfile
import shutil
import tempfile
from pathlib import Path
from typing import Optional
import json

from database import ConversationDatabase


def validate_zip_file(zip_path: Path, provider: str) -> bool:
    """
    Validate that zip file contains expected structure.

    Args:
        zip_path: Path to zip file
        provider: 'claude' or 'chatgpt'

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_files = {
        'claude': ['conversations.json'],
        'chatgpt': ['conversations.json']
    }

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            files = zf.namelist()
            for required in required_files[provider]:
                if not any(required in f for f in files):
                    raise ValueError(
                        f"Invalid {provider} export: missing {required}\n"
                        f"Expected structure: {required_files[provider]}"
                    )
        return True
    except zipfile.BadZipFile:
        raise ValueError(f"Invalid zip file: {zip_path}")


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    """
    Extract zip file to temporary directory.

    Args:
        zip_path: Path to zip file
        extract_to: Extraction destination

    Returns:
        Path to extracted directory
    """
    print(f"📦 Extracting {zip_path.name}...")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)

    # Find the root directory (might be nested)
    contents = list(extract_to.iterdir())
    if len(contents) == 1 and contents[0].is_dir():
        return contents[0]
    return extract_to


def check_output_exists(output_path: Path, force: bool) -> bool:
    """
    Check if output directory exists and handle overwrite confirmation.

    Args:
        output_path: Target output directory
        force: If True, skip confirmation

    Returns:
        True to proceed, False to cancel
    """
    if not output_path.exists():
        return True

    if force:
        print(f"⚠️  Overwriting existing vault: {output_path}")
        shutil.rmtree(output_path)
        return True

    # Interactive confirmation
    print(f"\n⚠️  Output directory already exists: {output_path}")
    print("    This will DELETE the existing vault and all its contents.")
    response = input("    Continue? [y/N]: ").strip().lower()

    if response == 'y':
        shutil.rmtree(output_path)
        return True

    print("❌ Cancelled.")
    return False


def determine_vault_name(zip_path: Path, custom_name: Optional[str]) -> str:
    """
    Determine output vault name from zip file or custom name.

    Args:
        zip_path: Path to input zip
        custom_name: Optional custom name

    Returns:
        Vault name to use
    """
    if custom_name:
        return custom_name

    # Extract from filename: "data-2025-11-20-22-45-30-batch-0000.zip" -> "data-2025-11-20-22-45-30-batch-0000"
    name = zip_path.stem
    # Clean up common patterns
    if name.startswith('data-'):
        name = name[5:]  # Remove 'data-' prefix
    if '-batch-' in name:
        name = name.split('-batch-')[0]  # Remove batch suffix

    return name or "converted_vault"


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert LLM chat exports to searchable knowledge bases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert Claude export
  %(prog)s claude /path/to/claude-export.zip

  # Convert ChatGPT export with custom vault name
  %(prog)s chatgpt /path/to/chatgpt-export.zip --name my-chatgpt-vault

  # Force overwrite existing vault
  %(prog)s claude /path/to/export.zip --force

  # Disable embeddings generation
  %(prog)s claude /path/to/export.zip --no-embeddings

Output:
  Converted vaults are created in: output/<vault-name>/
  - conversations/           (Markdown files organized by date)
  - conversations.db         (SQLite database with full-text search)
  - .obsidian/              (Obsidian configuration)

Search your converted chats:
  python src/search_chats.py output/<vault-name> "your query"
        """
    )

    parser.add_argument(
        'provider',
        choices=['claude', 'chatgpt'],
        help='Chat provider (claude or chatgpt)'
    )

    parser.add_argument(
        'zip_file',
        help='Path to exported chat archive (.zip)'
    )

    parser.add_argument(
        '--name', '-n',
        help='Custom vault name (default: derived from filename)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        default='output',
        help='Output directory (default: output/)'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force overwrite without confirmation'
    )

    parser.add_argument(
        '--no-embeddings',
        action='store_true',
        help='Skip generating embeddings (faster but no semantic search)'
    )

    parser.add_argument(
        '--skip-tags',
        action='store_true',
        help='Skip interactive tag configuration'
    )

    args = parser.parse_args()

    # Validate input
    zip_path = Path(args.zip_file)
    if not zip_path.exists():
        print(f"❌ Error: File not found: {zip_path}", file=sys.stderr)
        return 1

    if not zip_path.suffix == '.zip':
        print(f"❌ Error: Expected a .zip file, got: {zip_path.suffix}", file=sys.stderr)
        return 1

    try:
        # Validate zip structure
        validate_zip_file(zip_path, args.provider)
    except ValueError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1

    # Determine output path
    vault_name = determine_vault_name(zip_path, args.name)
    output_path = Path(args.output_dir) / vault_name

    # Check for overwrite
    if not check_output_exists(output_path, args.force):
        return 1

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Converting {args.provider.upper()} export...")
    print(f"   Input:  {zip_path}")
    print(f"   Output: {output_path}")
    print()

    # Extract to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        extracted_path = extract_zip(zip_path, temp_path)

        # Import and run appropriate converter
        if args.provider == 'claude':
            from convert_enhanced import convert_claude_history
            success = convert_claude_history(
                extracted_path,
                output_path,
                skip_tags=args.skip_tags,
                generate_embeddings=not args.no_embeddings
            )
        else:  # chatgpt
            from convert_chatgpt import convert_chatgpt_history
            success = convert_chatgpt_history(
                extracted_path,
                output_path,
                skip_tags=args.skip_tags,
                generate_embeddings=not args.no_embeddings
            )

    if success:
        print(f"\n✅ Conversion complete!")
        print(f"   Vault location: {output_path}")
        print(f"\n   Search your chats:")
        print(f"   python src/search_chats.py {output_path} \"your query\"")
        return 0
    else:
        print(f"\n❌ Conversion failed.", file=sys.stderr)
        return 1


if __name__ == '__main__':
    # If no arguments provided, print help
    if len(sys.argv) == 1:
        sys.argv.append('--help')

    sys.exit(main())
