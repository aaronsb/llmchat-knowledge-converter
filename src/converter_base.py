#!/usr/bin/env python3
"""
Base converter functionality shared between Claude and ChatGPT converters.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from decimal import Decimal

class DecimalEncoder(json.JSONEncoder):
    """Handle Decimal types in JSON serialization"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def create_conversation_structure(base_path: Path, date_info: Dict[str, str], 
                                conversation_name: str) -> Path:
    """Create the folder structure for a conversation"""
    # Create year/month/day structure
    year_folder = base_path / date_info['year']
    month_folder = year_folder / f"{date_info['month']}-{date_info['month_name']}"
    day_folder = month_folder / date_info['day']
    conv_folder = day_folder / conversation_name
    
    # Create all directories
    conv_folder.mkdir(parents=True, exist_ok=True)
    
    return conv_folder


def detect_markdown(text: str) -> bool:
    """Detect if text contains markdown formatting"""
    if not text:
        return False
    
    markdown_patterns = [
        r'^#{1,6}\s+',  # Headers
        r'```[\s\S]*?```',  # Code blocks
        r'`[^`]+`',  # Inline code
        r'\*\*[^*]+\*\*',  # Bold
        r'\*[^*]+\*',  # Italic
        r'\[[^\]]+\]\([^)]+\)',  # Links
        r'^\s*[-*+]\s+',  # Lists
        r'^\s*\d+\.\s+',  # Numbered lists
        r'^\s*>\s+',  # Blockquotes
        r'^\|.*\|$',  # Tables
    ]
    
    for pattern in markdown_patterns:
        if re.search(pattern, text, re.MULTILINE):
            return True
    
    return False


def extract_code_snippets(text: str, output_folder: Path) -> List[Dict[str, str]]:
    """Extract code blocks from text and save as separate files"""
    code_pattern = r'```(\w+)?\n(.*?)```'
    code_blocks = re.findall(code_pattern, text, re.DOTALL)
    
    if not code_blocks:
        return []
    
    snippets_folder = output_folder / 'code_snippets'
    snippets_folder.mkdir(exist_ok=True)
    
    # Language to extension mapping
    lang_extensions = {
        'python': 'py',
        'javascript': 'js',
        'typescript': 'ts',
        'java': 'java',
        'cpp': 'cpp',
        'c': 'c',
        'html': 'html',
        'css': 'css',
        'sql': 'sql',
        'bash': 'sh',
        'shell': 'sh',
        'json': 'json',
        'yaml': 'yaml',
        'xml': 'xml',
        'markdown': 'md',
    }
    
    saved_snippets = []
    
    for idx, (lang, code) in enumerate(code_blocks):
        lang = lang.lower() if lang else 'txt'
        extension = lang_extensions.get(lang, lang if lang else 'txt')
        
        snippet_filename = f"snippet_{idx:02d}.{extension}"
        snippet_path = snippets_folder / snippet_filename
        
        snippet_path.write_text(code.strip(), encoding='utf-8')
        
        saved_snippets.append({
            'filename': snippet_filename,
            'language': lang,
            'size': len(code.strip())
        })
    
    return saved_snippets


def enhance_markdown_content(content: str, conv_title: str, msg_idx: int,
                           sender: str, date_info: Dict[str, str], 
                           conv_tag: str, keywords: List[str],
                           platform: str = "Unknown") -> str:
    """Enhance markdown content with title and metadata"""
    lines = []
    
    # Add title
    sender_title = sender.replace('_', ' ').title()
    lines.append(f"# {conv_title} - {msg_idx:03d} {sender_title} Message")
    lines.append("")
    
    # Add original content
    lines.append(content)
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Add hashtags
    hashtags = [f"#{conv_tag}"]
    hashtags.extend(f"#{keyword.replace(' ', '-')}" for keyword in keywords)
    lines.append(' '.join(hashtags))
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Add metadata table
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Date | {date_info['year']}-{date_info['month_name']}-{date_info['day']} |")
    lines.append(f"| Conversation ID | {conv_tag.split('-')[-1]} |")
    lines.append(f"| Platform | {platform} |")
    
    return '\n'.join(lines)


def save_message_files(message: Dict[str, Any], idx: int, messages_folder: Path,
                      conv_folder: Path, conv_title: str, date_info: Dict[str, str],
                      conv_tag: str, keywords: List[str], platform: str = "Unknown") -> Dict[str, Any]:
    """Save message JSON and extract markdown/code if present"""
    msg_id = message['uuid'][:8] if len(message.get('uuid', '')) >= 8 else f"{idx:04d}"
    msg_filename = f"{idx:03d}_{message['sender']}_{msg_id}.json"
    
    # Check for markdown content
    has_markdown = detect_markdown(message.get('text', ''))
    
    # Create message metadata
    msg_metadata = {
        'uuid': message.get('uuid', ''),
        'sender': message['sender'],
        'created_at': message.get('created_at', ''),
        'updated_at': message.get('updated_at', ''),
        'text': message.get('text', ''),
        'has_attachments': bool(message.get('files')) or bool(message.get('attachments')),
        'has_files': bool(message.get('files')),
        'content_count': len(message.get('content', [])),
        'platform': platform
    }
    
    # Save message JSON
    msg_path = messages_folder / msg_filename
    with open(msg_path, 'w', encoding='utf-8') as f:
        json.dump(msg_metadata, f, indent=2, ensure_ascii=False, cls=DecimalEncoder)
    
    markdown_file = None
    
    # Extract and save markdown if detected
    if has_markdown and message.get('text'):
        md_filename = f"{conv_title.replace(' ', '_')}-{idx:03d}_{message['sender'].title()}_Message.md"
        md_path = messages_folder / md_filename  # Save to messages/ directory
        
        # Add metadata to markdown
        markdown_content = enhance_markdown_content(
            message['text'],
            conv_title,
            idx,
            message['sender'],
            date_info,
            conv_tag,
            keywords,
            platform
        )
        
        md_path.write_text(markdown_content, encoding='utf-8')
        markdown_file = md_filename
        msg_metadata['markdown_file'] = md_filename

        # Extract code snippets (also save to messages/ directory)
        snippets = extract_code_snippets(message['text'], messages_folder)
        if snippets:
            msg_metadata['code_snippets'] = snippets
    
    # Save updated message metadata if markdown was found
    if 'markdown_file' in msg_metadata:
        with open(msg_path, 'w', encoding='utf-8') as f:
            json.dump(msg_metadata, f, indent=2, ensure_ascii=False, cls=DecimalEncoder)
    
    return {
        'filename': msg_filename,
        'has_markdown': has_markdown,
        'markdown_file': markdown_file,
        'message_metadata': msg_metadata
    }