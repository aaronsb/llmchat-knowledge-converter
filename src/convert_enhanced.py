#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import ijson
import re
from typing import Dict, Any, List, Tuple, Optional, Set
from decimal import Decimal
from collections import Counter, defaultdict
import math
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tag_analyzer import TagAnalyzer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class DecimalEncoder(json.JSONEncoder):
    """Handle Decimal types in JSON serialization"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

class KeywordExtractor:
    """Extract keywords using TF-IDF and Bayesian-like scoring"""
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Add common programming and conversation stop words
        self.stop_words.update({
            'would', 'could', 'should', 'might', 'must', 'shall', 'will',
            'just', 'like', 'use', 'using', 'used', 'make', 'making', 'made',
            'want', 'need', 'help', 'please', 'thanks', 'thank', 'hello', 'hi',
            'yes', 'no', 'okay', 'ok', 'sure', 'right', 'left', 'top', 'bottom',
            'first', 'second', 'third', 'last', 'next', 'previous', 'current',
            'new', 'old', 'good', 'bad', 'best', 'worst', 'better', 'worse'
        })
        self.doc_freq = defaultdict(int)
        self.total_docs = 0
        
    def preprocess_text(self, text: str) -> List[str]:
        """Tokenize and clean text"""
        # Remove markdown formatting
        text = re.sub(r'```[\s\S]*?```', '', text)  # Remove code blocks
        text = re.sub(r'`[^`]+`', '', text)  # Remove inline code
        text = re.sub(r'[#*_\[\]()]', ' ', text)  # Remove markdown symbols
        text = re.sub(r'https?://\S+', '', text)  # Remove URLs
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Filter tokens
        tokens = [
            token for token in tokens
            if token not in self.stop_words
            and token not in string.punctuation
            and len(token) > 2
            and not token.isdigit()
        ]
        
        return tokens
    
    def extract_keywords(self, text: str, max_keywords: int = 7) -> List[str]:
        """Extract top keywords from text"""
        tokens = self.preprocess_text(text)
        if not tokens:
            return []
        
        # Calculate term frequency
        term_freq = Counter(tokens)
        total_terms = len(tokens)
        
        # Calculate TF-IDF scores
        scores = {}
        for term, freq in term_freq.items():
            tf = freq / total_terms
            # Simple IDF approximation (will be refined as we process more docs)
            idf = math.log(max(2, self.total_docs / (1 + self.doc_freq.get(term, 0))))
            scores[term] = tf * idf
        
        # Get top keywords
        top_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
        return [keyword for keyword, _ in top_keywords]
    
    def update_corpus_stats(self, text: str):
        """Update document frequency statistics"""
        tokens = set(self.preprocess_text(text))
        for token in tokens:
            self.doc_freq[token] += 1
        self.total_docs += 1

def sanitize_filename(name: str, max_length: int = 50) -> str:
    """Convert a string to a safe filename"""
    # Remove or replace unsafe characters
    name = re.sub(r'[<>:"/\\|?*]', '_', str(name))
    # Replace multiple spaces/underscores with single underscore
    name = re.sub(r'[_\s]+', '_', name)
    # Trim and remove trailing periods/spaces
    name = name.strip(' ._')
    # Limit length
    if len(name) > max_length:
        name = name[:max_length].rstrip(' ._')
    # Default name if empty
    return name or 'unnamed'

def humanize_title(title: str) -> str:
    """Convert underscored title to human-readable format"""
    # Replace underscores with spaces
    title = title.replace('_', ' ')
    # Capitalize words (but keep existing capitalization if present)
    words = title.split()
    humanized = []
    for word in words:
        if word.islower():
            humanized.append(word.capitalize())
        else:
            humanized.append(word)
    return ' '.join(humanized)

def detect_markdown_content(text: str) -> bool:
    """Detect if text contains significant markdown formatting"""
    if not text or len(text) < 20:
        return False
    
    markdown_patterns = [
        r'^#{1,6}\s+.+',  # Headers
        r'^\*{3,}$|^-{3,}$|^_{3,}$',  # Horizontal rules
        r'```[\s\S]*?```',  # Code blocks
        r'`[^`]+`',  # Inline code
        r'\[.+?\]\(.+?\)',  # Links
        r'!\[.*?\]\(.+?\)',  # Images
        r'^\s*[-*+]\s+',  # Unordered lists
        r'^\s*\d+\.\s+',  # Ordered lists
        r'\*\*[^*]+\*\*',  # Bold
        r'\*[^*]+\*',  # Italic
        r'^>\s+',  # Blockquotes
        r'\|.*\|.*\|',  # Tables
    ]
    
    # Count markdown indicators
    markdown_count = 0
    for pattern in markdown_patterns:
        if re.search(pattern, text, re.MULTILINE):
            markdown_count += 1
    
    # Consider it markdown if we find multiple indicators or code blocks
    return markdown_count >= 2 or '```' in text

def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """Extract code blocks from markdown text"""
    code_blocks = []
    pattern = r'```(\w*)\n([\s\S]*?)```'
    
    for match in re.finditer(pattern, text):
        language = match.group(1) or 'txt'
        code = match.group(2).strip()
        code_blocks.append((language, code))
    
    return code_blocks

def get_title_from_markdown(text: str, default: str = "content") -> str:
    """Extract title from markdown content"""
    # Try to find first header
    header_match = re.search(r'^#{1,6}\s+(.+)$', text, re.MULTILINE)
    if header_match:
        return sanitize_filename(header_match.group(1).strip())
    
    # Try to get first non-empty line
    lines = text.strip().split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if line and not line.startswith('#') and len(line) > 5:
            return sanitize_filename(line[:50])
    
    return default

def save_markdown_content(content: str, base_path: Path, filename: str = None, 
                         conversation_title: str = None, keywords: List[str] = None,
                         date_info: Dict[str, str] = None, uuid_short: str = None,
                         conversation_tag: str = None, tag_analyzer: TagAnalyzer = None) -> Optional[Path]:
    """Save markdown content to a .md file with enhanced metadata"""
    if not content or not detect_markdown_content(content):
        return None
    
    # Determine filename
    if not filename:
        filename = get_title_from_markdown(content)
    
    # Ensure .md extension
    if not filename.endswith('.md'):
        filename += '.md'
    
    filepath = base_path / filename
    
    # Avoid overwriting by adding number suffix
    counter = 1
    original_filepath = filepath
    while filepath.exists():
        filepath = original_filepath.with_stem(f"{original_filepath.stem}_{counter}")
        counter += 1
    
    # Prepare content with title and hashtags
    enhanced_content = ""
    
    # Add title if provided
    if conversation_title:
        # Check if content already has a header
        if not re.match(r'^#\s+', content):
            enhanced_content = f"# {conversation_title}\n\n"
    
    enhanced_content += content
    
    # Add hashtags at the end
    hashtags = []
    
    # Add conversation tag first for grouping
    if conversation_tag:
        hashtags.append(f'#{conversation_tag}')
        if tag_analyzer:
            tag_analyzer.add_tag(conversation_tag, 'conversation')
    
    # Add keyword hashtags
    if keywords:
        for tag in keywords:
            hashtags.append(f'#{tag}')
            if tag_analyzer:
                tag_analyzer.add_tag(tag, 'keyword')
    
    if hashtags:
        enhanced_content += f"\n\n---\n\n{' '.join(hashtags)}"
    
    # Add metadata table
    metadata_table = ["\n\n---\n"]
    metadata_table.append("| Field | Value |")
    metadata_table.append("|-------|-------|")
    
    if date_info:
        year = date_info.get('year', 'N/A')
        month = date_info.get('month', 'N/A').capitalize()
        day = date_info.get('day', 'N/A')
        metadata_table.append(f"| Date | {year}-{month}-{day} |")
    
    if uuid_short:
        metadata_table.append(f"| Conversation ID | {uuid_short} |")
    
    enhanced_content += '\n'.join(metadata_table)
    
    # Write content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(enhanced_content)
    
    # Extract and save code blocks separately
    code_blocks = extract_code_blocks(content)
    if code_blocks:
        code_dir = base_path / 'code_snippets'
        code_dir.mkdir(exist_ok=True)
        
        for idx, (language, code) in enumerate(code_blocks):
            ext = {
                'python': 'py',
                'javascript': 'js',
                'typescript': 'ts',
                'java': 'java',
                'cpp': 'cpp',
                'c': 'c',
                'bash': 'sh',
                'shell': 'sh',
                'sql': 'sql',
                'html': 'html',
                'css': 'css',
                'json': 'json',
                'yaml': 'yaml',
                'yml': 'yml',
                'xml': 'xml',
            }.get(language.lower(), 'txt')
            
            code_filename = f"snippet_{idx:02d}.{ext}"
            with open(code_dir / code_filename, 'w', encoding='utf-8') as f:
                f.write(code)
    
    return filepath

def create_conversation_structure(conversation: Dict[str, Any], base_path: Path) -> Tuple[Path, str, Dict[str, str]]:
    """Create folder structure for a conversation and return the path, title, and date info"""
    # Parse date for organization
    created_at = conversation.get('created_at', '')
    date_info = {}
    try:
        date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        year = date_obj.strftime('%Y')
        month = date_obj.strftime('%m-%B')
        day = date_obj.strftime('%d')
        
        # Store date info for hashtags
        date_info = {
            'year': date_obj.strftime('%Y'),
            'month': date_obj.strftime('%B').lower(),
            'day': date_obj.strftime('%d')
        }
    except:
        year, month, day = 'unknown', 'unknown', 'unknown'
    
    # Create conversation name
    conv_name = conversation.get('name', '').strip()
    if not conv_name and conversation.get('chat_messages'):
        # Try to extract name from first message
        first_msg = conversation['chat_messages'][0]
        text = first_msg.get('text', '')
        if text:
            conv_name = text[:50] + '...' if len(text) > 50 else text
    
    conv_name = sanitize_filename(conv_name or f"conversation_{conversation['uuid'][:8]}")
    conv_folder_name = f"{conv_name}_{conversation['uuid'][:8]}"
    
    # Create folder structure: year/month/day/conversation_name_uuid
    conv_folder = base_path / year / month / day / conv_folder_name
    conv_folder.mkdir(parents=True, exist_ok=True)
    
    # Return path, human-readable title, and date info
    conv_title = humanize_title(conv_name)
    
    return conv_folder, conv_title, date_info

def save_conversation(conversation: Dict[str, Any], conv_folder: Path, conv_title: str, 
                     date_info: Dict[str, str], keyword_extractor: KeywordExtractor,
                     tag_analyzer: TagAnalyzer = None):
    """Save conversation data to folder structure with markdown extraction and keywords"""
    # Create a unique conversation tag by combining title and short UUID
    conv_tag = f"conv-{conv_title.replace(' ', '-').lower()}-{conversation['uuid'][:8]}"
    
    # Collect all text for keyword extraction
    all_text = []
    
    # Add conversation name
    if conversation.get('name'):
        all_text.append(conversation['name'])
    
    # Collect message texts
    for message in conversation.get('chat_messages', []):
        if message.get('text'):
            all_text.append(message['text'])
        for content_item in message.get('content', []):
            if isinstance(content_item, dict) and 'text' in content_item:
                all_text.append(content_item['text'])
    
    # Extract keywords for the entire conversation
    full_text = ' '.join(all_text)
    conversation_keywords = keyword_extractor.extract_keywords(full_text) if full_text else []
    
    # Update corpus statistics
    if full_text:
        keyword_extractor.update_corpus_stats(full_text)
    
    # Save metadata
    metadata = {
        'uuid': conversation['uuid'],
        'name': conversation.get('name', ''),
        'created_at': conversation['created_at'],
        'updated_at': conversation['updated_at'],
        'account_uuid': conversation['account']['uuid'],
        'message_count': len(conversation.get('chat_messages', [])),
        'has_markdown_content': False,
        'keywords': conversation_keywords
    }
    
    # Save messages
    messages_folder = conv_folder / 'messages'
    messages_folder.mkdir(exist_ok=True)
    
    markdown_files = []
    
    for idx, message in enumerate(conversation.get('chat_messages', [])):
        msg_filename = f"{idx:03d}_{message['sender']}_{message['uuid'][:8]}.json"
        
        # Extract message data
        msg_data = {
            'uuid': message['uuid'],
            'sender': message['sender'],
            'created_at': message['created_at'],
            'updated_at': message['updated_at'],
            'text': message.get('text', ''),
            'has_attachments': len(message.get('attachments', [])) > 0,
            'has_files': len(message.get('files', [])) > 0,
            'content_count': len(message.get('content', []))
        }
        
        # Create enhanced title for markdown files
        msg_title = f"{conv_title} - {idx:03d} {message['sender'].capitalize()} Message"
        
        # Create better filename (without UUID in filename)
        sender_cap = message['sender'].capitalize()
        md_filename_base = f"{conv_title.replace(' ', '_')}-{idx:03d}_{sender_cap}_Message"
        
        # Check for markdown in message text
        text = message.get('text', '')
        if text and detect_markdown_content(text):
            md_path = save_markdown_content(text, messages_folder, md_filename_base, 
                                          msg_title, conversation_keywords, 
                                          date_info, conversation['uuid'][:8],
                                          conv_tag, tag_analyzer)
            if md_path:
                msg_data['markdown_file'] = md_path.name
                markdown_files.append(md_path.name)
                metadata['has_markdown_content'] = True
        
        # Check for markdown in content items
        for content_idx, content_item in enumerate(message.get('content', [])):
            if isinstance(content_item, dict) and 'text' in content_item:
                content_text = content_item.get('text', '')
                if content_text and detect_markdown_content(content_text):
                    content_title = f"{conv_title} - {idx:03d} {message['sender'].capitalize()} Content {content_idx:02d}"
                    md_filename_content = f"{conv_title.replace(' ', '_')}-{idx:03d}_{sender_cap}_Content_{content_idx:02d}"
                    md_path = save_markdown_content(content_text, messages_folder, md_filename_content,
                                                  content_title, conversation_keywords,
                                                  date_info, conversation['uuid'][:8],
                                                  conv_tag, tag_analyzer)
                    if md_path:
                        markdown_files.append(md_path.name)
                        metadata['has_markdown_content'] = True
        
        # Save message metadata
        with open(messages_folder / msg_filename, 'w') as f:
            json.dump(msg_data, f, indent=2, cls=DecimalEncoder)
        
        # Save full content if exists
        if message.get('content'):
            content_file = messages_folder / f"{idx:03d}_content.json"
            with open(content_file, 'w') as f:
                json.dump(message['content'], f, indent=2, cls=DecimalEncoder)
        
        # Save attachments info if exists
        if message.get('attachments'):
            attach_file = messages_folder / f"{idx:03d}_attachments.json"
            with open(attach_file, 'w') as f:
                json.dump(message['attachments'], f, indent=2, cls=DecimalEncoder)
    
    metadata['markdown_files'] = markdown_files
    
    with open(conv_folder / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, cls=DecimalEncoder)

def save_project(project: Dict[str, Any], projects_folder: Path, keyword_extractor: KeywordExtractor,
                tag_analyzer: TagAnalyzer = None):
    """Save project data to folder structure with markdown extraction and keywords"""
    project_name = sanitize_filename(project.get('name', f"project_{project['uuid'][:8]}"))
    project_folder = projects_folder / f"{project_name}_{project['uuid'][:8]}"
    project_folder.mkdir(parents=True, exist_ok=True)
    
    # Parse date for hashtags
    date_info = {}
    created_at = project.get('created_at', '')
    try:
        date_obj = datetime.fromisoformat(created_at.replace('+00:00', '+00:00').replace('Z', '+00:00'))
        date_info = {
            'year': date_obj.strftime('%Y'),
            'month': date_obj.strftime('%B').lower(),
            'day': date_obj.strftime('%d')
        }
    except:
        pass
    
    # Collect all text for keyword extraction
    all_text = []
    if project.get('name'):
        all_text.append(project['name'])
    if project.get('description'):
        all_text.append(project['description'])
    if project.get('prompt_template'):
        all_text.append(project['prompt_template'])
    
    # Add document content
    for doc in project.get('docs', []):
        if doc.get('content'):
            all_text.append(doc['content'])
    
    # Extract keywords
    full_text = ' '.join(all_text)
    project_keywords = keyword_extractor.extract_keywords(full_text) if full_text else []
    
    # Update corpus statistics
    if full_text:
        keyword_extractor.update_corpus_stats(full_text)
    
    # Human-readable project title
    project_title = humanize_title(project_name) + f" {project['uuid'][:8]}"
    
    # Save project metadata
    metadata = {
        'uuid': project['uuid'],
        'name': project['name'],
        'description': project.get('description', ''),
        'is_private': project.get('is_private', True),
        'is_starter_project': project.get('is_starter_project', False),
        'created_at': project['created_at'],
        'updated_at': project['updated_at'],
        'creator': project.get('creator', {}),
        'prompt_template': project.get('prompt_template', ''),
        'has_markdown_content': False,
        'keywords': project_keywords
    }
    
    markdown_files = []
    
    # Check for markdown in description
    if project.get('description') and detect_markdown_content(project['description']):
        md_path = save_markdown_content(project['description'], project_folder, 
                                      'project_description', f"{project_title} - Description",
                                      project_keywords, date_info, project['uuid'][:8], 
                                      None, tag_analyzer)
        if md_path:
            markdown_files.append(md_path.name)
            metadata['has_markdown_content'] = True
    
    # Check for markdown in prompt template
    if project.get('prompt_template') and detect_markdown_content(project['prompt_template']):
        md_path = save_markdown_content(project['prompt_template'], project_folder, 
                                      'prompt_template', f"{project_title} - Prompt Template",
                                      project_keywords, date_info, project['uuid'][:8],
                                      None, tag_analyzer)
        if md_path:
            markdown_files.append(md_path.name)
            metadata['has_markdown_content'] = True
    
    # Save documents
    docs = project.get('docs', [])
    if docs:
        docs_folder = project_folder / 'documents'
        docs_folder.mkdir(exist_ok=True)
        
        for idx, doc in enumerate(docs):
            doc_name = sanitize_filename(doc.get('filename', f"doc_{idx}"))
            
            # Check if content is markdown
            content = doc.get('content', '')
            if content and detect_markdown_content(content):
                # Save as markdown file
                doc_title = f"{project_title} - {humanize_title(doc_name)}"
                md_path = save_markdown_content(content, docs_folder, doc_name, 
                                              doc_title, project_keywords, date_info, project['uuid'][:8],
                                              None, tag_analyzer)
                if md_path:
                    markdown_files.append(f"documents/{md_path.name}")
                    metadata['has_markdown_content'] = True
            
            # Always save the full document data
            doc_file = docs_folder / f"{doc_name}_{doc['uuid'][:8]}.json"
            with open(doc_file, 'w') as f:
                json.dump(doc, f, indent=2, cls=DecimalEncoder)
    
    metadata['markdown_files'] = markdown_files
    
    with open(project_folder / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, cls=DecimalEncoder)

def convert_conversations(input_file: str, output_base: Path, keyword_extractor: KeywordExtractor,
                         tag_analyzer: TagAnalyzer = None):
    """Convert conversations.json to folder structure with markdown extraction"""
    print(f"\nConverting {input_file} with enhanced markdown extraction...")
    conversations_folder = output_base / 'conversations'
    conversations_folder.mkdir(exist_ok=True)
    
    count = 0
    markdown_count = 0
    
    with open(input_file, 'rb') as f:
        parser = ijson.items(f, 'item')
        
        for conversation in parser:
            try:
                conv_folder, conv_title, date_info = create_conversation_structure(conversation, conversations_folder)
                save_conversation(conversation, conv_folder, conv_title, date_info, keyword_extractor, tag_analyzer)
                count += 1
                
                # Check if this conversation had markdown
                metadata_file = conv_folder / 'metadata.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r') as mf:
                        metadata = json.load(mf)
                        if metadata.get('has_markdown_content'):
                            markdown_count += 1
                
                if count % 100 == 0:
                    print(f"  Processed {count} conversations ({markdown_count} with markdown)...", end='\r')
                    
            except Exception as e:
                print(f"\nError processing conversation {conversation.get('uuid', 'unknown')}: {e}")
    
    print(f"  Converted {count} conversations total ({markdown_count} with markdown content)    ")
    
    # Create index file
    create_index(conversations_folder, 'conversations')

def convert_projects(input_file: str, output_base: Path, keyword_extractor: KeywordExtractor,
                    tag_analyzer: TagAnalyzer = None):
    """Convert projects.json to folder structure with markdown extraction"""
    print(f"\nConverting {input_file} with enhanced markdown extraction...")
    projects_folder = output_base / 'projects'
    projects_folder.mkdir(exist_ok=True)
    
    markdown_count = 0
    
    with open(input_file, 'r') as f:
        projects = json.load(f)
    
    for project in projects:
        try:
            save_project(project, projects_folder, keyword_extractor, tag_analyzer)
            
            # Check if this project had markdown
            project_name = sanitize_filename(project.get('name', f"project_{project['uuid'][:8]}"))
            project_folder = projects_folder / f"{project_name}_{project['uuid'][:8]}"
            metadata_file = project_folder / 'metadata.json'
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as mf:
                    metadata = json.load(mf)
                    if metadata.get('has_markdown_content'):
                        markdown_count += 1
                        
        except Exception as e:
            print(f"Error processing project {project.get('uuid', 'unknown')}: {e}")
    
    print(f"  Converted {len(projects)} projects ({markdown_count} with markdown content)")
    
    # Create index file
    create_index(projects_folder, 'projects')

def create_index(folder: Path, data_type: str):
    """Create an index file for quick lookups"""
    index_data = []
    
    if data_type == 'conversations':
        # Walk through year/month/day structure
        for year_dir in sorted(folder.iterdir()):
            if not year_dir.is_dir() or year_dir.name.startswith('.'):
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                for day_dir in sorted(month_dir.iterdir()):
                    if not day_dir.is_dir():
                        continue
                    for conv_dir in sorted(day_dir.iterdir()):
                        if not conv_dir.is_dir():
                            continue
                        
                        # Read metadata
                        metadata_file = conv_dir / 'metadata.json'
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                                index_data.append({
                                    'path': str(conv_dir.relative_to(folder)),
                                    'uuid': metadata['uuid'],
                                    'name': metadata['name'],
                                    'created_at': metadata['created_at'],
                                    'message_count': metadata['message_count'],
                                    'has_markdown': metadata.get('has_markdown_content', False),
                                    'markdown_files': metadata.get('markdown_files', []),
                                    'keywords': metadata.get('keywords', [])
                                })
    
    elif data_type == 'projects':
        for project_dir in sorted(folder.iterdir()):
            if not project_dir.is_dir() or project_dir.name.startswith('.'):
                continue
            
            metadata_file = project_dir / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    index_data.append({
                        'path': str(project_dir.relative_to(folder)),
                        'uuid': metadata['uuid'],
                        'name': metadata['name'],
                        'created_at': metadata['created_at'],
                        'has_markdown': metadata.get('has_markdown_content', False),
                        'markdown_files': metadata.get('markdown_files', []),
                        'keywords': metadata.get('keywords', [])
                    })
    
    # Save index
    index_file = folder.parent / f'{data_type}_index.json'
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    # Count markdown items
    markdown_count = sum(1 for item in index_data if item.get('has_markdown'))
    print(f"  Created index with {len(index_data)} entries ({markdown_count} with markdown): {index_file}")

def convert_claude_history(input_path: Path, output_path: Path,
                          skip_tags: bool = False,
                          generate_embeddings: bool = True) -> bool:
    """
    Convert Claude chat history to searchable knowledge base.

    Args:
        input_path: Path to extracted Claude export directory
        output_path: Path to output vault directory
        skip_tags: Skip interactive tag configuration
        generate_embeddings: Generate semantic embeddings (requires Nomic)

    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        from database import ConversationDatabase
        from embeddings import EmbeddingGenerator, generate_conversation_embedding, NOMIC_AVAILABLE
    except ImportError as e:
        print(f"❌ Error importing required modules: {e}")
        return False

    try:
        output_base = Path(output_path)
        output_base.mkdir(parents=True, exist_ok=True)
        input_dir = Path(input_path)

        # Initialize keyword extractor and tag analyzer
        keyword_extractor = KeywordExtractor()
        tag_analyzer = TagAnalyzer()

        # Initialize database
        db_path = output_base / 'conversations.db'
        db = ConversationDatabase(str(db_path))
        print(f"📊 Creating database: {db_path}")

        # Initialize embedding generator if requested
        embedding_generator = None
        if generate_embeddings:
            if not NOMIC_AVAILABLE:
                print("⚠️  Nomic not available, skipping embeddings")
                generate_embeddings = False
            else:
                embedding_generator = EmbeddingGenerator()
                print("🔮 Embedding generator initialized")

        # Save user info
        users_file = input_dir / 'users.json'
        if users_file.exists():
            print("\nCopying users.json...")
            with open(users_file, 'r') as f:
                users = json.load(f)
            with open(output_base / 'users.json', 'w') as f:
                json.dump(users, f, indent=2)
    
        # Convert conversations with database population
        conversations_file = input_dir / 'conversations.json'
        if conversations_file.exists():
            print(f"\n🔄 Converting conversations...")
            conversations_folder = output_base / 'conversations'
            conversations_folder.mkdir(exist_ok=True)
    
            count = 0
            markdown_count = 0
    
            # Collect all conversations for batch embedding
            conversations_to_embed = []
    
            with open(conversations_file, 'rb') as f:
                parser = ijson.items(f, 'item')
    
                for conversation in parser:
                    try:
                        conv_folder, conv_title, date_info = create_conversation_structure(conversation, conversations_folder)
                        save_conversation(conversation, conv_folder, conv_title, date_info, keyword_extractor, tag_analyzer)
                        count += 1
    
                        # Read metadata to populate database
                        metadata_file = conv_folder / 'metadata.json'
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as mf:
                                metadata = json.load(mf)
    
                                if metadata.get('has_markdown_content'):
                                    markdown_count += 1
    
                                # Insert conversation into database
                                relative_path = str(conv_folder.relative_to(output_base / 'conversations'))
                                conv_id = db.add_conversation(
                                    uuid=metadata['uuid'],
                                    name=metadata['name'],
                                    created_at=metadata['created_at'],
                                    relative_path=relative_path,
                                    source='claude',
                                    updated_at=metadata.get('updated_at'),
                                    message_count=metadata['message_count'],
                                    has_markdown=metadata.get('has_markdown_content', False)
                                )
    
                                # Add keywords
                                keywords = metadata.get('keywords', [])
                                if keywords:
                                    keyword_tuples = [(kw, 1.0) for kw in keywords]
                                    db.add_keywords(conv_id, keyword_tuples)
    
                                # Add messages to database
                                messages_dir = conv_folder / 'messages'
                                if messages_dir.exists():
                                    # Read original conversation data for message content
                                    for idx, msg in enumerate(conversation.get('chat_messages', [])):
                                        sender = msg.get('sender', 'unknown')
                                        content_parts = []
                                        for content in msg.get('content', []):
                                            if isinstance(content, dict) and content.get('type') == 'text':
                                                content_parts.append(content.get('text', ''))
    
                                        full_content = '\n'.join(content_parts)
                                        has_code = '```' in full_content
    
                                        db.add_message(
                                            conversation_id=conv_id,
                                            sender=sender,
                                            content=full_content,
                                            index_in_conversation=idx,
                                            message_uuid=msg.get('uuid'),
                                            created_at=msg.get('created_at'),
                                            has_code=has_code
                                        )
    
                                # Collect for embedding generation
                                if generate_embeddings:
                                    first_message = None
                                    if conversation.get('chat_messages'):
                                        first_msg = conversation['chat_messages'][0]
                                        content_parts = []
                                        for content in first_msg.get('content', []):
                                            if isinstance(content, dict) and content.get('type') == 'text':
                                                content_parts.append(content.get('text', ''))
                                        first_message = '\n'.join(content_parts)[:500]
    
                                    embed_text = generate_conversation_embedding(
                                        title=metadata['name'],
                                        keywords=keywords,
                                        first_message=first_message
                                    )
                                    conversations_to_embed.append((conv_id, embed_text))
    
                        if count % 100 == 0:
                            print(f"  Processed {count} conversations ({markdown_count} with markdown)...", end='\r')
    
                    except Exception as e:
                        print(f"\n⚠️  Error processing conversation {conversation.get('uuid', 'unknown')}: {e}")
    
            print(f"  Converted {count} conversations total ({markdown_count} with markdown content)    ")
    
            # Generate embeddings in batch
            if generate_embeddings and conversations_to_embed:
                print(f"\n🔮 Generating embeddings for {len(conversations_to_embed)} conversations...")
                texts = [text for _, text in conversations_to_embed]
                embeddings = embedding_generator.generate_batch(texts, task_type='search_document')
    
                for (conv_id, _), embedding in zip(conversations_to_embed, embeddings):
                    db.add_embedding(conv_id, embedding, 'nomic-embed-text-v1.5')
    
                print("✅ Embeddings generated and stored")
    
        # Convert projects
        projects_file = input_dir / 'projects.json'
        if projects_file.exists():
            convert_projects(str(projects_file), output_base, keyword_extractor, tag_analyzer)
    
            # Also add projects to database
            print("📊 Adding projects to database...")
            projects_folder = output_base / 'projects'
            if projects_folder.exists():
                for project_dir in projects_folder.iterdir():
                    if not project_dir.is_dir() or project_dir.name.startswith('.'):
                        continue
    
                    metadata_file = project_dir / 'metadata.json'
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            relative_path = str(project_dir.relative_to(output_base / 'projects'))
                            db.add_project(
                                uuid=metadata['uuid'],
                                name=metadata['name'],
                                created_at=metadata['created_at'],
                                relative_path=relative_path,
                                description=metadata.get('description')
                            )
    
        # Show database statistics
        stats = db.get_statistics()
        print(f"\n📊 Database populated:")
        print(f"   - {stats['total_conversations']} conversations")
        print(f"   - {stats['total_messages']} messages")
        print(f"   - {stats['total_keywords']} unique keywords")
        if generate_embeddings:
            print(f"   - {stats['conversations_with_embeddings']} conversations with embeddings")
        if stats['total_projects'] > 0:
            print(f"   - {stats['total_projects']} projects")
    
        # Create summary
        summary = {
            'created_at': datetime.now().isoformat(),
            'source_files': {
                'conversations.json': conversations_file.exists(),
                'projects.json': projects_file.exists(),
                'users.json': users_file.exists()
            },
            'output_structure': {
                'conversations': 'conversations/{year}/{month}/{day}/{conversation_name}/',
                'projects': 'projects/{project_name}/',
                'database': 'conversations.db',
                'markdown_extraction': True,
                'code_snippet_extraction': True,
                'keyword_extraction': True,
                'embeddings_generated': generate_embeddings
            },
            'features': [
                'Enhanced markdown titles with full conversation context',
                'Automatic keyword extraction and hashtag generation',
                'Human-readable titles throughout',
                'Markdown files saved as .md with proper headers',
                'Code blocks extracted to separate files',
                'Keywords indexed for discovery and search',
                'SQLite database with full-text search',
                'Semantic embeddings for similarity search' if generate_embeddings else None
            ],
            'statistics': stats
        }
    
        with open(output_base / 'conversion_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
        # Tag analysis and Obsidian config
        if not skip_tags:
            print("\n" + "="*60)
            print("Generating Obsidian graph configuration...")
            print("="*60)
    
            # Scan all markdown files for complete tag analysis
            tag_analyzer.scan_markdown_files_for_tags(output_base)
    
            # Interactive water level adjustment for both tags and file patterns
            tag_water_level, file_water_level, tag_color_scheme, file_color_scheme = tag_analyzer.interactive_water_level_adjustment()
    
            # Generate Obsidian config files with dual grouping
            print(f"\nCreating Obsidian configuration with dual-layer grouping...")
            print(f"Using {tag_color_scheme} colors for tags and {file_color_scheme} colors for file patterns")
            tag_analyzer.create_obsidian_config(output_base, tag_water_level, file_water_level,
                                              tag_color_scheme, file_color_scheme)
    
            # Save analysis report
            report_file = tag_analyzer.save_analysis_report(output_base, tag_water_level, file_water_level,
                                                           tag_color_scheme, file_color_scheme)
            print(f"Tag analysis report saved to: {report_file}")
        else:
            # Create basic Obsidian config without interactive setup
            print("\n📝 Creating basic Obsidian configuration...")
            tag_analyzer.create_obsidian_config(output_base, 30, 30, 'rainbow', 'ocean')
    
        print("\n" + "="*60)
        print("✅ CONVERSION COMPLETE!")
        print("="*60)
        print(f"Your knowledge base is ready in: {output_base}")
        print(f"\nDatabase: {db_path}")
        print(f"  - Full-text search enabled")
        print(f"  - {stats['total_conversations']} conversations indexed")
        if generate_embeddings:
            print(f"  - Semantic search ready")
        print("\nTo search your conversations:")
        print(f"  python src/search_chats.py {output_base} \"your query\"")
        print("\nTo use with Obsidian:")
        print("1. Open Obsidian")
        print("2. Open this folder as a vault")
        print("3. Open Graph View to see your color-coded knowledge network!")

        db.close()
        return True

    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Get output directory from command line argument or use default
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = 'claude_history'
    
    output_base = Path(output_dir)
    output_base.mkdir(exist_ok=True)
    
    # Initialize keyword extractor and tag analyzer
    keyword_extractor = KeywordExtractor()
    tag_analyzer = TagAnalyzer()
    
    # Define input directory (relative to src/)
    input_dir = Path('../input')
    
    # Save user info
    users_file = input_dir / 'users.json'
    if users_file.exists():
        print("\nCopying users.json...")
        with open(users_file, 'r') as f:
            users = json.load(f)
        with open(output_base / 'users.json', 'w') as f:
            json.dump(users, f, indent=2)
    
    # Convert conversations
    conversations_file = input_dir / 'conversations.json'
    if conversations_file.exists():
        convert_conversations(str(conversations_file), output_base, keyword_extractor, tag_analyzer)
    
    # Convert projects  
    projects_file = input_dir / 'projects.json'
    if projects_file.exists():
        convert_projects(str(projects_file), output_base, keyword_extractor, tag_analyzer)
    
    # Create summary
    summary = {
        'created_at': datetime.now().isoformat(),
        'source_files': {
            'conversations.json': conversations_file.exists(),
            'projects.json': projects_file.exists(),
            'users.json': users_file.exists()
        },
        'output_structure': {
            'conversations': 'claude_history_enhanced/conversations/{year}/{month}/{day}/{conversation_name}/',
            'projects': 'claude_history_enhanced/projects/{project_name}/',
            'indexes': ['conversations_index.json', 'projects_index.json'],
            'markdown_extraction': True,
            'code_snippet_extraction': True,
            'keyword_extraction': True
        },
        'features': [
            'Enhanced markdown titles with full conversation context',
            'Automatic keyword extraction and hashtag generation',
            'Human-readable titles throughout',
            'Markdown files saved as .md with proper headers',
            'Code blocks extracted to separate files',
            'Keywords indexed for discovery and search'
        ]
    }
    
    with open(output_base / 'conversion_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Output directory: {output_base}")
    print(f"\nStructure:")
    print(f"  - conversations/ (organized by date)")
    print(f"  - projects/ (organized by name)")
    print(f"  - conversations_index.json (for quick lookups)")
    print(f"  - projects_index.json (for quick lookups)")
    print(f"  - users.json (user information)")
    print(f"  - conversion_summary.json (this conversion info)")
    print(f"\nEnhanced Features:")
    print(f"  - Full conversation context in markdown titles")
    print(f"  - Automatic keyword extraction with hashtags")
    print(f"  - Human-readable titles throughout")
    print(f"  - Code blocks saved as separate files")
    print(f"  - Keywords indexed for search and discovery")
    
    # Run tag analysis and generate Obsidian config
    print("\n" + "="*60)
    print("Generating Obsidian graph configuration...")
    print("="*60)
    
    # Scan all markdown files for complete tag analysis
    tag_analyzer.scan_markdown_files_for_tags(output_base)
    
    # Interactive water level adjustment for both tags and file patterns
    tag_water_level, file_water_level, tag_color_scheme, file_color_scheme = tag_analyzer.interactive_water_level_adjustment()
    
    # Generate Obsidian config files with dual grouping
    print("\nCreating Obsidian configuration with dual-layer grouping...")
    print(f"Using {tag_color_scheme} colors for tags and {file_color_scheme} colors for file patterns")
    tag_analyzer.create_obsidian_config(output_base, tag_water_level, file_water_level, 
                                      tag_color_scheme, file_color_scheme)
    
    # Save analysis report
    report_file = tag_analyzer.save_analysis_report(output_base, tag_water_level, file_water_level,
                                                   tag_color_scheme, file_color_scheme)
    print(f"Tag analysis report saved to: {report_file}")
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETE!")
    print("="*60)
    print(f"Your knowledge base is ready in: {output_base}")
    print("\nTo use with Obsidian:")
    print("1. Open Obsidian")
    print("2. Create new vault or open existing vault")
    print("3. Copy contents of output folder to your vault")
    print("4. Open Graph View to see your color-coded knowledge network!")

if __name__ == "__main__":
    main()