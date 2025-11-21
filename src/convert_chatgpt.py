#!/usr/bin/env python3
"""
ChatGPT conversation converter that outputs the same format as Claude converter.
"""

import json
import os
import sys
import shutil
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

# Import shared components from Claude converter
from convert_enhanced import KeywordExtractor
from tag_analyzer import TagAnalyzer
from converter_base import (
    DecimalEncoder, create_conversation_structure, detect_markdown,
    extract_code_snippets, enhance_markdown_content, save_message_files
)

class ChatGPTConverter:
    """Convert ChatGPT export to Claude converter format"""
    
    def __init__(self, output_dir: Path, input_dir: Path = None):
        self.output_dir = output_dir
        self.input_dir = input_dir or Path('input')
        self.keyword_extractor = KeywordExtractor()
        self.tag_analyzer = TagAnalyzer()  # Initialize for tag analysis
        
    def parse_export(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse ChatGPT conversations.json and convert to Claude format"""
        conversations = []
        
        with open(file_path, 'rb') as file:
            parser = ijson.items(file, 'item')
            
            for conv_data in parser:
                try:
                    conversation = self._convert_conversation(conv_data)
                    if conversation:
                        conversations.append(conversation)
                except Exception as e:
                    print(f"Error parsing conversation: {e}")
                    continue
        
        return conversations
    
    def _convert_conversation(self, conv_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert ChatGPT conversation to Claude format"""
        # Extract basic metadata
        conv_id = conv_data.get('id', conv_data.get('conversation_id', ''))
        if not conv_id:
            # Generate a UUID-like ID if missing
            import uuid
            conv_id = str(uuid.uuid4())
            
        title = conv_data.get('title', 'Untitled Conversation')
        
        # Convert timestamps
        created_at = datetime.fromtimestamp(float(conv_data.get('create_time', 0))).isoformat() + 'Z'
        updated_at = datetime.fromtimestamp(float(conv_data.get('update_time', 0))).isoformat() + 'Z'
        
        # Extract messages from the mapping structure
        mapping = conv_data.get('mapping', {})
        chat_messages = self._extract_messages_from_mapping(mapping)
        
        if not chat_messages:
            return None
        
        # Create Claude-compatible conversation structure
        conversation = {
            'uuid': conv_id,
            'name': title,
            'created_at': created_at,
            'updated_at': updated_at,
            'account': {
                'uuid': 'chatgpt-account'  # Placeholder since ChatGPT doesn't have account UUIDs
            },
            'chat_messages': chat_messages
        }
        
        return conversation
    
    def _extract_messages_from_mapping(self, mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract messages in Claude format from ChatGPT's mapping structure"""
        messages = []
        
        # Find the root node
        root_nodes = []
        for node_id, node in mapping.items():
            parent = node.get('parent')
            if parent is None or parent == 'client-created-root':
                root_nodes.append(node_id)
        
        # Traverse the tree from each root
        for root_id in root_nodes:
            self._traverse_message_tree(mapping, root_id, messages)
        
        # Sort messages by timestamp
        messages.sort(key=lambda m: m.get('created_at', ''))
        
        return messages
    
    def _traverse_message_tree(self, mapping: Dict[str, Any], node_id: str, 
                               messages: List[Dict[str, Any]], visited: set = None):
        """Recursively traverse the message tree"""
        if visited is None:
            visited = set()
        
        if node_id in visited or node_id not in mapping:
            return
        
        visited.add(node_id)
        node = mapping[node_id]
        
        # Extract message if present
        if 'message' in node and node['message']:
            msg_data = node['message']
            message = self._parse_message(msg_data)
            if message and message.get('text'):  # Only add non-empty messages
                messages.append(message)
        
        # Traverse children
        children = node.get('children', [])
        for child_id in children:
            self._traverse_message_tree(mapping, child_id, messages, visited)
    
    def _parse_message(self, msg_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a single message to Claude format"""
        # Extract author info
        author = msg_data.get('author', {})
        role = author.get('role', 'unknown')
        
        # Skip system messages that are visually hidden
        metadata = msg_data.get('metadata', {})
        if metadata.get('is_visually_hidden_from_conversation'):
            return None
        
        # Map ChatGPT roles to Claude roles
        sender_map = {
            'user': 'human',
            'assistant': 'assistant',
            'system': 'system',
            'tool': 'assistant'
        }
        sender = sender_map.get(role, role)
        
        # Extract content
        content_data = msg_data.get('content', {})
        content_parts = content_data.get('parts', [])
        text = '\n\n'.join(str(part) for part in content_parts if part)
        
        # Extract timestamp
        create_time = msg_data.get('create_time')
        if create_time:
            timestamp = datetime.fromtimestamp(float(create_time)).isoformat() + 'Z'
        else:
            timestamp = datetime.now().isoformat() + 'Z'
        
        # Build Claude-compatible message
        message = {
            'uuid': msg_data.get('id', ''),
            'sender': sender,
            'created_at': timestamp,
            'updated_at': timestamp,
            'text': text,
            'files': [],
            'content': []
        }
        
        # Handle attachments if any
        if 'attachments' in metadata:
            for att in metadata['attachments']:
                file_info = {
                    'file_name': att.get('name', 'Unnamed'),
                    'file_type': att.get('mime_type', 'unknown'),
                    'file_size': att.get('size', 0),
                    'file_id': att.get('id', '')  # Store file ID for copying
                }
                message['files'].append(file_info)
        
        return message
    
    def save_conversation(self, conversation: Dict[str, Any], conv_folder: Path, 
                         conv_title: str, date_info: Dict[str, str]):
        """Save conversation using Claude converter format"""
        # Create a unique conversation tag
        conv_id = conversation['uuid'][:8] if len(conversation['uuid']) >= 8 else conversation['uuid']
        conv_tag = f"conv-{conv_title.replace(' ', '-').lower()}-{conv_id}"
        
        # Collect all text for keyword extraction
        all_text = []
        
        # Add conversation name
        if conversation.get('name'):
            all_text.append(conversation['name'])
        
        # Collect message texts
        for message in conversation.get('chat_messages', []):
            if message.get('text'):
                all_text.append(message['text'])
        
        # Extract keywords
        full_text = ' '.join(all_text)
        conversation_keywords = self.keyword_extractor.extract_keywords(full_text) if full_text else []
        
        # Update corpus statistics
        if full_text:
            self.keyword_extractor.update_corpus_stats(full_text)
        
        # Save metadata
        metadata = {
            'uuid': conversation['uuid'],
            'name': conversation.get('name', ''),
            'created_at': conversation['created_at'],
            'updated_at': conversation['updated_at'],
            'account_uuid': conversation['account']['uuid'],
            'message_count': len(conversation.get('chat_messages', [])),
            'has_markdown_content': False,
            'keywords': conversation_keywords,
            'source': 'chatgpt'  # Add source indicator
        }
        
        # Save messages
        messages_folder = conv_folder / 'messages'
        messages_folder.mkdir(exist_ok=True)
        
        markdown_files = []
        
        for idx, message in enumerate(conversation.get('chat_messages', [])):
            # Use shared save_message_files function
            result = save_message_files(
                message, idx, messages_folder, conv_folder,
                conv_title, date_info, conv_tag, conversation_keywords,
                platform='ChatGPT'
            )
            
            if result['has_markdown']:
                metadata['has_markdown_content'] = True
                if result['markdown_file']:
                    markdown_files.append(result['markdown_file'])
        
        # Update metadata with markdown files
        if markdown_files:
            metadata['markdown_files'] = markdown_files
        
        # Save conversation metadata
        metadata_path = conv_folder / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, cls=DecimalEncoder)
        
        # Update tag analyzer if available
        if self.tag_analyzer:
            self.tag_analyzer.add_tag(conv_tag, 'conversation')
            for keyword in conversation_keywords:
                self.tag_analyzer.add_tag(keyword, 'keyword')
                
        # Copy images if any exist in the conversation
        self._copy_conversation_images(conversation, conv_folder)
    
    def _copy_conversation_images(self, conversation: Dict[str, Any], conv_folder: Path):
        """Copy images referenced in the conversation to the output folder"""
        images_copied = 0
        images_folder = None
        
        for message in conversation.get('chat_messages', []):
            # Check for file references in the message text
            if message.get('text'):
                # Find file references like file-service://file-XYZ or just file-XYZ
                import re
                file_refs = re.findall(r'file-[a-zA-Z0-9\-]+', message['text'])
                
                for file_ref in file_refs:
                    # Try to find the file in various locations
                    possible_files = [
                        self.input_dir / f"{file_ref}.png",
                        self.input_dir / f"{file_ref}.jpg", 
                        self.input_dir / f"{file_ref}.webp",
                        self.input_dir / f"{file_ref}-*.png",
                        self.input_dir / f"{file_ref}-*.jpg",
                        self.input_dir / f"{file_ref}-*.webp",
                        self.input_dir / "dalle-generations" / f"{file_ref}-*.webp",
                        self.input_dir / "user-*" / f"{file_ref}-*.png"
                    ]
                    
                    file_found = None
                    for pattern in possible_files:
                        if '*' in str(pattern):
                            # Use glob for wildcard patterns
                            matches = list(self.input_dir.glob(str(pattern.relative_to(self.input_dir))))
                            if matches:
                                file_found = matches[0]
                                break
                        elif pattern.exists():
                            file_found = pattern
                            break
                    
                    if file_found:
                        # Create images folder if needed
                        if images_folder is None:
                            images_folder = conv_folder / 'images'
                            images_folder.mkdir(exist_ok=True)
                        
                        # Copy the file
                        dest_file = images_folder / file_found.name
                        shutil.copy2(file_found, dest_file)
                        images_copied += 1
                        
                        # Update the message text to use local path
                        relative_path = f"images/{file_found.name}"
                        message['text'] = message['text'].replace(f"file-service://{file_ref}", relative_path)
                        message['text'] = message['text'].replace(file_ref, relative_path)
            
            # Also check for files in the message metadata
            for file_info in message.get('files', []):
                file_id = file_info.get('file_id', '')
                if file_id:
                    # Similar search logic for metadata files
                    file_patterns = [
                        self.input_dir / f"{file_id}*",
                        self.input_dir / "dalle-generations" / f"{file_id}*",
                        self.input_dir / "user-*" / f"*{file_id}*"
                    ]
                    
                    for pattern in file_patterns:
                        matches = list(self.input_dir.glob(str(pattern.relative_to(self.input_dir))))
                        if matches:
                            if images_folder is None:
                                images_folder = conv_folder / 'images'
                                images_folder.mkdir(exist_ok=True)
                            
                            dest_file = images_folder / matches[0].name
                            shutil.copy2(matches[0], dest_file)
                            images_copied += 1
                            
                            # Update file info with local path
                            file_info['local_path'] = f"images/{matches[0].name}"
                            break
        
        if images_copied > 0:
            print(f"  Copied {images_copied} images to {conv_folder.name}/images/")
    
    def convert(self, input_file: Path):
        """Main conversion method"""
        print(f"Converting ChatGPT export: {input_file}")
        
        # Parse conversations
        conversations = self.parse_export(input_file)
        print(f"Found {len(conversations)} conversations")
        
        # Create output structure
        conv_output_dir = self.output_dir / 'conversations'
        conv_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each conversation
        conversation_index = []
        
        for conversation in conversations:
            try:
                # Get date info
                created_at = datetime.fromisoformat(conversation['created_at'].replace('Z', '+00:00'))
                date_info = {
                    'year': created_at.strftime('%Y'),
                    'month': created_at.strftime('%m'),
                    'month_name': created_at.strftime('%B'),
                    'day': created_at.strftime('%d')
                }
                
                # Create conversation folder
                conv_title = conversation['name'].replace('_', ' ')
                conv_id = conversation['uuid'][:8] if len(conversation['uuid']) >= 8 else conversation['uuid']
                conv_folder_name = f"{conv_title.replace(' ', '_')}_{conv_id}"
                
                conv_folder = create_conversation_structure(
                    conv_output_dir,
                    date_info,
                    conv_folder_name
                )
                
                # Save conversation
                self.save_conversation(conversation, conv_folder, conv_title, date_info)
                
                # Add to index
                relative_path = f"{date_info['year']}/{date_info['month']}-{date_info['month_name']}/{date_info['day']}/{conv_folder_name}"
                index_entry = {
                    'path': relative_path,
                    'uuid': conversation['uuid'],
                    'name': conversation['name'],
                    'created_at': conversation['created_at'],
                    'message_count': len(conversation.get('chat_messages', [])),
                    'has_markdown': any('markdown_file' in msg for msg in conversation.get('chat_messages', [])),
                    'keywords': self.keyword_extractor.extract_keywords(' '.join([
                        conversation.get('name', ''),
                        ' '.join(msg.get('text', '') for msg in conversation.get('chat_messages', []))
                    ])),
                    'source': 'chatgpt'
                }
                conversation_index.append(index_entry)
                
            except Exception as e:
                print(f"Error processing conversation {conversation.get('name', 'Unknown')}: {e}")
                continue
        
        # Save conversation index
        index_path = self.output_dir / 'conversations_index.json'
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_index, f, indent=2, ensure_ascii=False, cls=DecimalEncoder)
        
        # Save conversion summary
        summary = {
            'created_at': datetime.now().isoformat() + 'Z',
            'source_files': {
                'conversations.json': True
            },
            'output_structure': {
                'conversations': f"{self.output_dir}/conversations/{{year}}/{{month}}/{{day}}/{{conversation_name}}/",
                'indexes': ['conversations_index.json'],
                'markdown_extraction': True,
                'code_snippet_extraction': True,
                'keyword_extraction': True
            },
            'features': [
                'ChatGPT to Claude format conversion',
                'Enhanced markdown titles with full conversation context',
                'Automatic keyword extraction and hashtag generation',
                'Human-readable titles throughout',
                'Markdown files saved as .md with proper headers',
                'Code blocks extracted to separate files',
                'Keywords indexed for discovery and search'
            ],
            'statistics': {
                'total_conversations': len(conversations),
                'conversations_with_markdown': sum(1 for c in conversation_index if c['has_markdown']),
                'total_keywords': len(set(kw for c in conversation_index for kw in c.get('keywords', [])))
            }
        }
        
        summary_path = self.output_dir / 'conversion_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, cls=DecimalEncoder)
        
        print(f"\nConversion complete!")
        print(f"Output directory: {self.output_dir}")
        print(f"Total conversations: {len(conversations)}")
        
        # Run tag analysis and generate Obsidian config
        print("\n" + "="*60)
        print("Generating Obsidian graph configuration...")
        print("="*60)
        
        # Scan all markdown files for complete tag analysis
        self.tag_analyzer.scan_markdown_files_for_tags(self.output_dir)
        
        # Interactive water level adjustment for both tags and file patterns
        tag_water_level, file_water_level, tag_color_scheme, file_color_scheme = self.tag_analyzer.interactive_water_level_adjustment()
        
        # Generate Obsidian config files with dual grouping
        print("\nCreating Obsidian configuration with dual-layer grouping...")
        print(f"Using {tag_color_scheme} colors for tags and {file_color_scheme} colors for file patterns")
        self.tag_analyzer.create_obsidian_config(self.output_dir, tag_water_level, file_water_level, 
                                              tag_color_scheme, file_color_scheme)
        
        # Save analysis report
        report_file = self.tag_analyzer.save_analysis_report(self.output_dir, tag_water_level, file_water_level,
                                                           tag_color_scheme, file_color_scheme)
        print(f"Tag analysis report saved to: {report_file}")
        
        print("\n" + "="*60)
        print("CONVERSION COMPLETE!")


def convert_chatgpt_history(input_path: Path, output_path: Path,
                           skip_tags: bool = False,
                           generate_embeddings: bool = True) -> bool:
    """
    Convert ChatGPT chat history to searchable knowledge base.

    Args:
        input_path: Path to extracted ChatGPT export directory
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

        # Create converter and process
        converter = ChatGPTConverter(output_base, input_dir)
        conversations_file = input_dir / 'conversations.json'

        if not conversations_file.exists():
            print(f"❌ Error: conversations.json not found in {input_dir}")
            return False

        print(f"\n🔄 Converting conversations...")

        # Parse all conversations
        conversations = converter.parse_export(conversations_file)
        total_count = len(conversations)
        print(f"   Found {total_count} conversations")

        # Collect for embedding generation
        conversations_to_embed = []
        conversations_folder = output_base / 'conversations'

        # Process each conversation (reuse existing converter logic)
        for idx, conversation in enumerate(conversations):
            try:
                # Get date info
                created_at = datetime.fromisoformat(conversation['created_at'].replace('Z', '+00:00'))
                date_info = {
                    'year': created_at.strftime('%Y'),
                    'month': created_at.strftime('%m'),
                    'month_name': created_at.strftime('%B'),
                    'day': created_at.strftime('%d')
                }

                # Create conversation folder
                conv_title = conversation['name'].replace('_', ' ')
                conv_id = conversation['uuid'][:8] if len(conversation['uuid']) >= 8 else conversation['uuid']
                conv_folder_name = f"{conv_title.replace(' ', '_')}_{conv_id}"

                # Create folder path from converter_base
                from converter_base import create_conversation_structure as create_conv_struct
                conv_folder = create_conv_struct(conversations_folder, date_info, conv_folder_name)

                # Save conversation metadata and messages
                converter.save_conversation(conversation, conv_folder, conv_title, date_info)

                # Copy images if present
                converter._copy_conversation_images(conversation, conv_folder)

                # Read metadata to populate database
                metadata_file = conv_folder / 'metadata.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r') as mf:
                        metadata = json.load(mf)

                        # Insert conversation into database
                        relative_path = str(conv_folder.relative_to(output_base / 'conversations'))
                        conv_id = db.add_conversation(
                            uuid=metadata['uuid'],
                            name=metadata['name'],
                            created_at=metadata['created_at'],
                            relative_path=relative_path,
                            source='chatgpt',
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
                        # ChatGPT uses 'text' field directly, not 'content' array
                        for msg_idx, msg in enumerate(conversation.get('chat_messages', [])):
                            sender = msg.get('sender', 'unknown')
                            full_content = msg.get('text', '')  # ChatGPT stores text directly
                            has_code = '```' in full_content

                            db.add_message(
                                conversation_id=conv_id,
                                sender=sender,
                                content=full_content,
                                index_in_conversation=msg_idx,
                                message_uuid=msg.get('uuid'),
                                created_at=msg.get('created_at'),
                                has_code=has_code
                            )

                        # Collect for embedding generation
                        if generate_embeddings:
                            first_message = None
                            if conversation.get('chat_messages'):
                                first_msg = conversation['chat_messages'][0]
                                first_message = first_msg.get('text', '')[:500]  # ChatGPT uses 'text' field

                            embed_text = generate_conversation_embedding(
                                title=metadata['name'],
                                keywords=keywords,
                                first_message=first_message
                            )
                            conversations_to_embed.append((conv_id, embed_text))

                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{total_count} conversations...", end='\r')

            except Exception as e:
                print(f"\n⚠️  Error processing conversation {conversation.get('uuid', 'unknown')}: {e}")

        print(f"  Converted {total_count} conversations total                    ")

        # Generate embeddings in batch
        if generate_embeddings and conversations_to_embed:
            print(f"\n🔮 Generating embeddings for {len(conversations_to_embed)} conversations...")
            texts = [text for _, text in conversations_to_embed]
            embeddings = embedding_generator.generate_batch(texts, task_type='search_document')

            for (conv_id, _), embedding in zip(conversations_to_embed, embeddings):
                db.add_embedding(conv_id, embedding, 'nomic-embed-text-v1.5')

            print("✅ Embeddings generated and stored")

        # Show database statistics
        stats = db.get_statistics()
        print(f"\n📊 Database populated:")
        print(f"   - {stats['total_conversations']} conversations")
        print(f"   - {stats['total_messages']} messages")
        print(f"   - {stats['total_keywords']} unique keywords")
        if generate_embeddings:
            print(f"   - {stats['conversations_with_embeddings']} conversations with embeddings")

        # Create summary
        summary = {
            'created_at': datetime.now().isoformat(),
            'source_files': {
                'conversations.json': conversations_file.exists()
            },
            'output_structure': {
                'conversations': 'conversations/{year}/{month}/{day}/{conversation_name}/',
                'database': 'conversations.db',
                'markdown_extraction': True,
                'code_snippet_extraction': True,
                'keyword_extraction': True,
                'embeddings_generated': generate_embeddings
            },
            'statistics': stats
        }

        with open(output_base / 'conversion_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Tag analysis and Obsidian config
        if not skip_tags:
            print("\n" + "="*60)
            print("Generating Obsidian graph configuration...")
            print("="*60)

            tag_analyzer = TagAnalyzer()
            tag_analyzer.scan_markdown_files_for_tags(output_base)
            tag_water_level, file_water_level, tag_color_scheme, file_color_scheme = tag_analyzer.interactive_water_level_adjustment()
            tag_analyzer.create_obsidian_config(output_base, tag_water_level, file_water_level,
                                              tag_color_scheme, file_color_scheme)
            report_file = tag_analyzer.save_analysis_report(output_base, tag_water_level, file_water_level,
                                                           tag_color_scheme, file_color_scheme)
            print(f"Tag analysis report saved to: {report_file}")
        else:
            print("\n📝 Creating basic Obsidian configuration...")
            tag_analyzer = TagAnalyzer()
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
    if len(sys.argv) < 2:
        print("Usage: python convert_chatgpt.py <path_to_conversations.json> [output_dir]")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    # Set output directory
    if len(sys.argv) >= 3:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = Path('claude_history_enhanced')
    
    # Determine input directory (parent of conversations.json)
    input_dir = input_file.parent
    
    # Create converter and run
    converter = ChatGPTConverter(output_dir, input_dir)
    converter.convert(input_file)


if __name__ == "__main__":
    main()