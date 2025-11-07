"""
Advanced Hierarchical Document Chunking System

Implements sophisticated document chunking strategies for manufacturing knowledge base.
Supports text, tables, images, and mixed content with context preservation.
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import tiktoken
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

class ChunkStrategy(str, Enum):
    """Chunking strategies for different content types"""
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    FIXED_SIZE = "fixed_size"
    HIERARCHICAL = "hierarchical"
    SLIDING_WINDOW = "sliding_window"
    MIXED = "mixed"

class ContentType(str, Enum):
    """Content types for specialized processing"""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    CODE = "code"
    LIST = "list"
    HEADING = "heading"
    MIXED = "mixed"

@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata"""
    content: str
    chunk_id: str
    doc_id: str
    chunk_type: ContentType
    chunk_strategy: ChunkStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Hierarchical information
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    level: int = 0
    section_path: List[str] = field(default_factory=list)

    # Content analysis
    token_count: int = 0
    char_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0

    # Quality metrics
    coherence_score: float = 0.0
    semantic_similarity: float = 0.0
    overlap_ratio: float = 0.0

    # Embeddings
    embedding: Optional[np.ndarray] = None
    multimodal_embedding: Optional[np.ndarray] = None

@dataclass
class ChunkingConfig:
    """Configuration for document chunking"""

    # Basic parameters
    max_tokens: int = 512
    min_tokens: int = 50
    overlap_tokens: int = 50
    overlap_ratio: float = 0.1

    # Strategy-specific parameters
    chunk_strategy: ChunkStrategy = ChunkStrategy.HIERARCHICAL

    # Hierarchical chunking
    max_section_depth: int = 5
    preserve_structure: bool = True
    section_headers: List[str] = field(default_factory=lambda: [
        "h1", "h2", "h3", "h4", "h5", "h6"
    ])

    # Semantic chunking
    semantic_threshold: float = 0.7
    similarity_model: str = "all-MiniLM-L6-v2"

    # Table processing
    max_table_rows: int = 100
    max_table_cols: int = 20
    table_chunk_strategy: str = "row_based"  # "row_based", "column_based", "cell_based"

    # Code processing
    preserve_code_blocks: bool = True
    max_code_block_lines: int = 50

    # Language-specific
    language: str = "en"  # "en", "zh", "auto"

    # Quality control
    min_chunk_quality: float = 0.5
    filter_empty_chunks: bool = True
    normalize_whitespace: bool = True

class HierarchicalDocumentChunker:
    """
    Advanced document chunker with hierarchical structure preservation
    and multi-modal content handling for manufacturing knowledge base.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Initialize sentence transformer for semantic chunking
        if self.config.chunk_strategy in [ChunkStrategy.SEMANTIC, ChunkStrategy.MIXED]:
            self.semantic_model = SentenceTransformer(self.config.similarity_model)
        else:
            self.semantic_model = None

        # Manufacturing-specific patterns
        self.manufacturing_patterns = {
            "specifications": r"(?:规格|specification|spec|参数|parameter)[:：]\s*",
            "materials": r"(?:材料|material|材质)[:：]\s*",
            "dimensions": r"(?:尺寸|dimension|大小|size)[:：]\s*",
            "tolerances": r"(?:公差|tolerance|精度)[:：]\s*",
            "process": r"(?:工艺|process|procedure|流程)[:：]\s*",
            "quality": r"(?:质量|quality|检验|inspection)[:：]\s*",
        }

        # Section headers for different languages
        self.section_patterns = {
            "en": r"^(#{1,6})\s+(.+)$",
            "zh": r"^(#{1,6})\s+(.+)$|^(第[一二三四五六七八九十\d]+章|第[一二三四五六七八九十\d]+节|第[一二三四五六七八九十\d]+部分)\s*(.+)$"
        }

    def chunk_document(self,
                      doc_id: str,
                      content: str,
                      content_type: ContentType = ContentType.MIXED,
                      metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Main method to chunk a document using the configured strategy.

        Args:
            doc_id: Document identifier
            content: Document content
            content_type: Type of content
            metadata: Additional metadata

        Returns:
            List of document chunks
        """
        print(f"开始分块文档 {doc_id}, 内容类型: {content_type}, 策略: {self.config.chunk_strategy}")

        # Preprocess content
        processed_content = self._preprocess_content(content)

        # Choose chunking strategy
        if self.config.chunk_strategy == ChunkStrategy.HIERARCHICAL:
            chunks = self._hierarchical_chunking(doc_id, processed_content, content_type, metadata)
        elif self.config.chunk_strategy == ChunkStrategy.SEMANTIC:
            chunks = self._semantic_chunking(doc_id, processed_content, content_type, metadata)
        elif self.config.chunk_strategy == ChunkStrategy.RECURSIVE:
            chunks = self._recursive_chunking(doc_id, processed_content, content_type, metadata)
        elif self.config.chunk_strategy == ChunkStrategy.FIXED_SIZE:
            chunks = self._fixed_size_chunking(doc_id, processed_content, content_type, metadata)
        elif self.config.chunk_strategy == ChunkStrategy.SLIDING_WINDOW:
            chunks = self._sliding_window_chunking(doc_id, processed_content, content_type, metadata)
        else:  # MIXED
            chunks = self._mixed_chunking(doc_id, processed_content, content_type, metadata)

        # Post-process chunks
        chunks = self._postprocess_chunks(chunks)

        print(f"文档 {doc_id} 分块完成，共生成 {len(chunks)} 个块")
        return chunks

    def _preprocess_content(self, content: str) -> str:
        """Preprocess content for better chunking"""

        if self.config.normalize_whitespace:
            # Normalize whitespace
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()

        # Add section markers for better structure preservation
        content = self._add_section_markers(content)

        # Process manufacturing-specific content
        content = self._process_manufacturing_content(content)

        return content

    def _add_section_markers(self, content: str) -> str:
        """Add explicit section markers for better structure detection"""

        # Detect and mark sections
        lines = content.split('\n')
        processed_lines = []

        for line in lines:
            line = line.rstrip()

            # Detect headings
            if re.match(r'^#{1,6}\s+', line):
                processed_lines.append(f"\n{line}\n")
            # Detect numbered sections (Chinese)
            elif re.match(r'^第[一二三四五六七八九十\d]+[章节部分]', line):
                processed_lines.append(f"\n{line}\n")
            # Detect manufacturing sections
            elif any(re.search(pattern, line, re.IGNORECASE) for pattern in self.manufacturing_patterns.values()):
                processed_lines.append(f"\n## {line}\n")
            else:
                processed_lines.append(line)

        return '\n'.join(processed_lines)

    def _process_manufacturing_content(self, content: str) -> str:
        """Process manufacturing-specific content patterns"""

        # Add markers for manufacturing entities
        for entity_type, pattern in self.manufacturing_patterns.items():
            content = re.sub(
                pattern,
                f"[{entity_type.upper()}]\\g<0>",
                content,
                flags=re.IGNORECASE
            )

        return content

    def _hierarchical_chunking(self,
                               doc_id: str,
                               content: str,
                               content_type: ContentType,
                               metadata: Optional[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Implement hierarchical chunking that preserves document structure.
        """

        chunks = []

        # Parse document structure
        sections = self._parse_document_structure(content)

        # Process each section
        for section in sections:
            section_chunks = self._process_section(doc_id, section, content_type, metadata)
            chunks.extend(section_chunks)

        return chunks

    def _parse_document_structure(self, content: str) -> List[Dict[str, Any]]:
        """Parse document into hierarchical sections"""

        sections = []
        lines = content.split('\n')
        current_section = None
        section_stack = []

        for i, line in enumerate(lines):
            line = line.strip()

            # Check for section headers
            if re.match(r'^#{1,6}\s+', line) or re.match(r'^第[一二三四五六七八九十\d]+[章节部分]', line):

                # Calculate section level
                if line.startswith('#'):
                    level = len(line) - len(line.lstrip('#'))
                    title = line.lstrip('#').strip()
                else:
                    # Chinese sections - default to level 2
                    level = 2
                    title = line

                # Create new section
                section = {
                    'title': title,
                    'level': level,
                    'start_line': i,
                    'content_lines': [],
                    'subsections': [],
                    'metadata': {
                        'section_type': 'heading',
                        'hierarchy_level': level
                    }
                }

                # Handle section hierarchy
                while section_stack and section_stack[-1]['level'] >= level:
                    section_stack.pop()

                if section_stack:
                    section_stack[-1]['subsections'].append(section)
                    section['parent'] = section_stack[-1]
                else:
                    sections.append(section)

                section_stack.append(section)
                current_section = section

            elif current_section:
                current_section['content_lines'].append(line)

        # Convert section structure to chunk-friendly format
        processed_sections = []
        for section in sections:
            processed_sections.extend(self._flatten_section(section))

        return processed_sections

    def _flatten_section(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten hierarchical section structure for chunking"""

        flattened = []

        # Add current section
        section_content = '\n'.join(section['content_lines'])
        if section_content.strip():
            flattened.append({
                'title': section['title'],
                'level': section['level'],
                'content': section_content,
                'metadata': section.get('metadata', {}),
                'section_path': self._get_section_path(section)
            })

        # Recursively add subsections
        for subsection in section.get('subsections', []):
            flattened.extend(self._flatten_section(subsection))

        return flattened

    def _get_section_path(self, section: Dict[str, Any]) -> List[str]:
        """Get the hierarchical path for a section"""

        path = []
        current = section

        while current:
            path.append(current['title'])
            current = current.get('parent')

        return list(reversed(path))

    def _process_section(self,
                         doc_id: str,
                         section: Dict[str, Any],
                         content_type: ContentType,
                         metadata: Optional[Dict[str, Any]]) -> List[DocumentChunk]:
        """Process a single section into chunks"""

        content = section['content']
        title = section['title']
        level = section['level']
        section_path = section['section_path']

        # Create section metadata
        section_metadata = {
            **(metadata or {}),
            'section_title': title,
            'section_level': level,
            'section_path': section_path,
            'content_type': content_type.value
        }

        # Add title to content if not present
        if not content.startswith(title):
            content = f"{title}\n\n{content}"

        # Token-based chunking within section
        section_chunks = self._token_based_chunking(doc_id, content, section_metadata)

        # Set hierarchical relationships
        for i, chunk in enumerate(section_chunks):
            chunk.level = level
            chunk.section_path = section_path

            # Set parent-child relationships
            if i > 0:
                chunk.parent_chunk_id = section_chunks[i-1].chunk_id
                section_chunks[i-1].child_chunk_ids.append(chunk.chunk_id)

        return section_chunks

    def _token_based_chunking(self,
                             doc_id: str,
                             content: str,
                             metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split content based on token limits with overlap"""

        chunks = []
        sentences = self._split_into_sentences(content)

        current_chunk = ""
        chunk_index = 0

        for sentence in sentences:
            test_chunk = current_chunk + (" " if current_chunk else "") + sentence
            token_count = len(self.tokenizer.encode(test_chunk))

            if token_count <= self.config.max_tokens:
                current_chunk = test_chunk
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunk = self._create_chunk(
                        doc_id=doc_id,
                        content=current_chunk,
                        chunk_index=chunk_index,
                        metadata=metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk
                current_chunk = sentence

        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                doc_id=doc_id,
                content=current_chunk,
                chunk_index=chunk_index,
                metadata=metadata
            )
            chunks.append(chunk)

        # Add overlap
        chunks = self._add_overlap(chunks)

        return chunks

    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences with manufacturing-aware logic"""

        # Manufacturing-specific sentence patterns
        manufacturing_endings = [
            r'[.!?]+(?=\s|$)',  # Standard sentence endings
            r'[:：](?=\s|$)',   # Colon endings (common in specs)
            r'[;；](?=\s|$)',   # Semicolon endings
            r'[)]\s*[:：]',    # Parentheses followed by colon
        ]

        # Split using multiple patterns
        sentences = []
        for pattern in manufacturing_endings:
            content = re.sub(pattern, '<SPLIT>' + r'\g<0>', content)

        raw_sentences = content.split('<SPLIT>')

        # Clean and filter sentences
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 3:  # Filter very short fragments
                sentences.append(sentence)

        return sentences

    def _add_overlap(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Add overlapping content between chunks"""

        if not chunks or self.config.overlap_tokens == 0:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            # Add overlap from previous chunk
            if i > 0:
                prev_chunk = chunks[i-1]
                prev_sentences = prev_chunk.content.split('.')

                # Add last few sentences from previous chunk
                overlap_sentences = prev_sentences[-2:] if len(prev_sentences) >= 2 else prev_sentences[-1:]
                overlap_content = '. '.join(overlap_sentences).strip()

                if overlap_content and not overlap_content.endswith('.'):
                    overlap_content += '.'

                enhanced_content = overlap_content + "\n\n" + chunk.content
                chunk.content = enhanced_content

            overlapped_chunks.append(chunk)

        return overlapped_chunks

    def _semantic_chunking(self,
                           doc_id: str,
                           content: str,
                           content_type: ContentType,
                           metadata: Optional[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Implement semantic chunking based on content similarity.
        """

        if not self.semantic_model:
            # Fallback to hierarchical chunking
            return self._hierarchical_chunking(doc_id, content, content_type, metadata)

        sentences = self._split_into_sentences(content)
        if len(sentences) == 0:
            return []

        # Generate embeddings for sentences
        sentence_embeddings = self.semantic_model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Group sentences based on semantic similarity
        chunks = self._group_by_semantic_similarity(
            doc_id, sentences, sentence_embeddings, metadata
        )

        return chunks

    def _group_by_semantic_similarity(self,
                                     doc_id: str,
                                     sentences: List[str],
                                     embeddings: np.ndarray,
                                     metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Group sentences based on semantic similarity"""

        chunks = []
        current_group = [0]
        current_length = len(self.tokenizer.encode(sentences[0]))

        for i in range(1, len(sentences)):
            # Calculate similarity with last sentence in current group
            last_idx = current_group[-1]
            similarity = np.dot(embeddings[last_idx], embeddings[i])

            # Calculate new length if we add this sentence
            new_length = current_length + len(self.tokenizer.encode(sentences[i])) + 1

            # Decide whether to start new chunk
            if (similarity < self.config.semantic_threshold or
                new_length > self.config.max_tokens):

                # Create chunk from current group
                chunk_content = ' '.join(sentences[idx] for idx in current_group)
                chunk = self._create_chunk(
                    doc_id=doc_id,
                    content=chunk_content,
                    chunk_index=len(chunks),
                    metadata={**metadata, 'similarity_threshold': self.config.semantic_threshold}
                )
                chunks.append(chunk)

                # Start new group
                current_group = [i]
                current_length = len(self.tokenizer.encode(sentences[i]))
            else:
                current_group.append(i)
                current_length = new_length

        # Add final group
        if current_group:
            chunk_content = ' '.join(sentences[idx] for idx in current_group)
            chunk = self._create_chunk(
                doc_id=doc_id,
                content=chunk_content,
                chunk_index=len(chunks),
                metadata=metadata
            )
            chunks.append(chunk)

        return chunks

    def _recursive_chunking(self,
                           doc_id: str,
                           content: str,
                           content_type: ContentType,
                           metadata: Optional[Dict[str, Any]]) -> List[DocumentChunk]:
        """Implement recursive character-based chunking"""

        separators = ["\n\n", "\n", ". ", " ", ""]
        chunks = []

        def _recursive_split(text: str, separators: List[str]) -> List[str]:
            if len(text) <= self.config.max_tokens * 4:  # Approximate token to char ratio
                return [text]

            for sep in separators:
                if sep not in text:
                    continue

                parts = text.split(sep)
                if len(parts) == 1:
                    continue

                # Try to split with current separator
                valid_chunks = []
                for part in parts:
                    sub_chunks = _recursive_split(part, separators[separators.index(sep) + 1:])
                    valid_chunks.extend(sub_chunks)

                return valid_chunks

            # If no separator worked, split by character count
            return [text[i:i+self.config.max_tokens*4]
                   for i in range(0, len(text), self.config.max_tokens*4)]

        raw_chunks = _recursive_split(content, separators)

        # Create DocumentChunk objects
        for i, chunk_content in enumerate(raw_chunks):
            if len(chunk_content.strip()) > 0:
                chunk = self._create_chunk(
                    doc_id=doc_id,
                    content=chunk_content.strip(),
                    chunk_index=i,
                    metadata={**metadata, 'chunking_method': 'recursive'}
                )
                chunks.append(chunk)

        return chunks

    def _fixed_size_chunking(self,
                            doc_id: str,
                            content: str,
                            content_type: ContentType,
                            metadata: Optional[Dict[str, Any]]) -> List[DocumentChunk]:
        """Implement fixed-size chunking with overlap"""

        chunks = []
        tokens = self.tokenizer.encode(content)

        start_idx = 0
        chunk_index = 0

        while start_idx < len(tokens):
            # Calculate end index
            end_idx = min(start_idx + self.config.max_tokens, len(tokens))

            # Decode chunk content
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_content = self.tokenizer.decode(chunk_tokens)

            # Create chunk
            chunk = self._create_chunk(
                doc_id=doc_id,
                content=chunk_content,
                chunk_index=chunk_index,
                metadata={**metadata, 'chunking_method': 'fixed_size'}
            )
            chunks.append(chunk)

            # Move start index with overlap
            start_idx = end_idx - self.config.overlap_tokens
            chunk_index += 1

        return chunks

    def _sliding_window_chunking(self,
                                doc_id: str,
                                content: str,
                                content_type: ContentType,
                                metadata: Optional[Dict[str, Any]]) -> List[DocumentChunk]:
        """Implement sliding window chunking"""

        tokens = self.tokenizer.encode(content)
        chunks = []

        window_size = self.config.max_tokens
        step_size = max(1, window_size - self.config.overlap_tokens)

        for start_idx in range(0, len(tokens), step_size):
            end_idx = min(start_idx + window_size, len(tokens))

            if start_idx >= len(tokens):
                break

            chunk_tokens = tokens[start_idx:end_idx]
            chunk_content = self.tokenizer.decode(chunk_tokens)

            chunk = self._create_chunk(
                doc_id=doc_id,
                content=chunk_content,
                chunk_index=len(chunks),
                metadata={**metadata, 'chunking_method': 'sliding_window'}
            )
            chunks.append(chunk)

        return chunks

    def _mixed_chunking(self,
                       doc_id: str,
                       content: str,
                       content_type: ContentType,
                       metadata: Optional[Dict[str, Any]]) -> List[DocumentChunk]:
        """Implement mixed strategy chunking"""

        # First, try hierarchical chunking
        hierarchical_chunks = self._hierarchical_chunking(doc_id, content, content_type, metadata)

        # Check if any chunks are too large
        oversized_chunks = [c for c in hierarchical_chunks if c.token_count > self.config.max_tokens]

        if not oversized_chunks:
            return hierarchical_chunks

        # Process oversized chunks with fixed-size strategy
        final_chunks = []
        for chunk in hierarchical_chunks:
            if chunk.token_count <= self.config.max_tokens:
                final_chunks.append(chunk)
            else:
                # Rechunk oversized content
                sub_chunks = self._fixed_size_chunking(
                    doc_id, chunk.content, content_type,
                    {**chunk.metadata, 'parent_chunk_id': chunk.chunk_id}
                )
                final_chunks.extend(sub_chunks)

        return final_chunks

    def _create_chunk(self,
                     doc_id: str,
                     content: str,
                     chunk_index: int,
                     metadata: Dict[str, Any]) -> DocumentChunk:
        """Create a DocumentChunk object with calculated metrics"""

        # Calculate basic metrics
        tokens = self.tokenizer.encode(content)
        token_count = len(tokens)
        char_count = len(content)
        sentence_count = len([s for s in content.split('.') if s.strip()])
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])

        # Generate chunk ID
        chunk_id = f"{doc_id}_chunk_{chunk_index:04d}"

        # Determine content type
        chunk_type = self._detect_content_type(content)

        # Calculate quality metrics
        coherence_score = self._calculate_coherence_score(content)

        chunk = DocumentChunk(
            content=content,
            chunk_id=chunk_id,
            doc_id=doc_id,
            chunk_type=chunk_type,
            chunk_strategy=self.config.chunk_strategy,
            metadata=metadata,
            token_count=token_count,
            char_count=char_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            coherence_score=coherence_score
        )

        return chunk

    def _detect_content_type(self, content: str) -> ContentType:
        """Detect the type of content in a chunk"""

        content_lower = content.lower()

        # Check for code blocks
        if re.search(r'```[\w]*\n.*?```', content, re.DOTALL):
            return ContentType.CODE

        # Check for tables
        if re.search(r'\|.*\|.*\|', content) and content.count('|') >= 6:
            return ContentType.TABLE

        # Check for lists
        if re.search(r'^[\s]*[-*+]\s+', content, re.MULTILINE):
            return ContentType.LIST

        # Check for headings
        if re.search(r'^#{1,6}\s+', content, re.MULTILINE):
            return ContentType.HEADING

        # Check for manufacturing specifications
        if any(re.search(pattern, content, re.IGNORECASE) for pattern in self.manufacturing_patterns.values()):
            return ContentType.TEXT

        return ContentType.TEXT

    def _calculate_coherence_score(self, content: str) -> float:
        """Calculate coherence score for a chunk"""

        # Basic coherence metrics
        sentences = [s.strip() for s in content.split('.') if s.strip()]

        if not sentences:
            return 0.0

        # Length score (prefer chunks with reasonable length)
        length_score = min(1.0, len(sentences) / 5.0)

        # Sentence length variation (prefer varied sentence lengths)
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) > 1:
            length_variation = np.std(sentence_lengths) / np.mean(sentence_lengths)
            variation_score = min(1.0, length_variation / 2.0)
        else:
            variation_score = 0.0

        # Manufacturing-specific indicators
        manufacturing_terms = len([
            term for pattern in self.manufacturing_patterns.values()
            for term in re.findall(pattern, content, re.IGNORECASE)
        ])
        manufacturing_score = min(1.0, manufacturing_terms / len(sentences))

        # Combine scores
        coherence_score = (length_score * 0.4 +
                          variation_score * 0.3 +
                          manufacturing_score * 0.3)

        return coherence_score

    def _postprocess_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Post-process chunks to ensure quality and consistency"""

        processed_chunks = []

        for chunk in chunks:
            # Filter empty chunks
            if self.config.filter_empty_chunks and not chunk.content.strip():
                continue

            # Filter chunks below minimum token count
            if chunk.token_count < self.config.min_tokens:
                continue

            # Filter chunks below quality threshold
            if chunk.coherence_score < self.config.min_chunk_quality:
                continue

            # Enhance metadata
            chunk.metadata['processing_timestamp'] = pd.Timestamp.now().isoformat()
            chunk.metadata['chunk_quality'] = chunk.coherence_score

            processed_chunks.append(chunk)

        return processed_chunks

# Factory function for easy instantiation
def create_chunker(strategy: ChunkStrategy = ChunkStrategy.HIERARCHICAL,
                    max_tokens: int = 512,
                    **kwargs) -> HierarchicalDocumentChunker:
    """Create a document chunker with specified strategy"""

    config = ChunkingConfig(
        chunk_strategy=strategy,
        max_tokens=max_tokens,
        **kwargs
    )

    return HierarchicalDocumentChunker(config)

# Example usage and testing
if __name__ == "__main__":
    # Test the chunker with manufacturing content
    manufacturing_content = """
    # 制造规格说明书

    ## 1. 产品规格

    ### 1.1 尺寸规格
    - 长度: 100mm ± 0.1mm
    - 宽度: 50mm ± 0.05mm
    - 高度: 25mm ± 0.02mm

    ### 1.2 材料规格
    - 主要材料: 不锈钢 304
    - 硬度: HRC 45-50
    - 表面处理: 镀铬 0.01mm

    ## 2. 工艺要求

    ### 2.1 加工工艺
    1. 粗加工: 留余量 0.5mm
    2. 精加工: 达到图纸要求
    3. 检验: 全检

    ### 2.2 质量控制
    - 首件检验
    - 过程检验
    - 最终检验
    """

    # Create chunker
    chunker = create_chunker(
        strategy=ChunkStrategy.HIERARCHICAL,
        max_tokens=256,
        preserve_structure=True
    )

    # Chunk the document
    chunks = chunker.chunk_document(
        doc_id="test_manual",
        content=manufacturing_content,
        content_type=ContentType.MIXED,
        metadata={"document_type": "specification", "language": "zh"}
    )

    # Print results
    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Type: {chunk.chunk_type}")
        print(f"Level: {chunk.level}")
        print(f"Tokens: {chunk.token_count}")
        print(f"Coherence: {chunk.coherence_score:.2f}")
        print(f"Section Path: {' > '.join(chunk.section_path)}")
        print(f"Content: {chunk.content[:100]}...")

    print(f"\n分块测试完成！")