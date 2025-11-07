#!/usr/bin/env python3
"""
RAG-Multimodal Integration Layer
Integrates the advanced RAG system with the existing multimodal document processing capabilities.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, field
import redis
import asyncpg
from pathlib import Path

# Import existing systems
from advanced_rag_system import (
    AdvancedRAGSystem, DocumentChunk, ChunkType, RetrievalStrategy,
    GeneratedAnswer, RetrievalResult, ConversationContext
)
from multimodal_document_processor import MultimodalDocumentProcessor
from multimodal_vector_index import MultimodalVectorIndex
from cross_modal_search_engine import CrossModalSearchEngine

logger = logging.getLogger(__name__)

@dataclass
class MultimodalDocumentChunk(DocumentChunk):
    """Enhanced document chunk with multimodal capabilities"""
    modality: str = "text"  # text, image, table, chart, mixed
    visual_features: Optional[Dict[str, Any]] = None
    table_data: Optional[Dict[str, Any]] = None
    chart_data: Optional[Dict[str, Any]] = None
    cross_modal_embeddings: Optional[Dict[str, np.ndarray]] = None
    extracted_entities: List[str] = field(default_factory=list)
    semantic_tags: List[str] = field(default_factory=list)

@dataclass
class MultimodalQuery:
    """Enhanced query with multimodal capabilities"""
    text_query: str
    image_query: Optional[str] = None  # Base64 encoded image
    table_query: Optional[Dict[str, Any]] = None
    modality_preference: str = "auto"  # auto, text, image, table, mixed
    cross_modal_weight: float = 0.5

class MultimodalRAGIntegrator:
    """Integration layer for RAG and multimodal processing"""

    def __init__(self,
                 redis_client: redis.Redis = None,
                 db_connection: asyncpg.Connection = None,
                 openai_api_key: str = None,
                 multimodal_processor: MultimodalDocumentProcessor = None,
                 vector_index: MultimodalVectorIndex = None,
                 cross_modal_search: CrossModalSearchEngine = None):

        self.redis_client = redis_client
        self.db_connection = db_connection

        # Initialize multimodal components
        self.multimodal_processor = multimodal_processor or MultimodalDocumentProcessor()
        self.vector_index = vector_index or MultimodalVectorIndex()
        self.cross_modal_search = cross_modal_search or CrossModalSearchEngine()

        # Initialize RAG system
        self.rag_system = AdvancedRAGSystem(
            openai_api_key=openai_api_key,
            redis_client=redis_client
        )

        # Integration cache
        self.processing_cache = {}
        self.embedding_cache = {}

        logger.info("Multimodal RAG Integration initialized")

    async def process_multimodal_document(self,
                                        file_path: str,
                                        document_id: str = None) -> Dict[str, Any]:
        """Process document with multimodal capabilities and create enhanced chunks"""

        if not document_id:
            document_id = str(uuid.uuid4())

        logger.info(f"Processing multimodal document: {file_path}")

        # Check cache first
        cache_key = f"multimodal_doc:{file_path}:{hash(Path(file_path).read_text()[:1000])}"
        if cache_key in self.processing_cache:
            logger.info(f"Using cached processing result for {file_path}")
            return self.processing_cache[cache_key]

        try:
            # Process with multimodal engine
            multimodal_result = await self.multimodal_processor.process_document(file_path)

            # Create enhanced chunks
            enhanced_chunks = await self._create_multimodal_chunks(
                multimodal_result, document_id, file_path
            )

            # Extract cross-modal features
            cross_modal_features = await self._extract_cross_modal_features(enhanced_chunks)

            # Generate multimodal embeddings
            await self._generate_multimodal_embeddings(enhanced_chunks, cross_modal_features)

            # Store in multimodal vector index
            await self._store_in_multimodal_index(enhanced_chunks, document_id)

            # Add to RAG system
            for chunk in enhanced_chunks:
                await self.rag_system.process_document(
                    chunk.content,
                    chunk.source_document,
                    chunk.page_number
                )

            # Cache result
            processing_result = {
                'document_id': document_id,
                'chunk_count': len(enhanced_chunks),
                'modalities': list(set(chunk.modality for chunk in enhanced_chunks)),
                'processing_time': datetime.now().isoformat(),
                'chunks': enhanced_chunks
            }
            self.processing_cache[cache_key] = processing_result

            logger.info(f"Successfully processed {file_path}: {len(enhanced_chunks)} chunks created")
            return processing_result

        except Exception as e:
            logger.error(f"Error processing multimodal document {file_path}: {e}")
            raise

    async def _create_multimodal_chunks(self,
                                      multimodal_result: Dict[str, Any],
                                      document_id: str,
                                      file_path: str) -> List[MultimodalDocumentChunk]:
        """Create enhanced multimodal chunks from processing results"""
        chunks = []

        # Process text content
        if 'text_content' in multimodal_result:
            text_chunks = await self._create_text_chunks(
                multimodal_result['text_content'], document_id, file_path
            )
            chunks.extend(text_chunks)

        # Process table content
        if 'tables' in multimodal_result:
            table_chunks = await self._create_table_chunks(
                multimodal_result['tables'], document_id, file_path
            )
            chunks.extend(table_chunks)

        # Process chart content
        if 'charts' in multimodal_result:
            chart_chunks = await self._create_chart_chunks(
                multimodal_result['charts'], document_id, file_path
            )
            chunks.extend(chart_chunks)

        # Process image content
        if 'images' in multimodal_result:
            image_chunks = await self._create_image_chunks(
                multimodal_result['images'], document_id, file_path
            )
            chunks.extend(image_chunks)

        # Create cross-modal chunks
        cross_modal_chunks = await self._create_cross_modal_chunks(chunks, document_id)
        chunks.extend(cross_modal_chunks)

        return chunks

    async def _create_text_chunks(self,
                                text_content: Dict[str, Any],
                                document_id: str,
                                file_path: str) -> List[MultimodalDocumentChunk]:
        """Create chunks from text content"""
        chunks = []

        if 'pages' in text_content:
            for page_num, page_content in enumerate(text_content['pages']):
                # Use RAG system's chunker for hierarchical chunking
                rag_chunks = self.rag_system.chunker.chunk_document(
                    page_content, file_path, page_num + 1
                )

                # Convert to multimodal chunks
                for rag_chunk in rag_chunks:
                    multimodal_chunk = MultimodalDocumentChunk(
                        id=rag_chunk.id,
                        content=rag_chunk.content,
                        chunk_type=rag_chunk.chunk_type,
                        level=rag_chunk.level,
                        parent_id=rag_chunk.parent_id,
                        children_ids=rag_chunk.children_ids,
                        metadata=rag_chunk.metadata,
                        embedding=rag_chunk.embedding,
                        source_document=rag_chunk.source_document,
                        page_number=rag_chunk.page_number,
                        bbox=rag_chunk.bbox,
                        tokens=rag_chunk.tokens,
                        created_at=rag_chunk.created_at,
                        modality="text",
                        semantic_tags=self._extract_semantic_tags(rag_chunk.content)
                    )
                    chunks.append(multimodal_chunk)

        return chunks

    async def _create_table_chunks(self,
                                 tables: List[Dict[str, Any]],
                                 document_id: str,
                                 file_path: str) -> List[MultimodalDocumentChunk]:
        """Create chunks from table content"""
        chunks = []

        for i, table in enumerate(tables):
            # Create table description
            table_text = f"Table {i+1}: {table.get('title', 'Untitled Table')}\n"

            if 'headers' in table and 'rows' in table:
                table_text += "Columns: " + ", ".join(table['headers']) + "\n"

                # Add first few rows as sample
                for j, row in enumerate(table['rows'][:5]):
                    row_text = " | ".join(str(cell) for cell in row)
                    table_text += f"Row {j+1}: {row_text}\n"

                if len(table['rows']) > 5:
                    table_text += f"... and {len(table['rows']) - 5} more rows\n"

            # Create chunk
            chunk = MultimodalDocumentChunk(
                id=str(uuid.uuid4()),
                content=table_text,
                chunk_type=ChunkType.TABLE,
                level=1,
                source_document=file_path,
                page_number=table.get('page_number'),
                modality="table",
                table_data=table,
                extracted_entities=self._extract_entities_from_table(table),
                semantic_tags=["tabular_data", "structured_information"]
            )

            chunks.append(chunk)

        return chunks

    async def _create_chart_chunks(self,
                                 charts: List[Dict[str, Any]],
                                 document_id: str,
                                 file_path: str) -> List[MultimodalDocumentChunk]:
        """Create chunks from chart content"""
        chunks = []

        for i, chart in enumerate(charts):
            # Create chart description
            chart_text = f"Chart {i+1}: {chart.get('title', 'Untitled Chart')}\n"
            chart_text += f"Type: {chart.get('type', 'Unknown')}\n"

            if 'data_points' in chart:
                chart_text += "Data: " + str(chart['data_points'][:10]) + "\n"

            if 'insights' in chart:
                chart_text += "Insights: " + str(chart['insights']) + "\n"

            # Create chunk
            chunk = MultimodalDocumentChunk(
                id=str(uuid.uuid4()),
                content=chart_text,
                chunk_type=ChunkType.PARAGRAPH, # Charts are treated as paragraphs for now
                level=1,
                source_document=file_path,
                page_number=chart.get('page_number'),
                modality="chart",
                chart_data=chart,
                extracted_entities=self._extract_entities_from_chart(chart),
                semantic_tags=["chart", "visualization", "data_visualization"]
            )

            chunks.append(chunk)

        return chunks

    async def _create_image_chunks(self,
                                 images: List[Dict[str, Any]],
                                 document_id: str,
                                 file_path: str) -> List[MultimodalDocumentChunk]:
        """Create chunks from image content"""
        chunks = []

        for i, image in enumerate(images):
            # Create image description
            image_text = f"Image {i+1}: {image.get('description', 'No description available')}\n"

            if 'tags' in image:
                image_text += f"Tags: {', '.join(image['tags'])}\n"

            if 'features' in image:
                image_text += f"Visual features: {str(image['features'])}\n"

            # Create chunk
            chunk = MultimodalDocumentChunk(
                id=str(uuid.uuid4()),
                content=image_text,
                chunk_type=ChunkType.PARAGRAPH,
                level=1,
                source_document=file_path,
                page_number=image.get('page_number'),
                modality="image",
                visual_features=image.get('features'),
                extracted_entities=self._extract_entities_from_image(image),
                semantic_tags=["image", "visual_content"] + image.get('tags', [])
            )

            chunks.append(chunk)

        return chunks

    async def _create_cross_modal_chunks(self,
                                       base_chunks: List[MultimodalDocumentChunk],
                                       document_id: str) -> List[MultimodalDocumentChunk]:
        """Create cross-modal chunks that combine information from different modalities"""
        cross_modal_chunks = []

        # Group chunks by page
        page_chunks = {}
        for chunk in base_chunks:
            page_num = chunk.page_number or 0
            if page_num not in page_chunks:
                page_chunks[page_num] = []
            page_chunks[page_num].append(chunk)

        # Create cross-modal summaries for each page
        for page_num, chunks_on_page in page_chunks.items():
            if len(chunks_on_page) > 1:  # Only create cross-modal chunks if multiple modalities exist
                modalities = set(chunk.modality for chunk in chunks_on_page)

                if len(modalities) > 1:  # Cross-modal content
                    # Create summary
                    summary_parts = []

                    # Add text summary
                    text_chunks = [c for c in chunks_on_page if c.modality == "text"]
                    if text_chunks:
                        summary_parts.append("Text content: " +
                                           " ".join([c.content[:200] for c in text_chunks[:2]]))

                    # Add table summary
                    table_chunks = [c for c in chunks_on_page if c.modality == "table"]
                    if table_chunks:
                        summary_parts.append(f"Contains {len(table_chunks)} table(s) with structured data")

                    # Add chart summary
                    chart_chunks = [c for c in chunks_on_page if c.modality == "chart"]
                    if chart_chunks:
                        summary_parts.append(f"Contains {len(chart_chunks)} chart(s) with data visualizations")

                    # Add image summary
                    image_chunks = [c for c in chunks_on_page if c.modality == "image"]
                    if image_chunks:
                        summary_parts.append(f"Contains {len(image_chunks)} image(s) with visual content")

                    cross_modal_content = f"Cross-modal content on page {page_num}: " + "; ".join(summary_parts)

                    # Create cross-modal chunk
                    cross_modal_chunk = MultimodalDocumentChunk(
                        id=str(uuid.uuid4()),
                        content=cross_modal_content,
                        chunk_type=ChunkType.SECTION,
                        level=0,
                        source_document=chunks_on_page[0].source_document,
                        page_number=page_num,
                        modality="mixed",
                        semantic_tags=["cross_modal", "multimodal_summary"],
                        metadata={
                            "related_chunks": [c.id for c in chunks_on_page],
                            "modalities_present": list(modalities)
                        }
                    )

                    cross_modal_chunks.append(cross_modal_chunk)

        return cross_modal_chunks

    async def _extract_cross_modal_features(self, chunks: List[MultimodalDocumentChunk]) -> Dict[str, Any]:
        """Extract cross-modal features from chunks"""
        features = {
            'entity_cooccurrence': {},
            'modality_patterns': {},
            'semantic_relationships': {}
        }

        # Analyze entity co-occurrence across modalities
        all_entities = []
        for chunk in chunks:
            all_entities.extend(chunk.extracted_entities)

        # Count entity frequencies
        entity_counts = {}
        for entity in all_entities:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1

        features['entity_cooccurrence'] = entity_counts

        # Analyze modality patterns
        modality_counts = {}
        for chunk in chunks:
            modality_counts[chunk.modality] = modality_counts.get(chunk.modality, 0) + 1

        features['modality_patterns'] = modality_counts

        # Analyze semantic relationships
        all_tags = []
        for chunk in chunks:
            all_tags.extend(chunk.semantic_tags)

        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        features['semantic_relationships'] = tag_counts

        return features

    async def _generate_multimodal_embeddings(self,
                                            chunks: List[MultimodalDocumentChunk],
                                            cross_modal_features: Dict[str, Any]) -> None:
        """Generate multimodal embeddings for chunks"""
        for chunk in chunks:
            embeddings = {}

            # Text embedding (already generated by RAG system)
            if chunk.embedding is not None:
                embeddings['text'] = chunk.embedding

            # Generate embeddings for other modalities based on content
            if chunk.modality == "table" and chunk.table_data:
                # Create table-specific embedding
                table_text = json.dumps(chunk.table_data, sort_keys=True)
                table_embedding = self.rag_system.retriever.embedding_model.encode(table_text)
                embeddings['table'] = table_embedding

            elif chunk.modality == "chart" and chunk.chart_data:
                # Create chart-specific embedding
                chart_text = json.dumps(chunk.chart_data, sort_keys=True)
                chart_embedding = self.rag_system.retriever.embedding_model.encode(chart_text)
                embeddings['chart'] = chart_embedding

            elif chunk.modality == "image" and chunk.visual_features:
                # Create image-specific embedding from features
                image_text = json.dumps(chunk.visual_features, sort_keys=True)
                image_embedding = self.rag_system.retriever.embedding_model.encode(image_text)
                embeddings['image'] = image_embedding

            # Create cross-modal embedding
            if len(embeddings) > 1:
                # Average all available embeddings
                cross_modal_embedding = np.mean(list(embeddings.values()), axis=0)
                embeddings['cross_modal'] = cross_modal_embedding

            chunk.cross_modal_embeddings = embeddings

    async def _store_in_multimodal_index(self,
                                       chunks: List[MultimodalDocumentChunk],
                                       document_id: str) -> None:
        """Store chunks in multimodal vector index"""
        for chunk in chunks:
            # Prepare multimodal data for storage
            multimodal_data = {
                'id': chunk.id,
                'document_id': document_id,
                'content': chunk.content,
                'modality': chunk.modality,
                'embeddings': chunk.cross_modal_embeddings,
                'metadata': {
                    'source_document': chunk.source_document,
                    'page_number': chunk.page_number,
                    'chunk_type': chunk.chunk_type.value,
                    'semantic_tags': chunk.semantic_tags,
                    'extracted_entities': chunk.extracted_entities
                }
            }

            # Store in vector index
            await self.vector_index.store_multimodal_vector(multimodal_data)

    async def multimodal_query(self,
                             query: MultimodalQuery,
                             session_id: str = None,
                             user_id: str = "default",
                             top_k: int = 10) -> GeneratedAnswer:
        """Process multimodal query with advanced RAG capabilities"""

        logger.info(f"Processing multimodal query: {query.text_query[:100]}...")

        # Determine optimal retrieval strategy based on query
        strategy = self._determine_retrieval_strategy(query)

        # Perform cross-modal search if needed
        cross_modal_results = None
        if query.image_query or query.modality_preference in ["image", "mixed"]:
            cross_modal_results = await self._perform_cross_modal_search(query)

        # Perform text-based RAG query
        text_answer = await self.rag_system.query(
            query.text_query,
            session_id,
            user_id,
            strategy,
            top_k
        )

        # If cross-modal results exist, enhance the answer
        if cross_modal_results:
            enhanced_answer = await self._enhance_answer_with_multimodal(
                text_answer, cross_modal_results, query
            )
            return enhanced_answer

        return text_answer

    def _determine_retrieval_strategy(self, query: MultimodalQuery) -> RetrievalStrategy:
        """Determine optimal retrieval strategy based on query characteristics"""

        # Use context-aware for most queries
        if query.modality_preference == "auto":
            return RetrievalStrategy.CONTEXT_AWARE

        # Use multi-hop for complex queries
        if len(query.text_query.split()) > 10 or "compare" in query.text_query.lower():
            return RetrievalStrategy.MULTI_HOP

        # Use hybrid for image-enhanced queries
        if query.image_query:
            return RetrievalStrategy.HYBRID

        return RetrievalStrategy.CONTEXT_AWARE

    async def _perform_cross_modal_search(self, query: MultimodalQuery) -> List[Dict[str, Any]]:
        """Perform cross-modal search using the multimodal search engine"""

        try:
            # Prepare search query for cross-modal engine
            search_params = {
                'text': query.text_query,
                'modalities': ['text', 'image', 'table', 'chart'],
                'weights': {
                    'text': 1.0 - query.cross_modal_weight,
                    'image': query.cross_modal_weight * 0.4,
                    'table': query.cross_modal_weight * 0.3,
                    'chart': query.cross_modal_weight * 0.3
                }
            }

            # Add image query if provided
            if query.image_query:
                search_params['image'] = query.image_query

            # Perform search
            results = await self.cross_modal_search.search(search_params)

            return results[:5]  # Limit to top 5 cross-modal results

        except Exception as e:
            logger.error(f"Cross-modal search failed: {e}")
            return []

    async def _enhance_answer_with_multimodal(self,
                                            text_answer: GeneratedAnswer,
                                            cross_modal_results: List[Dict[str, Any]],
                                            query: MultimodalQuery) -> GeneratedAnswer:
        """Enhance text answer with cross-modal information"""

        enhanced_content = text_answer.answer
        enhanced_citations = text_answer.citations.copy()

        # Add cross-modal insights
        if cross_modal_results:
            cross_modal_section = "\n\n**Cross-Modal Insights:**\n"

            for result in cross_modal_results:
                if result['modality'] == 'image':
                    cross_modal_section += f"- Visual Analysis: {result['description']}\n"
                elif result['modality'] == 'table':
                    cross_modal_section += f"- Table Data: {result['summary']}\n"
                elif result['modality'] == 'chart':
                    cross_modal_section += f"- Chart Analysis: {result['insights']}\n"

            enhanced_content += cross_modal_section

            # Add cross-modal citations
            for result in cross_modal_results:
                citation = {
                    'chunk_id': result.get('id', 'unknown'),
                    'content_snippet': result.get('description', '')[:200],
                    'relevance_score': result.get('score', 0.5),
                    'source_document': result.get('source', 'cross_modal_search'),
                    'confidence': result.get('confidence', 0.5)
                }
                enhanced_citations.append(citation)

        # Create enhanced answer
        enhanced_answer = GeneratedAnswer(
            answer=enhanced_content,
            citations=enhanced_citations,
            confidence=min(text_answer.confidence * 1.1, 1.0),  # Slightly boost confidence
            sources_used=text_answer.sources_used + ['cross_modal_search'],
            generation_metadata={
                **text_answer.generation_metadata,
                'cross_modal_enhanced': True,
                'cross_modal_results_count': len(cross_modal_results),
                'query_modality': query.modality_preference
            }
        )

        return enhanced_answer

    def _extract_semantic_tags(self, content: str) -> List[str]:
        """Extract semantic tags from content"""
        tags = []
        content_lower = content.lower()

        # Define tag patterns
        tag_patterns = {
            'technical': ['technical', 'specification', 'parameter', 'configuration'],
            'financial': ['cost', 'price', 'budget', 'revenue', 'financial'],
            'legal': ['legal', 'contract', 'agreement', 'terms', 'conditions'],
            'process': ['process', 'procedure', 'workflow', 'steps'],
            'data': ['data', 'information', 'statistics', 'metrics'],
            'analysis': ['analysis', 'report', 'study', 'research']
        }

        # Apply patterns
        for tag, patterns in tag_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                tags.append(tag)

        return tags

    def _extract_entities_from_table(self, table: Dict[str, Any]) -> List[str]:
        """Extract entities from table data"""
        entities = []

        # Extract from headers
        if 'headers' in table:
            entities.extend([h.strip() for h in table['headers'] if len(h.strip()) > 2])

        # Extract from first few rows
        if 'rows' in table:
            for row in table['rows'][:3]:  # Only first 3 rows
                for cell in row:
                    cell_str = str(cell).strip()
                    if len(cell_str) > 2 and cell_str.replace(' ', '').isalnum():
                        entities.append(cell_str)

        return list(set(entities))  # Remove duplicates

    def _extract_entities_from_chart(self, chart: Dict[str, Any]) -> List[str]:
        """Extract entities from chart data"""
        entities = []

        # Extract from title
        if 'title' in chart:
            title_words = chart['title'].split()
            entities.extend([word for word in title_words if len(word) > 3])

        # Extract from data points
        if 'data_points' in chart:
            for point in chart['data_points'][:5]:  # Only first 5 points
                if isinstance(point, dict):
                    for key, value in point.items():
                        if isinstance(key, str) and len(key) > 2:
                            entities.append(key)
                        if isinstance(value, str) and len(value) > 2:
                            entities.append(value)

        return list(set(entities))

    def _extract_entities_from_image(self, image: Dict[str, Any]) -> List[str]:
        """Extract entities from image data"""
        entities = []

        # Extract from description
        if 'description' in image:
            desc_words = image['description'].split()
            entities.extend([word for word in desc_words if len(word) > 3])

        # Extract from tags
        if 'tags' in image:
            entities.extend(image['tags'])

        # Extract from features
        if 'features' in image and isinstance(image['features'], dict):
            for key, value in image['features'].items():
                if isinstance(key, str) and len(key) > 2:
                    entities.append(key)
                if isinstance(value, str) and len(value) > 2:
                    entities.append(value)

        return list(set(entities))

    async def get_multimodal_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the multimodal RAG system"""

        # Get base RAG statistics
        rag_stats = self.rag_system.get_statistics()

        # Add multimodal-specific statistics
        multimodal_stats = {
            'multimodal_documents_processed': len(self.processing_cache),
            'supported_modalities': ['text', 'image', 'table', 'chart', 'mixed'],
            'cross_modal_capabilities': True,
            'vector_index_stats': await self.vector_index.get_statistics(),
            'cross_modal_search_stats': await self.cross_modal_search.get_statistics()
        }

        # Analyze modality distribution in processed documents
        if self.processing_cache:
            modality_counts = {}
            total_chunks = 0

            for doc_result in self.processing_cache.values():
                for chunk in doc_result['chunks']:
                    modality = chunk.modality
                    modality_counts[modality] = modality_counts.get(modality, 0) + 1
                    total_chunks += 1

            multimodal_stats['modality_distribution'] = modality_counts
            multimodal_stats['total_multimodal_chunks'] = total_chunks

        # Combine statistics
        combined_stats = {**rag_stats, **multimodal_stats}
        combined_stats['system_type'] = "advanced_multimodal_rag"

        return combined_stats

# Example usage and testing
if __name__ == "__main__":
    async def test_multimodal_rag_integration():
        """Test the multimodal RAG integration"""

        # Initialize integration
        integrator = MultimodalRAGIntegrator()

        # Test with a sample document (this would normally be a real file)
        sample_content = """
        # Sample Document with Mixed Content

        This document contains various types of content including text, tables, and charts.

        ## Sales Data Table

        | Product | Q1 Sales | Q2 Sales | Q3 Sales | Q4 Sales |
        |---------|----------|----------|----------|----------|
        | Product A | $100,000 | $120,000 | $110,000 | $130,000 |
        | Product B | $80,000 | $90,000 | $95,000 | $105,000 |
        | Product C | $60,000 | $75,000 | $85,000 | $90,000 |

        ## Revenue Chart

        The revenue chart shows steady growth throughout the year with Q4 being the strongest quarter.

        [Chart Description: Bar chart showing quarterly revenue growth]

        ## Product Images

        [Image: Product A photograph showing professional packaging]
        [Image: Product B lifestyle photo]

        ## Analysis Summary

        The data indicates positive growth trends across all product lines, with Product A leading in sales volume.
        """

        # Create a temporary file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(sample_content)
            temp_file = f.name

        try:
            # Process the document
            result = await integrator.process_multimodal_document(temp_file)
            print(f"Processed document: {result['document_id']}")
            print(f"Created {result['chunk_count']} chunks")
            print(f"Modalities: {result['modalities']}")

            # Test multimodal queries
            queries = [
                MultimodalQuery(text_query="What are the sales figures for Product A?"),
                MultimodalQuery(text_query="Show me the revenue trends", modality_preference="chart"),
                MultimodalQuery(text_query="Compare product performance", modality_preference="mixed")
            ]

            for query in queries:
                print(f"\nQuery: {query.text_query}")
                answer = await integrator.multimodal_query(query, session_id="test_session")
                print(f"Answer: {answer.answer[:200]}...")
                print(f"Confidence: {answer.confidence:.2f}")

            # Get statistics
            stats = await integrator.get_multimodal_statistics()
            print(f"\nSystem Statistics: {stats}")

        finally:
            # Clean up temporary file
            import os
            os.unlink(temp_file)

    # Run test
    asyncio.run(test_multimodal_rag_integration())