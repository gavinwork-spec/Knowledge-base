#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete RAG System Example
完整的RAG系统示例

This example demonstrates all major features of the advanced RAG system
including document processing, querying, conversation memory, and citations.
"""

import asyncio
import json
import sys
import os

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.advanced_rag_system import (
    create_advanced_rag_system,
    RAGQuery,
    RAGSystemConfig,
    ContentType,
    RetrievalStrategy
)

async def main():
    """Main example function demonstrating all RAG features"""

    print("🚀 Advanced RAG System - Complete Example")
    print("=" * 50)

    # Step 1: Initialize the RAG system
    print("\n📋 Step 1: Initializing RAG System...")
    config = RAGSystemConfig(
        db_path="knowledge_base.db",
        default_max_results=8,
        default_retrieval_strategy=RetrievalStrategy.HYBRID,
        enable_multi_modal=True,
        enable_conversation_memory=True,
        enable_query_decomposition=True,
        enable_citation_tracking=True,
        log_level="INFO"
    )

    rag_system = create_advanced_rag_system(config)

    try:
        await rag_system.initialize()
        print("✅ RAG System initialized successfully")

        # Step 2: Add sample manufacturing documents
        print("\n📚 Step 2: Adding Sample Manufacturing Documents...")
        sample_docs = await add_sample_documents(rag_system)
        print(f"✅ Added {len(sample_docs)} sample documents")

        # Step 3: Demonstrate different query types
        print("\n🔍 Step 3: Demonstrating Query Types...")
        await demonstrate_query_types(rag_system)

        # Step 4: Show conversation memory
        print("\n💬 Step 4: Demonstrating Conversation Memory...")
        await demonstrate_conversation_memory(rag_system)

        # Step 5: Show query decomposition
        print("\n🧩 Step 5: Demonstrating Query Decomposition...")
        await demonstrate_query_decomposition(rag_system)

        # Step 6: Show multi-modal retrieval
        print("\n🖼️ Step 6: Demonstrating Multi-Modal Retrieval...")
        await demonstrate_multi_modal_retrieval(rag_system)

        # Step 7: Show citation tracking
        print("\n📄 Step 7: Demonstrating Citation Tracking...")
        await demonstrate_citation_tracking(rag_system)

        # Step 8: Show streaming responses
        print("\n🌊 Step 8: Demonstrating Streaming Responses...")
        await demonstrate_streaming_responses(rag_system)

        # Step 9: Show system statistics
        print("\n📊 Step 9: System Statistics...")
        await show_system_statistics(rag_system)

        print("\n🎉 Complete example finished successfully!")

    except Exception as e:
        print(f"❌ Error during example execution: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        await rag_system.close()
        print("🧹 System cleanup completed")

async def add_sample_documents(rag_system):
    """Add sample manufacturing documents to the system"""

    sample_documents = [
        {
            "content": """
            CNC Machine Safety Procedures - HAAS VF-2

            1. Personal Protective Equipment (PPE)
            - Safety glasses with side shields
            - Steel-toed boots
            - Hearing protection (required when machine is running)
            - No loose clothing, jewelry, or long hair

            2. Machine Pre-Operation Check
            - Verify all machine guards are in place
            - Check emergency stop functionality
            - Ensure proper lighting conditions
            - Verify tool holder is properly secured

            3. Emergency Procedures
            - Emergency Stop: Large red button on control panel
            - Power Off: Main switch on rear of machine
            - Fire Extinguisher: Located at machine exit

            4. Daily Maintenance Requirements
            - Clean chip pan and work area
            - Check fluid levels (coolant, lubricants)
            - Inspect tools for wear and damage
            - Verify spindle cooling system operation
            """,
            "content_type": ContentType.TEXT,
            "metadata": {
                "title": "HAAS VF-2 Safety Procedures",
                "document_type": "Safety Manual",
                "equipment_model": "HAAS-VF2",
                "department": "Manufacturing",
                "version": "2.1",
                "author": "Safety Department",
                "creation_date": "2024-01-15"
            }
        },
        {
            "content": """
            DMG MORI DMU 50 - Quality Control Specifications

            Dimensional Tolerances:
            - Linear positioning accuracy: ±0.005mm
            - Repeatability: ±0.002mm
            - Spindle runout: <0.001mm
            - Table flatness: 0.01mm per 300mm

            Surface Finish Requirements:
            - Roughness (Ra): 0.8μm or better
            - Waviness (W): 0.2μm maximum
            - No visible tool marks on critical surfaces

            Inspection Procedures:
            1. Use calibrated CMM equipment
            2. Measure at 20°C ± 1°C
            3. Allow parts to stabilize for 2 hours before measurement
            4. Use appropriate probing force (≤0.5N)

            Acceptance Criteria:
            - All dimensions within specified tolerances
            - Surface finish meeting or exceeding requirements
            - No burrs or surface defects
            - Geometric tolerances per GD&T standards
            """,
            "content_type": ContentType.TEXT,
            "metadata": {
                "title": "DMU 50 Quality Control Specifications",
                "document_type": "Quality Specification",
                "equipment_model": "DMG-MORI-DMU50",
                "department": "Quality Assurance",
                "standard": "ISO-9001",
                "version": "3.0",
                "creation_date": "2024-02-20"
            }
        },
        {
            "content": """
            Manufacturing Process Validation - ISO 9001:2015 Section 8.5.1

            Validation Requirements:
            1. Process Capability Studies
               - Cpk ≥ 1.33 for critical characteristics
               - Control charts for statistical process control
               - Regular capability assessments (minimum quarterly)

            2. Equipment Qualification
               - Installation Qualification (IQ): Verify proper setup
               - Operational Qualification (OQ): Confirm function across operating range
               - Performance Qualification (PQ): Demonstrate consistent output

            3. Personnel Training and Competence
               - documented training records
               - competency assessments
               - periodic refresher training

            4. Monitoring and Measurement
               - real-time process monitoring where feasible
               - statistical process control charts
               - trend analysis and early warning systems

            Validation Documentation:
            - Validation protocol
            - Execution records
            - Summary report with conclusions
            - Ongoing monitoring plan
            """,
            "content_type": ContentType.TEXT,
            "metadata": {
                "title": "Process Validation Requirements",
                "document_type": "Quality Standard",
                "standard": "ISO-9001:2015",
                "section": "8.5.1",
                "department": "Quality Assurance",
                "version": "4.1",
                "creation_date": "2024-03-10"
            }
        }
    ]

    doc_ids = []
    for doc in sample_documents:
        doc_id = await rag_system.add_document(
            content=doc["content"],
            content_type=doc["content_type"],
            metadata=doc["metadata"]
        )
        doc_ids.append(doc_id)
        print(f"   Added document: {doc['metadata']['title']}")

    return doc_ids

async def demonstrate_query_types(rag_system):
    """Demonstrate different types of queries"""

    queries = [
        {
            "name": "Simple Factual Query",
            "query": "What safety equipment is required for CNC machine operation?"
        },
        {
            "name": "Technical Specification Query",
            "query": "What are the dimensional tolerances for DMG MORI DMU 50?"
        },
        {
            "name": "Procedural Query",
            "query": "Describe the validation process according to ISO 9001:2015"
        },
        {
            "name": "Equipment-Specific Query",
            "query": "What are the daily maintenance requirements for HAAS VF-2?"
        }
    ]

    for query_info in queries:
        print(f"\n   {query_info['name']}:")
        print(f"   Query: {query_info['query']}")

        query = RAGQuery(
            query_id=f"demo_{query_info['name'].lower().replace(' ', '_')}",
            text=query_info['query'],
            max_results=3,
            include_citations=True
        )

        response = await rag_system.query(query)

        print(f"   Answer: {response.answer[:200]}...")
        print(f"   Confidence: {response.confidence_score:.2f}")
        print(f"   Sources: {len(response.sources)}")
        print(f"   Processing Time: {response.processing_time_ms:.0f}ms")

async def demonstrate_conversation_memory(rag_system):
    """Demonstrate conversation memory and context awareness"""

    session_id = "demo_conversation_001"

    # First query
    query1 = RAGQuery(
        query_id="conv_001",
        text="What are the safety procedures for HAAS machines?",
        session_id=session_id
    )

    response1 = await rag_system.query(query1)
    print(f"   Query 1: {query1.text}")
    print(f"   Response: {response1.answer[:150]}...")

    # Follow-up query that references previous context
    query2 = RAGQuery(
        query_id="conv_002",
        text="What about the DMG MORI machines?",
        session_id=session_id
    )

    response2 = await rag_system.query(query2)
    print(f"\n   Query 2: {query2.text} (follow-up)")
    print(f"   Response: {response2.answer[:150]}...")
    print(f"   Context used: {response2.context_used}")

async def demonstrate_query_decomposition(rag_system):
    """Demonstrate complex query decomposition"""

    complex_query = RAGQuery(
        query_id="decomp_001",
        text="Compare the safety procedures and maintenance requirements between HAAS VF-2 and DMG MORI DMU 50 machines, focusing on daily operations and quality standards compliance",
        max_results=6,
        include_citations=True
    )

    response = await rag_system.query(complex_query)

    print(f"   Complex Query: {complex_query.text}")
    print(f"   Answer: {response.answer[:300]}...")
    print(f"   Sources used: {len(response.sources)}")
    print(f"   Processing time: {response.processing_time_ms:.0f}ms")

    # Show sub-queries if decomposition occurred
    if hasattr(response, 'decomposition_plan') and response.decomposition_plan:
        print(f"   Sub-queries created: {len(response.decomposition_plan.sub_queries)}")
        for i, sub_query in enumerate(response.decomposition_plan.sub_queries, 1):
            print(f"     {i}. {sub_query.text}")

async def demonstrate_multi_modal_retrieval(rag_system):
    """Demonstrate multi-modal content retrieval"""

    multi_modal_query = RAGQuery(
        query_id="multimodal_001",
        text="Show me technical specifications and quality requirements",
        content_types=[ContentType.TEXT, ContentType.TABLE],
        max_results=4
    )

    response = await rag_system.query(multi_modal_query)

    print(f"   Multi-Modal Query: {multi_modal_query.text}")
    print(f"   Content Types: {[ct.value for ct in multi_modal_query.content_types]}")
    print(f"   Answer: {response.answer[:200]}...")
    print(f"   Sources by type:")

    for source in response.sources:
        source_type = source.metadata.get('content_type', 'unknown')
        print(f"     - {source_type}: {source.metadata.get('title', 'Unknown')[:30]}...")

async def demonstrate_citation_tracking(rag_system):
    """Demonstrate citation tracking and source verification"""

    cited_query = RAGQuery(
        query_id="citation_001",
        text="What are the ISO 9001 requirements for process validation?",
        max_results=5,
        include_citations=True
    )

    response = await rag_system.query(cited_query)

    print(f"   Query with Citations: {cited_query.text}")
    print(f"   Answer: {response.answer[:250]}...")
    print(f"\n   Citations ({len(response.citations)}):")

    for i, citation in enumerate(response.citations, 1):
        print(f"     {i}. Source ID: {citation.source_id}")
        print(f"        Content: {citation.content_snippet[:100]}...")
        print(f"        Confidence: {citation.confidence_score:.2f}")
        print(f"        Type: {citation.citation_type.value}")
        if citation.metadata:
            print(f"        Metadata: {citation.metadata.get('title', 'No title')}")
        print()

async def demonstrate_streaming_responses(rag_system):
    """Demonstrate streaming response capability"""

    stream_query = RAGQuery(
        query_id="stream_001",
        text="Explain the complete quality control process for manufactured parts",
        session_id="demo_stream_session"
    )

    print(f"   Streaming Query: {stream_query.text}")
    print(f"   Response: ", end="", flush=True)

    sources_count = 0
    error_occurred = False

    try:
        async for chunk in rag_system.stream_query(stream_query):
            data = json.loads(chunk)

            if data['type'] == 'chunk':
                print(data['content'], end='', flush=True)
            elif data['type'] == 'metadata':
                sources_count = data['sources_count']
            elif data['type'] == 'error':
                error_occurred = True
                print(f"\n   Error: {data['message']}")
                break

    except Exception as e:
        print(f"\n   Streaming error: {e}")
        error_occurred = True

    if not error_occurred:
        print(f"\n   ✓ Streaming completed successfully")
        print(f"   Sources used: {sources_count}")

async def show_system_statistics(rag_system):
    """Display comprehensive system statistics"""

    stats = await rag_system.get_system_stats()

    print(f"   System Health: {stats['system_health']}")
    print(f"   Database Version: {stats['database']['schema_version']}")

    print("\n   Database Components:")
    for component, count in stats['database']['rag_components'].items():
        print(f"     - {component}: {count}")

    print(f"\n   Active Sessions: {stats['sessions']['active_sessions']}")
    print(f"   Total Sessions: {stats['sessions']['total_sessions']}")

    print("\n   Performance Metrics:")
    perf = stats['performance']
    print(f"     - Average response time: {perf['avg_response_time_ms']:.0f}ms")
    print(f"     - Queries processed: {perf['queries_processed']}")
    print(f"     - Cache hit rate: {perf['cache_hit_rate']:.1%}")

    print("\n   System Capabilities:")
    capabilities = stats['capabilities']
    print(f"     - Multi-modal: {'✅' if capabilities['multi_modal_enabled'] else '❌'}")
    print(f"     - Conversation memory: {'✅' if capabilities['conversation_memory_enabled'] else '❌'}")
    print(f"     - Query decomposition: {'✅' if capabilities['query_decomposition_enabled'] else '❌'}")
    print(f"     - Citation tracking: {'✅' if capabilities['citation_tracking_enabled'] else '❌'}")

if __name__ == "__main__":
    print("🎯 Starting Complete RAG System Example...")
    print("This example demonstrates all major features of the advanced RAG system.")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    print("\n🏁 Example execution completed")