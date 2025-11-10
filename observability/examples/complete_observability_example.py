#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Observability System Example
完整可观测性系统示例

This example demonstrates all major features of the comprehensive observability system
including AI interaction tracking, manufacturing metrics, real-time monitoring, and alerting.
"""

import asyncio
import json
import time
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

# Add the parent directory to the path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observability import (
    create_observability_orchestrator,
    ObservabilityConfig,
    get_observability_orchestrator
)

async def main():
    """Main example demonstrating complete observability features"""

    print("🚀 Comprehensive Observability System - Complete Example")
    print("=" * 60)

    # Step 1: Initialize the observability system
    print("\n📋 Step 1: Initializing Observability System...")
    config = ObservabilityConfig(
        db_path="knowledge_base.db",
        enable_langfuse=False,  # Disable for demo (would need actual LangFuse keys)
        enable_dashboard=True,
        dashboard_websocket_port=8765,
        enable_alerts=True,
        enable_detailed_logging=True,
        cost_tracking_enabled=True,
        user_analytics_enabled=True,
        manufacturing_metrics_enabled=True
    )

    orchestrator = await create_observability_orchestrator(config)
    print("✅ Observability system initialized successfully")

    # Step 2: Demonstrate AI interaction tracking
    print("\n🔍 Step 2: Demonstrating AI Interaction Tracking...")
    await demonstrate_ai_interactions(orchestrator)

    # Step 3: Demonstrate manufacturing event tracking
    print("\n🏭 Step 3: Demonstrating Manufacturing Event Tracking...")
    await demonstrate_manufacturing_events(orchestrator)

    # Step 4: Demonstrate user behavior analytics
    print("\n👥 Step 4: Demonstrating User Behavior Analytics...")
    await demonstrate_user_analytics(orchestrator)

    # Step 5: Demonstrate real-time monitoring
    print("\n📊 Step 5: Demonstrating Real-time Monitoring...")
    await demonstrate_real_time_monitoring(orchestrator)

    # Step 6: Demonstrate system health reporting
    print("\n🏥 Step 6: Demonstrating System Health Reporting...")
    await demonstrate_system_health(orchestrator)

    # Step 7: Show comprehensive observability summary
    print("\n📈 Step 7: Comprehensive Observability Summary...")
    await show_observability_summary(orchestrator)

    print("\n🎉 Complete observability example finished successfully!")
    print(f"\n📊 Dashboard available at: ws://localhost:{config.dashboard_websocket_port}")
    print("💡 Connect a WebSocket client to see real-time updates")

async def demonstrate_ai_interactions(orchestrator):
    """Demonstrate AI interaction tracking"""
    interactions = [
        {
            "session_id": "mfg_session_001",
            "user_id": "operator_001",
            "query": "What are the safety procedures for HAAS VF-2 CNC machine?",
            "response": "The HAAS VF-2 safety procedures include: 1. Always wear appropriate PPE including safety glasses and steel-toed boots. 2. Ensure all machine guards are in place before operation. 3. Perform daily pre-operation checks. 4. Use proper lockout/tagout procedures during maintenance. 5. Keep work area clean and organized.",
            "context": {"equipment_model": "HAAS_VF-2", "safety_level": "critical"},
            "performance_data": {
                "response_time_ms": 1250,
                "prompt_tokens": 18,
                "completion_tokens": 95,
                "satisfaction_score": 5
            }
        },
        {
            "session_id": "mfg_session_002",
            "user_id": "quality_inspector_001",
            "query": "What are the quality control requirements for DMG MORI DMU 50 according to ISO 9001?",
            "response": "According to ISO 9001:2015, the quality control requirements for DMG MORI DMU 50 include: 1. Regular calibration of measurement equipment. 2. Documented inspection procedures. 3. Statistical process control (SPC) implementation. 4. Traceability of all quality records. 5. Regular internal audits and management reviews.",
            "context": {"equipment_model": "DMG_MORI_DMU50", "standard": "ISO_9001"},
            "performance_data": {
                "response_time_ms": 1850,
                "prompt_tokens": 22,
                "completion_tokens": 110,
                "satisfaction_score": 4
            }
        },
        {
            "session_id": "mfg_session_003",
            "user_id": "process_engineer_001",
            "query": "Explain the welding requirements per ANSI AWS D17.1 for aerospace applications",
            "response": "ANSI AWS D17.1 welding requirements for aerospace applications specify: 1. Welder certification and qualification procedures. 2. Material traceability requirements. 3. Welding procedure specifications (WPS). 4. Non-destructive testing (NDT) requirements. 5. Acceptance criteria for weld discontinuities. 6. Documentation and record-keeping standards.",
            "context": {"standard": "ANSI_AWS_D17.1", "industry": "aerospace"},
            "performance_data": {
                "response_time_ms": 2100,
                "prompt_tokens": 15,
                "completion_tokens": 125,
                "satisfaction_score": 5
            }
        }
    ]

    for i, interaction in enumerate(interactions, 1):
        print(f"   Interaction {i}: {interaction['query'][:50]}...")

        # Track the AI interaction
        async with orchestrator.trace_interaction(
            session_id=interaction["session_id"],
            user_id=interaction["user_id"],
            interaction_type="manufacturing_query"
        ):
            await orchestrator.log_ai_interaction(
                session_id=interaction["session_id"],
                user_id=interaction["user_id"],
                agent_type="knowledge_retriever",
                model_provider="openai",
                model_name="gpt-4",
                prompt=interaction["query"],
                response=interaction["response"],
                context=interaction["context"],
                performance_data=interaction["performance_data"]
            )

        # Track user query for analytics
        await orchestrator.track_user_query(
            session_id=interaction["session_id"],
            user_id=interaction["user_id"],
            query=interaction["query"],
            response=interaction["response"],
            satisfaction_score=interaction["performance_data"]["satisfaction_score"]
        )

        # Simulate processing time
        await asyncio.sleep(0.5)

async def demonstrate_manufacturing_events(orchestrator):
    """Demonstrate manufacturing event tracking"""
    manufacturing_events = [
        {
            "event_type": "quote_created",
            "data": {
                "quote_id": "Q2024-001",
                "customer_id": "AEROSPACE_CORP",
                "part_number": "WING_BRACKET_001",
                "quantity": 500,
                "quoted_price": 125000.00,
                "accuracy_score": 96.5,
                "processing_time_minutes": 15.2,
                "margin_percentage": 28.5,
                "metadata": {
                    "material": "Titanium Grade 5",
                    "tolerance": "±0.005mm",
                    "surface_finish": "Ra 0.8",
                    "certification_required": "AS9100"
                }
            }
        },
        {
            "event_type": "quality_inspection",
            "data": {
                "inspection_id": "INS-2024-001",
                "part_number": "WING_BRACKET_001",
                "batch_id": "BATCH-001",
                "process_type": "cnc_machining",
                "total_inspected": 500,
                "passed_count": 485,
                "failed_count": 10,
                "rework_count": 5,
                "defect_rate": 3.0,
                "first_pass_yield": 97.0,
                "inspection_time_minutes": 45.5,
                "result": "pass",
                "inspector_id": "INSPECTOR_001"
            }
        },
        {
            "event_type": "customer_feedback",
            "data": {
                "feedback_id": "FB-2024-001",
                "customer_id": "AEROSPACE_CORP",
                "order_id": "ORD-2024-001",
                "overall_satisfaction": 5,
                "quality_satisfaction": 5,
                "delivery_satisfaction": 4,
                "service_satisfaction": 5,
                "price_satisfaction": 4,
                "nps_score": 9,
                "feedback_text": "Excellent quality and precision. The parts met all aerospace specifications. Delivery was on time and the team was very responsive to our requirements."
            }
        },
        {
            "event_type": "document_processed",
            "data": {
                "processing_id": "DOC-2024-001",
                "document_type": "technical_manual",
                "file_size_mb": 15.8,
                "processing_time_seconds": 25.3,
                "success": True,
                "extracted_entities": 156,
                "processing_accuracy": 94.2,
                "status": "success"
            }
        },
        {
            "event_type": "production_efficiency",
            "data": {
                "production_id": "PROD-2024-001",
                "work_order_id": "WO-2024-001",
                "part_number": "WING_BRACKET_001",
                "process_type": "cnc_machining",
                "planned_time_hours": 8.0,
                "actual_time_hours": 7.2,
                "efficiency_percentage": 111.1,
                "downtime_minutes": 12,
                "scrap_percentage": 0.8,
                "rework_percentage": 1.2,
                "operator_id": "OPERATOR_001"
            }
        }
    ]

    for i, event in enumerate(manufacturing_events, 1):
        print(f"   Manufacturing Event {i}: {event['event_type']}")
        await orchestrator.record_manufacturing_event(event["event_type"], event["data"])
        await asyncio.sleep(0.3)

async def demonstrate_user_analytics(orchestrator):
    """Demonstrate user behavior analytics"""
    user_sessions = [
        {
            "session_id": "user_session_001",
            "user_id": "operator_001",
            "queries": [
                "How do I set up the tool for aluminum machining?",
                "What are the recommended cutting parameters?",
                "Safety procedures for tool change"
            ],
            "session_duration": 25.5,
            "satisfaction_scores": [4, 5, 4]
        },
        {
            "session_id": "user_session_002",
            "user_id": "quality_inspector_001",
            "queries": [
                "CMM calibration procedures",
                "GD&T tolerance requirements",
                "Statistical process control methods"
            ],
            "session_duration": 45.2,
            "satisfaction_scores": [5, 5, 5]
        }
    ]

    for session in user_sessions:
        print(f"   User Session: {session['user_id']} - {len(session['queries'])} queries")

        for i, query in enumerate(session["queries"]):
            satisfaction = session["satisfaction_scores"][i] if i < len(session["satisfaction_scores"]) else None

            await orchestrator.track_user_query(
                session_id=session["session_id"],
                user_id=session["user_id"],
                query=query,
                response=f"Response to {query}",
                satisfaction_score=satisfaction
            )

async def demonstrate_real_time_monitoring(orchestrator):
    """Demonstrate real-time monitoring capabilities"""
    print("   Simulating real-time metrics updates...")

    # Simulate some real-time metric updates
    metrics_updates = [
        {"cpu_usage": random.uniform(20, 80)},
        {"memory_usage": random.uniform(30, 70)},
        {"api_response_time": random.uniform(100, 500)},
        {"error_rate": random.uniform(0, 2)},
        {"active_sessions": random.randint(5, 25)}
    ]

    for i, metric_update in enumerate(metrics_updates):
        for metric_name, value in metric_update.items():
            # Record the metric (this would normally be done automatically)
            if orchestrator.performance_tracker:
                from observability.core.performance_tracker import MetricType

                # Map metric names to MetricType enum
                metric_type_map = {
                    "cpu_usage": MetricType.RESPONSE_TIME,  # Using available enum
                    "memory_usage": MetricType.RESPONSE_TIME,
                    "api_response_time": MetricType.RESPONSE_TIME,
                    "error_rate": MetricType.RESPONSE_TIME,
                    "active_sessions": MetricType.RESPONSE_TIME
                }

                if metric_name in metric_type_map:
                    orchestrator.performance_tracker.record_metric(
                        metric_type_map[metric_name],
                        value,
                        "percent" if "usage" in metric_name else "count"
                    )

        await asyncio.sleep(0.2)

    print("   Real-time metrics simulation completed")

async def demonstrate_system_health(orchestrator):
    """Demonstrate system health reporting"""
    # Get comprehensive system health
    health_report = await orchestrator.get_system_health()

    print(f"   Overall Status: {health_report.overall_status}")
    print(f"   Active Alerts: {health_report.active_alerts_count}")
    print(f"   Component Status:")

    for component, status in health_report.component_status.items():
        print(f"     - {component}: {status}")

    if health_report.recommendations:
        print("   Recommendations:")
        for rec in health_report.recommendations:
            print(f"     - {rec}")

    print(f"   Timestamp: {health_report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")

async def show_observability_summary(orchestrator):
    """Show comprehensive observability summary"""
    summary = await orchestrator.get_observability_summary()

    print(f"   System Uptime: {summary['system_health']['uptime_hours']:.1f} hours")
    print(f"   Active Components: {sum(1 for active in summary['components'].values() if active)}")

    if summary.get('performance'):
        print(f"   Performance Metrics Available: {len(summary['performance'])}")

    if summary.get('costs'):
        print(f"   Cost Metrics Available: {len(summary['costs'])}")

    if summary.get('user_analytics'):
        print(f"   User Analytics Available: {len(summary['user_analytics'])}")

    if summary.get('manufacturing'):
        print(f"   Manufacturing Metrics Available: {len(summary['manufacturing'])}")

    # Show some specific metrics if available
    if summary.get('manufacturing'):
        mfg_metrics = summary['manufacturing']
        if 'quote_accuracy_7d' in mfg_metrics:
            print(f"   7-Day Quote Accuracy: {mfg_metrics['quote_accuracy_7d']:.1f}%")
        if 'quality_fpy_24h' in mfg_metrics:
            print(f"   24-Hour First Pass Yield: {mfg_metrics['quality_fpy_24h']:.1f}%")
        if 'customer_satisfaction_30d' in mfg_metrics:
            print(f"   30-Day Customer Satisfaction: {mfg_metrics['customer_satisfaction_30d']:.1f}/5.0")

async def simulate_background_activity(orchestrator):
    """Simulate background activity for demonstration"""
    while True:
        try:
            # Simulate random AI interactions
            await asyncio.sleep(random.uniform(5, 15))

            # Random query
            queries = [
                "What are the maintenance procedures for CNC machines?",
                "Explain the quality control requirements",
                "Safety guidelines for welding operations",
                "How to optimize machining processes?"
            ]

            query = random.choice(queries)
            session_id = f"bg_session_{int(time.time())}"
            user_id = f"user_{random.randint(1000, 9999)}"

            await orchestrator.track_user_query(
                session_id=session_id,
                user_id=user_id,
                query=query,
                response=f"Generated response for: {query[:30]}...",
                satisfaction_score=random.randint(3, 5)
            )

        except Exception as e:
            print(f"Background activity error: {e}")
            await asyncio.sleep(30)

async def dashboard_info():
    """Show dashboard connection information"""
    print("\n📊 Real-time Dashboard Information")
    print("=" * 40)
    print("WebSocket Dashboard: ws://localhost:8765")
    print("\nTo connect to the dashboard:")
    print("1. Use a WebSocket client (e.g., browser with JavaScript)")
    print("2. Send JSON messages to interact with the dashboard")
    print("\nExample WebSocket client code:")
    print("""
const ws = new WebSocket('ws://localhost:8765');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Dashboard update:', data);
};

// Request dashboard data
ws.send(JSON.stringify({
    type: 'get_dashboard',
    dashboard_id: 'manufacturing_metrics'
}));
    """)

if __name__ == "__main__":
    print("🎯 Starting Complete Observability System Example...")
    print("This example demonstrates all major features of the observability system.")
    print()

    try:
        # Start background activity simulation
        background_task = asyncio.create_task(simulate_background_activity(
            await create_observability_orchestrator()
        ))

        # Run the main example
        asyncio.run(main())

        # Show dashboard info
        await dashboard_info()

    except KeyboardInterrupt:
        print("\n\n👋 Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    print("\n🏁 Example execution completed")
    print("\n💡 Tips:")
    print("- Check the database files for stored observability data")
    print("- Connect to the WebSocket dashboard for real-time monitoring")
    print("- Monitor log files for detailed observability information")
    print("- The observability system continues running in the background")