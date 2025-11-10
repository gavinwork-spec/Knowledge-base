#!/usr/bin/env python3
"""
Integration Test Runner
Manufacturing Knowledge Base - Open-Source Component Integration Testing

This script provides an easy way to run comprehensive integration tests for all
open-source components with manufacturing-specific validation and performance
benchmarks.
"""

import asyncio
import argparse
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add integrations path
sys.path.insert(0, str(Path(__file__).parent))

from tests.test_framework import run_integration_tests
from shared.manager import IntegrationManager


def setup_logging(verbosity: int = 1):
    """Setup logging configuration"""
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(verbosity, 2)]

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('tests/integration_tests.log')
        ]
    )


async def run_health_checks(config_path: str) -> bool:
    """Run health checks on all integrations"""
    print("🔍 Running integration health checks...")

    manager = IntegrationManager()
    try:
        await manager.initialize(config_path)
        health_results = await manager.health_check_all()

        all_healthy = True
        for name, health in health_results.items():
            status = health.get("status", "unknown")
            if status == "healthy":
                print(f"  ✅ {name}: {status}")
            else:
                print(f"  ❌ {name}: {status}")
                all_healthy = False

        return all_healthy

    except Exception as e:
        print(f"  ❌ Health check failed: {str(e)}")
        return False


async def run_specific_tests(test_types: list, config_path: str) -> dict:
    """Run specific test types"""
    print(f"🧪 Running specific tests: {', '.join(test_types)}")

    # Map test types to test classes
    test_mapping = {
        "langchain": "LangChainIntegrationTest",
        "lobechat": "LobeChatIntegrationTest",
        "xagent": "XAgentIntegrationTest",
        "langfuse": "LangFuseIntegrationTest",
        "manager": "IntegrationManagerTest",
        "performance": "PerformanceTest"
    }

    # Filter test classes based on requested types
    requested_tests = []
    for test_type in test_types:
        if test_type in test_mapping:
            requested_tests.append(test_mapping[test_type])
        else:
            print(f"  ⚠️  Unknown test type: {test_type}")

    if not requested_tests:
        print("  ❌ No valid test types specified")
        return {"success": False, "error": "No valid test types"}

    # Import test classes dynamically and run tests
    from tests.test_framework import IntegrationTestCase
    import unittest

    results = {}
    for test_class_name in requested_tests:
        test_class = getattr(sys.modules[__name__], test_class_name, None)
        if test_class is None:
            # Try to import from test_framework
            try:
                test_class = getattr(sys.modules['tests.test_framework'], test_class_name)
            except AttributeError:
                print(f"  ❌ Test class not found: {test_class_name}")
                continue

        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)

        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        results[test_class_name] = {
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
        }

    return results


async def generate_test_report(results: dict, output_file: str = None):
    """Generate comprehensive test report"""
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_tests": sum(r["tests_run"] for r in results.values()),
            "total_failures": sum(r["failures"] for r in results.values()),
            "total_errors": sum(r["errors"] for r in results.values()),
        },
        "detailed_results": results
    }

    # Calculate overall success rate
    total_tests = report["summary"]["total_tests"]
    total_failed = report["summary"]["total_failures"] + report["summary"]["total_errors"]
    report["summary"]["success_rate"] = ((total_tests - total_failed) / total_tests * 100) if total_tests > 0 else 0

    # Print summary
    print(f"\n{'='*80}")
    print("INTEGRATION TEST REPORT")
    print('='*80)

    print(f"Generated: {report['generated_at']}")
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Failures: {report['summary']['total_failures']}")
    print(f"Errors: {report['summary']['total_errors']}")
    print(f"Overall Success Rate: {report['summary']['success_rate']:.1f}%")

    print(f"\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result["failures"] == 0 and result["errors"] == 0 else "❌ FAIL"
        print(f"  {test_name}: {status} ({result['tests_run']} tests, {result['success_rate']:.1%} success)")

    # Save report to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📊 Detailed report saved to: {output_file}")

    return report


async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Manufacturing Knowledge Base Integration Test Runner")
    parser.add_argument("--config", "-c",
                       default="config/integrations.yaml",
                       help="Path to integration configuration file")
    parser.add_argument("--tests", "-t",
                       nargs="+",
                       choices=["langchain", "lobechat", "xagent", "langfuse", "manager", "performance", "all"],
                       default=["all"],
                       help="Test types to run")
    parser.add_argument("--health-check", "-hc",
                       action="store_true",
                       help="Run health checks before tests")
    parser.add_argument("--report", "-r",
                       help="Output file for test report (JSON format)")
    parser.add_argument("--verbose", "-v",
                       action="count",
                       default=1,
                       help="Increase verbosity (use -vv for extra verbose)")
    parser.add_argument("--no-health-check",
                       action="store_true",
                       help="Skip health checks")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    print("🚀 Manufacturing Knowledge Base - Integration Test Runner")
    print("=" * 70)

    # Verify config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("Please ensure the integration configuration file exists.")
        return 1

    # Run health checks if requested
    if args.health_check and not args.no_health_check:
        health_ok = await run_health_checks(str(config_path))
        if not health_ok:
            print("❌ Health checks failed. Consider fixing issues before running tests.")
            response = input("Continue with tests anyway? (y/N): ")
            if response.lower() != 'y':
                return 1

    # Determine which tests to run
    test_types = args.tests
    if "all" in test_types:
        test_types = ["langchain", "lobechat", "xagent", "langfuse", "manager", "performance"]

    # Run tests
    try:
        if len(test_types) == 1 and test_types[0] in ["langchain", "lobechat", "xagent", "langfuse", "manager", "performance"]:
            # Run specific test suite
            results = await run_specific_tests(test_types, str(config_path))
        else:
            # Run all tests
            print("🧪 Running comprehensive integration tests...")
            results = await run_integration_tests()

        # Generate and display report
        report = await generate_test_report(results, args.report)

        # Return appropriate exit code
        total_failures = report["summary"]["total_failures"] + report["summary"]["total_errors"]
        return 1 if total_failures > 0 else 0

    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Test execution failed: {str(e)}")
        if args.verbose > 1:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)