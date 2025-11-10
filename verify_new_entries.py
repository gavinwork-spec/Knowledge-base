#!/usr/bin/env python3
"""
Knowledge Base Verification Script
Tests knowledge entries created in the last 24 hours and validates search functionality.
"""

import os
import sys
import sqlite3
import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any
import random
import time

# Configuration
DATABASE_PATH = '/Users/gavin/Knowledge base/database/trading_company.db'
API_BASE_URL = 'http://localhost:8001'  # Adjust to match your API server

class KnowledgeVerifier:
    def __init__(self, db_path: str = DATABASE_PATH, api_url: str = API_BASE_URL):
        self.db_path = db_path
        self.api_url = api_url
        self.verification_results = {
            'total_entries': 0,
            'retrievable_count': 0,
            'missing_fields_count': 0,
            'api_errors': 0,
            'tested_keywords': []
        }

    def get_recent_entries(self, hours: int = 24) -> List[Dict]:
        """Get knowledge entries created in the last specified hours."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Calculate the timestamp for 24 hours ago
            cutoff_time = datetime.now() - timedelta(hours=hours)

            cursor.execute('''
                SELECT id, file_path, file_type, extracted_at, attributes_json, raw_text
                FROM knowledge_entries
                WHERE extracted_at >= ?
                ORDER BY extracted_at DESC
            ''', (cutoff_time,))

            entries = []
            for row in cursor.fetchall():
                try:
                    attributes = json.loads(row[4])
                except json.JSONDecodeError:
                    attributes = {}

                entries.append({
                    'id': row[0],
                    'file_path': row[1],
                    'file_type': row[2],
                    'extracted_at': row[3],
                    'attributes': attributes,
                    'raw_text': row[5]
                })

            conn.close()
            return entries

        except Exception as e:
            print(f"❌ Error getting recent entries: {e}")
            return []

    def extract_product_keywords(self, entries: List[Dict]) -> List[str]:
        """Extract product keywords from knowledge entries for testing."""
        keywords = set()

        for entry in entries:
            attributes = entry.get('attributes', {})

            # Extract from specific fields
            for field in ['product_category', 'customer_needs', 'material', 'customer_name']:
                if field in attributes and attributes[field]:
                    # Split into individual words/phrases
                    text = str(attributes[field])
                    words = text.split()
                    for word in words:
                        # Add meaningful words (length > 2)
                        if len(word) > 2 and word.isalpha():
                            keywords.add(word.lower())

            # Extract common product terms
            raw_text = entry.get('raw_text', '').lower()
            product_terms = [
                'screw', 'bolt', 'nut', 'washer', 'fastener',
                '不锈钢', '螺钉', '螺栓', '螺母', '垫片',
                'steel', 'stainless', 'carbon', 'brass',
                'M4', 'M5', 'M6', 'M8', 'M10', 'M12',
                '304', '316', '4.8', '8.8', '12.9'
            ]

            for term in product_terms:
                if term in raw_text:
                    keywords.add(term)

        return list(keywords)

    def test_search_api(self, query: str) -> Dict:
        """Test the knowledge search API with a given query."""
        try:
            search_url = f"{self.api_url}/api/v1/knowledge/search"
            params = {'query': query, 'limit': 10}

            response = requests.get(search_url, params=params, timeout=10)

            if response.status_code == 200:
                return {
                    'success': True,
                    'status_code': response.status_code,
                    'results': response.json(),
                    'query': query
                }
            else:
                return {
                    'success': False,
                    'status_code': response.status_code,
                    'error': response.text,
                    'query': query
                }

        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e),
                'query': query
            }

    def check_missing_fields(self, entries: List[Dict]) -> List[Dict]:
        """Check for entries missing expected fields."""
        required_fields = ['customer_name', 'product_category', 'material', 'quantity']
        entries_with_missing = []

        for entry in entries:
            attributes = entry.get('attributes', {})
            missing = []

            for field in required_fields:
                if field not in attributes or not attributes[field]:
                    missing.append(field)

            if missing:
                entries_with_missing.append({
                    'id': entry['id'],
                    'file_path': entry['file_path'],
                    'missing_fields': missing,
                    'confidence_score': attributes.get('confidence_score', 0)
                })

        return entries_with_missing

    def verify_entries(self, max_tests: int = 20) -> Dict:
        """Main verification function."""
        print("🔍 Starting Knowledge Base Verification")
        print("=" * 50)

        # Get recent entries
        print("📋 Getting recent entries...")
        entries = self.get_recent_entries()
        self.verification_results['total_entries'] = len(entries)

        print(f"📊 Found {len(entries)} entries from the last 24 hours")

        if not entries:
            print("⚠️  No recent entries found to verify")
            return self.verification_results

        # Check for missing fields
        print("🔍 Checking for missing required fields...")
        missing_fields_entries = self.check_missing_fields(entries)
        self.verification_results['missing_fields_count'] = len(missing_fields_entries)

        if missing_fields_entries:
            print(f"⚠️  Found {len(missing_fields_entries)} entries with missing fields:")
            for entry in missing_fields_entries[:5]:  # Show first 5
                print(f"   - {entry['file_path']}: Missing {', '.join(entry['missing_fields'])}")

        # Extract keywords for testing
        print("🔤 Extracting keywords for search testing...")
        keywords = self.extract_product_keywords(entries)
        print(f"📝 Extracted {len(keywords)} unique keywords")

        # Test search functionality
        print("🔎 Testing search API functionality...")

        # Select random keywords for testing
        test_keywords = random.sample(keywords, min(max_tests, len(keywords))) if keywords else []
        self.verification_results['tested_keywords'] = test_keywords

        successful_searches = 0

        for i, keyword in enumerate(test_keywords, 1):
            print(f"   Test {i}/{len(test_keywords)}: Searching for '{keyword}'")

            result = self.test_search_api(keyword)

            if result['success']:
                successful_searches += 1
                results_count = len(result.get('results', []))
                print(f"      ✅ Found {results_count} results")
            else:
                self.verification_results['api_errors'] += 1
                print(f"      ❌ Error: {result.get('error', 'Unknown error')}")

            # Small delay between requests
            time.sleep(0.5)

        self.verification_results['retrievable_count'] = successful_searches

        # Calculate success rate
        success_rate = (successful_searches / len(test_keywords)) * 100 if test_keywords else 0

        print(f"\n📈 Search API Success Rate: {success_rate:.1f}% ({successful_searches}/{len(test_keywords)})")

        return self.verification_results

    def generate_report(self, results: Dict) -> str:
        """Generate a detailed verification report."""
        report = []
        report.append("📊 KNOWLEDGE BASE VERIFICATION REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary
        report.append("📋 SUMMARY")
        report.append("-" * 20)
        report.append(f"Total entries (24h): {results['total_entries']}")
        report.append(f"Entries with missing fields: {results['missing_fields_count']}")
        report.append(f"Search tests performed: {len(results['tested_keywords'])}")
        report.append(f"Successful searches: {results['retrievable_count']}")
        report.append(f"API errors: {results['api_errors']}")
        report.append("")

        # Success rates
        if results['tested_keywords']:
            search_success_rate = (results['retrievable_count'] / len(results['tested_keywords'])) * 100
            report.append("📈 SUCCESS RATES")
            report.append("-" * 20)
            report.append(f"Search API success rate: {search_success_rate:.1f}%")

            if results['total_entries'] > 0:
                completeness_rate = ((results['total_entries'] - results['missing_fields_count']) / results['total_entries']) * 100
                report.append(f"Data completeness rate: {completeness_rate:.1f}%")
        report.append("")

        # Tested keywords
        if results['tested_keywords']:
            report.append("🔤 TESTED KEYWORDS")
            report.append("-" * 20)
            for i, keyword in enumerate(results['tested_keywords'], 1):
                report.append(f"{i:2d}. {keyword}")
        report.append("")

        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 20)

        if results['missing_fields_count'] > 0:
            report.append("⚠️  Consider improving extraction patterns for missing fields")

        if results['api_errors'] > 0:
            report.append("⚠️  Investigate API errors and server connectivity")

        search_success_rate = (results['retrievable_count'] / len(results['tested_keywords'])) * 100 if results['tested_keywords'] else 100
        if search_success_rate < 80:
            report.append("⚠️  Search API success rate is below 80% - investigate indexing issues")
        elif search_success_rate >= 95:
            report.append("✅ Search API is performing well")

        completeness_rate = ((results['total_entries'] - results['missing_fields_count']) / results['total_entries']) * 100 if results['total_entries'] > 0 else 100
        if completeness_rate >= 90:
            report.append("✅ Data extraction completeness is excellent")
        elif completeness_rate >= 70:
            report.append("✅ Data extraction completeness is good")
        else:
            report.append("⚠️  Data extraction completeness needs improvement")

        return "\n".join(report)

    def save_report(self, results: Dict, filename: str = None):
        """Save verification report to file."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"knowledge_verification_report_{timestamp}.txt"

        report = self.generate_report(results)

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 Report saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")

        return filename


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Knowledge Base Verification Script')
    parser.add_argument('--db-path', type=str, default=DATABASE_PATH,
                       help='Path to SQLite database')
    parser.add_argument('--api-url', type=str, default=API_BASE_URL,
                       help='Base URL for knowledge API')
    parser.add_argument('--hours', type=int, default=24,
                       help='Hours to look back for recent entries')
    parser.add_argument('--max-tests', type=int, default=20,
                       help='Maximum number of search tests to perform')
    parser.add_argument('--save-report', action='store_true',
                       help='Save verification report to file')
    parser.add_argument('--report-file', type=str,
                       help='Custom filename for the report')

    args = parser.parse_args()

    print("🚀 Starting Knowledge Base Verification")
    print(f"📁 Database: {args.db_path}")
    print(f"🌐 API URL: {args.api_url}")
    print(f"⏰ Time range: Last {args.hours} hours")
    print("=" * 50)

    verifier = KnowledgeVerifier(args.db_path, args.api_url)
    results = verifier.verify_entries(args.max_tests)

    # Print summary
    print("\n" + "=" * 50)
    print("🎯 VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"📊 Total entries: {results['total_entries']}")
    print(f"🔍 Missing fields: {results['missing_fields_count']}")
    print(f"🔎 Search success: {results['retrievable_count']}/{len(results['tested_keywords'])}")

    if results['tested_keywords']:
        success_rate = (results['retrievable_count'] / len(results['tested_keywords'])) * 100
        print(f"📈 Success rate: {success_rate:.1f}%")

    print(f"❌ API errors: {results['api_errors']}")

    # Generate and save report if requested
    if args.save_report:
        report_file = verifier.save_report(results, args.report_file)
        print(f"📄 Detailed report saved to: {report_file}")

    # Return appropriate exit code
    if results['api_errors'] > 0:
        print("\n⚠️  Verification completed with API errors")
        sys.exit(1)
    elif results['missing_fields_count'] > results['total_entries'] * 0.5:
        print("\n⚠️  Verification completed with many missing fields")
        sys.exit(1)
    else:
        print("\n✅ Verification completed successfully")
        sys.exit(0)


if __name__ == '__main__':
    main()