#!/usr/bin/env python3
"""
Document Parser for Trading Company Files
Extracts key information from PDF and Excel files and stores in knowledge base.
"""

import os
import sys
import sqlite3
import json
import re
import pandas as pd
import fitz  # PyMuPDF
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Configuration
DATABASE_PATH = '/Users/gavin/Knowledge base/database/trading_company.db'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_parser.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class DocumentParser:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.ensure_database()

    def ensure_database(self):
        """Ensure knowledge_entries table exists."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create knowledge_entries table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    extracted_at DATETIME NOT NULL,
                    attributes_json TEXT NOT NULL,
                    raw_text TEXT,
                    confidence_score REAL,
                    processing_status TEXT DEFAULT 'completed',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(file_path)
                )
            ''')

            # Create indexes
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_knowledge_entries_file_path
                ON knowledge_entries(file_path)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_knowledge_entries_extracted_at
                ON knowledge_entries(extracted_at)
            ''')

            conn.commit()
            conn.close()
            logging.info("✅ Database schema ensured successfully")

        except Exception as e:
            logging.error(f"❌ Error ensuring database: {e}")
            sys.exit(1)

    def get_next_pending_file(self) -> Optional[Dict]:
        """Get the next pending file from file_processing_log."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, file_path, file_type, discovered_at
                FROM file_processing_log
                WHERE status = 'pending'
                ORDER BY discovered_at ASC
                LIMIT 1
            ''')

            result = cursor.fetchone()
            conn.close()

            if result:
                return {
                    'id': result[0],
                    'file_path': result[1],
                    'file_type': result[2],
                    'discovered_at': result[3]
                }
            return None

        except Exception as e:
            logging.error(f"❌ Error getting pending file: {e}")
            return None

    def extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF using PyMuPDF."""
        try:
            text = ""
            with fitz.open(file_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text += page.get_text()
            return text
        except Exception as e:
            logging.error(f"❌ Error extracting PDF text: {e}")
            return ""

    def extract_excel_data(self, file_path: str) -> pd.DataFrame:
        """Extract data from Excel file using pandas."""
        try:
            # Try to read the first sheet
            df = pd.read_excel(file_path, sheet_name=0)
            return df
        except Exception as e:
            logging.error(f"❌ Error reading Excel file: {e}")
            return pd.DataFrame()

    def extract_text_from_excel(self, file_path: str) -> str:
        """Extract text from Excel file."""
        try:
            text_parts = []
            # Read all sheets
            with pd.ExcelFile(file_path) as xls:
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    # Convert all cells to text
                    for column in df.columns:
                        for value in df[column]:
                            if pd.notna(value):
                                text_parts.append(str(value))
                    # Add sheet separator
                    text_parts.append(f"\n--- Sheet: {sheet_name} ---\n")

            return " ".join(text_parts)
        except Exception as e:
            logging.error(f"❌ Error extracting Excel text: {e}")
            return ""

    def extract_key_fields(self, text: str, file_type: str) -> Dict[str, Any]:
        """Extract key fields from document text."""
        if not text:
            return {}

        extracted = {}

        # Common patterns for trading documents
        patterns = {
            # Customer name patterns
            'customer_name': [
                r'customer[:\s]+([^\n]+)',
                r'client[:\s]+([^\n]+)',
                r'公司[:\s]+([^\n]+)',
                r'company[:\s]+([^\n]+)',
                r'from[:\s]+([^\n]+)',
                r'([^@\s]+@(?:[^@\s]+\.)+[^@\s]+)',  # Email addresses
                r'([A-Z][a-zA-Z\s]+(?:Ltd|Inc|Corp|GmbH|AB|SAS|SRL|BV|NV|SpA))',
            ],

            # Product category patterns
            'product_category': [
                r'product[:\s]+([^\n]+)',
                r'item[:\s]+([^\n]+)',
                r'产品[:\s]+([^\n]+)',
                r'screw|bolt|nut|washer|fastener|螺钉|螺栓|螺母|垫片',
            ],

            # Material patterns
            'material': [
                r'material[:\s]+([^\n]+)',
                r'stainless|steel|carbon|brass|尼龙|不锈钢|碳钢|黄铜',
                r'304|316|4\.8|8\.8|12\.9',
                r'A2|A4|Grade\s+\d+',
            ],

            # Quantity patterns
            'quantity': [
                r'qty[:\s]+([\d,]+)',
                r'quantity[:\s]+([\d,]+)',
                r'数量[:\s]+([\d,]+)',
                r'(\d+[,\d]*)\s*(?:pcs|pieces|个|件)',
                r'(\d+)[\s-]*(?:pcs|pc|ea|each)',
            ],

            # Date patterns
            'date': [
                r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
                r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
                r'(\d{1,2}\.\d{1,2}\.\d{4})',
                r'(\d{1,2}/\d{1,2}/\d{2,4})',
                r'(?:date|日期)[:\s]*([^\n]+)',
            ],

            # Price patterns
            'price': [
                r'price[:\s]*\$?([\d,]+\.?\d*)',
                r'cost[:\s]*\$?([\d,]+\.?\d*)',
                r'单价[:\s]*\$?([\d,]+\.?\d*)',
                r'价格[:\s]*\$?([\d,]+\.?\d*)',
                r'\$([\d,]+\.?\d*)',
                r'USD\s*([\d,]+\.?\d*)',
                r'EUR\s*([\d,]+\.?\d*)',
            ],

            # Customer needs/requirements
            'customer_needs': [
                r'requirement[:\s]+([^\n]+)',
                r'specification[:\s]+([^\n]+)',
                r'要求[:\s]+([^\n]+)',
                r'规格[:\s]+([^\n]+)',
                r'standard[:\s]+([^\n]+)',
                r'标准[:\s]+([^\n]+)',
            ],

            # Dimensions/size
            'dimensions': [
                r'(\d+\.?\d*)\s*[x×*]\s*(\d+\.?\d*)\s*[x×*]\s*(\d+\.?\d*)',
                r'M(\d+)[x×*](\d+)',
                r'Ø\s*(\d+\.?\d*)',
                r'dia\s*:?\s*(\d+\.?\d*)',
                r'长度[:\s]*(\d+\.?\d*)',
                r'length[:\s]*(\d+\.?\d*)',
            ],

            # Contact information
            'contact_info': [
                r'tel[:\s]*([^\n]+)',
                r'phone[:\s]*([^\n]+)',
                r'电话[:\s]*([^\n]+)',
                r'(\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4})',
            ]
        }

        # Extract fields using patterns
        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                try:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        # Take the first match and clean it
                        value = matches[0] if isinstance(matches[0], str) else str(matches[0])
                        value = value.strip()
                        if value and len(value) > 1:  # Avoid single characters
                            extracted[field] = value
                            break
                except Exception as e:
                    logging.warning(f"Warning extracting {field}: {e}")

        # Special handling for Excel dataframes
        if file_type == 'Excel':
            try:
                df = self.extract_excel_data(extracted.get('file_path', ''))
                if not df.empty:
                    # Look for common column names in Excel
                    for col in df.columns:
                        col_lower = str(col).lower()
                        if 'customer' in col_lower or 'client' in col_lower:
                            extracted['customer_name'] = df[col].dropna().iloc[0] if not df[col].dropna().empty else extracted.get('customer_name')
                        elif 'product' in col_lower or 'item' in col_lower:
                            extracted['product_category'] = df[col].dropna().iloc[0] if not df[col].dropna().empty else extracted.get('product_category')
                        elif 'quantity' in col_lower or 'qty' in col_lower:
                            extracted['quantity'] = df[col].dropna().iloc[0] if not df[col].dropna().empty else extracted.get('quantity')
                        elif 'price' in col_lower or 'cost' in col_lower:
                            extracted['price'] = df[col].dropna().iloc[0] if not df[col].dropna().empty else extracted.get('price')
                        elif 'material' in col_lower:
                            extracted['material'] = df[col].dropna().iloc[0] if not df[col].dropna().empty else extracted.get('material')
                        elif 'date' in col_lower:
                            extracted['date'] = df[col].dropna().iloc[0] if not df[col].dropna().empty else extracted.get('date')
            except Exception as e:
                logging.warning(f"Warning processing Excel columns: {e}")

        # Calculate confidence score
        confidence_score = len([v for v in extracted.values() if v]) / len(patterns)
        extracted['confidence_score'] = round(confidence_score, 2)

        return extracted

    def process_file(self, file_info: Dict) -> bool:
        """Process a single file and extract information."""
        file_path = file_info['file_path']
        file_type = file_info['file_type']
        file_id = file_info['id']

        logging.info(f"🔍 Processing file: {file_path}")

        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logging.error(f"❌ File not found: {file_path}")
                self.update_file_status(file_id, 'failed', 'File not found')
                return False

            # Extract text based on file type
            if file_type == 'PDF':
                raw_text = self.extract_pdf_text(file_path)
            elif file_type == 'Excel':
                raw_text = self.extract_text_from_excel(file_path)
            else:
                raw_text = ""

            if not raw_text:
                logging.warning(f"⚠️  No text extracted from: {file_path}")
                self.update_file_status(file_id, 'failed', 'No text extracted')
                return False

            # Extract key fields
            extracted_fields = self.extract_key_fields(raw_text, file_type)

            # Prepare attributes JSON
            attributes = {
                'extraction_timestamp': datetime.now().isoformat(),
                'source_file': file_path,
                'file_type': file_type,
                'confidence_score': extracted_fields.get('confidence_score', 0.0),
                **extracted_fields
            }

            # Store in knowledge_entries
            self.store_knowledge_entry(file_path, file_type, attributes, raw_text)

            # Update file processing status
            self.update_file_status(file_id, 'done')

            logging.info(f"✅ Successfully processed: {file_path}")
            return True

        except Exception as e:
            logging.error(f"❌ Error processing file {file_path}: {e}")
            self.update_file_status(file_id, 'failed', str(e))
            return False

    def store_knowledge_entry(self, file_path: str, file_type: str, attributes: Dict, raw_text: str):
        """Store extracted information in knowledge_entries table."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO knowledge_entries
                (file_path, file_type, extracted_at, attributes_json, raw_text, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                file_path,
                file_type,
                datetime.now(),
                json.dumps(attributes, ensure_ascii=False, indent=2),
                raw_text[:10000],  # Limit raw text size
                attributes.get('confidence_score', 0.0)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"❌ Error storing knowledge entry: {e}")
            raise

    def update_file_status(self, file_id: int, status: str, error_message: str = None):
        """Update file processing status."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if status == 'failed' and error_message:
                cursor.execute('''
                    UPDATE file_processing_log
                    SET status = ?, processed_at = ?, error_message = ?,
                        retry_count = retry_count + 1
                    WHERE id = ?
                ''', (status, datetime.now(), error_message, file_id))
            else:
                cursor.execute('''
                    UPDATE file_processing_log
                    SET status = ?, processed_at = ?
                    WHERE id = ?
                ''', (status, datetime.now(), file_id))

            conn.commit()
            conn.close()

        except Exception as e:
            logging.error(f"❌ Error updating file status: {e}")

    def process_all_pending(self, max_files: int = None) -> Dict[str, int]:
        """Process all pending files."""
        stats = {'processed': 0, 'failed': 0, 'total': 0}

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Count pending files
            cursor.execute("SELECT COUNT(*) FROM file_processing_log WHERE status = 'pending'")
            total_pending = cursor.fetchone()[0]
            stats['total'] = total_pending

            if max_files:
                total_pending = min(total_pending, max_files)

            conn.close()

            logging.info(f"📊 Found {total_pending} pending files to process")

            for i in range(total_pending):
                file_info = self.get_next_pending_file()
                if file_info:
                    if self.process_file(file_info):
                        stats['processed'] += 1
                    else:
                        stats['failed'] += 1

                    # Progress update
                    progress = (i + 1) / total_pending * 100
                    logging.info(f"📈 Progress: {progress:.1f}% ({i + 1}/{total_pending})")

            logging.info(f"🎉 Processing completed! Processed: {stats['processed']}, Failed: {stats['failed']}")

        except Exception as e:
            logging.error(f"❌ Error in batch processing: {e}")

        return stats

    def show_statistics(self):
        """Show processing statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # File processing stats
            cursor.execute('''
                SELECT status, COUNT(*)
                FROM file_processing_log
                GROUP BY status
            ''')
            file_stats = dict(cursor.fetchall())

            # Knowledge entries stats
            cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
            knowledge_count = cursor.fetchone()[0]

            # Average confidence score
            cursor.execute("SELECT AVG(confidence_score) FROM knowledge_entries")
            avg_confidence = cursor.fetchone()[0] or 0

            conn.close()

            print("\n📊 Processing Statistics:")
            print(f"Files in database: {sum(file_stats.values())}")
            print(f"  Pending: {file_stats.get('pending', 0)}")
            print(f"  Done: {file_stats.get('done', 0)}")
            print(f"  Failed: {file_stats.get('failed', 0)}")
            print(f"Knowledge entries: {knowledge_count}")
            print(f"Average confidence: {avg_confidence:.2f}")

        except Exception as e:
            logging.error(f"❌ Error getting statistics: {e}")

    def export_knowledge_base(self, output_file: str):
        """Export knowledge base to JSON file."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT file_path, file_type, extracted_at, attributes_json, raw_text
                FROM knowledge_entries
                ORDER BY extracted_at DESC
            ''')

            entries = []
            for row in cursor.fetchall():
                entries.append({
                    'file_path': row[0],
                    'file_type': row[1],
                    'extracted_at': row[2],
                    'attributes': json.loads(row[3]),
                    'raw_text': row[4][:500] + "..." if row[4] and len(row[4]) > 500 else row[4]
                })

            conn.close()

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)

            logging.info(f"✅ Knowledge base exported to: {output_file}")

        except Exception as e:
            logging.error(f"❌ Error exporting knowledge base: {e}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Document Parser for Trading Company Files')
    parser.add_argument('--mode', choices=['process', 'stats', 'export'], default='process',
                       help='Operation mode: process (parse files), stats (show statistics), export (export knowledge base)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--export-file', type=str, default='knowledge_base_export.json',
                       help='Output file for export mode')

    args = parser.parse_args()

    print("🚀 Starting Document Parser")
    print("=" * 50)

    parser = DocumentParser()

    if args.mode == 'process':
        print("🔍 Starting to process pending files...")
        stats = parser.process_all_pending(args.max_files)

        print(f"\n📈 Processing Summary:")
        print(f"Total processed: {stats['processed']}")
        print(f"Failed: {stats['failed']}")
        print(f"Remaining: {stats['total'] - stats['processed'] - stats['failed']}")

    elif args.mode == 'stats':
        parser.show_statistics()

    elif args.mode == 'export':
        parser.export_knowledge_base(args.export_file)


if __name__ == '__main__':
    main()