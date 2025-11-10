#!/usr/bin/env python3
"""
File Scanner for New Files Monitoring
Watches specified folders for new or unprocessed files and logs them to database.
"""

import os
import sys
import sqlite3
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set
import hashlib

# Configuration
WATCH_FOLDERS = [
    '/Users/gavin/Nutstore Files/.symlinks/坚果云/005-询盘询价',
    '/Users/gavin/Nutstore Files/002-重要客户'
]

SUPPORTED_EXTENSIONS = {
    '.pdf': 'PDF',
    '.xlsx': 'Excel',
    '.xls': 'Excel',
    '.png': 'Image',
    '.jpg': 'Image',
    '.jpeg': 'Image'
}

DATABASE_PATH = '/Users/gavin/Knowledge base/database/trading_company.db'

class FileScanner:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.processed_files: Set[str] = set()
        self.ensure_database()
        self.load_processed_files()

    def ensure_database(self):
        """Ensure database and table exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create file_processing_log table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_processing_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_type TEXT NOT NULL,
                    discovered_at DATETIME NOT NULL,
                    status TEXT DEFAULT 'pending',
                    file_hash TEXT,
                    file_size INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    processed_at DATETIME,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0
                )
            ''')

            # Create indexes for better performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_file_processing_log_status
                ON file_processing_log(status)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_file_processing_log_discovered_at
                ON file_processing_log(discovered_at)
            ''')

            conn.commit()
            conn.close()
            print("✅ Database and table ensured successfully")

        except Exception as e:
            print(f"❌ Error ensuring database: {e}")
            sys.exit(1)

    def load_processed_files(self):
        """Load already processed files from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT file_path FROM file_processing_log')
            self.processed_files = {row[0] for row in cursor.fetchall()}

            conn.close()
            print(f"📂 Loaded {len(self.processed_files)} previously processed files")

        except Exception as e:
            print(f"❌ Error loading processed files: {e}")

    def get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file for change detection."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""

    def scan_folder(self, folder_path: str) -> List[Dict]:
        """Scan a single folder for new files."""
        new_files = []

        if not os.path.exists(folder_path):
            print(f"⚠️  Folder does not exist: {folder_path}")
            return new_files

        try:
            folder = Path(folder_path)
            print(f"🔍 Scanning folder: {folder_path}")

            # Walk through all files in folder and subfolders
            for file_path in folder.rglob('*'):
                if file_path.is_file():
                    file_ext = file_path.suffix.lower()

                    # Check if file type is supported
                    if file_ext in SUPPORTED_EXTENSIONS:
                        full_path = str(file_path)

                        # Check if file is already processed
                        if full_path not in self.processed_files:
                            file_size = file_path.stat().st_size
                            file_hash = self.get_file_hash(full_path)

                            new_files.append({
                                'file_path': full_path,
                                'file_type': SUPPORTED_EXTENSIONS[file_ext],
                                'discovered_at': datetime.now(),
                                'file_hash': file_hash,
                                'file_size': file_size
                            })

                            print(f"📄 Found new file: {full_path}")

        except Exception as e:
            print(f"❌ Error scanning folder {folder_path}: {e}")

        return new_files

    def log_new_files(self, new_files: List[Dict]):
        """Log new files to database."""
        if not new_files:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for file_info in new_files:
                cursor.execute('''
                    INSERT OR REPLACE INTO file_processing_log
                    (file_path, file_type, discovered_at, file_hash, file_size, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    file_info['file_path'],
                    file_info['file_type'],
                    file_info['discovered_at'],
                    file_info['file_hash'],
                    file_info['file_size'],
                    'pending'
                ))

                # Add to processed files set to avoid duplicates in same run
                self.processed_files.add(file_info['file_path'])

            conn.commit()
            conn.close()

            print(f"✅ Logged {len(new_files)} new files to database")

        except Exception as e:
            print(f"❌ Error logging files to database: {e}")

    def scan_all_folders(self) -> int:
        """Scan all configured folders and return count of new files found."""
        total_new_files = 0

        for folder in WATCH_FOLDERS:
            new_files = self.scan_folder(folder)
            if new_files:
                self.log_new_files(new_files)
                total_new_files += len(new_files)

        return total_new_files

    def get_statistics(self) -> Dict:
        """Get current processing statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get counts by status
            cursor.execute('''
                SELECT status, COUNT(*)
                FROM file_processing_log
                GROUP BY status
            ''')
            status_counts = dict(cursor.fetchall())

            # Get counts by file type
            cursor.execute('''
                SELECT file_type, COUNT(*)
                FROM file_processing_log
                GROUP BY file_type
            ''')
            type_counts = dict(cursor.fetchall())

            # Get total files
            cursor.execute('SELECT COUNT(*) FROM file_processing_log')
            total_files = cursor.fetchone()[0]

            conn.close()

            return {
                'total_files': total_files,
                'by_status': status_counts,
                'by_type': type_counts
            }

        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            return {}

    def watch_continuously(self, interval: int = 60):
        """Watch folders continuously, scanning at specified intervals."""
        print(f"👁️  Starting continuous file watching (interval: {interval}s)")
        print(f"📁 Watching {len(WATCH_FOLDERS)} folders:")
        for i, folder in enumerate(WATCH_FOLDERS, 1):
            print(f"   {i}. {folder}")

        try:
            while True:
                print(f"\n🕐 Scanning at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                new_files_count = self.scan_all_folders()

                if new_files_count > 0:
                    print(f"🎉 Found {new_files_count} new files!")
                else:
                    print("✨ No new files found")

                # Print statistics
                stats = self.get_statistics()
                if stats.get('total_files', 0) > 0:
                    print(f"📊 Database stats: {stats['total_files']} total files")
                    if stats.get('by_status'):
                        status_str = ', '.join([f"{k}: {v}" for k, v in stats['by_status'].items()])
                        print(f"   Status: {status_str}")

                print(f"⏳ Waiting {interval} seconds...")
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n👋 File watching stopped by user")
        except Exception as e:
            print(f"\n❌ Error in continuous watching: {e}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='File Scanner for New Files Monitoring')
    parser.add_argument('--mode', choices=['scan', 'watch'], default='scan',
                       help='Operation mode: scan (once) or watch (continuous)')
    parser.add_argument('--interval', type=int, default=60,
                       help='Interval in seconds for watch mode (default: 60)')
    parser.add_argument('--stats', action='store_true',
                       help='Show current statistics and exit')

    args = parser.parse_args()

    print("🚀 Starting File Scanner")
    print("=" * 50)

    scanner = FileScanner()

    if args.stats:
        # Show statistics
        stats = scanner.get_statistics()
        print("\n📊 Current Statistics:")
        print(f"Total files in database: {stats.get('total_files', 0)}")

        if stats.get('by_status'):
            print("\nBy status:")
            for status, count in stats['by_status'].items():
                print(f"  {status}: {count}")

        if stats.get('by_type'):
            print("\nBy file type:")
            for file_type, count in stats['by_type'].items():
                print(f"  {file_type}: {count}")

    elif args.mode == 'scan':
        # Single scan
        print("🔍 Performing single scan...")
        new_files_count = scanner.scan_all_folders()

        if new_files_count > 0:
            print(f"\n🎉 Scan completed! Found {new_files_count} new files.")
        else:
            print("\n✨ Scan completed! No new files found.")

        # Show final statistics
        stats = scanner.get_statistics()
        print(f"📊 Total files in database: {stats.get('total_files', 0)}")

    elif args.mode == 'watch':
        # Continuous watching
        scanner.watch_continuously(args.interval)


if __name__ == '__main__':
    main()