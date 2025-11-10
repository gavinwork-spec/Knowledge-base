#!/usr/bin/env python3
"""
Trading Company Database Migration Runner
Version: 1.0
Date: 2025-01-10
Description: Executes database migrations for trading company schema
"""

import sqlite3
import os
import sys
from datetime import datetime
import shutil

# Database paths
DB_PATH = "/Users/gavin/Knowledge base/data/db.sqlite"
BACKUP_DIR = "/Users/gavin/Knowledge base/data/backups"
MIGRATIONS_DIR = "/Users/gavin/Knowledge base/database/migrations"

def create_backup():
    """Create a backup of the current database"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{BACKUP_DIR}/db_trading_migration_{timestamp}.sqlite"

    try:
        shutil.copy2(DB_PATH, backup_file)
        print(f"✅ Database backup created: {backup_file}")
        return backup_file
    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return None

def execute_migration(migration_file):
    """Execute a single migration file"""
    try:
        with open(migration_file, 'r', encoding='utf-8') as f:
            migration_sql = f.read()

        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")

        # Execute migration
        cursor.executescript(migration_sql)
        conn.commit()

        print(f"✅ Migration executed successfully: {migration_file}")
        return True

    except sqlite3.Error as e:
        print(f"❌ SQLite error in {migration_file}: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"❌ Error executing {migration_file}: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def check_database_integrity():
    """Check database integrity after migration"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Check integrity
        cursor.execute("PRAGMA integrity_check")
        integrity_result = cursor.fetchone()

        # Check foreign key constraints
        cursor.execute("PRAGMA foreign_key_check")
        fk_violations = cursor.fetchall()

        conn.close()

        if integrity_result[0] == "ok" and len(fk_violations) == 0:
            print("✅ Database integrity check passed")
            return True
        else:
            print(f"❌ Database integrity issues found:")
            print(f"Integrity: {integrity_result[0]}")
            if fk_violations:
                print(f"Foreign key violations: {len(fk_violations)}")
                for violation in fk_violations:
                    print(f"  - {violation}")
            return False

    except Exception as e:
        print(f"❌ Error checking database integrity: {e}")
        return False

def get_migration_status():
    """Get current migration status"""
    try:
        # Check if migrations table exists
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='migration_history'
        """)
        table_exists = cursor.fetchone()

        if not table_exists:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS migration_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    migration_file TEXT NOT NULL,
                    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT 1
                )
            """)
            conn.commit()

        # Get executed migrations
        cursor.execute("SELECT migration_file FROM migration_history ORDER BY executed_at")
        executed_migrations = [row[0] for row in cursor.fetchall()]

        conn.close()
        return executed_migrations

    except Exception as e:
        print(f"❌ Error getting migration status: {e}")
        return []

def record_migration(migration_file, success=True):
    """Record migration execution in history"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO migration_history (migration_file, success)
            VALUES (?, ?)
        """, (os.path.basename(migration_file), success))

        conn.commit()
        conn.close()

    except Exception as e:
        print(f"❌ Error recording migration: {e}")

def print_database_stats():
    """Print database statistics after migration"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get table counts
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        print("\n📊 Database Statistics:")
        print("=" * 50)

        total_records = 0
        for table in tables:
            table_name = table[0]
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                total_records += count
                print(f"{table_name:<30}: {count:>10} records")
            except sqlite3.Error:
                print(f"{table_name:<30}: {'Error':>10}")

        print("=" * 50)
        print(f"{'Total':<30}: {total_records:>10} records")

        conn.close()

    except Exception as e:
        print(f"❌ Error getting database stats: {e}")

def main():
    """Main migration function"""
    print("🚀 Trading Company Database Migration")
    print("=" * 50)

    # Check if database exists
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at: {DB_PATH}")
        return False

    # Create backup
    backup_file = create_backup()
    if not backup_file:
        print("❌ Cannot proceed without backup")
        return False

    # Get migration status
    executed_migrations = get_migration_status()

    # List migration files
    migration_files = [
        "001_trading_company_schema.sql",
        "002_seed_trading_data.sql"
    ]

    migrations_to_run = []
    for migration_file in migration_files:
        full_path = os.path.join(MIGRATIONS_DIR, migration_file)
        if os.path.exists(full_path) and migration_file not in executed_migrations:
            migrations_to_run.append(full_path)

    if not migrations_to_run:
        print("✅ All migrations are up to date")
        return True

    print(f"\n📝 Migrations to execute: {len(migrations_to_run)}")
    for migration in migrations_to_run:
        print(f"  - {os.path.basename(migration)}")

    # Execute migrations
    print(f"\n🔧 Executing migrations...")
    success_count = 0

    for migration_file in migrations_to_run:
        print(f"\n▶️  Executing: {os.path.basename(migration_file)}")

        success = execute_migration(migration_file)
        if success:
            record_migration(migration_file, True)
            success_count += 1
        else:
            record_migration(migration_file, False)
            print(f"❌ Migration failed: {migration_file}")
            print(f"📦 Please restore from backup: {backup_file}")
            return False

    # Check integrity
    print(f"\n🔍 Checking database integrity...")
    if not check_database_integrity():
        print(f"❌ Database integrity check failed")
        print(f"📦 Please restore from backup: {backup_file}")
        return False

    # Print statistics
    print_database_stats()

    print(f"\n🎉 Migration completed successfully!")
    print(f"✅ {success_count} migrations executed")
    print(f"📦 Backup available at: {backup_file}")

    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)