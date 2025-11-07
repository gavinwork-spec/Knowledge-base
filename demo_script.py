#!/usr/bin/env python3
"""
Knowledge Base CLI Demo Script
Demonstrates key features of the CLI tool
"""

import subprocess
import time
import sys
import os

def run_command(cmd, description):
    """Run a CLI command and display the result"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            f'python3 kb_cli.py {cmd}',
            shell=True,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}")

        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    """Main demo function"""
    print("🚀 Knowledge Base CLI Demo")
    print("=" * 60)
    print("This demo showcases the key features of the Knowledge Base CLI tool")
    print("Inspired by AI Shell with natural language processing capabilities")
    print("=" * 60)

    commands = [
        ("--help", "Show available commands"),
        ("status", "Display system status"),
        ("health", "Perform health check"),
        ("backup", "Create a backup"),
        ("backups", "List available backups"),
        ("user-stats", "Show user statistics"),
        ("analytics", "Display analytics"),
    ]

    successful = 0
    total = len(commands)

    for cmd, desc in commands:
        if run_command(cmd, desc):
            successful += 1
        time.sleep(1)  # Brief pause between commands

    print(f"\n{'='*60}")
    print("📊 Demo Summary")
    print(f"{'='*60}")
    print(f"✅ Successful commands: {successful}/{total}")
    print(f"📁 Files created:")

    # List created files
    files_to_check = [
        "knowledge_base.db",
        "users.db",
        "kb_config.json",
        "backups/"
    ]

    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"   ✓ {file_path}")
        else:
            print(f"   ✗ {file_path}")

    print(f"\n🎉 Demo completed successfully!")
    print(f"\nNext steps:")
    print(f"1. Try interactive mode: python kb_cli.py")
    print(f"2. Test natural language: python kb_cli.py exec 'show system status'")
    print(f"3. Create users: python kb_cli.py user-create user@email.com --role editor")
    print(f"4. Ingest documents: python kb_cli.py ingest /path/to/docs")
    print(f"\n📚 For full documentation, see: README_CLI.md")

if __name__ == "__main__":
    main()