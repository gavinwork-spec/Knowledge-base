# File Scanner for New Files Monitoring

## 📋 Overview

`scan_new_files.py` is a Python script that watches specified folders for new or unprocessed files and logs them to a SQLite database. It's designed to monitor trading company folders for incoming inquiry documents, quotes, and drawings.

## 🎯 Features

- **Multi-folder monitoring**: Watch multiple folders simultaneously
- **File type filtering**: Supports PDF, Excel, and Image files (PNG/JPG/JPEG)
- **Duplicate detection**: Uses file paths and MD5 hashes to avoid duplicates
- **Database logging**: Stores file metadata in SQLite database
- **Two operation modes**: Single scan or continuous watching
- **Statistics reporting**: Track processing status and file types

## 📁 Supported File Types

| Extension | Type | Description |
|-----------|------|-------------|
| `.pdf` | PDF | Portable Document Format files |
| `.xlsx`, `.xls` | Excel | Microsoft Excel spreadsheets |
| `.png`, `.jpg`, `.jpeg` | Image | Image files (photos, screenshots, etc.) |

## 🗂️ Monitored Folders

By default, the script monitors these folders:

1. `/Users/gavin/Nutstore Files/.symlinks/坚果云/005-询盘询价` - Inquiry and pricing folder
2. `/Users/gavin/Nutstore Files/002-重要客户` - Important customers folder

## 🗃️ Database Schema

The script creates a `file_processing_log` table with the following columns:

```sql
CREATE TABLE file_processing_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,           -- Full path to the file
    file_type TEXT NOT NULL,                  -- File type (PDF, Excel, Image)
    discovered_at DATETIME NOT NULL,          -- When the file was discovered
    status TEXT DEFAULT 'pending',            -- Processing status
    file_hash TEXT,                           -- MD5 hash for change detection
    file_size INTEGER,                        -- File size in bytes
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    processed_at DATETIME,                    -- When processing was completed
    error_message TEXT,                       -- Error details if processing failed
    retry_count INTEGER DEFAULT 0             -- Number of processing attempts
);
```

## 🚀 Usage

### Basic Commands

```bash
# Show help
python3 scan_new_files.py --help

# Single scan (find new files once)
python3 scan_new_files.py --mode scan

# Show current statistics
python3 scan_new_files.py --stats

# Continuous monitoring (scans every 60 seconds)
python3 scan_new_files.py --mode watch

# Continuous monitoring with custom interval (30 seconds)
python3 scan_new_files.py --mode watch --interval 30
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Operation mode: `scan` (once) or `watch` (continuous) | `scan` |
| `--interval` | Scan interval in seconds for watch mode | `60` |
| `--stats` | Show current statistics and exit | - |

## 📊 Statistics

The script provides detailed statistics including:

- **Total files**: Overall number of files in database
- **By status**: Files grouped by processing status (pending, processed, error)
- **By file type**: Files grouped by type (PDF, Excel, Image)

Example output:
```
📊 Current Statistics:
Total files in database: 2788

By status:
  pending: 2788

By file type:
  Excel: 591
  Image: 1733
  PDF: 464
```

## 🔧 Configuration

### Changing Monitored Folders

Edit the `WATCH_FOLDERS` list in the script:

```python
WATCH_FOLDERS = [
    '/path/to/your/first/folder',
    '/path/to/your/second/folder',
    # Add more folders as needed
]
```

### Changing Database Location

Edit the `DATABASE_PATH` variable:

```python
DATABASE_PATH = '/path/to/your/database.db'
```

### Adding New File Types

Edit the `SUPPORTED_EXTENSIONS` dictionary:

```python
SUPPORTED_EXTENSIONS = {
    '.pdf': 'PDF',
    '.xlsx': 'Excel',
    '.xls': 'Excel',
    '.png': 'Image',
    '.jpg': 'Image',
    '.jpeg': 'Image',
    '.docx': 'Word',      # Add new file type
    '.txt': 'Text',       # Add another file type
}
```

## 🔄 Continuous Monitoring

For production use, you can run the script in continuous monitoring mode:

```bash
# Start continuous monitoring
python3 scan_new_files.py --mode watch --interval 300

# Run in background (Linux/macOS)
nohup python3 scan_new_files.py --mode watch --interval 300 > scanner.log 2>&1 &

# Run in background (Windows)
start /B python3 scan_new_files.py --mode watch --interval 300
```

## 🐛 Troubleshooting

### Common Issues

1. **Permission denied**: Make sure the script has read access to the monitored folders
2. **Database locked**: Ensure only one instance of the script is running at a time
3. **Folder not found**: Check that the folder paths in `WATCH_FOLDERS` are correct

### Debug Mode

For debugging, you can add print statements or check the database directly:

```bash
# Check database content
sqlite3 "/Users/gavin/Knowledge base/database/trading_company.db"
.tables
SELECT * FROM file_processing_log LIMIT 10;
```

## 📈 Integration with Trading Workflow

This script is part of a larger trading company automation system:

1. **File Discovery**: Scanner detects new inquiry files
2. **Processing**: Other scripts process files based on their status
3. **Analysis**: Files are analyzed for customer inquiries, pricing requests, etc.
4. **Response**: Automated or manual responses are generated

### Processing Workflow

Files marked as `pending` can be processed by other scripts:

```python
-- Get pending Excel files
SELECT * FROM file_processing_log
WHERE status = 'pending' AND file_type = 'Excel'
ORDER BY discovered_at ASC;

-- Update file status after processing
UPDATE file_processing_log
SET status = 'processed', processed_at = CURRENT_TIMESTAMP
WHERE file_path = '/path/to/file.xlsx';
```

## 📝 Development Notes

- **Python Version**: Compatible with Python 3.6+
- **Dependencies**: Uses only Python standard library (sqlite3, pathlib, hashlib, etc.)
- **Performance**: Efficient file scanning with MD5 hashing for change detection
- **Error Handling**: Graceful handling of missing folders, permission issues, and database errors

## 🔒 Security Considerations

- The script only reads files; it doesn't modify or delete anything
- File paths are stored in the database but file contents are not
- MD5 hashes are used for detecting file changes, not for security
- Database should be protected with appropriate file permissions

## 📞 Support

For issues or questions:

1. Check the troubleshooting section above
2. Verify folder paths and permissions
3. Check database connectivity
4. Review the log files for error messages