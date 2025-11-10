# Document Processing and Knowledge Verification System

## 📋 Overview

This system consists of three integrated Python scripts that form a complete document processing pipeline for trading companies:

1. **`scan_new_files.py`** - File scanner that detects new documents
2. **`parse_document.py`** - Document parser that extracts key information
3. **`verify_new_entries.py`** - Knowledge verification and testing system

## 🔄 Complete Workflow

```
1. File Scanner → 2. Document Parser → 3. Knowledge Verification
```

1. **File Discovery**: Monitor folders for new files (PDF, Excel, Images)
2. **Information Extraction**: Parse documents and extract trading-specific fields
3. **Quality Assurance**: Verify extraction quality and search functionality

## 🛠️ Installation Requirements

```bash
# Install required Python packages
pip3 install PyMuPDF pandas requests xlrd openpyxl

# The scripts use these main libraries:
# - PyMuPDF (fitz): PDF text extraction
# - pandas: Excel file processing
# - requests: HTTP requests for API testing
# - xlrd: Legacy Excel file support
# - openpyxl: Modern Excel file support
# - sqlite3: Database operations (built-in)
# - re: Regular expressions (built-in)
```

## 📁 File Scanner (`scan_new_files.py`)

### Purpose
Monitors specified folders for new files and logs them to a SQLite database.

### Key Features
- **Multi-folder monitoring**: Watch multiple trading folders
- **File type detection**: Supports PDF, Excel, and Image files
- **Duplicate prevention**: Uses file paths and MD5 hashes
- **Database logging**: Stores file metadata with timestamps

### Usage
```bash
# Single scan (find new files once)
python3 scan_new_files.py --mode scan

# Continuous monitoring (every 60 seconds)
python3 scan_new_files.py --mode watch

# Show current statistics
python3 scan_new_files.py --stats

# Custom monitoring interval (30 seconds)
python3 scan_new_files.py --mode watch --interval 30
```

### Configuration
Edit the script to change:
- **Monitored folders**: `WATCH_FOLDERS` list
- **Database path**: `DATABASE_PATH` variable
- **Supported file types**: `SUPPORTED_EXTENSIONS` dictionary

### Database Schema
```sql
CREATE TABLE file_processing_log (
    id INTEGER PRIMARY KEY,
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
);
```

## 📄 Document Parser (`parse_document.py`)

### Purpose
Processes pending files from the database and extracts key trading information.

### Key Features
- **Multi-format support**: PDF and Excel file processing
- **Intelligent extraction**: Uses regex patterns and heuristics
- **Field extraction**: Customer names, products, materials, quantities, prices
- **Confidence scoring**: Quality assessment of extracted data
- **Knowledge storage**: Stores extracted information in structured format

### Extracted Fields
| Field | Description | Pattern Types |
|-------|-------------|---------------|
| `customer_name` | Customer/client name | Email addresses, company names, patterns |
| `product_category` | Product type or category | Product terms, categories |
| `material` | Material specifications | Material types, grades, standards |
| `quantity` | Order quantities | Number patterns with units |
| `date` | Document dates | Various date formats |
| `price` | Pricing information | Currency patterns, numbers |
| `customer_needs` | Requirements/specifications | Requirement keywords |
| `dimensions` | Product dimensions | Measurement patterns |

### Usage
```bash
# Process all pending files
python3 parse_document.py --mode process

# Process limited number of files (for testing)
python3 parse_document.py --mode process --max-files 10

# Show processing statistics
python3 parse_document.py --mode stats

# Export knowledge base to JSON
python3 parse_document.py --mode export --export-file knowledge_export.json
```

### Knowledge Base Schema
```sql
CREATE TABLE knowledge_entries (
    id INTEGER PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    file_type TEXT NOT NULL,
    extracted_at DATETIME NOT NULL,
    attributes_json TEXT NOT NULL,
    raw_text TEXT,
    confidence_score REAL,
    processing_status TEXT DEFAULT 'completed',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Extraction Patterns
The parser uses comprehensive regex patterns for multiple languages:

**English Patterns**:
- `customer[:\s]+([^\n]+)`
- `product[:\s]+([^\n]+)`
- `price[:\s]*\$?([\d,]+\.?\d*)`
- `(\d{4}[-/]\d{1,2}[-/]\d{1,2})`

**Chinese Patterns**:
- `公司[:\s]+([^\n]+)`
- `产品[:\s]+([^\n]+)`
- `价格[:\s]*\$?([\d,]+\.?\d*)`
- `不锈钢|碳钢|黄铜`

**Technical Patterns**:
- `M(\d+)[x×*](\d+)` (Metric sizes)
- `304|316|4\.8|8\.8|12\.9` (Materials/grades)
- `Ø\s*(\d+\.?\d*)` (Diameters)

## ✅ Knowledge Verification (`verify_new_entries.py`)

### Purpose
Tests and validates the knowledge base entries and search functionality.

### Key Features
- **Data quality assessment**: Checks for missing required fields
- **Search testing**: Validates API search functionality
- **Keyword extraction**: Generates test queries from extracted data
- **Success rate monitoring**: Tracks processing and search quality
- **Detailed reporting**: Comprehensive verification reports

### Usage
```bash
# Basic verification (last 24 hours)
python3 verify_new_entries.py

# Custom time range (last 48 hours)
python3 verify_new_entries.py --hours 48

# Limited search tests
python3 verify_new_entries.py --max-tests 10

# Save detailed report
python3 verify_new_entries.py --save-report

# Custom database and API URLs
python3 verify_new_entries.py --db-path /path/to/db.sqlite --api-url http://localhost:8080
```

### Verification Process
1. **Database Analysis**: Queries entries from specified time period
2. **Field Validation**: Checks for missing required fields
3. **Keyword Extraction**: Generates search terms from content
4. **API Testing**: Tests knowledge search endpoints
5. **Report Generation**: Creates detailed quality reports

### Quality Metrics
- **Data completeness**: Percentage of entries with all required fields
- **Extraction confidence**: Average confidence scores
- **Search success rate**: API search functionality performance
- **Error tracking**: Failed extractions and API errors

## 📊 Monitoring and Statistics

### File Scanner Statistics
```bash
python3 scan_new_files.py --stats
```
Output includes:
- Total files by type (PDF, Excel, Image)
- Processing status distribution
- File discovery timeline

### Document Parser Statistics
```bash
python3 parse_document.py --mode stats
```
Output includes:
- Files processed vs. pending
- Knowledge entries created
- Average extraction confidence
- Error rates and reasons

### Verification Results
The verification script generates detailed reports including:
- Entry completeness analysis
- Missing field identification
- Search API performance metrics
- Quality improvement recommendations

## 🔧 Configuration and Customization

### Adding New File Types
In `scan_new_files.py`:
```python
SUPPORTED_EXTENSIONS = {
    '.pdf': 'PDF',
    '.xlsx': 'Excel',
    '.docx': 'Word',  # Add new type
    '.txt': 'Text',   # Add another type
}
```

### Custom Extraction Patterns
In `parse_document.py`, add new patterns to the `patterns` dictionary:
```python
patterns = {
    'custom_field': [
        r'custom_pattern[:\s]+([^\n]+)',
        r'another_pattern',
    ],
    # ... existing patterns
}
```

### Database Paths
Update these variables in each script:
```python
DATABASE_PATH = '/path/to/your/database.db'
API_BASE_URL = 'http://your-api-server:port'
```

## 🚀 Production Deployment

### Continuous File Monitoring
```bash
# Start file scanner in background
nohup python3 scan_new_files.py --mode watch --interval 300 > scanner.log 2>&1 &

# Start document processor in background
nohup python3 parse_document.py --mode process > parser.log 2>&1 &
```

### Automated Verification
```bash
# Schedule daily verification
0 9 * * * cd /path/to/scripts && python3 verify_new_entries.py --save-report
```

### Logging and Monitoring
- All scripts generate detailed logs
- File scanner: `document_parser.log`
- Document parser: Same log file with processing details
- Verification: Console output and optional report files

## 🐛 Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure read access to monitored folders
2. **Database Locked**: Only one script instance at a time
3. **Missing Dependencies**: Install required Python packages
4. **API Connection**: Verify knowledge API server is running
5. **File Not Found**: Check file paths in database are accessible

### Debug Mode
Enable detailed logging by modifying log levels:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Database Inspection
```bash
# Connect to database
sqlite3 trading_company.db

# Check tables
.tables

# View recent entries
SELECT * FROM knowledge_entries ORDER BY extracted_at DESC LIMIT 10;

# Check processing status
SELECT status, COUNT(*) FROM file_processing_log GROUP BY status;
```

## 📈 Performance Optimization

### Batch Processing
- Process files in batches to manage memory usage
- Use `--max-files` parameter for controlled processing
- Monitor database size and performance

### Search Optimization
- Index knowledge entries on frequently searched fields
- Cache common search results
- Optimize API response times

### Storage Management
- Archive old processed files
- Clean up failed entries periodically
- Monitor database growth

## 🔒 Security Considerations

- **File Access**: Scripts only read files, no modification
- **Database Security**: Use appropriate file permissions
- **API Security**: Validate API endpoints and authentication
- **Data Privacy**: Sensitive information extracted and stored in database

## 📞 Support and Maintenance

### Regular Tasks
1. **Weekly**: Run verification and review reports
2. **Monthly**: Check processing statistics and error rates
3. **Quarterly**: Update extraction patterns and keywords
4. **Annually**: Review and optimize the complete workflow

### Contact and Issues
For issues with:
- **File scanning**: Check folder paths and permissions
- **Document parsing**: Review extraction patterns and file formats
- **Verification**: Verify API connectivity and database access
- **Performance**: Monitor system resources and database size

---

**Version**: 1.0
**Last Updated**: 2025-11-10
**Compatible**: Python 3.6+, SQLite 3.x