#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document Parser Script
文档解析脚本

This script parses various document formats (PDF, Excel, Word) to extract
structured knowledge information and store it in the knowledge database.
"""

import sqlite3
import json
import logging
import argparse
import os
import re
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/processed/parse_documents.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentParser:
    """文档解析器"""

    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = db_path
        self.conn = None
        self.parsing_rules = self.load_parsing_rules()
        self.processed_files = set()

    def load_parsing_rules(self) -> Dict:
        """加载解析规则"""
        return {
            'customer_extraction': {
                'patterns': {
                    'company_name': [
                        r'(?:)(公司|有限公司|股份公司|企业|实业|科技|制造)',
                        r'[A-Za-z][\w\u4e00-\u9fa5]+(?:公司|有限公司|股份公司|企业|实业|科技|制造)',
                    ],
                    'contact_person': [
                        r'(?:)(经理|主管|负责人|先生|女士)\s*[A-Za-z\u4e00-\u9fa5]+',
                    ],
                    'phone_number': [
                        r'1[3-9]\d{9}',
                        r'\d{3}-\d{4}-\d{4}',
                        r'\d{11}',
                    ],
                    'email': [
                        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                    ]
                },
                'confidence_threshold': 0.7,
                'context_window': 50
            },
            'product_extraction': {
                'patterns': {
                    'product_name': [
                        r'(?:)(螺栓|螺钉|螺丝|紧固件|垫片|轴承)',
                        r'[A-Z]+-\d+',
                        r'M\d+',
                    ],
                    'specification': [
                        r'M\d+',
                        r'\d+\s*mm',
                        r'φ\d+',
                        r'\d+\s*×\s*\d+',
                    ],
                    'material': [
                        r'(?:)(不锈钢|碳钢|合金钢|铜|铝|锌|尼龙|塑料)',
                        r'SS\d+',
                        r'Q\d+',
                        r'45#',
                        r'304',
                        r'316',
                    ],
                    'quantity': [
                        r'\d+\s*(?:个|件|套|箱|包)',
                        r'(?:)quantity\s*[：:]?\s*\d+',
                    ],
                    'unit_price': [
                        r'￥\s*\d+\.?\d*',
                        r'RMB\s*\d+\.?\d*',
                        r'\$\s*\d+\.?\d*',
                    ]
                },
                'confidence_threshold': 0.7,
                'context_window': 30
            }
        }

    def connect(self):
        """连接数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""

    def is_file_processed(self, file_path: str) -> bool:
        """检查文件是否已处理"""
        file_hash = self.calculate_file_hash(file_path)
        return file_hash in self.processed_files

    def mark_file_processed(self, file_path: str):
        """标记文件为已处理"""
        file_hash = self.calculate_file_hash(file_path)
        self.processed_files.add(file_hash)

    def extract_text_from_pdf(self, file_path: str) -> str:
        """从PDF文件提取文本"""
        try:
            import pdfplumber

            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        text += page_text + "\n"
                        logger.debug(f"Extracted text from page {page_num + 1}")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")

            return text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {file_path}: {e}")
            return ""

    def extract_data_from_excel(self, file_path: str) -> List[Dict]:
        """从Excel文件提取数据"""
        try:
            import pandas as pd

            data = []

            # 读取Excel文件
            if file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_excel(file_path, engine='openpyxl')

            # 将数据转换为字典列表
            for index, row in df.iterrows():
                row_dict = {}
                for col in df.columns:
                    if pd.notna(row[col]):
                        row_dict[col] = str(row[col])
                data.append(row_dict)

            logger.info(f"Extracted {len(data)} rows from Excel file")
            return data

        except Exception as e:
            logger.error(f"Failed to extract data from Excel {file_path}: {e}")
            return []

    def extract_text_from_docx(self, file_path: str) -> str:
        """从Word文档提取文本"""
        try:
            from docx import Document

            doc = Document(file_path)
            text = ""

            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"

            return text
        except Exception as e:
            logger.error(f"Failed to extract text from DOCX {file_path}: {e}")
            return ""

    def extract_entities(self, text: str, file_info: Dict) -> List[Dict]:
        """从文本中提取实体"""
        entities = []

        try:
            # 客户信息提取
            if self.parsing_rules.get('customer_extraction'):
                customer_entities = self.extract_entities_by_rule(
                    text, self.parsing_rules['customer_extraction'], 'customer', file_info
                )
                entities.extend(customer_entities)

            # 产品信息提取
            if self.parsing_rules.get('product_extraction'):
                product_entities = self.extract_entities_by_rule(
                    text, self.parsing_rules['product_extraction'], 'product', file_info
                )
                entities.extend(product_entities)

        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")

        return entities

    def extract_entities_by_rule(self, text: str, rule: Dict, entity_type: str, file_info: Dict) -> List[Dict]:
        """根据规则提取实体"""
        entities = []
        patterns = rule.get('patterns', {})
        threshold = rule.get('confidence_threshold', 0.7)

        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        entity = {
                            'category': category,
                            'keyword': match.group(0),
                            'value': match.group(0),
                            'confidence_score': self.calculate_confidence(match, text, rule),
                            'context_text': text[max(0, match.start()-50):match.end()+50],
                            'start_position': match.start(),
                            'end_position': match.end(),
                            'file_path': file_info['path'],
                            'file_name': file_info['name'],
                            'extracted_at': datetime.now().isoformat()
                        }

                        if entity['confidence_score'] >= threshold:
                            entities.append(entity)

                except Exception as e:
                    logger.warning(f"Error in pattern matching for {category}: {e}")

        return entities

    def calculate_confidence(self, match, full_text: str, rule: Dict) -> float:
        """计算匹配置信度"""
        try:
            # 基础置信度
            base_confidence = 0.8

            # 根据匹配长度调整
            match_length = len(match.group(0))
            if match_length >= 10:
                length_bonus = 0.1
            elif match_length >= 5:
                length_bonus = 0.05
            else:
                length_bonus = 0.0

            # 根据上下文相关性调整
            context_window = rule.get('context_window', 50)
            start_pos = max(0, match.start() - context_window)
            end_pos = min(len(full_text), match.end() + context_window)
            context = full_text[start_pos:end_pos]

            # 检查上下文中是否有相关关键词
            relevant_keywords = ['报价', '客户', '产品', '规格', '价格', '订单', '合同']
            context_relevance = sum(1 for keyword in relevant_keywords if keyword.lower() in context.lower())

            if context_relevance > 0:
                context_bonus = min(0.2, context_relevance * 0.05)
            else:
                context_bonus = 0.0

            confidence = min(1.0, base_confidence + length_bonus + context_bonus)
            return confidence

        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.5

    def create_knowledge_entry(self, file_info: Dict, entities: List[Dict]) -> Dict:
        """创建知识条目"""
        try:
            # 确定实体类型
            entity_type = self.determine_entity_type(file_info, entities)

            # 提取主要信息
            name = self.extract_main_name(file_info, entities)
            description = self.generate_description(file_info, entities)

            # 构建属性数据
            attributes = {
                'file_path': file_info['path'],
                'file_name': file_info['name'],
                'file_size': file_info.get('size', 0),
                'file_modified': file_info.get('modified', ''),
                'extraction_date': datetime.now().isoformat(),
                'entity_count': len(entities),
                'entities': entities
            }

            # 添加提取的实体信息
            if entities:
                for entity in entities:
                    if entity['category'] not in attributes:
                        attributes[entity['category']] = []
                    attributes[entity['category']].append(entity['value'])

            knowledge_entry = {
                'entity_type': entity_type,
                'name': name,
                'related_file': file_info['path'],
                'description': description,
                'attributes_json': json.dumps(attributes, ensure_ascii=False),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }

            return knowledge_entry

        except Exception as e:
            logger.error(f"Failed to create knowledge entry: {e}")
            return {}

    def determine_entity_type(self, file_info: Dict, entities: List[Dict]) -> str:
        """确定实体类型"""
        file_name = file_info.get('name', '').lower()
        file_path = file_info.get('path', '').lower()

        # 根据文件路径确定类型
        if 'customer' in file_path or '客户' in file_path:
            return 'customer'
        elif 'quote' in file_path or '报价' in file_path or 'inquiry' in file_path:
            return 'quote'
        elif 'drawing' in file_path or '图纸' in file_path:
            return 'drawing'
        elif 'factory' in file_path or '工厂' in file_path:
            return 'factory'
        elif entities:
            # 根据提取的实体确定类型
            if any(e['category'] in ['company_name', 'contact_person'] for e in entities):
                return 'customer'
            elif any(e['category'] in ['product_name', 'specification', 'material'] for e in entities):
                return 'product'
            elif any(e['category'] in ['unit_price', 'total_price'] for e in entities):
                return 'quote'
            else:
                return 'general'
        else:
            return 'document'

    def extract_main_name(self, file_info: Dict, entities: List[Dict]) -> str:
        """提取主要名称"""
        # 优先使用文件名
        file_name = file_info.get('name', '')
        if file_name:
            # 移除文件扩展名
            name_without_ext = os.path.splitext(file_name)[0]
            return name_without_ext

        # 从实体中提取名称
        if entities:
            # 优先使用公司名称
            for entity in entities:
                if entity['category'] == 'company_name':
                    return entity['value']

            # 使用第一个高置信度的实体
            high_confidence_entities = [e for e in entities if e['confidence_score'] >= 0.8]
            if high_confidence_entities:
                return high_confidence_entities[0]['value']

        return file_name or "未知文档"

    def generate_description(self, file_info: Dict, entities: List[Dict]) -> str:
        """生成描述"""
        description_parts = []

        # 添加文件基本信息
        if file_info.get('name'):
            description_parts.append(f"文档名称: {file_info['name']}")

        # 添加提取的实体信息
        if entities:
            entity_summary = {}
            for entity in entities:
                category = entity['category']
                if category not in entity_summary:
                    entity_summary[category] = []
                entity_summary[category].append(entity['value'])

            for category, values in entity_summary.items():
                if len(values) > 3:
                    description_parts.append(f"{category}: {', '.join(values[:3])} 等{len(values)-3}个")
                else:
                    description_parts.append(f"{category}: {', '.join(values)}")

        # 添加提取时间
        description_parts.append(f"提取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "; ".join(description_parts)

    def save_knowledge_entry(self, entry: Dict) -> Optional[int]:
        """保存知识条目到数据库"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO knowledge_entries (
                    entity_type, name, related_file, description, attributes_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                entry['entity_type'],
                entry['name'],
                entry['related_file'],
                entry['description'],
                entry['attributes_json'],
                entry['created_at'],
                entry['updated_at']
            ))

            entry_id = cursor.lastrowid
            self.conn.commit()

            # 保存NLP实体
            self.save_nlp_entities(entry_id, entry.get('attributes_json', '{}').get('entities', []))

            logger.info(f"Saved knowledge entry: {entry['name']} (ID: {entry_id})")
            return entry_id

        except Exception as e:
            logger.error(f"Failed to save knowledge entry: {e}")
            return None

    def save_nlp_entities(self, entry_id: int, entities: List[Dict]):
        """保存NLP实体到数据库"""
        try:
            cursor = self.conn.cursor()

            for entity in entities:
                cursor.execute("""
                    INSERT INTO nlp_entities (
                        entry_id, keyword, value, category, confidence_score,
                        context_text, start_position, end_position, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry_id,
                    entity['keyword'],
                    entity['value'],
                    entity['category'],
                    entity['confidence_score'],
                    entity['context_text'],
                    entity['start_position'],
                    entity['end_position'],
                    datetime.now()
                ))

            self.conn.commit()
            logger.info(f"Saved {len(entities)} NLP entities for entry {entry_id}")

        except Exception as e:
            logger.error(f"Failed to save NLP entities: {e}")

    def parse_document(self, file_path: str) -> Optional[Dict]:
        """解析单个文档"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None

            # 检查文件是否已处理
            if self.is_file_processed(str(file_path)):
                logger.info(f"File already processed: {file_path}")
                return None

            # 获取文件信息
            file_info = {
                'path': str(file_path),
                'name': file_path.name,
                'size': file_path.stat().st_size,
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                'extension': file_path.suffix.lower()
            }

            logger.info(f"Processing document: {file_info['name']}")

            # 提取文本内容
            text = ""
            if file_info['extension'] == '.pdf':
                text = self.extract_text_from_pdf(str(file_path))
            elif file_info['extension'] in ['.xlsx', '.xls']:
                # Excel文件需要特殊处理
                excel_data = self.extract_data_from_excel(str(file_path))
                if excel_data:
                    text = "\n".join([str(row) for row in excel_data[:5]])  # 只取前5行用于实体提取
            elif file_info['extension'] in ['.docx', '.doc']:
                text = self.extract_text_from_docx(str(file_path))
            elif file_info['extension'] in ['.txt', '.csv']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='gbk') as f:
                        text = f.read()
            else:
                logger.warning(f"Unsupported file format: {file_info['extension']}")
                return None

            if not text.strip():
                logger.warning(f"No text extracted from: {file_info['name']}")
                return None

            # 提取实体
            entities = self.extract_entities(text, file_info)

            if not entities:
                logger.warning(f"No entities extracted from: {file_info['name']}")
                return None

            # 创建知识条目
            knowledge_entry = self.create_knowledge_entry(file_info, entities)

            # 保存到数据库
            entry_id = self.save_knowledge_entry(knowledge_entry)

            if entry_id:
                # 标记文件为已处理
                self.mark_file_processed(str(file_path))

                # 保存解析结果
                self.save_parsing_result(file_info, entities, entry_id)

                return {
                    'success': True,
                    'entry_id': entry_id,
                    'entity_count': len(entities),
                    'file_info': file_info
                }
            else:
                return {'success': False, 'error': 'Failed to save knowledge entry'}

        except Exception as e:
            logger.error(f"Failed to parse document {file_path}: {e}")
            return {'success': False, 'error': str(e)}

    def save_parsing_result(self, file_info: Dict, entities: List[Dict], entry_id: int):
        """保存解析结果"""
        try:
            os.makedirs('data/processed', exist_ok=True)

            result = {
                'file_info': file_info,
                'parsing_timestamp': datetime.now().isoformat(),
                'entry_id': entry_id,
                'entity_count': len(entities),
                'entities': entities,
                'success': True
            }

            # 保存到JSON文件
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"parsed_docs_{date_str}.json"

            existing_results = []
            json_file = Path('data/processed') / filename

            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                except:
                    existing_results = []

            existing_results.append(result)

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(existing_results, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved parsing result to {filename}")

        except Exception as e:
            logger.error(f"Failed to save parsing result: {e}")

    def parse_directory(self, directory: str, recursive: bool = True) -> Dict:
        """解析目录中的文档"""
        directory_path = Path(directory)
        if not directory_path.exists():
            logger.error(f"Directory not found: {directory}")
            return {'success': False, 'error': 'Directory not found'}

        results = {
            'directory': str(directory_path),
            'start_time': datetime.now().isoformat(),
            'files_processed': 0,
            'entries_created': 0,
            'files_failed': 0,
            'errors': [],
            'processed_files': []
        }

        # 查找支持的文件
        supported_extensions = ['.pdf', '.xlsx', '.xls', '.docx', '.doc', '.txt', '.csv']

        if recursive:
            files = list(directory_path.rglob('*'))
        else:
            files = list(directory_path.glob('*'))

        files = [f for f in files if f.suffix.lower() in supported_extensions]

        logger.info(f"Found {len(files)} supported files in {directory}")

        for file_path in files:
            try:
                result = self.parse_document(str(file_path))
                if result:
                    if result['success']:
                        results['files_processed'] += 1
                        results['entries_created'] += 1
                        results['processed_files'].append({
                            'file': str(file_path),
                            'entry_id': result.get('entry_id'),
                            'entity_count': result.get('entity_count', 0)
                        })
                    else:
                        results['files_failed'] += 1
                        results['errors'].append({
                            'file': str(file_path),
                            'error': result.get('error', 'Unknown error')
                        })
                else:
                    results['files_failed'] += 1
                    results['errors'].append({
                        'file': str(file_path),
                        'error': 'No result returned'
                    })

            except Exception as e:
                results['files_failed'] += 1
                results['errors'].append({
                    'file': str(file_path),
                    'error': str(e)
                })

        results['end_time'] = datetime.now().isoformat()
        results['duration_seconds'] = (
            datetime.fromisoformat(results['end_time']) -
            datetime.fromisoformat(results['start_time'])
        ).total_seconds()

        # 保存批量处理结果
        self.save_batch_results(results)

        logger.info(f"Directory parsing completed: {results['files_processed']} processed, {results['files_failed']} failed")

        return results

    def save_batch_results(self, results: Dict):
        """保存批量处理结果"""
        try:
            os.makedirs('data/processed', exist_ok=True)

            filename = f"batch_parsing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_path = Path('data/processed') / filename

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved batch results to {filename}")

        except Exception as e:
            logger.error(f"Failed to save batch results: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Document parsing tool')
    parser.add_argument('--mode', choices=['single', 'batch'], default='batch',
                       help='Parsing mode: single file or directory')
    parser.add_argument('--file', type=str, help='File path (for single mode)')
    parser.add_argument('--directory', type=str, help='Directory path (for batch mode)')
    parser.add_argument('--recursive', action='store_true', default=True,
                       help='Search directories recursively')
    parser.add_argument('--db-path', type=str, default='knowledge_base.db',
                       help='Database file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        parser = DocumentParser(args.db_path)
        parser.connect()

        if args.mode == 'single':
            if not args.file:
                print("Error: --file argument is required for single mode")
                return 1

            result = parser.parse_document(args.file)
            if result and result.get('success'):
                print(f"✅ Successfully parsed document: {args.file}")
                print(f"   Entry ID: {result.get('entry_id')}")
                print(f"   Entities extracted: {result.get('entity_count', 0)}")
            else:
                print(f"❌ Failed to parse document: {args.file}")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                return 1

        elif args.mode == 'batch':
            directory = args.directory or "/Users/gavin/Nutstore Files/.symlinks/坚果云/005-询盘询价和/"
            result = parser.parse_directory(directory, args.recursive)

            print(f"\n📊 Parsing Results:")
            print(f"   Directory: {result['directory']}")
            print(f"   Files processed: {result['files_processed']}")
            print(f"   Entries created: {result['entries_created']}")
            print(f"   Files failed: {result['files_failed']}")
            print(f"   Duration: {results['duration_seconds']:.2f} seconds")

            if result['errors']:
                print(f"\n❌ Errors ({len(result['errors'])}):")
                for error in result['errors']:
                    print(f"   • {error['file']}: {error['error']}")
            else:
                print("\n✅ All files processed successfully!")

        parser.close()
        return 0

    except Exception as e:
        logger.error(f"Document parsing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())