#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learn From Updates - 自动学习脚本
根据新增文件和提醒记录更新知识库

This script scans directories for new/modified files and updates the knowledge base
with extracted information, maintaining embeddings and relationships.
"""

import os
import sqlite3
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import re
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/processed/learning_log.json', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LearningManager:
    """自动学习管理器"""

    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = db_path
        self.conn = None
        self.learning_stats = {
            'scan_time': datetime.now().isoformat(),
            'files_scanned': 0,
            'files_processed': 0,
            'entries_created': 0,
            'entries_updated': 0,
            'embeddings_regenerated': 0,
            'errors': []
        }

    def connect(self):
        """连接数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("PRAGMA foreign_keys = ON")
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def get_file_hash(self, file_path: Path) -> str:
        """获取文件哈希值用于检测变化"""
        try:
            # 使用文件大小和修改时间生成简单哈希
            stat = file_path.stat()
            hash_input = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to get hash for {file_path}: {e}")
            return ""

    def get_processed_files(self) -> Dict[str, str]:
        """获取已处理的文件记录"""
        try:
            # 创建处理记录表（如果不存在）
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_files (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT,
                    processed_at DATETIME,
                    entry_id INTEGER,
                    FOREIGN KEY (entry_id) REFERENCES knowledge_entries(id) ON DELETE SET NULL
                )
            """)

            cursor = self.conn.cursor()
            cursor.execute("SELECT file_path, file_hash FROM processed_files")
            return dict(cursor.fetchall())
        except Exception as e:
            logger.error(f"Failed to get processed files: {e}")
            return {}

    def scan_directories(self, directories: List[str], days_back: int = 7) -> List[Dict]:
        """扫描指定目录中的新文件"""
        scanned_files = []
        cutoff_time = datetime.now() - timedelta(days=days_back)
        processed_files = self.get_processed_files()

        for directory in directories:
            if not os.path.exists(directory):
                logger.warning(f"Directory not found: {directory}")
                continue

            logger.info(f"Scanning directory: {directory}")

            for file_path in Path(directory).rglob('*'):
                if not file_path.is_file():
                    continue

                # 检查文件类型
                if file_path.suffix.lower() not in ['.pdf', '.xlsx', '.xls', '.docx', '.doc', '.txt', '.csv']:
                    continue

                # 检查文件修改时间
                try:
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mod_time < cutoff_time:
                        continue
                except OSError:
                    continue

                # 检查是否已处理
                rel_path = str(file_path)
                current_hash = self.get_file_hash(file_path)

                if rel_path in processed_files:
                    if processed_files[rel_path] == current_hash:
                        continue  # 文件未变化，跳过

                # 收集文件信息
                file_info = {
                    'path': str(file_path),
                    'relative_path': rel_path,
                    'name': file_path.name,
                    'size': file_path.stat().st_size,
                    'modified': mod_time.isoformat(),
                    'hash': current_hash,
                    'type': file_path.suffix.lower()
                }

                scanned_files.append(file_info)

        self.learning_stats['files_scanned'] = len(scanned_files)
        logger.info(f"Found {len(scanned_files)} new/modified files")
        return scanned_files

    def determine_entity_type(self, file_path: str, content: str = "") -> str:
        """根据文件路径和内容确定实体类型"""
        path_lower = file_path.lower()
        content_lower = content.lower() if content else ""

        # 报价相关
        if any(keyword in path_lower or keyword in content_lower for keyword in
               ['报价', 'quote', '价格', 'price', '询价', 'inquiry']):
            if '报价' in path_lower or 'quote' in path_lower or 'price' in path_lower:
                return 'quote'
            else:
                return 'inquiry'

        # 客户相关
        if any(keyword in path_lower for keyword in ['客户', 'customer', '公司']):
            return 'customer'

        # 工厂相关
        if any(keyword in path_lower for keyword in ['工厂', 'factory', '供应商', 'supplier']):
            return 'factory'

        # 图纸相关
        if any(keyword in path_lower for keyword in ['图纸', 'drawing', 'dwg']):
            return 'drawing'

        # 产品规格相关
        if any(keyword in path_lower for keyword in ['规格', 'specification', 'standard']):
            return 'specification'

        # 材料相关
        if any(keyword in path_lower for keyword in ['材料', 'material']):
            return 'material'

        # 默认为产品
        return 'product'

    def extract_text_from_file(self, file_path: str) -> Optional[str]:
        """从文件中提取文本内容"""
        try:
            file_path = Path(file_path)

            if file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()

            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                return df.to_string()

            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8')
                return df.to_string()

            elif file_path.suffix.lower() == '.pdf':
                try:
                    import pdfplumber
                    text = ""
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            text += page.extract_text() + "\n"
                    return text
                except ImportError:
                    logger.warning("pdfplumber not available, skipping PDF processing")
                    return None

            elif file_path.suffix.lower() in ['.docx', '.doc']:
                try:
                    import docx
                    doc = docx.Document(file_path)
                    text = ""
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\n"
                    return text
                except ImportError:
                    logger.warning("python-docx not available, skipping DOCX processing")
                    return None

            else:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                return None

        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            return None

    def extract_attributes_from_text(self, text: str, entity_type: str) -> Dict[str, Any]:
        """从文本中提取属性信息"""
        attributes = {}

        try:
            # 提取产品规格
            spec_patterns = [
                r'M\d+\.?\d*',     # M规格
                r'φ\d+\.?\d*',     # 直径
                r'\d+\.?\d*mm',    # 毫米
                r'φ\d+\.?\d*',     # 直径符号
                r'GB/T\s*\d+',     # 国标
                r'ISO\s*\d+',      # ISO标准
                r'DIN\s*\d+',      # DIN标准
            ]

            specs = []
            for pattern in spec_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                specs.extend(matches)

            if specs:
                attributes['specifications'] = list(set(specs))

            # 提取材料信息
            materials = [
                '不锈钢', '碳钢', '合金钢', '铜', '铝', '锌', '尼龙', '塑料', '橡胶',
                '304', '316', '201', '202', '45#', 'Q235', 'Q345', '20#', '40Cr'
            ]

            found_materials = []
            text_lower = text.lower()
            for material in materials:
                if material.lower() in text_lower:
                    found_materials.append(material)

            if found_materials:
                attributes['materials'] = list(set(found_materials))

            # 提取数量信息
            quantity_patterns = [
                r'(\d+)\s*个',
                r'(\d+)\s*件',
                r'(\d+)\s*套',
                r'(\d+)\s*箱',
                r'(\d+)\s*包',
                r'quantity[:\s]+(\d+)',
                r'数量[:\s]+(\d+)',
            ]

            quantities = []
            for pattern in quantity_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                quantities.extend(matches)

            if quantities:
                try:
                    attributes['quantity'] = int(quantities[0])
                except ValueError:
                    pass

            # 提取价格信息（脱敏处理）
            price_patterns = [
                r'￥\s*([\d,]+\.?\d*)',
                r'¥\s*([\d,]+\.?\d*)',
                r'([\d,]+\.?\d*)\s*元',
                r'RMB\s*([\d,]+\.?\d*)',
                r'([\d,]+\.?\d*)',
            ]

            prices = []
            for pattern in price_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        price = float(match.replace(',', ''))
                        if 0.01 <= price <= 1000000:  # 合理的价格范围
                            prices.append(price)
                    except ValueError:
                        continue

            if prices:
                # 计算价格统计，但不存储具体价格
                attributes['price_count'] = len(prices)
                attributes['price_range_min'] = min(prices)
                attributes['price_range_max'] = max(prices)
                if len(prices) > 1:
                    attributes['price_average'] = statistics.mean(prices)

            # 根据实体类型提取特定属性
            if entity_type == 'quote':
                # 提取有效期
                validity_patterns = [
                    r'(\d+)\s*天',
                    r'(\d+)\s*日',
                    r'有效期[:\s]*(\d+)',
                ]
                for pattern in validity_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        try:
                            attributes['validity_days'] = int(matches[0])
                            break
                        except ValueError:
                            continue

            elif entity_type == 'customer':
                # 提取联系方式（脱敏）
                phone_patterns = [
                    r'1[3-9]\d{9}',  # 中国手机号
                    r'\d{3}-\d{4}-\d{4}',  # 座机
                ]

                phones = []
                for pattern in phone_patterns:
                    matches = re.findall(pattern, text)
                    phones.extend(matches)

                if phones:
                    attributes['has_phone'] = True
                    attributes['phone_count'] = len(phones)

                # 提取邮箱（脱敏）
                email_patterns = [
                    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                ]

                emails = re.findall(email_patterns[0], text)
                if emails:
                    attributes['has_email'] = True
                    attributes['email_count'] = len(emails)

        except Exception as e:
            logger.error(f"Failed to extract attributes: {e}")

        return attributes

    def create_or_update_knowledge_entry(self, file_info: Dict, content: str, entity_type: str) -> Optional[int]:
        """创建或更新知识条目"""
        try:
            # 检查是否已存在相关条目
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, name, description FROM knowledge_entries
                WHERE related_file = ?
            """, (file_info['relative_path'],))

            existing = cursor.fetchone()

            # 提取属性
            attributes = self.extract_attributes_from_text(content, entity_type)

            # 生成名称和描述
            name = file_info['name']
            if len(name) > 200:
                name = name[:200]

            description = content[:1000] if content else "自动提取的文档内容"

            if existing:
                # 更新现有条目
                entry_id = existing[0]
                cursor.execute("""
                    UPDATE knowledge_entries
                    SET name = ?, description = ?, attributes_json = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (
                    name,
                    description,
                    json.dumps(attributes, ensure_ascii=False),
                    entry_id
                ))
                self.learning_stats['entries_updated'] += 1
                logger.info(f"Updated entry {entry_id}: {name}")
            else:
                # 创建新条目
                cursor.execute("""
                    INSERT INTO knowledge_entries
                    (entity_type, name, description, related_file, attributes_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (
                    entity_type,
                    name,
                    description,
                    file_info['relative_path'],
                    json.dumps(attributes, ensure_ascii=False)
                ))
                entry_id = cursor.lastrowid
                self.learning_stats['entries_created'] += 1
                logger.info(f"Created entry {entry_id}: {name}")

            # 更新处理记录
            cursor.execute("""
                INSERT OR REPLACE INTO processed_files
                (file_path, file_hash, processed_at, entry_id)
                VALUES (?, ?, CURRENT_TIMESTAMP, ?)
            """, (file_info['relative_path'], file_info['hash'], entry_id))

            self.conn.commit()
            return entry_id

        except Exception as e:
            logger.error(f"Failed to create/update knowledge entry: {e}")
            self.learning_stats['errors'].append(str(e))
            return None

    def learn_from_files(self, directories: List[str], days_back: int = 7) -> Dict:
        """从文件学习"""
        try:
            logger.info(f"🚀 Starting learning from files (last {days_back} days)")

            # 扫描文件
            scanned_files = self.scan_directories(directories, days_back)

            if not scanned_files:
                logger.info("No new files to process")
                return self.learning_stats

            # 处理每个文件
            for file_info in scanned_files:
                try:
                    # 提取文本内容
                    content = self.extract_text_from_file(file_info['path'])
                    if not content or len(content.strip()) < 50:
                        logger.warning(f"Insufficient content in {file_info['path']}, skipping")
                        continue

                    # 确定实体类型
                    entity_type = self.determine_entity_type(file_info['path'], content)

                    # 创建或更新知识条目
                    entry_id = self.create_or_update_knowledge_entry(file_info, content, entity_type)
                    if entry_id:
                        self.learning_stats['files_processed'] += 1

                except Exception as e:
                    logger.error(f"Failed to process file {file_info['path']}: {e}")
                    self.learning_stats['errors'].append(f"File {file_info['path']}: {str(e)}")
                    continue

            logger.info(f"✅ Learning from files completed")
            logger.info(f"📊 Processed: {self.learning_stats['files_processed']}/{self.learning_stats['files_scanned']} files")
            logger.info(f"📝 Created: {self.learning_stats['entries_created']} entries")
            logger.info(f"🔄 Updated: {self.learning_stats['entries_updated']} entries")

            return self.learning_stats

        except Exception as e:
            logger.error(f"❌ Learning from files failed: {e}")
            self.learning_stats['errors'].append(str(e))
            return self.learning_stats

    def regenerate_embeddings_if_needed(self) -> bool:
        """如果需要，重新生成嵌入索引"""
        try:
            # 检查是否有新的知识条目需要建立嵌入
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM knowledge_entries ke
                LEFT JOIN embedding_index ei ON ke.id = ei.entry_id
                WHERE ei.entry_id IS NULL
            """)
            missing_embeddings = cursor.fetchone()[0]

            if missing_embeddings > 0:
                logger.info(f"🔄 Regenerating embeddings for {missing_embeddings} new entries")

                # 调用嵌入构建脚本
                import subprocess
                result = subprocess.run([
                    'python3', 'build_embeddings.py', '--build'
                ], capture_output=True, text=True, cwd='.')

                if result.returncode == 0:
                    self.learning_stats['embeddings_regenerated'] = missing_embeddings
                    logger.info(f"✅ Embeddings regenerated successfully")
                    return True
                else:
                    logger.error(f"❌ Failed to regenerate embeddings: {result.stderr}")
                    return False
            else:
                logger.info("ℹ️ All entries have embeddings, no regeneration needed")
                return True

        except Exception as e:
            logger.error(f"Failed to regenerate embeddings: {e}")
            return False

    def save_learning_stats(self) -> bool:
        """保存学习统计"""
        try:
            stats_file = Path("data/processed/learning_stats.json")
            stats_file.parent.mkdir(parents=True, exist_ok=True)

            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.learning_stats, f, ensure_ascii=False, indent=2)

            logger.info(f"📊 Learning stats saved to {stats_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save learning stats: {e}")
            return False

    def run_learning(self, mode: str = "weekly") -> Dict:
        """运行学习过程"""
        try:
            # 连接数据库
            self.connect()

            # 根据模式配置扫描参数
            if mode == "weekly":
                days_back = 7
                directories = [
                    "/Users/gavin/Nutstore Files/.symlinks/坚果云/005-询盘询价和/",
                    "/Users/gavin/Nutstore Files/.symlinks/坚果云/002-客户中/"
                ]
                auto_embed = True
            elif mode == "daily":
                days_back = 1
                directories = [
                    "/Users/gavin/Nutstore Files/.symlinks/坚果云/005-询盘询价和/",
                    "/Users/gavin/Nutstore Files/.symlinks/坚果云/002-客户中/"
                ]
                auto_embed = False
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # 执行学习
            stats = self.learn_from_files(directories, days_back)

            # 重新生成嵌入（如果需要）
            if auto_embed:
                self.regenerate_embeddings_if_needed()

            # 保存统计
            self.save_learning_stats()

            return stats

        except Exception as e:
            logger.error(f"❌ Learning process failed: {e}")
            self.learning_stats['errors'].append(str(e))
            return self.learning_stats
        finally:
            self.close()

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Learn From Updates')
    parser.add_argument('--mode', choices=['weekly', 'daily'], default='weekly',
                       help='Learning mode (weekly or daily)')
    parser.add_argument('--directories', nargs='+',
                       help='Specific directories to scan')
    parser.add_argument('--days-back', type=int, help='Number of days to look back')
    parser.add_argument('--force-embed', action='store_true',
                       help='Force embedding regeneration')

    args = parser.parse_args()

    learner = LearningManager()

    if args.directories:
        # 自定义目录扫描
        days = args.days_back or 7
        stats = learner.learn_from_files(args.directories, days)
    else:
        # 标准学习模式
        stats = learner.run_learning(args.mode)

    # 输出结果
    print(f"\n🎓 Learning Results ({args.mode} mode)")
    print("=" * 50)
    print(f"Files scanned: {stats['files_scanned']}")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Entries created: {stats['entries_created']}")
    print(f"Entries updated: {stats['entries_updated']}")
    print(f"Embeddings regenerated: {stats['embeddings_regenerated']}")
    print(f"Errors: {len(stats['errors'])}")

    if stats['errors']:
        print(f"\n❌ Errors:")
        for error in stats['errors'][:5]:  # Show first 5
            print(f"  - {error}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")

if __name__ == "__main__":
    main()