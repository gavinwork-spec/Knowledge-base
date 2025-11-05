#!/usr/bin/env python3
"""
客户资料自动导入脚本
扫描指定文件夹中的Excel/CSV/文本文件，提取客户信息并插入数据库
"""

import os
import re
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from models import DatabaseManager, Customer

class CustomerIngestor:
    """客户资料导入器"""

    def __init__(self, db_path: str = "./data/db.sqlite"):
        self.db_manager = DatabaseManager(db_path)
        self.customer = Customer(self.db_manager)
        self.processed_log = []
        self.errors = []

        # 创建处理日志目录
        self.log_dir = Path("./data/processed")
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "customer_ingest_log.json"

    def scan_directory(self, directory_path: str) -> List[Path]:
        """
        扫描目录，查找支持的文件类型

        Args:
            directory_path: 要扫描的目录路径

        Returns:
            List[Path]: 找到的文件列表
        """
        supported_extensions = {'.xlsx', '.xls', '.csv', '.txt'}
        files = []

        if not os.path.exists(directory_path):
            print(f"❌ 目录不存在: {directory_path}")
            return files

        directory = Path(directory_path)
        print(f"📁 扫描目录: {directory}")

        # 递归查找文件
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(file_path)

        print(f"✓ 找到 {len(files)} 个支持的文件")
        return files

    def extract_info_from_filename(self, filename: str) -> Dict[str, str]:
        """
        从文件名中提取可能的客户信息

        Args:
            filename: 文件名

        Returns:
            Dict: 提取的信息
        """
        info = {}

        # 常见的邮箱模式
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, filename)
        if emails:
            info['contact_email'] = emails[0]

        # 公司名称模式 (通常包含 "公司", "有限公司", "Co", "Ltd" 等)
        company_patterns = [
            r'([^/\\]+(?:公司|有限公司|集团|企业|Co\.?|Ltd\.?|Inc\.?|Corp\.?))',
            r'([^/\\]{3,20}(?:制造|科技|电子|机械|工业))',
        ]

        for pattern in company_patterns:
            matches = re.findall(pattern, filename, re.IGNORECASE)
            if matches:
                info['company_name'] = matches[0].strip()
                break

        # 联系人姓名模式 (中文或英文姓名)
        name_patterns = [
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)',  # 英文姓名
            r'([\u4e00-\u9fff]{2,4})',       # 中文姓名
        ]

        for pattern in name_patterns:
            matches = re.findall(pattern, filename)
            if matches and len(matches[0]) > 1:
                info['contact_name'] = matches[0].strip()
                break

        return info

    def parse_excel_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        解析Excel文件

        Args:
            file_path: Excel文件路径

        Returns:
            List[Dict]: 解析出的客户数据
        """
        customers = []

        try:
            # 尝试读取所有工作表
            excel_file = pd.ExcelFile(file_path)
            print(f"  📊 Excel文件工作表: {excel_file.sheet_names}")

            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    customers.extend(self.parse_dataframe(df, f"{file_path.name} - {sheet_name}"))
                except Exception as e:
                    print(f"  ❌ 读取工作表 {sheet_name} 失败: {e}")
                    continue

        except Exception as e:
            print(f"  ❌ 解析Excel文件失败: {e}")
            self.errors.append({
                'file': str(file_path),
                'error': f'Excel解析失败: {str(e)}',
                'timestamp': datetime.now().isoformat()
            })

        return customers

    def parse_csv_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        解析CSV文件

        Args:
            file_path: CSV文件路径

        Returns:
            List[Dict]: 解析出的客户数据
        """
        customers = []

        try:
            # 尝试不同的编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"  📄 CSV文件编码: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue

            if df is not None:
                customers.extend(self.parse_dataframe(df, file_path.name))
            else:
                raise Exception("无法确定文件编码")

        except Exception as e:
            print(f"  ❌ 解析CSV文件失败: {e}")
            self.errors.append({
                'file': str(file_path),
                'error': f'CSV解析失败: {str(e)}',
                'timestamp': datetime.now().isoformat()
            })

        return customers

    def parse_text_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        解析文本文件

        Args:
            file_path: 文本文件路径

        Returns:
            List[Dict]: 解析出的客户数据
        """
        customers = []

        try:
            # 尝试不同编码
            encodings = ['utf-8', 'gbk', 'gb2312']
            content = None

            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    print(f"  📝 文本文件编码: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                raise Exception("无法确定文件编码")

            # 尝试从文件名提取信息
            filename_info = self.extract_info_from_filename(file_path.name)

            # 简单的文本解析逻辑
            lines = content.split('\n')
            customer_data = {'source_file': str(file_path)}

            # 合并文件名提取的信息
            customer_data.update(filename_info)

            # 尝试从文本中提取更多信息
            for line in lines:
                line = line.strip()

                # 邮箱
                if '@' in line and 'contact_email' not in customer_data:
                    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', line)
                    if email_match:
                        customer_data['contact_email'] = email_match.group()

                # 电话
                phone_pattern = r'(\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9})'
                phone_match = re.search(phone_pattern, line)
                if phone_match and 'phone' not in customer_data:
                    customer_data['phone'] = phone_match.group()

                # 国家信息
                countries = ['中国', '美国', '日本', '韩国', '德国', '英国', '法国', 'China', 'USA', 'Japan', 'Korea', 'Germany', 'UK', 'France']
                for country in countries:
                    if country in line and 'country' not in customer_data:
                        customer_data['country'] = country
                        break

            # 如果提取到了基本信息，添加到结果中
            if 'company_name' in customer_data or 'contact_email' in customer_data:
                customers.append(customer_data)

        except Exception as e:
            print(f"  ❌ 解析文本文件失败: {e}")
            self.errors.append({
                'file': str(file_path),
                'error': f'文本解析失败: {str(e)}',
                'timestamp': datetime.now().isoformat()
            })

        return customers

    def parse_dataframe(self, df: pd.DataFrame, source: str) -> List[Dict[str, Any]]:
        """
        解析DataFrame数据

        Args:
            df: pandas DataFrame
            source: 数据源描述

        Returns:
            List[Dict]: 解析出的客户数据
        """
        customers = []

        try:
            print(f"  📋 DataFrame形状: {df.shape}, 列名: {list(df.columns)}")

            # 列名映射字典
            column_mapping = {
                # 公司名称
                'company': 'company_name',
                '公司': 'company_name',
                'company_name': 'company_name',
                '客户名称': 'company_name',
                '客户': 'company_name',

                # 联系人姓名
                'name': 'contact_name',
                'contact': 'contact_name',
                'contact_name': 'contact_name',
                '联系人': 'contact_name',
                '姓名': 'contact_name',

                # 邮箱
                'email': 'contact_email',
                'mail': 'contact_email',
                'contact_email': 'contact_email',
                '邮箱': 'contact_email',
                '电子邮件': 'contact_email',

                # 国家
                'country': 'country',
                '国家': 'country',
                '地区': 'country',

                # 语言
                'language': 'language',
                '语言': 'language',

                # 电话
                'phone': 'phone',
                'tel': 'phone',
                'telephone': 'phone',
                '电话': 'phone',
                '手机': 'phone',

                # 首次联系日期
                'date': 'first_contact_date',
                'contact_date': 'first_contact_date',
                'first_contact': 'first_contact_date',
                '联系日期': 'first_contact_date',
                '首次联系': 'first_contact_date',

                # 备注
                'notes': 'notes',
                'note': 'notes',
                '备注': 'notes',
                '说明': 'notes'
            }

            # 重命名列
            df_renamed = df.rename(columns=column_mapping)

            # 转换每一行为客户数据
            for index, row in df_renamed.iterrows():
                customer_data = {
                    'source_file': source,
                    'row_number': index + 1
                }

                # 提取各字段
                for field in ['company_name', 'contact_name', 'contact_email', 'country',
                            'language', 'phone', 'first_contact_date', 'notes']:
                    if field in df_renamed.columns:
                        value = row[field]
                        if pd.notna(value) and str(value).strip():
                            customer_data[field] = str(value).strip()

                # 只有至少有公司名称或邮箱才保存
                if 'company_name' in customer_data or 'contact_email' in customer_data:
                    customers.append(customer_data)

        except Exception as e:
            print(f"  ❌ 解析DataFrame失败: {e}")
            self.errors.append({
                'file': source,
                'error': f'DataFrame解析失败: {str(e)}',
                'timestamp': datetime.now().isoformat()
            })

        return customers

    def insert_customers(self, customers: List[Dict[str, Any]]) -> int:
        """
        将客户数据插入数据库

        Args:
            customers: 客户数据列表

        Returns:
            int: 成功插入的客户数量
        """
        inserted_count = 0

        with self.db_manager:
            for customer_data in customers:
                try:
                    # 检查是否已存在相同的客户（基于公司名称+邮箱）
                    company_name = customer_data.get('company_name', '').strip()
                    contact_email = customer_data.get('contact_email', '').strip()

                    if company_name and contact_email:
                        existing = self.customer.get_by_company_and_email(company_name, contact_email)
                        if existing:
                            print(f"  ⚠️  客户已存在: {company_name} / {contact_email}")
                            self.processed_log.append({
                                'status': 'duplicate',
                                'data': customer_data,
                                'timestamp': datetime.now().isoformat()
                            })
                            continue

                    # 准备插入数据
                    insert_data = {}
                    for field in ['company_name', 'contact_name', 'contact_email', 'country',
                                'language', 'phone', 'first_contact_date', 'notes']:
                        if field in customer_data:
                            insert_data[field] = customer_data[field]

                    if insert_data:
                        customer_id = self.customer.create(**insert_data)
                        inserted_count += 1
                        print(f"  ✅ 插入客户 #{customer_id}: {insert_data.get('company_name', 'Unknown')} / {insert_data.get('contact_email', 'No email')}")

                        self.processed_log.append({
                            'status': 'inserted',
                            'customer_id': customer_id,
                            'data': customer_data,
                            'timestamp': datetime.now().isoformat()
                        })

                except Exception as e:
                    print(f"  ❌ 插入客户失败: {e}")
                    self.errors.append({
                        'customer_data': customer_data,
                        'error': f'插入失败: {str(e)}',
                        'timestamp': datetime.now().isoformat()
                    })

        return inserted_count

    def save_log(self):
        """保存处理日志"""
        log_data = {
            'scan_time': datetime.now().isoformat(),
            'processed_count': len(self.processed_log),
            'error_count': len(self.errors),
            'processed_items': self.processed_log,
            'errors': self.errors
        }

        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            print(f"📝 处理日志已保存: {self.log_file}")
        except Exception as e:
            print(f"❌ 保存日志失败: {e}")

    def process_directory(self, directory_path: str) -> Dict[str, int]:
        """
        处理整个目录

        Args:
            directory_path: 要处理的目录路径

        Returns:
            Dict: 处理结果统计
        """
        print("=" * 60)
        print("客户资料自动导入脚本")
        print("=" * 60)

        # 扫描文件
        files = self.scan_directory(directory_path)
        if not files:
            return {'scanned_files': 0, 'extracted_customers': 0, 'inserted_customers': 0}

        total_customers = []
        scanned_count = 0

        # 处理每个文件
        for file_path in files:
            print(f"\n📄 处理文件: {file_path.name}")
            scanned_count += 1

            # 根据文件类型解析
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                customers = self.parse_excel_file(file_path)
            elif file_path.suffix.lower() == '.csv':
                customers = self.parse_csv_file(file_path)
            else:  # .txt
                customers = self.parse_text_file(file_path)

            print(f"  📊 提取到 {len(customers)} 个客户信息")
            total_customers.extend(customers)

        # 插入数据库
        print(f"\n📤 开始插入数据库...")
        print(f"总共提取到 {len(total_customers)} 个客户信息")
        inserted_count = self.insert_customers(total_customers)

        # 保存日志
        self.save_log()

        # 返回统计结果
        result = {
            'scanned_files': scanned_count,
            'extracted_customers': len(total_customers),
            'inserted_customers': inserted_count,
            'errors': len(self.errors)
        }

        print("\n" + "=" * 60)
        print("处理完成!")
        print(f"扫描文件: {result['scanned_files']}")
        print(f"提取客户: {result['extracted_customers']}")
        print(f"插入成功: {result['inserted_customers']}")
        print(f"处理错误: {result['errors']}")
        print("=" * 60)

        return result

def main():
    """主函数"""
    # 配置路径
    customer_directory = "/Users/gavin/Nutstore Files/.symlinks/坚果云/002-客户/"
    db_path = "./data/db.sqlite"

    # 创建导入器并处理
    ingestor = CustomerIngestor(db_path)
    result = ingestor.process_directory(customer_directory)

    return result

if __name__ == "__main__":
    main()