#!/usr/bin/env python3
"""
图纸资料自动导入脚本
扫描指定文件夹中的PDF/图片文件，提取元数据并插入数据库
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from models import DatabaseManager, Drawing, Customer

class DrawingIngestor:
    """图纸资料导入器"""

    def __init__(self, db_path: str = "./data/db.sqlite"):
        self.db_manager = DatabaseManager(db_path)
        self.drawing = Drawing(self.db_manager)
        self.customer = Customer(self.db_manager)
        self.processed_log = []
        self.errors = []

        # 创建处理日志目录
        self.log_dir = Path("./data/processed")
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "drawing_ingest_log.json"

        # 支持的文件扩展名
        self.supported_extensions = {
            # 图片格式
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp',
            # 文档格式
            '.pdf', '.dwg', '.dxf', '.svg', '.psd', '.ai', '.eps',
            # 压缩格式（可能包含图纸文件）
            '.zip', '.rar', '.7z'
        }

    def scan_directory(self, directory_path: str) -> List[Path]:
        """
        扫描目录，查找支持的文件类型

        Args:
            directory_path: 要扫描的目录路径

        Returns:
            List[Path]: 找到的文件列表
        """
        files = []

        if not os.path.exists(directory_path):
            print(f"❌ 目录不存在: {directory_path}")
            return files

        directory = Path(directory_path)
        print(f"📁 扫描目录: {directory}")

        # 递归查找文件
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                files.append(file_path)

        print(f"✓ 找到 {len(files)} 个支持的文件")
        return files

    def extract_info_from_filename(self, filename: str) -> Dict[str, Any]:
        """
        从文件名中提取图纸信息

        Args:
            filename: 文件名

        Returns:
            Dict: 提取的信息
        """
        info = {}
        clean_name = Path(filename).stem  # 去除扩展名

        # 提取图纸编号模式 (如: ABC-001, DWG-2024-001等)
        drawing_number_patterns = [
            r'([A-Z]{2,4}-\d{3,6})',           # ABC-001, XYZ-1234
            r'([A-Z]{1,3}\d{4,6})',            # A1234, XYZ12345
            r'(DWG[-_]?\d{3,6})',              # DWG-001, DWG1234
            r'(图[-_]?\d{3,6})',               # 图-001, 图1234
            r'(\d{4}[-_]\d{2}[-_]\d{2})',      # 日期格式: 2024-03-01
        ]

        for pattern in drawing_number_patterns:
            matches = re.findall(pattern, clean_name, re.IGNORECASE)
            if matches:
                info['drawing_number'] = matches[0].strip()
                break

        # 提取产品类别
        product_categories = [
            '螺丝', '齿轮', '轴承', '弹簧', '垫片', '销子', '铆钉', '螺母',
            'screw', 'gear', 'bearing', 'spring', 'washer', 'pin', 'rivet', 'nut',
            '支架', '外壳', '盖子', '底座', '连接器', '法兰', '轴', '套筒',
            'bracket', 'housing', 'cover', 'base', 'connector', 'flange', 'shaft', 'sleeve'
        ]

        for category in product_categories:
            if category.lower() in clean_name.lower():
                info['product_category'] = category
                break

        # 提取公司名称 (参考客户导入的逻辑)
        company_patterns = [
            r'([^/\\]+(?:公司|有限公司|集团|企业|Co\.?|Ltd\.?|Inc\.?|Corp\.?))',
            r'([^/\\]{3,20}(?:制造|科技|电子|机械|工业))',
        ]

        for pattern in company_patterns:
            matches = re.findall(pattern, clean_name, re.IGNORECASE)
            if matches:
                info['possible_company'] = matches[0].strip()
                break

        # 提取联系人信息
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, clean_name)
        if emails:
            info['possible_email'] = emails[0]

        # 提取尺寸信息 (如: M10x20, 50x30x10等)
        size_pattern = r'([A-Z]?\d+[x×*]\d+(?:[x×*]\d+)?)'
        size_matches = re.findall(size_pattern, clean_name, re.IGNORECASE)
        if size_matches:
            info['dimensions'] = size_matches[0]

        # 提取材料信息
        materials = [
            '不锈钢', '碳钢', '铝合金', '铜', '铁', '塑料', '橡胶',
            'SS', 'SUS', 'Carbon', 'Aluminum', 'Brass', 'Steel', 'Plastic', 'Rubber'
        ]

        for material in materials:
            if material.lower() in clean_name.lower():
                info['material'] = material
                break

        # 提取版本信息 (如: V1.0, Rev2, 修订A等)
        version_patterns = [
            r'(V\d+\.?\d*)',
            r'(Rev\d+)',
            r'(修订?[A-Z]?)',
            r'(v\d+\.?\d*)',
        ]

        for pattern in version_patterns:
            matches = re.findall(pattern, clean_name, re.IGNORECASE)
            if matches:
                info['version'] = matches[0].strip()
                break

        return info

    def find_matching_customer(self, file_info: Dict[str, Any]) -> Optional[int]:
        """
        根据文件信息尝试匹配客户

        Args:
            file_info: 从文件名提取的信息

        Returns:
            Optional[int]: 匹配的客户ID，如果没有匹配则返回None
        """
        try:
            # 优先使用邮箱匹配
            if 'possible_email' in file_info:
                email = file_info['possible_email']
                customer = self.customer.get_by_email(email)
                if customer:
                    print(f"  🎯 通过邮箱匹配客户: {customer['company_name']} ({email})")
                    return customer['id']

            # 使用公司名称匹配
            if 'possible_company' in file_info:
                company_name = file_info['possible_company']
                # 模糊匹配公司名称
                all_customers = self.customer.get_all()
                for customer in all_customers:
                    if company_name.lower() in customer['company_name'].lower() or \
                       customer['company_name'].lower() in company_name.lower():
                        print(f"  🎯 通过公司名匹配客户: {customer['company_name']}")
                        return customer['id']

        except Exception as e:
            print(f"  ⚠️  客户匹配失败: {e}")

        return None

    def get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        获取文件元数据

        Args:
            file_path: 文件路径

        Returns:
            Dict: 文件元数据
        """
        metadata = {}

        try:
            # 基本文件信息
            stat = file_path.stat()
            metadata['file_size'] = stat.st_size
            metadata['modified_time'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
            metadata['created_time'] = datetime.fromtimestamp(stat.st_ctime).isoformat()

            # 尝试读取PDF元数据
            if file_path.suffix.lower() == '.pdf':
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        if pdf_reader.metadata:
                            pdf_info = pdf_reader.metadata
                            metadata['pdf_title'] = pdf_info.get('/Title', '')
                            metadata['pdf_author'] = pdf_info.get('/Author', '')
                            metadata['pdf_creator'] = pdf_info.get('/Creator', '')
                            metadata['pdf_producer'] = pdf_info.get('/Producer', '')
                            metadata['pdf_creation_date'] = str(pdf_info.get('/CreationDate', ''))
                            metadata['pdf_mod_date'] = str(pdf_info.get('/ModDate', ''))
                        metadata['pdf_page_count'] = len(pdf_reader.pages)
                except ImportError:
                    print("  ⚠️  PyPDF2未安装，无法读取PDF元数据")
                except Exception as e:
                    print(f"  ⚠️  读取PDF元数据失败: {e}")

            # 尝试读取图片元数据
            elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                try:
                    from PIL import Image
                    from PIL.ExifTags import TAGS

                    with Image.open(file_path) as img:
                        metadata['image_format'] = img.format
                        metadata['image_mode'] = img.mode
                        metadata['image_size'] = f"{img.width}x{img.height}"

                        # EXIF数据
                        if hasattr(img, '_getexif') and img._getexif():
                            exif_data = img._getexif()
                            for tag_id, value in exif_data.items():
                                tag = TAGS.get(tag_id, tag_id)
                                if tag in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                                    metadata[f'exif_{tag.lower()}'] = str(value)

                except ImportError:
                    print("  ⚠️  Pillow未安装，无法读取图片元数据")
                except Exception as e:
                    print(f"  ⚠️  读取图片元数据失败: {e}")

        except Exception as e:
            print(f"  ❌ 获取文件元数据失败: {e}")

        return metadata

    def create_drawing_record(self, file_path: Path, file_info: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建图纸记录

        Args:
            file_path: 文件路径
            file_info: 从文件名提取的信息
            metadata: 文件元数据

        Returns:
            Dict: 图纸记录数据
        """
        # 构建图纸名称
        drawing_name = file_path.stem
        if 'drawing_number' in file_info:
            drawing_name = f"{file_info['drawing_number']} - {drawing_name}"

        # 确定产品类别
        product_category = file_info.get('product_category', '未分类')

        # 准备备注信息
        notes_parts = []
        if 'dimensions' in file_info:
            notes_parts.append(f"尺寸: {file_info['dimensions']}")
        if 'material' in file_info:
            notes_parts.append(f"材料: {file_info['material']}")
        if 'version' in file_info:
            notes_parts.append(f"版本: {file_info['version']}")

        # 添加文件元数据到备注
        if metadata.get('file_size'):
            size_mb = metadata['file_size'] / (1024 * 1024)
            notes_parts.append(f"文件大小: {size_mb:.2f}MB")

        if metadata.get('pdf_page_count'):
            notes_parts.append(f"页数: {metadata['pdf_page_count']}")

        if metadata.get('image_size'):
            notes_parts.append(f"图片尺寸: {metadata['image_size']}")

        notes = '; '.join(notes_parts) if notes_parts else None

        return {
            'drawing_name': drawing_name,
            'product_category': product_category,
            'file_path': str(file_path),
            'upload_date': metadata.get('modified_time', datetime.now().isoformat()),
            'notes': notes,
            'file_info': file_info,
            'metadata': metadata
        }

    def insert_drawings(self, drawings: List[Dict[str, Any]]) -> int:
        """
        将图纸数据插入数据库

        Args:
            drawings: 图纸数据列表

        Returns:
            int: 成功插入的图纸数量
        """
        inserted_count = 0

        with self.db_manager:
            for drawing_data in drawings:
                try:
                    # 尝试匹配客户
                    customer_id = self.find_matching_customer(drawing_data.get('file_info', {}))

                    # 准备插入数据
                    insert_data = {
                        'drawing_name': drawing_data['drawing_name'],
                        'product_category': drawing_data['product_category'],
                        'file_path': drawing_data['file_path'],
                        'upload_date': drawing_data['upload_date'],
                        'notes': drawing_data['notes']
                    }

                    if customer_id:
                        insert_data['customer_id'] = customer_id

                    # 检查是否已存在相同的文件路径
                    existing_drawing = None
                    # 这里可以添加检查逻辑，暂时跳过

                    drawing_id = self.drawing.create(**insert_data)
                    inserted_count += 1

                    print(f"  ✅ 插入图纸 #{drawing_id}: {drawing_data['drawing_name']}")
                    if customer_id:
                        print(f"      关联客户ID: {customer_id}")

                    self.processed_log.append({
                        'status': 'inserted',
                        'drawing_id': drawing_id,
                        'customer_id': customer_id,
                        'data': drawing_data,
                        'timestamp': datetime.now().isoformat()
                    })

                except Exception as e:
                    print(f"  ❌ 插入图纸失败: {e}")
                    self.errors.append({
                        'drawing_data': drawing_data,
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
        print("图纸资料自动导入脚本")
        print("=" * 60)

        # 扫描文件
        files = self.scan_directory(directory_path)
        if not files:
            return {'scanned_files': 0, 'processed_drawings': 0, 'inserted_drawings': 0}

        total_drawings = []
        scanned_count = 0

        # 处理每个文件
        for file_path in files:
            print(f"\n📄 处理文件: {file_path.name}")
            scanned_count += 1

            try:
                # 从文件名提取信息
                file_info = self.extract_info_from_filename(file_path.name)
                print(f"  📋 文件名分析: {len(file_info)} 个信息项")

                # 获取文件元数据
                metadata = self.get_file_metadata(file_path)
                print(f"  📊 元数据: {len(metadata)} 项")

                # 创建图纸记录
                drawing_data = self.create_drawing_record(file_path, file_info, metadata)
                total_drawings.append(drawing_data)

            except Exception as e:
                print(f"  ❌ 处理文件失败: {e}")
                self.errors.append({
                    'file': str(file_path),
                    'error': f'文件处理失败: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                })

        # 插入数据库
        print(f"\n📤 开始插入数据库...")
        print(f"总共处理 {len(total_drawings)} 个图纸文件")
        inserted_count = self.insert_drawings(total_drawings)

        # 保存日志
        self.save_log()

        # 返回统计结果
        result = {
            'scanned_files': scanned_count,
            'processed_drawings': len(total_drawings),
            'inserted_drawings': inserted_count,
            'errors': len(self.errors)
        }

        print("\n" + "=" * 60)
        print("处理完成!")
        print(f"扫描文件: {result['scanned_files']}")
        print(f"处理图纸: {result['processed_drawings']}")
        print(f"插入成功: {result['inserted_drawings']}")
        print(f"处理错误: {result['errors']}")
        print("=" * 60)

        return result

def main():
    """主函数"""
    # 配置路径
    drawing_directory = "/Users/gavin/Nutstore Files/.symlinks/坚果云/005-询盘询价/"
    db_path = "./data/db.sqlite"

    # 创建导入器并处理
    ingestor = DrawingIngestor(db_path)
    result = ingestor.process_directory(drawing_directory)

    return result

if __name__ == "__main__":
    main()