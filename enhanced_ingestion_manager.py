#!/usr/bin/env python3
"""
增强型数据录入管理器
提供完整的日志记录、错误处理和批量运行能力
"""

import os
import logging
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import sqlite3

from models import DatabaseManager, Customer, Drawing

@dataclass
class IngestionResult:
    """导入结果数据类"""
    success: bool
    message: str
    file_path: str = ""
    record_count: int = 0
    error_details: str = ""
    processing_time: float = 0.0

class EnhancedIngestionManager:
    """增强型数据录入管理器"""

    def __init__(self, db_path: str = "./data/db.sqlite"):
        self.db_manager = DatabaseManager(db_path)
        self.setup_logging()
        self.setup_directories()

        # 统计信息
        self.stats = {
            'total_processed': 0,
            'successful_imports': 0,
            'failed_imports': 0,
            'total_records': 0,
            'start_time': None,
            'end_time': None
        }

    def setup_logging(self):
        """设置详细的日志系统"""
        # 创建日志目录
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        # 设置主日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'ingestion.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        # 设置错误日志
        error_handler = logging.FileHandler(log_dir / 'ingestion_errors.log', encoding='utf-8')
        error_handler.setLevel(logging.ERROR)

        self.logger = logging.getLogger('EnhancedIngestion')
        self.logger.addHandler(error_handler)

        # 设置详细处理日志
        self.process_logger = logging.getLogger('Processing')
        process_handler = logging.FileHandler(log_dir / 'processing_details.log', encoding='utf-8')
        process_handler.setLevel(logging.DEBUG)
        process_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.process_logger.addHandler(process_handler)

    def setup_directories(self):
        """创建必要的目录结构"""
        dirs = [
            './data/processed',
            './data/failed',
            './data/backups',
            './logs',
            './reports'
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(exist_ok=True)

    def log_ingestion_start(self, operation: str, source_path: str):
        """记录导入开始"""
        self.logger.info(f"🚀 开始{operation}: {source_path}")
        self.process_logger.info(f"START: {operation} | {source_path} | {datetime.now().isoformat()}")

    def log_ingestion_end(self, operation: str, result: IngestionResult):
        """记录导入结束"""
        status = "✅ 成功" if result.success else "❌ 失败"
        self.logger.info(f"{status} {operation}: {result.message}")

        if result.success:
            self.process_logger.info(f"SUCCESS: {operation} | 文件: {result.file_path} | "
                                   f"记录数: {result.record_count} | 耗时: {result.processing_time:.2f}s")
        else:
            self.process_logger.error(f"FAILED: {operation} | 文件: {result.file_path} | "
                                    f"错误: {result.error_details} | 耗时: {result.processing_time:.2f}s")

    def safe_execute_with_retry(self, operation_func, operation_name: str, max_retries: int = 3) -> IngestionResult:
        """带重试机制的安全执行"""
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                result = operation_func()
                processing_time = time.time() - start_time

                if isinstance(result, IngestionResult):
                    result.processing_time = processing_time
                else:
                    result = IngestionResult(
                        success=True,
                        message=f"{operation_name}完成",
                        processing_time=processing_time
                    )

                if attempt > 0:
                    self.logger.info(f"🔄 重试成功: {operation_name} (第{attempt + 1}次尝试)")

                return result

            except Exception as e:
                error_msg = f"{operation_name}失败 (第{attempt + 1}次尝试): {str(e)}"
                self.logger.error(error_msg)

                if attempt == max_retries - 1:
                    # 最后一次尝试失败，返回错误结果
                    return IngestionResult(
                        success=False,
                        message=f"{operation_name}失败",
                        error_details=error_msg,
                        processing_time=0.0
                    )
                else:
                    # 等待后重试
                    wait_time = (attempt + 1) * 2
                    self.logger.info(f"⏳ 等待 {wait_time}s 后重试...")
                    time.sleep(wait_time)

    def ingest_customers_batch(self, source_paths: List[str]) -> List[IngestionResult]:
        """批量导入客户数据"""
        self.logger.info(f"📦 开始批量导入客户数据: {len(source_paths)} 个文件")
        self.stats['start_time'] = datetime.now()

        results = []
        total_records = 0

        for source_path in source_paths:
            self.log_ingestion_start("客户导入", source_path)

            result = self.safe_execute_with_retry(
                lambda: self._ingest_single_customer_file(source_path),
                f"客户文件导入: {Path(source_path).name}",
                max_retries=2
            )

            results.append(result)
            self.log_ingestion_end("客户导入", result)

            # 更新统计
            self.stats['total_processed'] += 1
            if result.success:
                self.stats['successful_imports'] += 1
                total_records += result.record_count
            else:
                self.stats['failed_imports'] += 1

                # 移动失败文件到失败目录
                self._move_failed_file(source_path, "customer")

        self.stats['total_records'] = total_records
        self.stats['end_time'] = datetime.now()

        # 生成批量报告
        self._generate_batch_report("customer_import", results)

        return results

    def ingest_drawings_batch(self, source_paths: List[str]) -> List[IngestionResult]:
        """批量导入图纸数据"""
        self.logger.info(f"📦 开始批量导入图纸数据: {len(source_paths)} 个文件")
        self.stats['start_time'] = datetime.now()

        results = []
        total_records = 0

        for source_path in source_paths:
            self.log_ingestion_start("图纸导入", source_path)

            result = self.safe_execute_with_retry(
                lambda: self._ingest_single_drawing_file(source_path),
                f"图纸文件导入: {Path(source_path).name}",
                max_retries=2
            )

            results.append(result)
            self.log_ingestion_end("图纸导入", result)

            # 更新统计
            self.stats['total_processed'] += 1
            if result.success:
                self.stats['successful_imports'] += 1
                total_records += result.record_count
            else:
                self.stats['failed_imports'] += 1

                # 移动失败文件到失败目录
                self._move_failed_file(source_path, "drawing")

        self.stats['total_records'] = total_records
        self.stats['end_time'] = datetime.now()

        # 生成批量报告
        self._generate_batch_report("drawing_import", results)

        return results

    def _ingest_single_customer_file(self, file_path: str) -> IngestionResult:
        """导入单个客户文件"""
        try:
            from ingest_customers import CustomerIngestor
            ingestor = CustomerIngestor(self.db_manager.db_path)

            # 读取和处理文件
            customers = ingestor.process_file(file_path)

            if not customers:
                return IngestionResult(
                    success=True,
                    message="没有找到客户数据",
                    file_path=file_path,
                    record_count=0
                )

            # 插入数据库
            inserted_count = 0
            for customer_data in customers:
                try:
                    ingestor.customer.create(**customer_data)
                    inserted_count += 1
                except Exception as e:
                    self.logger.warning(f"插入客户失败: {customer_data.get('company_name', 'Unknown')} - {e}")

            return IngestionResult(
                success=True,
                message=f"成功导入 {inserted_count} 个客户",
                file_path=file_path,
                record_count=inserted_count
            )

        except Exception as e:
            return IngestionResult(
                success=False,
                message=f"客户文件处理失败",
                file_path=file_path,
                error_details=f"{str(e)}\n{traceback.format_exc()}"
            )

    def _ingest_single_drawing_file(self, file_path: str) -> IngestionResult:
        """导入单个图纸文件"""
        try:
            from ingest_drawings import DrawingIngestor
            ingestor = DrawingIngestor(self.db_manager.db_path)

            # 处理文件
            drawing_data = ingestor.process_drawing_file(file_path)

            if not drawing_data:
                return IngestionResult(
                    success=True,
                    message="没有提取到图纸数据",
                    file_path=file_path,
                    record_count=0
                )

            # 插入数据库
            drawing_id = ingestor.drawing.create(**drawing_data)

            return IngestionResult(
                success=True,
                message=f"成功导入图纸: {drawing_data.get('drawing_name', 'Unknown')}",
                file_path=file_path,
                record_count=1
            )

        except Exception as e:
            return IngestionResult(
                success=False,
                message=f"图纸文件处理失败",
                file_path=file_path,
                error_details=f"{str(e)}\n{traceback.format_exc()}"
            )

    def _move_failed_file(self, file_path: str, file_type: str):
        """移动失败的文件到失败目录"""
        try:
            source = Path(file_path)
            if source.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{file_type}_{source.name}"
                destination = Path("./data/failed") / filename

                # 如果是移动文件，而不是删除
                if source.is_file():
                    import shutil
                    shutil.move(str(source), str(destination))
                    self.logger.info(f"📁 失败文件已移动: {destination}")
        except Exception as e:
            self.logger.error(f"移动失败文件出错: {e}")

    def _generate_batch_report(self, operation: str, results: List[IngestionResult]):
        """生成批量处理报告"""
        report = {
            'operation': operation,
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats,
            'results': [asdict(r) for r in results],
            'summary': {
                'total_files': len(results),
                'successful': len([r for r in results if r.success]),
                'failed': len([r for r in results if not r.success]),
                'total_records': sum(r.record_count for r in results if r.success),
                'total_processing_time': sum(r.processing_time for r in results)
            }
        }

        # 保存详细报告
        report_file = f"./reports/{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.logger.info(f"📄 批量报告已保存: {report_file}")

        # 保存最新报告
        latest_report = f"./reports/latest_{operation}_report.json"
        with open(latest_report, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def create_database_backup(self, operation: str):
        """创建数据库备份"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"./data/backups/db_backup_{operation}_{timestamp}.sqlite"

            # 简单的数据库复制
            import shutil
            shutil.copy2(self.db_manager.db_path, backup_file)

            self.logger.info(f"💾 数据库备份完成: {backup_file}")
            return backup_file

        except Exception as e:
            self.logger.error(f"数据库备份失败: {e}")
            return None

    def run_scheduled_import(self, customer_paths: List[str], drawing_paths: List[str]):
        """运行计划导入"""
        self.logger.info("🔄 开始计划导入流程")

        # 创建备份
        backup_file = self.create_database_backup("scheduled_import")

        try:
            # 导入客户数据
            if customer_paths:
                customer_results = self.ingest_customers_batch(customer_paths)
                self.logger.info(f"客户导入完成: 成功 {len([r for r in customer_results if r.success])}/{len(customer_results)}")

            # 导入图纸数据
            if drawing_paths:
                drawing_results = self.ingest_drawings_batch(drawing_paths)
                self.logger.info(f"图纸导入完成: 成功 {len([r for r in drawing_results if r.success])}/{len(drawing_results)}")

            self.logger.info("✅ 计划导入完成")
            return True

        except Exception as e:
            self.logger.error(f"计划导入失败: {e}")

            # 如果有备份，询问是否恢复
            if backup_file:
                self.logger.info(f"💡 备份文件可用: {backup_file}")

            return False

    def get_import_statistics(self) -> Dict[str, Any]:
        """获取导入统计信息"""
        return {
            'current_session': self.stats,
            'database_stats': self._get_database_stats(),
            'recent_logs': self._get_recent_logs()
        }

    def _get_database_stats(self) -> Dict[str, int]:
        """获取数据库统计"""
        try:
            with self.db_manager:
                conn = self.db_manager.connect()
                cursor = conn.cursor()

                stats = {}
                tables = ['customers', 'drawings', 'factories', 'factory_quotes', 'specifications']

                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[table] = cursor.fetchone()[0]

                return stats

        except Exception as e:
            self.logger.error(f"获取数据库统计失败: {e}")
            return {}

    def _get_recent_logs(self) -> List[str]:
        """获取最近的日志"""
        try:
            log_file = Path("./logs/ingestion.log")
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    return [line.strip() for line in lines[-10:]]  # 最近10行
            return []
        except Exception:
            return []

def main():
    """主函数 - 演示用法"""
    manager = EnhancedIngestionManager()

    # 示例：批量导入
    customer_files = [
        # "/path/to/customer_excel.xlsx",
        # "/path/to/customer_data.csv"
    ]

    drawing_files = [
        # "/path/to/drawings_folder"
    ]

    # 运行计划导入
    success = manager.run_scheduled_import(customer_files, drawing_files)

    # 显示统计
    stats = manager.get_import_statistics()
    print(f"导入统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")

if __name__ == "__main__":
    main()