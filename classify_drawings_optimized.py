#!/usr/bin/env python3
"""
优化版图纸自动分类脚本
通过增强关键词库和改进匹配算法来提高分类准确率
"""

import sqlite3
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/classify_drawings_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ClassifyDrawingsOptimized')

class OptimizedDrawingClassifier:
    """优化版图纸分类器"""

    def __init__(self):
        self.db_path = "./data/db.sqlite"

        # 增强的关键词库
        self.expanded_keywords = {
            '紧固件-螺栓螺钉': {
                'keywords': [
                    # 基础词汇
                    '螺栓', '螺钉', '螺丝', 'bolt', 'screw', 'thread', '螺纹',
                    '六角', '内六角', '外六角', 'hex', 'socket', 'allen',
                    # 具体类型
                    '机牙', '自攻', '钻尾', 'tap', 'drilling', 'self tapping',
                    '沉头', '盘头', '圆头', '平头', 'countersunk', 'pan', 'round', 'flat',
                    '十字', '一字', 'phillips', 'slotted',
                    # 英文
                    't-head', 'wood screw', 'machine screw', 'hex bolt',
                    # 德文（可能来自德国图纸）
                    'schraube', 'schrauben', 'gewinde',
                    # 常见规格模式
                    r'\d+\.\d+x\d+',  # 4.8x80
                    r'm\d+x\d+',      # M5x8
                    r'din\d+',         # DIN标准
                ],
                'weight': 1.0
            },
            '紧固件-螺母': {
                'keywords': [
                    '螺母', '螺帽', 'nut', 'hex nut', '六角螺母', '法兰螺母',
                    '锁紧螺母', 'lock nut', '盖形螺母', '蝶形螺母', 'wing nut',
                    '焊接螺母', 'weld nut', '嵌入螺母', 'square nut', 'nylon nut'
                ],
                'weight': 1.0
            },
            '紧固件-垫圈垫片': {
                'keywords': [
                    '垫圈', '垫片', 'washer', 'flat washer', 'spring washer',
                    '平垫圈', '弹簧垫圈', '齿形垫圈', '波形垫圈', 'lock washer',
                    '防松垫圈', 'split washer', 'external tooth', 'internal tooth'
                ],
                'weight': 1.0
            },
            '紧固件-销铆钉': {
                'keywords': [
                    '销', '铆钉', 'pin', 'rivet', '圆柱销', 'dowel pin', '开口销',
                    'cotter pin', '弹性销', 'spring pin', '半圆头铆钉', 'blind rivet',
                    '拉铆钉', 'pull rivet', '定位销', 'locating pin'
                ],
                'weight': 1.0
            },
            '传动件-齿轮齿条': {
                'keywords': [
                    '齿轮', '齿条', 'gear', 'rack', 'spur gear', 'helical gear',
                    '锥齿轮', 'bevel gear', '蜗轮', 'worm gear', '正齿轮',
                    '斜齿轮', '直齿轮', 'gearbox', '行星齿轮', 'planetary gear'
                ],
                'weight': 0.9
            },
            '传动件-轴承': {
                'keywords': [
                    '轴承', 'bearing', '球轴承', '滚珠轴承', '滚子轴承', 'needle bearing',
                    '深沟球轴承', '角接触轴承', '圆锥滚子轴承', 'thrust bearing'
                ],
                'weight': 0.9
            },
            '建材-金属材料': {
                'keywords': [
                    '钢', '钢材', 'steel', '不锈钢', 'stainless steel', 'ss',
                    '铝合金', 'aluminum', 'alu', '铜材', 'copper', '锌材', 'zinc',
                    '铁材', 'iron', '金属板', 'metal', 'sheet', 'plate',
                    '棒材', 'bar', '管材', 'tube', 'pipe'
                ],
                'weight': 0.8
            },
            '建材-木材制品': {
                'keywords': [
                    '木', '木材', 'wood', 'timber', '实木', 'plywood', '胶合板',
                    '密度板', 'mdf', '刨花板', 'particle board', '木板', 'plank'
                ],
                'weight': 0.8
            },
            '液压气动': {
                'keywords': [
                    '液压', '气动', 'hydraulic', 'pneumatic', '气缸', 'cylinder',
                    '油缸', 'valve', '阀门', '接头', 'fitting', 'seal', '密封件'
                ],
                'weight': 0.8
            },
            '电子电气': {
                'keywords': [
                    '电子', '电气', 'electronic', 'electrical', '电路', 'circuit',
                    'pcb', '电路板', 'connector', '连接器', 'cable', '线缆'
                ],
                'weight': 0.7
            }
        }

        # 定制件识别关键词
        self.custom_indicators = [
            '定制', '异形', '特殊', '非标', '来图', '客户设计', 'oem', 'odm',
            'custom', 'special', 'bespoke', 'tailored', 'made-to-order',
            'sample', '样品', 'prototype', '原型'
        ]

        # 标准件识别关键词
        self.standard_indicators = [
            'din', 'iso', 'gb', 'ansi', 'astm', 'jis', 'bs', 'nf',
            '标准件', 'standard', 'std', 'norm'
        ]

        # 数据源识别
        self.data_source_patterns = {
            'email': ['email', '邮件', 'gmail', 'outlook', '@', '.com'],
            'wechat': ['微信', 'wechat', '企业微信', 'wx'],
            'manual': ['手动', 'manual', '人工录入'],
            'scan': ['扫描', 'scan', 'scanner'],
            'cad': ['cad', 'dwg', 'dxf', 'autocad', 'solidworks'],
            'sap': ['sap', '报价资料', '报价单', 'quotation'],
            'sample': ['sample', 'photos', '图片', 'image']
        }

    def get_db_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def calculate_category_score(self, drawing_name: str, category_config: Dict) -> float:
        """计算分类分数"""
        drawing_name_lower = drawing_name.lower()
        keywords = category_config['keywords']
        weight = category_config['weight']

        score = 0.0

        for keyword in keywords:
            if isinstance(keyword, str):
                # 精确匹配
                if keyword in drawing_name_lower:
                    score += 1.0
                # 部分匹配
                elif keyword in drawing_name_lower.split():
                    score += 0.5
            elif isinstance(keyword, re.Pattern):
                # 正则表达式匹配
                if keyword.search(drawing_name_lower):
                    score += 0.8

        return score * weight

    def classify_drawing(self, drawing_name: str) -> Dict[str, Any]:
        """分类单个图纸"""
        if not drawing_name:
            return {
                'product_category': '未分类',
                'standard_or_custom': 0,
                'classification_confidence': 0.0,
                'data_source': 'unknown'
            }

        drawing_name_lower = drawing_name.lower()

        # 清理文件名 - 移除常见的无意义前缀
        cleaned_name = re.sub(r'^(xyz-|abc-|sap\d+\s*-\s*|图_\d+\s*-\s*)', '', drawing_name_lower)

        # 1. 识别数据源
        data_source = 'unknown'
        for source, patterns in self.data_source_patterns.items():
            if any(pattern in drawing_name_lower for pattern in patterns):
                data_source = source
                break

        # 2. 识别是否为定制件
        is_custom = any(indicator in drawing_name_lower for indicator in self.custom_indicators)
        is_standard = any(indicator in drawing_name_lower for indicator in self.standard_indicators)

        # 3. 产品分类
        category_scores = {}
        for category, config in self.expanded_keywords.items():
            score = self.calculate_category_score(cleaned_name, config)
            if score > 0:
                category_scores[category] = score

        # 选择最高分数的分类
        best_category = '未分类'
        max_score = 0
        if category_scores:
            best_category, max_score = max(category_scores.items(), key=lambda x: x[1])

        # 4. 计算置信度
        confidence = min(max_score / 3.0, 1.0)  # 标准化到0-1

        # 5. 判断标准件/定制件
        if is_custom:
            standard_or_custom = 1  # 定制件
        elif is_standard or max_score > 2:
            standard_or_custom = 0  # 标准件
        else:
            standard_or_custom = 0  # 默认为标准件

        return {
            'product_category': best_category,
            'standard_or_custom': standard_or_custom,
            'classification_confidence': confidence,
            'data_source': data_source
        }

    def classify_all_drawings(self) -> Dict[str, Any]:
        """分类所有未分类的图纸"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        try:
            # 获取未分类的图纸
            cursor.execute("""
                SELECT id, drawing_name, product_category, is_classified
                FROM drawings
                WHERE is_classified = 0 OR product_category = '未分类'
            """)
            unclassified_drawings = cursor.fetchall()

            logger.info(f"📊 找到 {len(unclassified_drawings)} 个需要分类的图纸")

            classification_results = []
            category_counts = {}
            total_processed = 0
            successful_classifications = 0

            for drawing in unclassified_drawings:
                try:
                    # 进行分类
                    result = self.classify_drawing(drawing['drawing_name'])

                    # 更新数据库
                    cursor.execute("""
                        UPDATE drawings
                        SET product_category = ?,
                            standard_or_custom = ?,
                            classification_confidence = ?,
                            classification_date = ?,
                            is_classified = 1
                        WHERE id = ?
                    """, (
                        result['product_category'],
                        result['standard_or_custom'],
                        result['classification_confidence'],
                        datetime.now().isoformat(),
                        drawing['id']
                    ))

                    total_processed += 1

                    if result['product_category'] != '未分类':
                        successful_classifications += 1

                    # 统计分类结果
                    category = result['product_category']
                    if category not in category_counts:
                        category_counts[category] = 0
                    category_counts[category] += 1

                    classification_results.append({
                        'id': drawing['id'],
                        'drawing_name': drawing['drawing_name'],
                        'classification': result
                    })

                except Exception as e:
                    logger.error(f"处理图纸 {drawing['drawing_name']} 时出错: {e}")
                    continue

            conn.commit()
            conn.close()

            classification_rate = (successful_classifications / total_processed * 100) if total_processed > 0 else 0

            # 生成分类报告
            report = {
                'generated_at': datetime.now().isoformat(),
                'summary': {
                    'total_processed': total_processed,
                    'successful_classifications': successful_classifications,
                    'classification_rate': round(classification_rate, 1),
                    'category_distribution': category_counts
                },
                'detailed_results': classification_results
            }

            return report

        except Exception as e:
            logger.error(f"分类过程中出错: {e}")
            if conn:
                conn.close()
            raise

    def save_classification_report(self, report: Dict[str, Any]):
        """保存分类报告"""
        try:
            output_dir = Path("./data/processed")
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 保存JSON格式报告
            json_file = output_dir / f"optimized_classification_report_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(report, f, ensure_ascii=False, indent=2)

            logger.info(f"📄 分类报告已保存: {json_file}")
            return str(json_file)

        except Exception as e:
            logger.error(f"保存分类报告失败: {e}")

def main():
    """主函数"""
    logger.info("🚀 开始优化版图纸分类...")

    try:
        classifier = OptimizedDrawingClassifier()

        # 执行分类
        logger.info("📊 开始处理未分类图纸...")
        report = classifier.classify_all_drawings()

        # 显示结果
        summary = report['summary']
        logger.info(f"✅ 分类完成: {summary['successful_classifications']}/{summary['total_processed']} ({summary['classification_rate']}%)")
        logger.info(f"📊 分类分布: {summary['category_distribution']}")

        # 保存报告
        classifier.save_classification_report(report)

        print(f"\n🎉 优化版分类完成!")
        print(f"📊 处理总数: {summary['total_processed']}")
        print(f"🏷️ 成功分类: {summary['successful_classifications']}")
        print(f"📈 分类率: {summary['classification_rate']}%")
        print(f"📋 分类分布:")
        for category, count in summary['category_distribution'].items():
            print(f"   {category}: {count}")

    except Exception as e:
        logger.error(f"❌ 分类失败: {e}")
        return

if __name__ == "__main__":
    main()