#!/usr/bin/env python3
"""
图纸自动分类脚本
根据图纸名称自动分类产品类别和标准件/定制件标识
"""

import sqlite3
import re
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class DrawingClassifier:
    """图纸分类器"""

    def __init__(self, db_path: str = "./data/db.sqlite"):
        self.db_path = db_path
        self.setup_logging()
        self.load_classification_rules()

    def setup_logging(self):
        """设置日志"""
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'classify_drawings.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DrawingClassifier')

    def load_classification_rules(self):
        """加载分类规则"""
        # 紧固件关键词
        self.fastener_keywords = {
            '螺栓螺钉': [
                '螺栓', '螺钉', '螺丝', 'bolt', 'screw', '机器螺钉', '六角螺栓',
                '内六角', '外六角', 'hex', 'hex bolt', 'socket screw', 'machine screw',
                '自攻', '自钻', '钻尾', 'self tapping', 'self drilling', 'tek screw',
                '沉头', '平头', '盘头', 'countersunk', 'pan head', 'round head',
                '马车螺栓', 'carriage bolt', '方头螺栓', 'square bolt'
            ],
            '螺母': [
                '螺母', 'nut', '六角螺母', 'hex nut', '法兰螺母', 'flange nut',
                '锁紧螺母', 'lock nut', '盖形螺母', 'cap nut', '蝶形螺母', 'wing nut',
                '焊接螺母', 'weld nut', '嵌入螺母', 'insert nut'
            ],
            '垫圈垫片': [
                '垫圈', '垫片', 'washer', 'flat washer', 'spring washer', 'lock washer',
                '平垫圈', '弹簧垫圈', '齿形垫圈', '波形垫圈', '防松垫圈'
            ],
            '销铆钉': [
                '销', '铆钉', 'pin', 'rivet', '圆柱销', 'dowel pin', '开口销', 'cotter pin',
                '弹性销', 'spring pin', '半圆头铆钉', 'blind rivet', '拉铆钉', 'pull rivet'
            ]
        }

        # 家具关键词
        self.furniture_keywords = {
            '座椅类': [
                '椅子', '座椅', '椅', 'chair', '座椅', '办公椅', 'office chair',
                '会议椅', 'conference chair', '接待椅', 'reception chair', '吧椅', 'bar stool'
            ],
            '桌台类': [
                '桌子', '桌', '台', 'table', 'desk', '办公桌', 'office desk',
                '会议桌', 'conference table', '茶几', 'coffee table', '接待台', 'reception desk'
            ],
            '沙发类': [
                '沙发', 'sofa', '组合沙发', 'sectional sofa', '办公沙发', 'office sofa',
                '休息椅', 'lounge chair', '贵妃椅', 'chaise lounge'
            ],
            '柜架类': [
                '柜', '架', 'cabinet', 'shelf', '衣柜', 'wardrobe', '书柜', 'bookcase',
                '文件柜', 'filing cabinet', '储物柜', 'storage cabinet', '展示柜', 'display cabinet'
            ],
            '床具类': [
                '床', 'bed', '双人床', 'double bed', '单人床', 'single bed', '上下铺', 'bunk bed'
            ]
        }

        # 建材关键词
        self.building_materials_keywords = {
            '金属材料': [
                '钢', '钢材', 'steel', '不锈钢', 'stainless steel', '铝合金', 'aluminum alloy',
                '铜材', 'copper', '锌材', 'zinc', '铁材', 'iron', '金属', 'metal'
            ],
            '木材材料': [
                '木', '木材', 'wood', 'timber', '实木', 'solid wood', '胶合板', 'plywood',
                '密度板', 'mdf', '刨花板', 'particle board', '细木工板', 'blockboard'
            ],
            '装饰材料': [
                '瓷砖', 'tile', '涂料', 'paint', '油漆', 'coating', '壁纸', 'wallpaper',
                '地板', 'flooring', '吊顶', 'ceiling', '门窗', 'door', 'window'
            ],
            '防水保温': [
                '防水', 'waterproof', '保温', 'insulation', '密封', 'sealing', '胶带', 'tape'
            ]
        }

        # 定制件标识关键词
        self.custom_keywords = [
            '定制', '异形', '特殊', '非标', '来图', '客户设计', 'custom', 'special',
            'bespoke', 'tailored', 'made-to-order', 'oem', 'odm'
        ]

        # 数据源标识
        self.data_source_patterns = {
            'email': ['email', '邮件', 'gmail', 'outlook', '@'],
            'wechat': ['微信', 'wechat', '企业微信'],
            'manual': ['手动', 'manual', '人工录入'],
            'scan': ['扫描', 'scan', 'scanner'],
            'cad': ['cad', 'dwg', 'dxf', 'autocad']
        }

    def classify_drawing_name(self, drawing_name: str) -> Dict[str, any]:
        """分类单个图纸名称"""
        if not drawing_name:
            return {
                'product_category': None,
                'standard_or_custom': False,
                'classification_confidence': 0.0,
                'data_source': 'unknown'
            }

        drawing_name_lower = drawing_name.lower()
        classification_result = {
            'product_category': None,
            'standard_or_custom': False,
            'classification_confidence': 0.0,
            'data_source': 'unknown'
        }

        # 1. 检查是否为定制件
        is_custom = any(keyword in drawing_name_lower for keyword in self.custom_keywords)
        classification_result['standard_or_custom'] = is_custom

        # 2. 产品分类
        category_scores = {}

        # 检查紧固件类别
        for category, keywords in self.fastener_keywords.items():
            score = sum(1 for keyword in keywords if keyword in drawing_name_lower)
            if score > 0:
                category_scores[f"紧固件-{category}"] = score

        # 检查家具类别
        for category, keywords in self.furniture_keywords.items():
            score = sum(1 for keyword in keywords if keyword in drawing_name_lower)
            if score > 0:
                category_scores[f"家具-{category}"] = score

        # 检查建材类别
        for category, keywords in self.building_materials_keywords.items():
            score = sum(1 for keyword in keywords if keyword in drawing_name_lower)
            if score > 0:
                category_scores[f"建材-{category}"] = score

        # 选择得分最高的类别
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            classification_result['product_category'] = best_category

            # 计算置信度
            max_score = category_scores[best_category]
            total_keywords = sum(len(kw_list) for kw_group in
                               [self.fastener_keywords, self.furniture_keywords, self.building_materials_keywords]
                               for kw_list in kw_group.values())
            classification_result['classification_confidence'] = min(max_score / 3.0, 1.0)  # 标准化到0-1

        # 3. 识别数据源
        for source, patterns in self.data_source_patterns.items():
            if any(pattern in drawing_name_lower for pattern in patterns):
                classification_result['data_source'] = source
                break

        return classification_result

    def classify_all_drawings(self) -> Dict[str, any]:
        """分类所有未分类的图纸"""
        self.logger.info("🚀 开始图纸分类...")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 获取需要分类的图纸
            cursor.execute("""
                SELECT id, drawing_name, product_category, is_classified
                FROM drawings
                WHERE product_category IS NULL
                   OR product_category = '未分类'
                   OR is_classified = 0
                   OR is_classified IS NULL
                ORDER BY id
            """)

            drawings_to_classify = cursor.fetchall()
            total_drawings = len(drawings_to_classify)

            self.logger.info(f"📊 找到 {total_drawings} 个需要分类的图纸")

            if total_drawings == 0:
                self.logger.info("✅ 没有需要分类的图纸")
                conn.close()
                return {'total_processed': 0, 'classified': 0, 'unclassified': 0}

            # 分类结果统计
            classified_count = 0
            unclassified_count = 0
            classification_results = []

            # 批量更新数据
            update_data = []

            for drawing_id, drawing_name, current_category, is_classified in drawings_to_classify:
                classification = self.classify_drawing_name(drawing_name)

                if classification['product_category']:
                    # 成功分类
                    update_data.append((
                        classification['product_category'],
                        classification['standard_or_custom'],
                        classification['classification_confidence'],
                        classification['data_source'],
                        True,  # is_classified
                        datetime.now().isoformat(),  # classification_date
                        drawing_id
                    ))
                    classified_count += 1
                    classification_results.append({
                        'id': drawing_id,
                        'name': drawing_name[:50] + '...' if len(drawing_name) > 50 else drawing_name,
                        'category': classification['product_category'],
                        'confidence': classification['classification_confidence'],
                        'is_custom': classification['standard_or_custom']
                    })
                else:
                    # 无法分类
                    update_data.append((
                        '未分类',
                        False,
                        0.0,
                        classification['data_source'],
                        True,
                        datetime.now().isoformat(),
                        drawing_id
                    ))
                    unclassified_count += 1
                    classification_results.append({
                        'id': drawing_id,
                        'name': drawing_name[:50] + '...' if len(drawing_name) > 50 else drawing_name,
                        'category': '未分类',
                        'confidence': 0.0,
                        'is_custom': False
                    })

            # 批量更新数据库
            if update_data:
                cursor.executemany("""
                    UPDATE drawings
                    SET product_category = ?,
                        standard_or_custom = ?,
                        classification_confidence = ?,
                        data_source = ?,
                        is_classified = ?,
                        classification_date = ?
                    WHERE id = ?
                """, update_data)

                conn.commit()

            # 生成分类统计
            cursor.execute("""
                SELECT product_category, COUNT(*) as count
                FROM drawings
                WHERE is_classified = 1
                GROUP BY product_category
                ORDER BY count DESC
            """)

            category_stats = cursor.fetchall()

            result = {
                'total_processed': total_drawings,
                'classified': classified_count,
                'unclassified': unclassified_count,
                'classification_rate': (classified_count / total_drawings * 100) if total_drawings > 0 else 0,
                'category_distribution': dict(category_stats),
                'sample_results': classification_results[:10],  # 前10个结果示例
                'timestamp': datetime.now().isoformat()
            }

            conn.close()

            self.logger.info(f"✅ 分类完成: {classified_count}/{total_drawings} ({result['classification_rate']:.1f}%)")
            self.logger.info(f"📊 分类分布: {dict(category_stats)}")

            return result

        except Exception as e:
            self.logger.error(f"❌ 分类失败: {e}")
            return {'error': str(e)}

    def reclassify_all_drawings(self) -> Dict[str, any]:
        """重新分类所有图纸"""
        self.logger.info("🔄 开始重新分类所有图纸...")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 重置所有分类状态
            cursor.execute("""
                UPDATE drawings
                SET is_classified = 0,
                    classification_date = NULL
            """)

            conn.commit()
            conn.close()

            self.logger.info("✅ 已重置所有分类状态")

            # 重新执行分类
            return self.classify_all_drawings()

        except Exception as e:
            self.logger.error(f"❌ 重新分类失败: {e}")
            return {'error': str(e)}

    def save_classification_report(self, result: Dict[str, any]) -> str:
        """保存分类报告"""
        try:
            report_dir = Path("./data/processed")
            report_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"classification_report_{timestamp}.json"

            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            self.logger.info(f"📄 分类报告已保存: {report_file}")
            return str(report_file)

        except Exception as e:
            self.logger.error(f"❌ 保存报告失败: {e}")
            return ""

    def get_classification_statistics(self) -> Dict[str, any]:
        """获取分类统计信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 总体统计
            cursor.execute("SELECT COUNT(*) FROM drawings")
            total_drawings = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM drawings WHERE is_classified = 1")
            classified_drawings = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM drawings WHERE standard_or_custom = 1")
            custom_drawings = cursor.fetchone()[0]

            # 按类别统计
            cursor.execute("""
                SELECT product_category, COUNT(*) as count
                FROM drawings
                GROUP BY product_category
                ORDER BY count DESC
            """)
            category_stats = cursor.fetchall()

            # 按数据源统计
            cursor.execute("""
                SELECT data_source, COUNT(*) as count
                FROM drawings
                GROUP BY data_source
                ORDER BY count DESC
            """)
            source_stats = cursor.fetchall()

            # 置信度分布
            cursor.execute("""
                SELECT
                    CASE
                        WHEN classification_confidence >= 0.8 THEN '高'
                        WHEN classification_confidence >= 0.5 THEN '中'
                        ELSE '低'
                    END as confidence_level,
                    COUNT(*) as count
                FROM drawings
                WHERE classification_confidence > 0
                GROUP BY confidence_level
            """)
            confidence_stats = cursor.fetchall()

            conn.close()

            return {
                'total_drawings': total_drawings,
                'classified_drawings': classified_drawings,
                'unclassified_drawings': total_drawings - classified_drawings,
                'classification_rate': (classified_drawings / total_drawings * 100) if total_drawings > 0 else 0,
                'custom_drawings': custom_drawings,
                'custom_rate': (custom_drawings / total_drawings * 100) if total_drawings > 0 else 0,
                'category_distribution': dict(category_stats),
                'source_distribution': dict(source_stats),
                'confidence_distribution': dict(confidence_stats),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"❌ 获取统计失败: {e}")
            return {'error': str(e)}

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='图纸自动分类工具')
    parser.add_argument('--db-path', default='./data/db.sqlite', help='数据库文件路径')
    parser.add_argument('--reclassify', action='store_true', help='重新分类所有图纸')
    parser.add_argument('--stats', action='store_true', help='仅显示分类统计')
    parser.add_argument('--report', action='store_true', help='生成详细报告')

    args = parser.parse_args()

    classifier = DrawingClassifier(args.db_path)

    if args.stats:
        stats = classifier.get_classification_statistics()
        print("📊 图纸分类统计:")
        print(f"  总图纸数: {stats.get('total_drawings', 0)}")
        print(f"  已分类: {stats.get('classified_drawings', 0)}")
        print(f"  未分类: {stats.get('unclassified_drawings', 0)}")
        print(f"  分类率: {stats.get('classification_rate', 0):.1f}%")
        print(f"  定制件: {stats.get('custom_drawings', 0)} ({stats.get('custom_rate', 0):.1f}%)")

        print("\n📋 类别分布:")
        for category, count in stats.get('category_distribution', {}).items():
            print(f"  {category}: {count}")

    else:
        # 执行分类
        if args.reclassify:
            result = classifier.reclassify_all_drawings()
        else:
            result = classifier.classify_all_drawings()

        if 'error' in result:
            print(f"❌ 分类失败: {result['error']}")
        else:
            print("✅ 图纸分类完成!")
            print(f"📊 处理总数: {result['total_processed']}")
            print(f"🏷️ 成功分类: {result['classified']}")
            print(f"❓ 未分类: {result['unclassified']}")
            print(f"📈 分类率: {result['classification_rate']:.1f}%")

            # 保存报告
            if args.report:
                report_file = classifier.save_classification_report(result)
                if report_file:
                    print(f"📄 详细报告: {report_file}")

if __name__ == "__main__":
    main()