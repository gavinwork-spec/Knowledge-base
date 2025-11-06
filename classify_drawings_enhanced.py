#!/usr/bin/env python3
"""
增强版图纸自动分类脚本
进一步优化分类算法，处理更多边界情况
"""

import sqlite3
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 设置日志
# 确保日志目录存在
Path('./logs').mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/classify_drawings_enhanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ClassifyDrawingsEnhanced')

class EnhancedDrawingClassifier:
    """增强版图纸分类器"""

    def __init__(self):
        self.db_path = "./data/db.sqlite"

        # 进一步扩展的关键词库
        self.keyword_library = {
            '紧固件-螺栓螺钉': {
                # 英文关键词
                'english': ['bolt', 'screw', 'thread', 'fastener', 'hex', 'socket', 'allen', 'cap', 'machine', 'wood', 'self', 'tapping', 'drilling', 'countersunk', 'pan', 'round', 'flat', 'phillips', 'slotted', 'torx', 'star'],
                # 中文关键词
                'chinese': ['螺栓', '螺钉', '螺丝', '螺纹', '六角', '内六角', '外六角', '沉头', '盘头', '圆头', '平头', '十字', '一字', '机牙', '自攻', '钻尾'],
                # 德文关键词
                'german': ['schraube', 'schrauben', 'gewinde', 'bolzen'],
                # 规格模式
                'patterns': [r'm\d+x\d+', r'\d+\.\d+x\d+', r'din\s*\d+', r'iso\s*\d+'],
                # 标准标识
                'standards': ['din933', 'din912', 'din7991', 'iso4014', 'iso4762']
            },
            '紧固件-螺母': {
                'english': ['nut', 'hex nut', 'lock nut', 'flange nut', 'wing nut', 'nylon nut', 'square nut', 'weld nut', 'cap nut'],
                'chinese': ['螺母', '螺帽', '六角螺母', '法兰螺母', '锁紧螺母', '蝶形螺母', '盖形螺母', '方形螺母', '焊接螺母'],
                'patterns': [r'm\d+', r'din\s*\d+']
            },
            '紧固件-垫圈垫片': {
                'english': ['washer', 'flat washer', 'spring washer', 'lock washer', 'split washer', 'tooth washer', 'internal tooth', 'external tooth'],
                'chinese': ['垫圈', '垫片', '平垫圈', '弹簧垫圈', '防松垫圈', '齿形垫圈', '波形垫圈'],
                'patterns': [r'din\s*\d+']
            },
            '传动件-齿轮齿条': {
                'english': ['gear', 'rack', 'spur', 'helical', 'bevel', 'worm', 'pinion', 'sprocket', 'timing'],
                'chinese': ['齿轮', '齿条', '正齿轮', '斜齿轮', '锥齿轮', '蜗轮', '蜗杆', '链轮', '同步轮'],
                'german': ['zahnrad', 'ritzel', 'stirnrad', 'kegelrad', 'schneckenrad'],
                'patterns': [r'module\s*\d+', r'z\d+']
            },
            '传动件-轴承': {
                'english': ['bearing', 'ball bearing', 'roller bearing', 'needle bearing', 'thrust bearing', 'angular contact', 'deep groove'],
                'chinese': ['轴承', '滚珠轴承', '滚子轴承', '圆锥滚子轴承', '推力轴承', '角接触轴承'],
                'patterns': [r'\d{3,4}rs', r'\d{3,4}zz', r'ucf\d+']
            },
            '传动件-皮带链条': {
                'english': ['belt', 'chain', 'timing belt', 'v-belt', 'synchronous', 'conveyor', 'roller chain'],
                'chinese': ['皮带', '链条', '同步带', 'v带', '传送带', '滚子链'],
                'patterns': [r'b\d+x\d+', r'no\.?\d+']
            },
            '建材-金属材料': {
                'english': ['steel', 'stainless', 'aluminum', 'copper', 'brass', 'bronze', 'iron', 'metal', 'sheet', 'plate', 'bar', 'rod', 'tube', 'pipe'],
                'chinese': ['钢', '不锈钢', '铝合金', '铜', '黄铜', '青铜', '铁', '金属', '板材', '棒材', '管材'],
                'abbreviations': ['ss', 'sus', 'alu', 'cu', 'fe'],
                'patterns': [r'sus\d+', r'ss\d+', r'alu\d+']
            },
            '建材-木材制品': {
                'english': ['wood', 'timber', 'plywood', 'mdf', 'particle board', 'lumber', 'plank'],
                'chinese': ['木', '木材', '胶合板', '密度板', '刨花板', '木板'],
                'patterns': [r'ply\d+', r'mdf']
            },
            '液压气动': {
                'english': ['hydraulic', 'pneumatic', 'cylinder', 'valve', 'pump', 'motor', 'fitting', 'connector', 'seal', 'o-ring'],
                'chinese': ['液压', '气动', '气缸', '油缸', '阀门', '泵', '电机', '接头', '密封圈', 'o型圈'],
                'abbreviations': ['hyd', 'pnu']
            },
            '电子电气': {
                'english': ['pcb', 'circuit', 'connector', 'cable', 'wire', 'terminal', 'switch', 'sensor', 'led'],
                'chinese': ['电路板', '连接器', '线缆', '电线', '端子', '开关', '传感器'],
                'abbreviations': ['ic', 'mcu', 'pcb']
            },
            '模具工具': {
                'english': ['mold', 'die', 'tool', 'cutter', 'drill', 'tap', 'reamer', 'broach'],
                'chinese': ['模具', '刀具', '钻头', '丝锥', '铰刀', '拉刀'],
                'patterns': [r'din\d+[a-z]*']
            }
        }

        # 无意义文件名模式（应该跳过分类）
        self.skip_patterns = [
            r'^图_\d+\s*-\s*企业微信截图',
            r'^screenshot',
            r'^img_\d+',
            r'^photo_\d+',
            r'^图片',
            r'^sample\s*photo',
            r'^企业微信截图',
            r'^wechat',
            r'^email\s*attachment'
        ]

        # 定制件指示词
        self.custom_indicators = [
            '定制', '异形', '特殊', '非标', '来图', '客户设计', 'oem', 'odm',
            'custom', 'special', 'bespoke', 'tailored', 'made-to-order',
            'prototype', '样品', 'sample', 'test', 'testing'
        ]

        # 标准件指示词
        self.standard_indicators = [
            'din', 'iso', 'gb', 'ansi', 'astm', 'jis', 'bs', 'nf',
            '标准件', 'standard', 'std', 'norm', 'normen'
        ]

    def get_db_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def should_skip_classification(self, drawing_name: str) -> bool:
        """检查是否应该跳过分类"""
        drawing_name_lower = drawing_name.lower()

        for pattern in self.skip_patterns:
            if re.match(pattern, drawing_name_lower):
                return True
        return False

    def calculate_category_scores(self, drawing_name: str) -> Dict[str, float]:
        """计算各分类的得分"""
        drawing_name_lower = drawing_name.lower()
        category_scores = {}

        for category, keywords_dict in self.keyword_library.items():
            score = 0.0

            # 检查各类关键词
            for keyword_type, keywords in keywords_dict.items():
                if keyword_type == 'patterns':
                    # 正则表达式匹配
                    for pattern in keywords:
                        if re.search(pattern, drawing_name_lower):
                            score += 1.0
                elif keyword_type == 'standards':
                    # 标准标识匹配
                    for standard in keywords:
                        if standard in drawing_name_lower:
                            score += 1.5  # 标准匹配加分
                else:
                    # 普通关键词匹配
                    for keyword in keywords:
                        if keyword in drawing_name_lower:
                            # 精确匹配
                            score += 1.0
                        elif keyword in drawing_name_lower.split():
                            # 单词匹配
                            score += 0.7
                        elif drawing_name_lower.find(keyword) != -1:
                            # 部分匹配
                            score += 0.3

            if score > 0:
                category_scores[category] = score

        return category_scores

    def classify_drawing(self, drawing_name: str) -> Dict[str, Any]:
        """分类单个图纸"""
        if not drawing_name:
            return {
                'product_category': '未分类',
                'standard_or_custom': 0,
                'classification_confidence': 0.0,
                'data_source': 'unknown',
                'skip_reason': '空文件名'
            }

        # 检查是否应该跳过
        if self.should_skip_classification(drawing_name):
            return {
                'product_category': '未分类',
                'standard_or_custom': 0,
                'classification_confidence': 0.0,
                'data_source': 'screenshot',
                'skip_reason': '截图文件'
            }

        drawing_name_lower = drawing_name.lower()

        # 1. 识别数据源
        data_source = 'unknown'
        if '企业微信' in drawing_name_lower or 'wechat' in drawing_name_lower:
            data_source = 'wechat'
        elif 'email' in drawing_name_lower or '@' in drawing_name_lower:
            data_source = 'email'
        elif 'sap' in drawing_name_lower or '报价' in drawing_name_lower:
            data_source = 'quotation'
        elif 'sample' in drawing_name_lower or 'photo' in drawing_name_lower:
            data_source = 'image'
        elif 'cad' in drawing_name_lower or 'dwg' in drawing_name_lower:
            data_source = 'cad'

        # 2. 计算分类得分
        category_scores = self.calculate_category_scores(drawing_name)

        # 3. 选择最佳分类
        max_score = 0
        if category_scores:
            best_category, max_score = max(category_scores.items(), key=lambda x: x[1])
            confidence = min(max_score / 4.0, 1.0)  # 标准化置信度
        else:
            best_category = '未分类'
            confidence = 0.0

        # 4. 判断标准件/定制件
        is_custom = any(indicator in drawing_name_lower for indicator in self.custom_indicators)
        is_standard = any(indicator in drawing_name_lower for indicator in self.standard_indicators)

        if is_custom:
            standard_or_custom = 1  # 定制件
        elif is_standard or max_score >= 2:
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
            # 获取所有图纸（包括已分类的，以便重新分类）
            cursor.execute("SELECT id, drawing_name, product_category, is_classified FROM drawings")
            all_drawings = cursor.fetchall()

            logger.info(f"📊 找到 {len(all_drawings)} 个图纸进行重新分类")

            classification_results = []
            category_counts = {}
            total_processed = 0
            successful_classifications = 0
            skipped_count = 0
            source_counts = {}

            for drawing in all_drawings:
                try:
                    # 进行分类
                    result = self.classify_drawing(drawing['drawing_name'])

                    # 统计数据源
                    source = result.get('data_source', 'unknown')
                    if source not in source_counts:
                        source_counts[source] = 0
                    source_counts[source] += 1

                    # 如果是截图文件，跳过
                    if result.get('skip_reason'):
                        skipped_count += 1
                        total_processed += 1
                        continue

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
                    'skipped_count': skipped_count,
                    'classification_rate': round(classification_rate, 1),
                    'category_distribution': category_counts,
                    'data_source_distribution': source_counts
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
            json_file = output_dir / f"enhanced_classification_report_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(report, f, ensure_ascii=False, indent=2)

            logger.info(f"📄 分类报告已保存: {json_file}")
            return str(json_file)

        except Exception as e:
            logger.error(f"保存分类报告失败: {e}")

def main():
    """主函数"""
    logger.info("🚀 开始增强版图纸分类...")

    try:
        classifier = EnhancedDrawingClassifier()

        # 执行分类
        logger.info("📊 开始处理图纸...")
        report = classifier.classify_all_drawings()

        # 显示结果
        summary = report['summary']
        logger.info(f"✅ 分类完成: {summary['successful_classifications']}/{summary['total_processed']} ({summary['classification_rate']}%)")
        logger.info(f"⏭️ 跳过文件: {summary['skipped_count']}")
        logger.info(f"📊 分类分布: {summary['category_distribution']}")
        logger.info(f"📡 数据源分布: {summary['data_source_distribution']}")

        # 保存报告
        classifier.save_classification_report(report)

        print(f"\n🎉 增强版分类完成!")
        print(f"📊 处理总数: {summary['total_processed']}")
        print(f"🏷️ 成功分类: {summary['successful_classifications']}")
        print(f"⏭️ 跳过截图: {summary['skipped_count']}")
        print(f"📈 分类率: {summary['classification_rate']}%")
        print(f"📋 分类分布:")
        for category, count in summary['category_distribution'].items():
            print(f"   {category}: {count}")
        print(f"📡 数据源分布:")
        for source, count in summary['data_source_distribution'].items():
            print(f"   {source}: {count}")

    except Exception as e:
        logger.error(f"❌ 分类失败: {e}")
        return

if __name__ == "__main__":
    main()