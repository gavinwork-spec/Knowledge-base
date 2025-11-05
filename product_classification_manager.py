#!/usr/bin/env python3
"""
产品分类管理脚本
管理和更新数据库中的产品分类，支持紧固件、家具、建材三大类别的分类体系
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from models import DatabaseManager, Drawing, Specification

class ProductClassificationManager:
    """产品分类管理器"""

    def __init__(self, db_path: str = "./data/db.sqlite"):
        self.db_manager = DatabaseManager(db_path)
        self.classification_data = self._load_classification_data()

    def _load_classification_data(self):
        """加载分类数据"""
        return {
            "fasteners": {
                "level_1": {
                    "standard": {
                        "code": "STANDARD",
                        "name": "标准件",
                        "description": "按国际/国家/行业标准生产的通用紧固件"
                    },
                    "custom": {
                        "code": "CUSTOM",
                        "name": "定制件",
                        "description": "根据客户特定要求设计的紧固件"
                    }
                },
                "level_2": {
                    "bolt_screw": {
                        "code": "BOL_SCR",
                        "name": "螺栓螺钉",
                        "parent": "both"
                    },
                    "nut": {
                        "code": "NUT",
                        "name": "螺母",
                        "parent": "both"
                    },
                    "washer": {
                        "code": "WAS",
                        "name": "垫圈",
                        "parent": "both"
                    },
                    "pin_rivet": {
                        "code": "PIN_RIV",
                        "name": "销铆钉",
                        "parent": "both"
                    }
                },
                "level_3": {
                    # 螺栓螺钉类细分
                    "hex_bolt": {"code": "HEX_BOL", "name": "六角螺栓", "parent": "bolt_screw"},
                    "socket_bolt": {"code": "SOK_BOL", "name": "内六角螺栓", "parent": "bolt_screw"},
                    "countersunk_bolt": {"code": "COUNTER_BOL", "name": "沉头螺栓", "parent": "bolt_screw"},
                    "carriage_bolt": {"code": "CARR_BOL", "name": "马车螺栓", "parent": "bolt_screw"},
                    "self_tapping": {"code": "SELF_TAP", "name": "自攻螺钉", "parent": "bolt_screw"},
                    "machine_screw": {"code": "MAC_SCR", "name": "机制螺钉", "parent": "bolt_screw"},
                    "wood_screw": {"code": "WD_SCR", "name": "木螺钉", "parent": "bolt_screw"},
                    "drywall_screw": {"code": "DW_SCR", "name": "干壁螺钉", "parent": "bolt_screw"},
                    "drilling_screw": {"code": "DRILL_SCR", "name": "钻尾螺钉", "parent": "bolt_screw"},

                    # 螺母类细分
                    "hex_nut": {"code": "HEX_NUT", "name": "六角螺母", "parent": "nut"},
                    "flange_nut": {"code": "FLA_NUT", "name": "法兰螺母", "parent": "nut"},
                    "lock_nut": {"code": "LOCK_NUT", "name": "锁紧螺母", "parent": "nut"},
                    "wing_nut": {"code": "WING_NUT", "name": "蝶形螺母", "parent": "nut"},
                    "cap_nut": {"code": "CAP_NUT", "name": "盖形螺母", "parent": "nut"},
                    "weld_nut": {"code": "WELD_NUT", "name": "焊接螺母", "parent": "nut"},

                    # 垫圈类细分
                    "flat_washer": {"code": "FLAT_WAS", "name": "平垫圈", "parent": "washer"},
                    "spring_washer": {"code": "SPR_WAS", "name": "弹簧垫圈", "parent": "washer"},
                    "lock_washer": {"code": "LOCK_WAS", "name": "锁紧垫圈", "parent": "washer"},

                    # 销铆钉类细分
                    "dowel_pin": {"code": "DOW_PIN", "name": "圆柱销", "parent": "pin_rivet"},
                    "taper_pin": {"code": "TAP_PIN", "name": "锥形销", "parent": "pin_rivet"},
                    "split_pin": {"code": "SPL_PIN", "name": "开口销", "parent": "pin_rivet"},
                    "solid_rivet": {"code": "SOL_RIV", "name": "实心铆钉", "parent": "pin_rivet"},
                    "blind_rivet": {"code": "BLD_RIV", "name": "盲铆钉", "parent": "pin_rivet"}
                }
            },
            "furniture": {
                "level_1": {
                    "office": {
                        "code": "OFFICE",
                        "name": "办公家具",
                        "description": "办公室使用的家具产品"
                    },
                    "residential": {
                        "code": "RESIDENTIAL",
                        "name": "民用家具",
                        "description": "家庭使用的家具产品"
                    },
                    "outdoor": {
                        "code": "OUTDOOR",
                        "name": "户外家具",
                        "description": "户外使用的家具产品"
                    },
                    "hotel": {
                        "code": "HOTEL",
                        "name": "酒店家具",
                        "description": "酒店专用的家具产品"
                    },
                    "commercial": {
                        "code": "COMMERCIAL",
                        "name": "商用家具",
                        "description": "商业用途的家具产品"
                    }
                },
                "level_2": {
                    "seating": {"code": "SEAT", "name": "座椅类", "parent": ["office", "residential", "hotel"]},
                    "tables": {"code": "TAB", "name": "桌类", "parent": ["office", "residential", "hotel"]},
                    "sofas": {"code": "SOF", "name": "沙发类", "parent": ["office", "residential", "hotel"]},
                    "storage": {"code": "STOR", "name": "收纳类", "parent": ["residential", "office"]},
                    "beds": {"code": "BED", "name": "床类", "parent": ["residential", "hotel"]},
                    "cabinets": {"code": "CAB", "name": "柜类", "parent": ["office", "residential", "hotel"]}
                },
                "level_3": {
                    # 座椅类细分
                    "office_chair": {"code": "OFF_CHR", "name": "办公椅", "parent": "seating"},
                    "conference_chair": {"code": "CONF_CHR", "name": "会议椅", "parent": "seating"},
                    "reception_chair": {"code": "REC_CHR", "name": "接待椅", "parent": "seating"},
                    "lounge_chair": {"code": "LOUN_CHR", "name": "休闲椅", "parent": "seating"},
                    "gaming_chair": {"code": "GAME_CHR", "name": "电竞椅", "parent": "seating"},

                    # 桌类细分
                    "office_desk": {"code": "OFF_DESK", "name": "办公桌", "parent": "tables"},
                    "conference_table": {"code": "CONF_TAB", "name": "会议桌", "parent": "tables"},
                    "reception_desk": {"code": "REC_DESK", "name": "接待台", "parent": "tables"},
                    "coffee_table": {"code": "COF_TAB", "name": "茶几", "parent": "tables"},

                    # 沙发类细分
                    "office_sofa": {"code": "OFF_SOF", "name": "办公沙发", "parent": "sofas"},
                    "reception_sofa": {"code": "REC_SOF", "name": "接待沙发", "parent": "sofas"},
                    "lounge_sofa": {"code": "LOUN_SOF", "name": "休闲沙发", "parent": "sofas"},

                    # 床类细分
                    "double_bed": {"code": "DOUB_BED", "name": "双人床", "parent": "beds"},
                    "single_bed": {"code": "SIN_BED", "name": "单人床", "parent": "beds"},
                    "bunk_bed": {"code": "BUNK_BED", "name": "上下铺", "parent": "beds"},

                    # 柜类细分
                    "wardrobe": {"code": "WARD", "name": "衣柜", "parent": "cabinets"},
                    "bookcase": {"code": "BOOK", "name": "书柜", "parent": "cabinets"},
                    "storage_cabinet": {"code": "STOR_CAB", "name": "储物柜", "parent": "cabinets"}
                }
            },
            "building_materials": {
                "level_1": {
                    "basic": {
                        "code": "BASIC",
                        "name": "基础建材",
                        "description": "建筑基础结构和主体材料"
                    },
                    "decorative": {
                        "code": "DECORATIVE",
                        "name": "装饰建材",
                        "description": "建筑装饰和饰面材料"
                    },
                    "specialized": {
                        "code": "SPECIALIZED",
                        "name": "专用建材",
                        "description": "具有特殊功能的建筑材料"
                    }
                },
                "level_2": {
                    "metal_materials": {"code": "METAL", "name": "金属材料", "parent": ["basic", "specialized"]},
                    "wood_materials": {"code": "WOOD", "name": "木材竹材", "parent": ["basic", "decorative"]},
                    "plastic_materials": {"code": "PLASTIC", "name": "塑料材料", "parent": ["basic", "specialized"]},
                    "finishing_materials": {"code": "FINISH", "name": "饰面材料", "parent": "decorative"},
                    "ceiling_materials": {"code": "CEILING", "name": "吊顶材料", "parent": "decorative"},
                    "door_window": {"code": "DOOR_WIN", "name": "门窗材料", "parent": "decorative"},
                    "waterproofing": {"code": "WATERPROOF", "name": "防水材料", "parent": "specialized"},
                    "insulation": {"code": "INSULATION", "name": "保温材料", "parent": "specialized"},
                    "sound_insulation": {"code": "SOUND", "name": "隔音材料", "parent": "specialized"}
                },
                "level_3": {
                    # 金属材料细分
                    "steel": {"code": "STL", "name": "钢材", "parent": "metal_materials"},
                    "aluminum": {"code": "ALUM", "name": "铝材", "parent": "metal_materials"},
                    "stainless_steel": {"code": "SS", "name": "不锈钢", "parent": "metal_materials"},
                    "copper": {"code": "COP", "name": "铜材", "parent": "metal_materials"},

                    # 木材竹材细分
                    "solid_wood": {"code": "SOLID_WD", "name": "实木", "parent": "wood_materials"},
                    "plywood": {"code": "PLY", "name": "胶合板", "parent": "wood_materials"},
                    "mdf": {"code": "MDF", "name": "中纤板", "parent": "wood_materials"},
                    "particle_board": {"code": "PART", "name": "刨花板", "parent": "wood_materials"},

                    # 饰面材料细分
                    "paint": {"code": "PAINT", "name": "涂料", "parent": "finishing_materials"},
                    "wallpaper": {"code": "WALLPAPER", "name": "壁纸", "parent": "finishing_materials"},
                    "tiles": {"code": "TILES", "name": "瓷砖", "parent": "finishing_materials"},

                    # 吊顶材料细分
                    "mineral_board": {"code": "MINERAL", "name": "矿棉板", "parent": "ceiling_materials"},
                    "gypsum_board": {"code": "GYPSUM", "name": "石膏板", "parent": "ceiling_materials"},
                    "aluminum_ceiling": {"code": "ALUM_CEIL", "name": "铝扣板", "parent": "ceiling_materials"},

                    # 防水材料细分
                    "waterproof_membrane": {"code": "WP_MEM", "name": "防水卷材", "parent": "waterproofing"},
                    "waterproof_coating": {"code": "WP_COAT", "name": "防水涂料", "parent": "waterproofing"},

                    # 保温材料细分
                    "rock_wool": {"code": "ROCK", "name": "岩棉", "parent": "insulation"},
                    "glass_wool": {"code": "GLASS", "name": "玻璃棉", "parent": "insulation"},
                    "eps_xps": {"code": "EPS_XPS", "name": "聚苯板/挤塑板", "parent": "insulation"}
                }
            }
        }

    def classify_product_by_name(self, product_name):
        """根据产品名称进行分类"""
        product_name = product_name.lower()

        # 紧固件关键词
        if any(keyword in product_name for keyword in ['螺栓', '螺钉', '螺丝', '螺母', '垫圈', '销', '铆钉', 'screw', 'bolt', 'nut', 'washer', 'pin', 'rivet']):
            return self._classify_fastener(product_name)

        # 家具关键词
        elif any(keyword in product_name for keyword in ['椅子', '沙发', '桌子', '床', '柜子', '家具', 'chair', 'sofa', 'table', 'bed', 'cabinet']):
            return self._classify_furniture(product_name)

        # 建材关键词
        elif any(keyword in product_name for keyword in ['钢材', '钢板', '瓷砖', '涂料', '油漆', '防水', '保温', '门', '窗', '吊顶', '钢筋', '钢管', '铝材']):
            return self._classify_building_material(product_name)

        # 默认未分类
        return {
            "main_category": "未分类",
            "sub_category": "未分类",
            "detail_category": "未分类",
            "confidence": 0.1,
            "reason": "关键词匹配失败"
        }

    def _classify_fastener(self, product_name):
        """分类紧固件"""
        fasteners = self.classification_data["fasteners"]

        # 检测是否为定制件
        custom_keywords = ['定制', '异形', '特殊', '非标', '来图', '客户设计']
        is_custom = any(keyword in product_name for keyword in custom_keywords)

        level_1_code = "custom" if is_custom else "standard"
        level_1_info = fasteners["level_1"][level_1_code]

        # 二级分类
        category = None
        category_mapping = {
            "bolt_screw": ["螺栓", "螺钉", "螺丝", "screw", "bolt"],
            "nut": ["螺母", "nut"],
            "washer": ["垫圈", "washer", "垫片"],
            "pin_rivet": ["销", "铆钉", "pin", "rivet"]
        }

        for cat, keywords in category_mapping.items():
            if any(keyword in product_name for keyword in keywords):
                category = cat
                break

        # 三级分类
        detail_category = None
        if category == "bolt_screw":
            bolt_mapping = {
                "hex_bolt": ["六角", "hex", "外六角"],
                "socket_bolt": ["内六角", "socket", "内六"],
                "countersunk_bolt": ["沉头", "平头", "countersunk"],
                "carriage_bolt": ["马车", "carriage"],
                "self_tapping": ["自攻", "self tapping", "自钻"],
                "drilling_screw": ["钻尾", "drilling", "尾牙"]
            }

            for detail, keywords in bolt_mapping.items():
                if any(keyword in product_name for keyword in keywords):
                    detail_category = detail
                    break

        elif category == "nut":
            nut_mapping = {
                "hex_nut": ["六角螺母", "hex nut"],
                "flange_nut": ["法兰", "flange"],
                "lock_nut": ["锁紧", "lock"],
                "wing_nut": ["蝶形", "wing"]
            }

            for detail, keywords in nut_mapping.items():
                if any(keyword in product_name for keyword in keywords):
                    detail_category = detail
                    break

        return {
            "main_category": "紧固件",
            "sub_category": level_1_info["name"],
            "detail_category": self._get_category_name(fasteners, detail_category or category),
            "code": self._generate_code("F", level_1_info["code"], category, detail_category),
            "confidence": 0.8 if detail_category else 0.6,
            "is_custom": is_custom,
            "level_1": level_1_code,
            "level_2": category,
            "level_3": detail_category
        }

    def _classify_furniture(self, product_name):
        """分类家具"""
        furniture = self.classification_data["furniture"]

        # 一级分类
        category_mapping = {
            "office": ["办公", "office", "职员", "员工"],
            "residential": ["住宅", "家用", "卧室", "客厅", "餐厅", "书房"],
            "outdoor": ["户外", "庭院", "露台", "阳台"],
            "hotel": ["酒店", "客房", "大堂", "套房"],
            "commercial": ["商业", "教育", "医疗", "展示"]
        }

        level_1 = None
        for cat, keywords in category_mapping.items():
            if any(keyword in product_name for keyword in keywords):
                level_1 = cat
                break

        level_1 = level_1 or "residential"  # 默认为住宅
        level_1_info = furniture["level_1"][level_1]

        # 二级分类
        category_mapping = {
            "seating": ["椅子", "座椅", "椅"],
            "tables": ["桌子", "桌", "台"],
            "sofas": ["沙发", "sofa", "组合"],
            "storage": ["柜", "架", "储物", "收纳"],
            "beds": ["床", "bed"],
            "cabinets": ["柜", "橱", "cabinet"]
        }

        level_2 = None
        for cat, keywords in category_mapping.items():
            # 检查父级是否匹配
            parent_list = furniture["level_2"][cat]["parent"]
            if isinstance(parent_list, list):
                if level_1 in parent_list:
                    level_2 = cat
                    break
            elif parent_list == "both":
                level_2 = cat
                break
            elif parent_list == level_1:
                level_2 = cat
                break

        # 三级分类
        detail_category = None
        if level_2 == "seating":
            seating_mapping = {
                "office_chair": ["办公椅", "office chair"],
                "conference_chair": ["会议椅", "conference chair"],
                "reception_chair": ["接待椅", "reception chair"],
                "lounge_chair": ["休闲椅", "lounge chair"]
            }

            for detail, keywords in seating_mapping.items():
                if any(keyword in product_name for keyword in keywords):
                    detail_category = detail
                    break

        return {
            "main_category": "家具",
            "sub_category": level_1_info["name"],
            "detail_category": self._get_category_name(furniture, detail_category or level_2),
            "code": self._generate_code("FUR", level_1_info["code"], level_2, detail_category),
            "confidence": 0.7 if detail_category else 0.5,
            "level_1": level_1,
            "level_2": level_2,
            "level_3": detail_category
        }

    def _classify_building_material(self, product_name):
        """分类建材"""
        building = self.classification_data["building_materials"]

        # 一级分类
        category_mapping = {
            "basic": ["基础", "结构", "承重", "主体"],
            "decorative": ["装饰", "饰面", "装修", "美观"],
            "specialized": ["防水", "保温", "隔音", "防火", "专用"]
        }

        level_1 = None
        for cat, keywords in category_mapping.items():
            if any(keyword in product_name for keyword in keywords):
                level_1 = cat
                break

        level_1 = level_1 or "basic"  # 默认为基础建材
        level_1_info = building["level_1"][level_1]

        # 二级分类
        category_mapping = {
            "metal_materials": ["钢", "铁", "铝", "铜", "金属", "合金"],
            "wood_materials": ["木", "竹", "木材", "纤维板", "胶合板"],
            "plastic_materials": ["塑料", "pvc", "pp", "pe", "abs"],
            "finishing_materials": ["涂料", "油漆", "壁纸", "饰面"],
            "ceiling_materials": ["吊顶", "天花", "矿棉", "石膏板"],
            "door_window": ["门", "窗", "五金", "合页", "门锁"],
            "waterproofing": ["防水", "卷材", "涂料", "防渗"],
            "insulation": ["保温", "隔热", "岩棉", "玻璃棉", "eps"],
            "sound_insulation": ["隔音", "吸音", "声学"]
        }

        level_2 = None
        for cat, keywords in category_mapping.items():
            # 检查父级是否匹配
            parent_list = building["level_2"][cat]["parent"]
            if isinstance(parent_list, list):
                if level_1 in parent_list:
                    level_2 = cat
                    break
            elif parent_list == level_1:
                level_2 = cat
                break

        # 三级分类
        detail_category = None
        if level_2 == "metal_materials":
            metal_mapping = {
                "steel": ["钢材", "碳钢", "不锈钢"],
                "aluminum": ["铝材", "铝合金", "铝板"],
                "stainless_steel": ["不锈钢", "ss304", "ss316"]
            }

            for detail, keywords in metal_mapping.items():
                if any(keyword in product_name for keyword in keywords):
                    detail_category = detail
                    break

        return {
            "main_category": "建材",
            "sub_category": level_1_info["name"],
            "detail_category": self._get_category_name(building, detail_category or level_2),
            "code": self._generate_code("BUL", level_1_info["code"], level_2, detail_category),
            "confidence": 0.6 if detail_category else 0.4,
            "level_1": level_1,
            "level_2": level_2,
            "level_3": detail_category
        }

    def _get_category_name(self, data, category_key):
        """获取分类名称"""
        if not category_key:
            return "未分类"

        # 在三级分类中查找
        if "level_3" in data and category_key in data["level_3"]:
            return data["level_3"][category_key]["name"]

        # 在二级分类中查找
        if "level_2" in data and category_key in data["level_2"]:
            return data["level_2"][category_key]["name"]

        return "未分类"

    def _generate_code(self, prefix, level_1_code, level_2, level_3):
        """生成分类编码"""
        code_parts = [prefix]

        if level_1_code:
            code_parts.append(level_1_code)

        if level_2 and level_2 in self.classification_data.get(prefix.lower(), {}).get("level_2", {}):
            code_parts.append(self.classification_data[prefix.lower()]["level_2"][level_2]["code"])

        if level_3 and level_3 in self.classification_data.get(prefix.lower(), {}).get("level_3", {}):
            code_parts.append(self.classification_data[prefix.lower()]["level_3"][level_3]["code"])

        return "-".join(code_parts)

    def update_database_categories(self):
        """更新数据库中的产品分类"""
        print("🔄 更新数据库产品分类...")

        with self.db_manager:
            conn = self.db_manager.connect()
            cursor = conn.cursor()

            # 更新图纸表
            cursor.execute("SELECT id, drawing_name, product_category FROM drawings")
            drawings = cursor.fetchall()

            updated_count = 0
            for drawing_id, drawing_name, current_category in drawings:
                # 智能分类
                classification = self.classify_product_by_name(drawing_name)

                if classification["confidence"] > 0.5:  # 置信度阈值
                    new_category = classification["detail_category"]
                    if new_category != "未分类":
                        cursor.execute(
                            "UPDATE drawings SET product_category = ? WHERE id = ?",
                            (new_category, drawing_id)
                        )
                        updated_count += 1
                        print(f"  更新: {drawing_name[:30]}... → {new_category}")

            # 更新规格表
            cursor.execute("SELECT id, product_category FROM specifications")
            specs = cursor.fetchall()

            spec_updated = 0
            for spec_id, current_category in specs:
                if current_category and current_category != "未分类":
                    classification = self.classify_product_by_name(current_category)
                    if classification["confidence"] > 0.5:
                        new_category = classification["detail_category"]
                        if new_category != "未分类":
                            cursor.execute(
                                "UPDATE specifications SET product_category = ? WHERE id = ?",
                                (new_category, spec_id)
                            )
                            spec_updated += 1

            conn.commit()
            print(f"✅ 更新完成: 图纸 {updated_count} 个, 规格 {spec_updated} 个")

    def get_classification_statistics(self):
        """获取分类统计"""
        print("📊 产品分类统计")
        print("=" * 50)

        with self.db_manager:
            conn = self.db_manager.connect()
            cursor = conn.cursor()

            # 统计各主类别的数量
            cursor.execute("""
                SELECT
                    CASE
                        WHEN product_category LIKE '%六角%' OR product_category LIKE '%螺丝%' OR product_category LIKE '%螺母%'
                        THEN '紧固件'
                        WHEN product_category LIKE '%椅子%' OR product_category LIKE '%沙发%' OR product_category LIKE '%桌%' OR product_category LIKE '%床%'
                        THEN '家具'
                        WHEN product_category LIKE '%钢%' OR product_category LIKE '%瓷砖%' OR product_category LIKE '%涂料%' OR product_category LIKE '%防水%'
                        THEN '建材'
                        ELSE '其他'
                    END as main_category,
                    COUNT(*) as count
                FROM drawings
                GROUP BY main_category
                ORDER BY count DESC
            """)

            results = cursor.fetchall()

            for main_category, count in results:
                print(f"  {main_category}: {count} 个图纸")

            print(f"\n📁 详细分类:")

            # 获取详细分类统计
            cursor.execute("""
                SELECT product_category, COUNT(*) as count
                FROM drawings
                WHERE product_category != '未分类'
                GROUP BY product_category
                ORDER BY count DESC
                LIMIT 20
            """)

            detailed_results = cursor.fetchall()
            for category, count in detailed_results:
                print(f"  {category}: {count} 个")

    def export_classification_data(self):
        """导出分类数据"""
        print("📤 导出分类数据...")

        # 导出分类规则
        rules_file = Path("./data/processed/classification_rules.json")
        with open(rules_file, 'w', encoding='utf-8') as f:
            json.dump(self.classification_data, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 分类规则: {rules_file}")

        # 导出当前分类结果
        self.update_database_categories()

        # 导出分类统计
        stats_file = Path("./data/processed/classification_stats.json")
        stats = {}

        with self.db_manager:
            conn = self.db_manager.connect()
            cursor = conn.cursor()

            cursor.execute("SELECT product_category, COUNT(*) FROM drawings GROUP BY product_category")
            for category, count in cursor.fetchall():
                stats[category] = count

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 分类统计: {stats_file}")

    def run_full_classification(self):
        """运行完整的分类流程"""
        print("🚀 开始产品分类流程...")
        print("=" * 60)

        try:
            self.export_classification_data()
            self.update_database_categories()
            self.get_classification_statistics()

            print("\n" + "=" * 60)
            print("✅ 产品分类完成!")
            print("=" * 60)

        except Exception as e:
            print(f"❌ 分类失败: {e}")
            raise

def main():
    """主函数"""
    classifier = ProductClassificationManager()
    classifier.run_full_classification()

if __name__ == "__main__":
    main()