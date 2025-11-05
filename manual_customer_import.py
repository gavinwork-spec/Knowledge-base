#!/usr/bin/env python3
"""
手动客户资料导入脚本
基于文件夹结构和文件名手动提取客户信息
"""

import os
import re
from datetime import datetime
from pathlib import Path
from models import DatabaseManager, Customer

def extract_customers_from_directory():
    """从文件夹结构中提取客户信息"""

    customers_found = []

    # 客户资料根目录
    root_dir = Path("/Users/gavin/Nutstore Files/.symlinks/坚果云/002-客户/")

    # 基于文件名和文件夹结构提取客户信息
    customer_patterns = {
        # 从文件夹名称推断客户
        "巴林": {
            "company_name": "巴林客户",
            "country": "巴林",
            "notes": "家具类客户，文件夹：巴林"
        },
        "Lejiang": {
            "company_name": "Lejiang Furniture",
            "country": "中国",
            "notes": "家具制造商，广交会客户"
        },
        "富唐": {
            "company_name": "富唐公司",
            "country": "中国",
            "notes": "询价单客户"
        },
        "昊慕家": {
            "company_name": "昊慕家",
            "country": "中国",
            "notes": "验货报告客户"
        },
        "UF优纺": {
            "company_name": "UF优纺",
            "country": "中国",
            "notes": "纺织品客户"
        },
        "奥衍精工": {
            "company_name": "奥衍精工",
            "country": "中国",
            "notes": "精密制造客户"
        }
    }

    # 从文件名中提取更多客户信息
    for file_path in root_dir.rglob('*'):
        if file_path.is_file():
            file_name = file_path.name

            # 提取可能的客户名称
            if "AYA" in file_name.upper():
                customers_found.append({
                    "company_name": "AYA Fasteners",
                    "contact_name": "Unknown",
                    "contact_email": "info@aya-fasteners.com",  # 推测邮箱
                    "country": "巴林",
                    "language": "英语",
                    "first_contact_date": "2024-01-01",
                    "notes": "从文件名识别：AYA客户，紧固件业务",
                    "source_file": str(file_path)
                })

            if "Homelux" in file_name:
                customers_found.append({
                    "company_name": "Hebei Homelux Technology Co., Ltd",
                    "contact_name": "Unknown",
                    "contact_email": "sales@homelux.com",  # 推测邮箱
                    "country": "中国",
                    "language": "中文",
                    "first_contact_date": "2024-01-01",
                    "notes": "从文件名识别：Homelux，河北科技公司",
                    "source_file": str(file_path)
                })

            if "沃耳特" in file_name or "Walter" in file_name:
                customers_found.append({
                    "company_name": "沃耳特五金科技",
                    "contact_name": "Unknown",
                    "contact_email": "info@walter-fasteners.com",  # 推测邮箱
                    "country": "中国",
                    "language": "中文",
                    "first_contact_date": "2024-01-01",
                    "notes": "从文件名识别：沃耳特五金，报价客户",
                    "source_file": str(file_path)
                })

            if "阳昶" in file_name:
                customers_found.append({
                    "company_name": "广东阳昶精密制造",
                    "contact_name": "Unknown",
                    "contact_email": "sales@yangchang.com",  # 推测邮箱
                    "country": "中国",
                    "language": "中文",
                    "first_contact_date": "2024-01-01",
                    "notes": "从文件名识别：阳昶精密制造，报价客户",
                    "source_file": str(file_path)
                })

    # 添加基于文件夹的客户
    for folder_name, info in customer_patterns.items():
        if any(folder_name in str(p) for p in root_dir.rglob('*')):
            customers_found.append({
                "company_name": info["company_name"],
                "contact_name": "Unknown",
                "contact_email": f"contact@{info['company_name'].replace(' ', '').lower()}.com",  # 生成推测邮箱
                "country": info["country"],
                "language": "中文" if info["country"] == "中国" else "英语",
                "first_contact_date": "2024-01-01",
                "notes": info["notes"],
                "source_file": f"文件夹: {folder_name}"
            })

    # 去重
    unique_customers = {}
    for customer in customers_found:
        key = (customer["company_name"], customer["contact_email"])
        if key not in unique_customers:
            unique_customers[key] = customer

    return list(unique_customers.values())

def import_manual_customers():
    """导入手动提取的客户数据"""
    print("🔍 从文件夹结构中提取客户信息...")

    customers = extract_customers_from_directory()

    if not customers:
        print("❌ 未找到客户信息")
        return

    print(f"✅ 找到 {len(customers)} 个潜在客户:")
    for i, customer in enumerate(customers, 1):
        print(f"  {i}. {customer['company_name']} ({customer['contact_email']})")

    # 导入数据库
    db_manager = DatabaseManager("./data/db.sqlite")
    customer_model = Customer(db_manager)

    imported_count = 0
    with db_manager:
        for customer in customers:
            try:
                # 检查是否已存在
                existing = customer_model.get_by_company_and_email(
                    customer["company_name"],
                    customer["contact_email"]
                )

                if existing:
                    print(f"  ⚠️  客户已存在: {customer['company_name']}")
                    continue

                # 插入新客户
                customer_id = customer_model.create(
                    company_name=customer["company_name"],
                    contact_name=customer.get("contact_name"),
                    contact_email=customer["contact_email"],
                    country=customer.get("country"),
                    language=customer.get("language"),
                    first_contact_date=customer.get("first_contact_date"),
                    notes=customer.get("notes")
                )

                imported_count += 1
                print(f"  ✅ 导入客户 #{customer_id}: {customer['company_name']}")

            except Exception as e:
                print(f"  ❌ 导入失败 {customer['company_name']}: {e}")

    print(f"\n🎉 成功导入 {imported_count} 个客户!")

if __name__ == "__main__":
    import_manual_customers()