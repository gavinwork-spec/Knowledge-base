"""
数据库初始化脚本
用于创建所有表结构并插入初始数据
"""

import os
import sqlite3
from datetime import datetime
from models import DatabaseManager, ALL_ENTITIES

def create_database_with_path(db_path: str = "./data/db.sqlite") -> DatabaseManager:
    """
    创建数据库并初始化所有表

    Args:
        db_path: 数据库文件路径

    Returns:
        DatabaseManager: 数据库管理器实例
    """
    print(f"正在初始化数据库: {db_path}")

    # 确保数据目录存在
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # 创建数据库管理器
    db_manager = DatabaseManager(db_path)

    # 创建所有表
    with db_manager:
        for entity_class in ALL_ENTITIES:
            entity_instance = entity_class(db_manager)
            entity_instance.create_table()
            print(f"✓ 已创建表: {entity_class.__name__.lower()}")

    print("✓ 数据库初始化完成!")
    return db_manager

def insert_sample_data(db_manager: DatabaseManager):
    """
    插入示例数据用于测试

    Args:
        db_manager: 数据库管理器实例
    """
    print("正在插入示例数据...")

    with db_manager:
        # 导入实体类
        from models import Customer, Factory, Drawing, FactoryQuote, Specification, ProcessStatus

        # 创建示例客户
        customer = Customer(db_manager)
        customer1_id = customer.create(
            company_name="ABC制造有限公司",
            contact_email="john@abc-manufacturing.com",
            contact_name="John Smith",
            country="美国",
            language="英语",
            phone="+1-555-0123",
            first_contact_date="2024-01-15",
            notes="主要生产精密零部件"
        )

        customer2_id = customer.create(
            company_name="XYZ科技公司",
            contact_email="zhang@xyz-tech.com",
            contact_name="张伟",
            country="中国",
            language="中文",
            phone="+86-138-0000-0000",
            first_contact_date="2024-02-20",
            notes="专注于电子产品制造"
        )

        print(f"✓ 创建了2个示例客户")

        # 创建示例工厂
        factory = Factory(db_manager)
        factory1_id = factory.create(
            factory_name="精密制造工厂A",
            location="广东省深圳市",
            capability="CNC加工、精密冲压",
            cost_reference="中等偏低",
            production_cycle="15-20天",
            notes="ISO9001认证"
        )

        factory2_id = factory.create(
            factory_name="金属制品厂B",
            location="江苏省苏州市",
            capability="铸造、锻造、热处理",
            cost_reference="低",
            production_cycle="25-30天",
            notes="擅长大批量生产"
        )

        print(f"✓ 创建了2个示例工厂")

        # 创建示例规格
        specification = Specification(db_manager)
        spec1_id = specification.create(
            product_category="精密螺丝",
            material="不锈钢304",
            standard_or_custom="standard",
            surface_treatment="镀镍",
            default_moq=1000,
            notes="M2-M6规格"
        )

        spec2_id = specification.create(
            product_category="定制齿轮",
            material="45号钢",
            standard_or_custom="custom",
            surface_treatment="渗碳淬火",
            default_moq=500,
            notes="模数1-5"
        )

        print(f"✓ 创建了2个示例规格")

        # 创建示例图纸
        drawing = Drawing(db_manager)
        drawing1_id = drawing.create(
            drawing_name="ABC-001-精密螺丝",
            customer_id=customer1_id,
            product_category="精密螺丝",
            file_path="/drawings/abc-001.pdf",
            upload_date="2024-03-01",
            status="confirmed",
            notes="客户确认版本"
        )

        drawing2_id = drawing.create(
            drawing_name="XYZ-002-定制齿轮",
            customer_id=customer2_id,
            product_category="定制齿轮",
            file_path="/drawings/xyz-002.dwg",
            upload_date="2024-03-15",
            status="pending",
            notes="等待客户确认"
        )

        print(f"✓ 创建了2个示例图纸")

        # 创建示例工厂报价
        quote = FactoryQuote(db_manager)
        quote.create(
            factory_id=factory1_id,
            product_category="精密螺丝",
            quote_date="2024-03-02",
            price=0.15,
            moq=1000,
            notes="不锈钢304材质"
        )

        quote.create(
            factory_id=factory2_id,
            product_category="定制齿轮",
            quote_date="2024-03-16",
            price=25.50,
            moq=500,
            notes="45号钢，含热处理"
        )

        print(f"✓ 创建了2个示例报价")

        # 创建示例流程状态
        process = ProcessStatus(db_manager)
        process.create(
            drawing_id=drawing1_id,
            customer_id=customer1_id,
            status="sample",
            last_update_date="2024-03-05",
            notes="样品制作中，预计3月10日完成"
        )

        process.create(
            drawing_id=drawing2_id,
            customer_id=customer2_id,
            status="drawing_confirmation",
            last_update_date="2024-03-15",
            notes="等待客户图纸确认"
        )

        print(f"✓ 创建了2个示例流程状态")

    print("✓ 示例数据插入完成!")

def verify_database_setup(db_manager: DatabaseManager):
    """
    验证数据库设置是否正确

    Args:
        db_manager: 数据库管理器实例
    """
    print("正在验证数据库设置...")

    with db_manager:
        conn = db_manager.connect()
        cursor = conn.cursor()

        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]

        expected_tables = ['customers', 'factories', 'drawings', 'factory_quotes', 'specifications', 'process_status']

        for expected_table in expected_tables:
            if expected_table in table_names:
                cursor.execute(f"SELECT COUNT(*) FROM {expected_table}")
                count = cursor.fetchone()[0]
                print(f"✓ 表 {expected_table}: {count} 条记录")
            else:
                print(f"✗ 表 {expected_table}: 未找到")

        # 检查外键约束是否启用
        cursor.execute("PRAGMA foreign_key_list")
        foreign_keys = cursor.fetchall()
        print(f"✓ 外键约束: {'已启用' if len(foreign_keys) > 0 else '未启用'}")

    print("✓ 数据库验证完成!")

def main():
    """
    主函数：完整的数据库初始化流程
    """
    print("=" * 50)
    print("数据库初始化脚本")
    print("=" * 50)

    # 数据库路径
    db_path = "./data/db.sqlite"

    try:
        # 创建数据库和表结构
        db_manager = create_database_with_path(db_path)

        # 插入示例数据
        insert_sample_data(db_manager)

        # 验证设置
        verify_database_setup(db_manager)

        print("\n" + "=" * 50)
        print("数据库初始化成功完成！")
        print(f"数据库文件位置: {os.path.abspath(db_path)}")
        print("=" * 50)

    except Exception as e:
        print(f"❌ 数据库初始化失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()