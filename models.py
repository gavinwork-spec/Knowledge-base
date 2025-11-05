"""
数据库模型定义
包含所有核心实体：Customer, Factory, Drawing, FactoryQuote, Specification, ProcessStatus
"""

import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any

class DatabaseManager:
    """数据库管理类"""

    def __init__(self, db_path: str = "./data/db.sqlite"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self):
        """连接数据库"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # 使结果可以按列名访问
        return self.conn

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class Customer:
    """客户实体"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_table(self):
        """创建客户表"""
        conn = self.db.connect()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name TEXT NOT NULL,
                contact_name TEXT,
                contact_email TEXT,
                country TEXT,
                language TEXT,
                phone TEXT,
                first_contact_date TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 创建索引以提高查询性能
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(contact_email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_customers_company ON customers(company_name)')

        conn.commit()

    def create(self, company_name: str, contact_email: str, **kwargs) -> int:
        """创建新客户"""
        conn = self.db.connect()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO customers (
                company_name, contact_name, contact_email, country,
                language, phone, first_contact_date, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            company_name,
            kwargs.get('contact_name'),
            contact_email,
            kwargs.get('country'),
            kwargs.get('language'),
            kwargs.get('phone'),
            kwargs.get('first_contact_date'),
            kwargs.get('notes')
        ))

        conn.commit()
        return cursor.lastrowid

    def get_by_id(self, customer_id: int) -> Optional[Dict[str, Any]]:
        """根据ID获取客户"""
        conn = self.db.connect()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM customers WHERE id = ?', (customer_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """根据邮箱获取客户"""
        conn = self.db.connect()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM customers WHERE contact_email = ?', (email,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_by_company_and_email(self, company_name: str, contact_email: str) -> Optional[Dict[str, Any]]:
        """根据公司名称和邮箱获取客户（核心查询方法）"""
        conn = self.db.connect()
        cursor = conn.cursor()

        cursor.execute(
            'SELECT * FROM customers WHERE company_name = ? AND contact_email = ?',
            (company_name, contact_email)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all(self) -> List[Dict[str, Any]]:
        """获取所有客户"""
        conn = self.db.connect()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM customers ORDER BY company_name')
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

class Factory:
    """工厂实体"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_table(self):
        """创建工厂表"""
        conn = self.db.connect()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS factories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                factory_name TEXT NOT NULL,
                location TEXT,
                capability TEXT,
                cost_reference TEXT,
                production_cycle TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_factories_name ON factories(factory_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_factories_location ON factories(location)')

        conn.commit()

    def create(self, factory_name: str, **kwargs) -> int:
        """创建新工厂"""
        conn = self.db.connect()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO factories (
                factory_name, location, capability, cost_reference,
                production_cycle, notes
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            factory_name,
            kwargs.get('location'),
            kwargs.get('capability'),
            kwargs.get('cost_reference'),
            kwargs.get('production_cycle'),
            kwargs.get('notes')
        ))

        conn.commit()
        return cursor.lastrowid

    def get_by_id(self, factory_id: int) -> Optional[Dict[str, Any]]:
        """根据ID获取工厂"""
        conn = self.db.connect()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM factories WHERE id = ?', (factory_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

class Drawing:
    """图纸记录实体"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_table(self):
        """创建图纸表"""
        conn = self.db.connect()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drawings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drawing_name TEXT NOT NULL,
                customer_id INTEGER,
                product_category TEXT,
                file_path TEXT,
                upload_date TEXT,
                status TEXT DEFAULT 'pending',
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (customer_id) REFERENCES customers (id)
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_drawings_customer ON drawings(customer_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_drawings_category ON drawings(product_category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_drawings_status ON drawings(status)')

        conn.commit()

    def create(self, drawing_name: str, **kwargs) -> int:
        """创建新图纸记录"""
        conn = self.db.connect()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO drawings (
                drawing_name, customer_id, product_category, file_path,
                upload_date, status, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            drawing_name,
            kwargs.get('customer_id'),  # 可以为None
            kwargs.get('product_category'),
            kwargs.get('file_path'),
            kwargs.get('upload_date', datetime.now().isoformat()),
            kwargs.get('status', 'pending'),
            kwargs.get('notes')
        ))

        conn.commit()
        return cursor.lastrowid

class FactoryQuote:
    """工厂报价实体"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_table(self):
        """创建工厂报价表"""
        conn = self.db.connect()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS factory_quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                factory_id INTEGER,
                product_category TEXT,
                quote_date TEXT,
                price REAL,
                moq INTEGER,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (factory_id) REFERENCES factories (id)
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_quotes_factory ON factory_quotes(factory_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_quotes_category ON factory_quotes(product_category)')

        conn.commit()

    def create(self, factory_id: int, product_category: str, price: float, **kwargs) -> int:
        """创建新工厂报价"""
        conn = self.db.connect()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO factory_quotes (
                factory_id, product_category, quote_date, price, moq, notes
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            factory_id,
            product_category,
            kwargs.get('quote_date', datetime.now().isoformat()),
            price,
            kwargs.get('moq'),
            kwargs.get('notes')
        ))

        conn.commit()
        return cursor.lastrowid

class Specification:
    """规格实体"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_table(self):
        """创建规格表"""
        conn = self.db.connect()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS specifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_category TEXT NOT NULL,
                material TEXT,
                standard_or_custom TEXT DEFAULT 'standard',
                surface_treatment TEXT,
                default_moq INTEGER,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_specifications_category ON specifications(product_category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_specifications_material ON specifications(material)')

        conn.commit()

    def create(self, product_category: str, **kwargs) -> int:
        """创建新规格"""
        conn = self.db.connect()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO specifications (
                product_category, material, standard_or_custom,
                surface_treatment, default_moq, notes
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            product_category,
            kwargs.get('material'),
            kwargs.get('standard_or_custom', 'standard'),
            kwargs.get('surface_treatment'),
            kwargs.get('default_moq'),
            kwargs.get('notes')
        ))

        conn.commit()
        return cursor.lastrowid

class ProcessStatus:
    """定制流程状态实体"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def create_table(self):
        """创建流程状态表"""
        conn = self.db.connect()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS process_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drawing_id INTEGER,
                customer_id INTEGER,
                status TEXT DEFAULT 'drawing_confirmation',
                last_update_date TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (drawing_id) REFERENCES drawings (id),
                FOREIGN KEY (customer_id) REFERENCES customers (id)
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_process_status_drawing ON process_status(drawing_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_process_status_customer ON process_status(customer_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_process_status_status ON process_status(status)')

        conn.commit()

    def create(self, drawing_id: int, customer_id: int, status: str = 'drawing_confirmation', **kwargs) -> int:
        """创建新流程状态"""
        conn = self.db.connect()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO process_status (
                drawing_id, customer_id, status, last_update_date, notes
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            drawing_id,
            customer_id,
            status,
            kwargs.get('last_update_date', datetime.now().isoformat()),
            kwargs.get('notes')
        ))

        conn.commit()
        return cursor.lastrowid

# 所有实体类的列表，用于批量创建表
ALL_ENTITIES = [
    Customer,
    Factory,
    Drawing,
    FactoryQuote,
    Specification,
    ProcessStatus
]

# 状态选项常量
PROCESS_STATUSES = [
    'drawing_confirmation',  # 图纸确认
    'sample',              # 样品
    'batch_production'     # 批量
]

STANDARD_CUSTOM_OPTIONS = [
    'standard',  # 标准
    'custom'     # 定制
]