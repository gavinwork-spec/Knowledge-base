-- ============================================================================
-- Trading Company Database Migration Script
-- Version: 1.0
-- Date: 2025-01-10
-- Description: Adds trading company specific tables and modifies existing tables
-- ============================================================================

BEGIN TRANSACTION;

-- ============================================================================
-- 1. Raw Materials Price Monitoring Tables
-- ============================================================================

-- Raw materials master data
CREATE TABLE IF NOT EXISTS raw_materials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_code TEXT UNIQUE NOT NULL,
    material_name TEXT NOT NULL,
    category TEXT NOT NULL, -- 钢材、不锈钢、铜材、铝材等
    unit TEXT NOT NULL DEFAULT '吨', -- 吨、公斤、件
    standard TEXT, -- 国标、美标、德标等
    description TEXT,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Raw material price history
CREATE TABLE IF NOT EXISTS raw_material_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_id INTEGER NOT NULL,
    price REAL NOT NULL,
    currency TEXT DEFAULT 'CNY',
    supplier TEXT,
    market TEXT DEFAULT 'SHFE', -- 上海期货交易所、LME等
    price_date DATE NOT NULL,
    change_amount REAL DEFAULT 0,
    change_percentage REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (material_id) REFERENCES raw_materials (id)
);

-- Exchange rates
CREATE TABLE IF NOT EXISTS exchange_rates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_currency TEXT NOT NULL,
    to_currency TEXT NOT NULL,
    rate REAL NOT NULL,
    date DATE NOT NULL,
    source TEXT DEFAULT '央行', -- 央行、银行、市场等
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(from_currency, to_currency, date)
);

-- Price alerts
CREATE TABLE IF NOT EXISTS price_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    material_id INTEGER,
    alert_type TEXT NOT NULL CHECK (alert_type IN ('price_above', 'price_below', 'price_change')),
    threshold_value REAL NOT NULL,
    is_active BOOLEAN DEFAULT 1,
    notification_sent BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (material_id) REFERENCES raw_materials (id)
);

-- ============================================================================
-- 2. Enhanced Customer Management
-- ============================================================================

-- Add trading specific fields to existing customers table
ALTER TABLE customers ADD COLUMN customer_tier TEXT DEFAULT 'B' CHECK (customer_tier IN ('A+', 'A', 'B', 'C', 'D'));
ALTER TABLE customers ADD COLUMN annual_revenue REAL DEFAULT 0;
ALTER TABLE customers ADD COLUMN payment_terms TEXT DEFAULT 'NET30';
ALTER TABLE customers ADD COLUMN credit_limit REAL DEFAULT 0;
ALTER TABLE customers ADD COLUMN sales_representative TEXT;
ALTER TABLE customers ADD COLUMN last_order_date DATE;
ALTER TABLE customers ADD COLUMN total_orders INTEGER DEFAULT 0;
ALTER TABLE customers ADD COLUMN customer_since DATE;
ALTER TABLE customers ADD COLUMN preferred_language TEXT DEFAULT '中文';
ALTER TABLE customers ADD COLUMN time_zone TEXT DEFAULT 'Asia/Shanghai';
ALTER TABLE customers ADD COLUMN discount_rate REAL DEFAULT 0;

-- Customer contacts table (expand contact management)
CREATE TABLE IF NOT EXISTS customer_contacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    contact_name TEXT NOT NULL,
    position TEXT,
    email TEXT UNIQUE,
    phone TEXT,
    mobile TEXT,
    is_primary BOOLEAN DEFAULT 0,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers (id) ON DELETE CASCADE
);

-- Customer interactions tracking
CREATE TABLE IF NOT EXISTS customer_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    contact_id INTEGER,
    interaction_type TEXT NOT NULL CHECK (interaction_type IN ('email', 'phone', 'meeting', 'visit', 'quote', 'order', 'complaint')),
    subject TEXT,
    content TEXT,
    direction TEXT CHECK (direction IN ('inbound', 'outbound')),
    status TEXT DEFAULT 'open',
    priority TEXT DEFAULT 'medium' CHECK (priority IN ('high', 'medium', 'low')),
    assigned_to TEXT,
    next_follow_up DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers (id) ON DELETE CASCADE,
    FOREIGN KEY (contact_id) REFERENCES customer_contacts (id)
);

-- ============================================================================
-- 3. Enhanced Supplier Management
-- ============================================================================

-- Add trading specific fields to factories table (rename to suppliers)
ALTER TABLE factories RENAME TO suppliers;

-- Add new columns to suppliers table
ALTER TABLE suppliers ADD COLUMN supplier_tier TEXT DEFAULT 'B' CHECK (supplier_tier IN ('A+', 'A', 'B', 'C', 'D'));
ALTER TABLE suppliers ADD COLUMN payment_terms TEXT DEFAULT 'NET30';
ALTER TABLE suppliers ADD COLUMN lead_time_days INTEGER DEFAULT 30;
ALTER TABLE suppliers ADD COLUMN quality_rating REAL DEFAULT 0 CHECK (quality_rating >= 0 AND quality_rating <= 5);
ALTER TABLE suppliers ADD COLUMN on_time_delivery_rate REAL DEFAULT 0;
ALTER TABLE suppliers ADD COLUMN total_orders INTEGER DEFAULT 0;
ALTER TABLE suppliers ADD COLUMN total_value REAL DEFAULT 0;
ALTER TABLE suppliers ADD COLUMN contact_person TEXT;
ALTER TABLE suppliers ADD COLUMN contact_email TEXT;
ALTER TABLE suppliers ADD COLUMN contact_phone TEXT;
ALTER TABLE suppliers ADD COLUMN certifications TEXT; -- JSON array of certifications
ALTER TABLE suppliers ADD COLUMN is_active BOOLEAN DEFAULT 1;
ALTER TABLE suppliers ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
ALTER TABLE suppliers ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

-- Supplier performance tracking
CREATE TABLE IF NOT EXISTS supplier_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    supplier_id INTEGER NOT NULL,
    period TEXT NOT NULL, -- YYYY-MM format
    orders_delivered INTEGER DEFAULT 0,
    on_time_deliveries INTEGER DEFAULT 0,
    quality_issues INTEGER DEFAULT 0,
    average_response_time_hours REAL DEFAULT 0,
    total_value REAL DEFAULT 0,
    performance_score REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (supplier_id) REFERENCES suppliers (id) ON DELETE CASCADE
);

-- ============================================================================
-- 4. Enhanced Order Management
-- ============================================================================

-- Sales orders table
CREATE TABLE IF NOT EXISTS sales_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_number TEXT UNIQUE NOT NULL,
    customer_id INTEGER NOT NULL,
    customer_po TEXT, -- Customer Purchase Order number
    order_date DATE NOT NULL,
    delivery_date DATE,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'in_production', 'ready', 'shipped', 'delivered', 'cancelled', 'on_hold')),
    currency TEXT DEFAULT 'CNY',
    exchange_rate REAL DEFAULT 1,
    subtotal REAL NOT NULL,
    discount_amount REAL DEFAULT 0,
    tax_amount REAL DEFAULT 0,
    shipping_cost REAL DEFAULT 0,
    total_amount REAL NOT NULL,
    margin_percentage REAL DEFAULT 0,
    profit_amount REAL DEFAULT 0,
    payment_status TEXT DEFAULT 'unpaid' CHECK (payment_status IN ('unpaid', 'partial', 'paid', 'overdue')),
    payment_terms TEXT,
    sales_representative TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers (id)
);

-- Sales order items
CREATE TABLE IF NOT EXISTS sales_order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    line_number INTEGER NOT NULL,
    drawing_id INTEGER,
    product_code TEXT NOT NULL,
    description TEXT NOT NULL,
    specification TEXT,
    material TEXT,
    surface_treatment TEXT,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    discount_rate REAL DEFAULT 0,
    total_price REAL NOT NULL,
    supplier_id INTEGER,
    supplier_quote_id INTEGER,
    delivery_date DATE,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'ordered', 'received', 'shipped', 'delivered', 'cancelled')),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES sales_orders (id) ON DELETE CASCADE,
    FOREIGN KEY (drawing_id) REFERENCES drawings (id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers (id),
    FOREIGN KEY (supplier_quote_id) REFERENCES factory_quotes (id)
);

-- Purchase orders
CREATE TABLE IF NOT EXISTS purchase_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    po_number TEXT UNIQUE NOT NULL,
    supplier_id INTEGER NOT NULL,
    order_date DATE NOT NULL,
    expected_delivery_date DATE,
    actual_delivery_date DATE,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'shipped', 'received', 'cancelled', 'partial_received')),
    currency TEXT DEFAULT 'CNY',
    subtotal REAL NOT NULL,
    tax_amount REAL DEFAULT 0,
    total_amount REAL NOT NULL,
    payment_status TEXT DEFAULT 'unpaid' CHECK (payment_status IN ('unpaid', 'partial', 'paid', 'overdue')),
    payment_terms TEXT,
    ordered_by TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (supplier_id) REFERENCES suppliers (id)
);

-- Purchase order items
CREATE TABLE IF NOT EXISTS purchase_order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    po_id INTEGER NOT NULL,
    line_number INTEGER NOT NULL,
    material_code TEXT,
    description TEXT NOT NULL,
    specification TEXT,
    quantity_ordered INTEGER NOT NULL,
    quantity_received INTEGER DEFAULT 0,
    unit_price REAL NOT NULL,
    total_price REAL NOT NULL,
    delivery_date DATE,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'ordered', 'shipped', 'received', 'cancelled')),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (po_id) REFERENCES purchase_orders (id) ON DELETE CASCADE
);

-- ============================================================================
-- 5. Project Management
-- ============================================================================

-- Projects table (enhanced from existing process tracking)
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name TEXT NOT NULL,
    project_code TEXT UNIQUE,
    customer_id INTEGER NOT NULL,
    sales_order_id INTEGER,
    project_type TEXT CHECK (project_type IN ('standard', 'custom', 'oem', 'prototype')),
    priority TEXT DEFAULT 'medium' CHECK (priority IN ('high', 'medium', 'low')),
    status TEXT DEFAULT 'inquiry' CHECK (status IN ('inquiry', 'quotation', 'order_confirmed', 'design', 'production', 'quality_inspection', 'shipment', 'delivered', 'cancelled')),
    project_manager TEXT,
    estimated_value REAL DEFAULT 0,
    actual_value REAL DEFAULT 0,
    start_date DATE,
    delivery_date DATE,
    actual_completion_date DATE,
    progress_percentage REAL DEFAULT 0,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers (id),
    FOREIGN KEY (sales_order_id) REFERENCES sales_orders (id)
);

-- Project milestones
CREATE TABLE IF NOT EXISTS project_milestones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    milestone_name TEXT NOT NULL,
    description TEXT,
    due_date DATE,
    completed_date DATE,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'delayed', 'cancelled')),
    progress_percentage REAL DEFAULT 0,
    assigned_to TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
);

-- Project documents
CREATE TABLE IF NOT EXISTS project_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    document_type TEXT CHECK (document_type IN ('drawing', 'specification', 'contract', 'certificate', 'quote', 'invoice', 'other')),
    document_name TEXT NOT NULL,
    file_path TEXT,
    file_size INTEGER,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    uploaded_by TEXT,
    version INTEGER DEFAULT 1,
    is_current BOOLEAN DEFAULT 1,
    notes TEXT,
    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
);

-- ============================================================================
-- 6. Financial Management
-- ============================================================================

-- Invoices
CREATE TABLE IF NOT EXISTS invoices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_number TEXT UNIQUE NOT NULL,
    invoice_type TEXT CHECK (invoice_type IN ('sales', 'purchase')),
    order_id INTEGER, -- Can be sales_order_id or purchase_order_id
    customer_id INTEGER,
    supplier_id INTEGER,
    invoice_date DATE NOT NULL,
    due_date DATE,
    currency TEXT DEFAULT 'CNY',
    subtotal REAL NOT NULL,
    tax_amount REAL DEFAULT 0,
    total_amount REAL NOT NULL,
    status TEXT DEFAULT 'draft' CHECK (status IN ('draft', 'sent', 'paid', 'overdue', 'cancelled', 'partial_paid')),
    payment_date DATE,
    payment_amount REAL DEFAULT 0,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers (id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers (id)
);

-- Invoice items
CREATE TABLE IF NOT EXISTS invoice_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id INTEGER NOT NULL,
    line_number INTEGER NOT NULL,
    description TEXT NOT NULL,
    quantity REAL NOT NULL,
    unit_price REAL NOT NULL,
    discount_rate REAL DEFAULT 0,
    tax_rate REAL DEFAULT 0,
    total_amount REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (invoice_id) REFERENCES invoices (id) ON DELETE CASCADE
);

-- Payments
CREATE TABLE IF NOT EXISTS payments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    payment_number TEXT UNIQUE NOT NULL,
    payment_type TEXT CHECK (payment_type IN ('receivable', 'payable')),
    invoice_id INTEGER,
    customer_id INTEGER,
    supplier_id INTEGER,
    payment_date DATE NOT NULL,
    amount REAL NOT NULL,
    currency TEXT DEFAULT 'CNY',
    exchange_rate REAL DEFAULT 1,
    payment_method TEXT CHECK (payment_method IN ('cash', 'bank_transfer', 'check', 'credit_card', 'other')),
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'failed', 'cancelled')),
    reference_number TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (invoice_id) REFERENCES invoices (id),
    FOREIGN KEY (customer_id) REFERENCES customers (id),
    FOREIGN KEY (supplier_id) REFERENCES suppliers (id)
);

-- ============================================================================
-- 7. Inventory Management
-- ============================================================================

-- Inventory master
CREATE TABLE IF NOT EXISTS inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_code TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL,
    specification TEXT,
    material TEXT,
    category TEXT,
    unit TEXT DEFAULT '件',
    current_stock REAL DEFAULT 0,
    minimum_stock REAL DEFAULT 0,
    maximum_stock REAL DEFAULT 0,
    reorder_point REAL DEFAULT 0,
    average_cost REAL DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);

-- Inventory transactions
CREATE TABLE IF NOT EXISTS inventory_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_code TEXT NOT NULL,
    transaction_type TEXT CHECK (transaction_type IN ('in', 'out', 'adjustment', 'transfer')),
    quantity REAL NOT NULL,
    unit_cost REAL DEFAULT 0,
    reference_type TEXT CHECK (reference_type IN ('purchase_order', 'sales_order', 'adjustment', 'transfer')),
    reference_id INTEGER,
    notes TEXT,
    created_by TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_code) REFERENCES inventory (product_code)
);

-- ============================================================================
-- 8. Shipping and Logistics
-- ============================================================================

-- Shipments
CREATE TABLE IF NOT EXISTS shipments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shipment_number TEXT UNIQUE NOT NULL,
    order_id INTEGER, -- sales_order_id
    customer_id INTEGER NOT NULL,
    shipping_method TEXT CHECK (shipping_method IN ('air', 'sea', 'land', 'express', 'standard')),
    carrier TEXT,
    tracking_number TEXT,
    origin_address TEXT,
    destination_address TEXT,
    ship_date DATE,
    expected_delivery_date DATE,
    actual_delivery_date DATE,
    status TEXT DEFAULT 'preparing' CHECK (status IN ('preparing', 'shipped', 'in_transit', 'delivered', 'delayed', 'lost')),
    shipping_cost REAL DEFAULT 0,
    currency TEXT DEFAULT 'CNY',
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES sales_orders (id),
    FOREIGN KEY (customer_id) REFERENCES customers (id)
);

-- Shipment items
CREATE TABLE IF NOT EXISTS shipment_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shipment_id INTEGER NOT NULL,
    order_item_id INTEGER,
    product_code TEXT NOT NULL,
    description TEXT NOT NULL,
    quantity_shipped INTEGER NOT NULL,
    package_type TEXT,
    weight REAL DEFAULT 0,
    volume REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (shipment_id) REFERENCES shipments (id) ON DELETE CASCADE,
    FOREIGN KEY (order_item_id) REFERENCES sales_order_items (id)
);

-- ============================================================================
-- 9. Indexes for Performance Optimization
-- ============================================================================

-- Raw materials indexes
CREATE INDEX idx_raw_materials_code ON raw_materials(material_code);
CREATE INDEX idx_raw_materials_category ON raw_materials(category);
CREATE INDEX idx_raw_material_prices_date ON raw_materials(price_date);
CREATE INDEX idx_raw_material_prices_material_date ON raw_material_prices(material_id, price_date);

-- Exchange rates indexes
CREATE INDEX idx_exchange_rates_date ON exchange_rates(date);
CREATE INDEX idx_exchange_rates_pair ON exchange_rates(from_currency, to_currency);

-- Customer indexes
CREATE INDEX idx_customers_tier ON customers(customer_tier);
CREATE INDEX idx_customers_rep ON customers(sales_representative);
CREATE INDEX idx_customers_last_order ON customers(last_order_date);
CREATE INDEX idx_customer_contacts_customer ON customer_contacts(customer_id);
CREATE INDEX idx_customer_contacts_email ON customer_contacts(email);
CREATE INDEX idx_customer_interactions_customer ON customer_interactions(customer_id);
CREATE INDEX idx_customer_interactions_date ON customer_interactions(created_at);
CREATE INDEX idx_customer_interactions_type ON customer_interactions(interaction_type);

-- Supplier indexes
CREATE INDEX idx_suppliers_tier ON suppliers(supplier_tier);
CREATE INDEX idx_suppliers_active ON suppliers(is_active);
CREATE INDEX idx_supplier_performance_supplier_period ON supplier_performance(supplier_id, period);

-- Sales orders indexes
CREATE INDEX idx_sales_orders_customer ON sales_orders(customer_id);
CREATE INDEX idx_sales_orders_date ON sales_orders(order_date);
CREATE INDEX idx_sales_orders_status ON sales_orders(status);
CREATE INDEX idx_sales_orders_rep ON sales_orders(sales_representative);
CREATE INDEX idx_sales_order_items_order ON sales_order_items(order_id);
CREATE INDEX idx_sales_order_items_drawing ON sales_order_items(drawing_id);
CREATE INDEX idx_sales_order_items_supplier ON sales_order_items(supplier_id);

-- Purchase orders indexes
CREATE INDEX idx_purchase_orders_supplier ON purchase_orders(supplier_id);
CREATE INDEX idx_purchase_orders_date ON purchase_orders(order_date);
CREATE INDEX idx_purchase_orders_status ON purchase_orders(status);
CREATE INDEX idx_purchase_order_items_po ON purchase_order_items(po_id);

-- Project indexes
CREATE INDEX idx_projects_customer ON projects(customer_id);
CREATE INDEX idx_projects_status ON projects(status);
CREATE INDEX idx_projects_manager ON projects(project_manager);
CREATE INDEX idx_projects_delivery ON projects(delivery_date);
CREATE INDEX idx_project_milestones_project ON project_milestones(project_id);
CREATE INDEX idx_project_milestones_date ON project_milestones(due_date);
CREATE INDEX idx_project_documents_project ON project_documents(project_id);

-- Financial indexes
CREATE INDEX idx_invoices_customer ON invoices(customer_id);
CREATE INDEX idx_invoices_supplier ON invoices(supplier_id);
CREATE INDEX idx_invoices_date ON invoices(invoice_date);
CREATE INDEX idx_invoices_status ON invoices(status);
CREATE INDEX idx_payments_customer ON payments(customer_id);
CREATE INDEX idx_payments_supplier ON payments(supplier_id);
CREATE INDEX idx_payments_date ON payments(payment_date);

-- Inventory indexes
CREATE INDEX idx_inventory_category ON inventory(category);
CREATE INDEX idx_inventory_stock ON inventory(current_stock);
CREATE INDEX idx_inventory_transactions_product ON inventory_transactions(product_code);
CREATE INDEX idx_inventory_transactions_date ON inventory_transactions(created_at);

-- Shipping indexes
CREATE INDEX idx_shipments_order ON shipments(order_id);
CREATE INDEX idx_shipments_customer ON shipments(customer_id);
CREATE INDEX idx_shipments_date ON shipments(ship_date);
CREATE INDEX idx_shipments_status ON shipments(status);
CREATE INDEX idx_shipment_items_shipment ON shipment_items(shipment_id);

-- ============================================================================
-- 10. Triggers for Data Consistency
-- ============================================================================

-- Update customer total orders when new order is created
CREATE TRIGGER update_customer_order_count
    AFTER INSERT ON sales_orders
    WHEN NEW.status NOT IN ('cancelled', 'on_hold')
BEGIN
    UPDATE customers
    SET total_orders = total_orders + 1,
        last_order_date = NEW.order_date
    WHERE id = NEW.customer_id;
END;

-- Update timestamp triggers
CREATE TRIGGER update_customers_timestamp
    AFTER UPDATE ON customers
BEGIN
    UPDATE customers SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_suppliers_timestamp
    AFTER UPDATE ON suppliers
BEGIN
    UPDATE suppliers SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_projects_timestamp
    AFTER UPDATE ON projects
BEGIN
    UPDATE projects SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_inventory_timestamp
    AFTER UPDATE ON inventory
BEGIN
    UPDATE inventory SET last_updated = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- ============================================================================
-- 11. Views for Common Queries
-- ============================================================================

-- Customer summary view
CREATE VIEW IF NOT EXISTS customer_summary AS
SELECT
    c.id,
    c.company_name,
    c.contact_email,
    c.customer_tier,
    c.customer_status,
    c.total_orders,
    c.last_order_date,
    c.annual_revenue,
    c.payment_terms,
    c.sales_representative,
    COUNT(ci.id) as total_interactions,
    COUNT(DISTINCT so.id) as sales_order_count,
    COALESCE(SUM(so.total_amount), 0) as total_sales_value
FROM customers c
LEFT JOIN customer_interactions ci ON c.id = ci.customer_id
LEFT JOIN sales_orders so ON c.id = so.customer_id AND so.status NOT IN ('cancelled', 'on_hold')
GROUP BY c.id;

-- Supplier performance view
CREATE VIEW IF NOT EXISTS supplier_performance_summary AS
SELECT
    s.id,
    s.factory_name,
    s.supplier_tier,
    s.is_active,
    s.total_orders,
    s.total_value,
    s.quality_rating,
    s.on_time_delivery_rate,
    COALESCE(AVG(sp.performance_score), 0) as avg_performance_score,
    COUNT(p.id) as active_purchase_orders
FROM suppliers s
LEFT JOIN supplier_performance sp ON s.id = sp.supplier_id
LEFT JOIN purchase_orders p ON s.id = p.supplier_id AND p.status IN ('pending', 'confirmed', 'shipped', 'partial_received')
GROUP BY s.id;

-- Project status summary view
CREATE VIEW IF NOT EXISTS project_status_summary AS
SELECT
    p.id,
    p.project_name,
    p.project_code,
    p.status,
    p.priority,
    c.company_name as customer_name,
    p.project_manager,
    p.estimated_value,
    p.actual_value,
    p.delivery_date,
    p.progress_percentage,
    COUNT(pm.id) as total_milestones,
    COUNT(CASE WHEN pm.status = 'completed' THEN 1 END) as completed_milestones,
    COUNT(pd.id) as total_documents
FROM projects p
JOIN customers c ON p.customer_id = c.id
LEFT JOIN project_milestones pm ON p.id = pm.project_id
LEFT JOIN project_documents pd ON p.id = pd.project_id
GROUP BY p.id;

COMMIT;

-- ============================================================================
-- Migration Complete
-- ============================================================================
-- Notes:
-- 1. All existing data is preserved
-- 2. New columns have sensible defaults
-- 3. Foreign key constraints maintain data integrity
-- 4. Indexes are optimized for trading company queries
-- 5. Views provide convenient access to common data patterns
-- ============================================================================