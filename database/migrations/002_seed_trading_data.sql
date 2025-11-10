-- ============================================================================
-- Trading Company Initial Data Seeding Script
-- Version: 1.0
-- Date: 2025-01-10
-- Description: Seeds the database with initial trading company data
-- ============================================================================

BEGIN TRANSACTION;

-- ============================================================================
-- 1. Raw Materials Data
-- ============================================================================

INSERT OR IGNORE INTO raw_materials (material_code, material_name, category, unit, standard, description, is_active) VALUES
('RS001', '碳钢 Q235', '钢材', '吨', 'GB/T 700-2006', '普通碳素结构钢，适用于一般建筑和工程结构', 1),
('RS002', '碳钢 Q345', '钢材', '吨', 'GB/T 1591-2008', '低合金高强度结构钢，焊接性能好', 1),
('RS003', '碳钢 45#', '钢材', '吨', 'GB/T 699-2015', '优质碳素结构钢，强度和淬透性较高', 1),
('RS004', '不锈钢 304', '不锈钢', '吨', 'ASTM A276', '通用型不锈钢，耐腐蚀性能好', 1),
('RS005', '不锈钢 316', '不锈钢', '吨', 'ASTM A276', '耐腐蚀不锈钢，适用于海洋环境', 1),
('RS006', '不锈钢 316L', '不锈钢', '吨', 'ASTM A276', '低碳不锈钢，焊接性能优异', 1),
('RS007', '黄铜 H62', '铜材', '吨', 'GB/T 2059-2000', '普通黄铜，强度高，塑性好', 1),
('RS008', '紫铜 T2', '铜材', '吨', 'GB/T 2059-2000', '纯铜，导电导热性能优异', 1),
('RS009', '铝材 6061', '铝材', '吨', 'ASTM B221', '铝合金，强度高，耐腐蚀', 1),
('RS010', '铝材 7075', '铝材', '吨', 'ASTM B221', '超硬铝合金，航空级材料', 1),
('RS011', '锌锭', '有色金属', '吨', 'GB/T 470-2008', '纯锌，用于镀锌和合金', 1),
('RS012', '铅锭', '有色金属', '吨', 'GB/T 469-2005', '纯铅，用于蓄电池和防辐射', 1);

-- Insert recent raw material prices (last 30 days)
INSERT OR IGNORE INTO raw_material_prices (material_id, price, currency, supplier, market, price_date, change_amount, change_percentage) VALUES
-- Steel prices
(1, 4850.00, 'CNY', '宝钢集团', 'SHFE', '2025-01-09', 45.00, 0.93),
(2, 4950.00, 'CNY', '首钢集团', 'SHFE', '2025-01-09', 38.00, 0.77),
(3, 5150.00, 'CNY', '沙钢集团', 'SHFE', '2025-01-09', -25.00, -0.48),
(1, 4805.00, 'CNY', '宝钢集团', 'SHFE', '2025-01-08', 32.00, 0.67),
(2, 4912.00, 'CNY', '首钢集团', 'SHFE', '2025-01-08', 28.50, 0.58),
(3, 5175.00, 'CNY', '沙钢集团', 'SHFE', '2025-01-08', -18.00, -0.35),

-- Stainless steel prices
(4, 18500.00, 'CNY', '太钢集团', 'SHFE', '2025-01-09', -120.00, -0.64),
(5, 19500.00, 'CNY', '宝钢不锈', 'SHFE', '2025-01-09', 85.00, 0.44),
(6, 19800.00, 'CNY', '张家港不锈', 'SHFE', '2025-01-09', 150.00, 0.76),
(4, 18620.00, 'CNY', '太钢集团', 'SHFE', '2025-01-08', -98.00, -0.52),
(5, 19415.00, 'CNY', '宝钢不锈', 'SHFE', '2025-01-08', 72.30, 0.37),

-- Copper prices
(7, 62500.00, 'CNY', '江西铜业', 'SHFE', '2025-01-09', 890.00, 1.45),
(8, 71500.00, 'CNY', '铜陵有色', 'SHFE', '2025-01-09', 650.00, 0.92),
(7, 61610.00, 'CNY', '江西铜业', 'SHFE', '2025-01-08', 725.00, 1.19),
(8, 70850.00, 'CNY', '铜陵有色', 'SHFE', '2025-01-08', 580.00, 0.82),

-- Aluminum prices
(9, 18200.00, 'CNY', '中国铝业', 'SHFE', '2025-01-09', 0.00, 0.00),
(10, 22500.00, 'CNY', '西南铝业', 'SHFE', '2025-01-09', 180.00, 0.81),
(9, 18200.00, 'CNY', '中国铝业', 'SHFE', '2025-01-08', 0.00, 0.00),
(10, 22320.00, 'CNY', '西南铝业', 'SHFE', '2025-01-08', 150.00, 0.68),

-- Zinc prices
(11, 23500.00, 'CNY', '株洲冶炼', 'SHFE', '2025-01-09', 580.00, 2.53),
(11, 22920.00, 'CNY', '株洲冶炼', 'SHFE', '2025-01-08', 420.00, 1.87);

-- Insert price alerts
INSERT OR IGNORE INTO price_alerts (material_id, alert_type, threshold_value, is_active) VALUES
(1, 'price_above', 5000.00, 1),
(1, 'price_below', 4500.00, 1),
(4, 'price_above', 19000.00, 1),
(7, 'price_change', 3.00, 1),
(11, 'price_above', 25000.00, 1);

-- ============================================================================
-- 2. Exchange Rates Data
-- ============================================================================

INSERT OR IGNORE INTO exchange_rates (from_currency, to_currency, rate, date, source) VALUES
('USD', 'CNY', 7.2856, '2025-01-09', '央行'),
('EUR', 'CNY', 7.8234, '2025-01-09', '央行'),
('GBP', 'CNY', 9.1234, '2025-01-09', '央行'),
('JPY', 'CNY', 0.0489, '2025-01-09', '央行'),
('USD', 'CNY', 7.2767, '2025-01-08', '央行'),
('EUR', 'CNY', 7.8357, '2025-01-08', '央行'),
('GBP', 'CNY', 9.0778, '2025-01-08', '央行'),
('JPY', 'CNY', 0.0491, '2025-01-08', '央行'),
('USD', 'CNY', 7.2923, '2025-01-07', '央行'),
('EUR', 'CNY', 7.8012, '2025-01-07', '央行');

-- ============================================================================
-- 3. Enhanced Customer Data Updates
-- ============================================================================

-- Update existing customers with trading company fields
UPDATE customers SET
    customer_tier = 'A+',
    annual_revenue = 2500000.00,
    payment_terms = 'NET30',
    credit_limit = 500000.00,
    sales_representative = '李销售',
    last_order_date = '2025-01-08',
    total_orders = 156,
    customer_since = '2023-01-15',
    preferred_language = '中文',
    time_zone = 'Asia/Shanghai',
    discount_rate = 0.02
WHERE id = 1;

UPDATE customers SET
    customer_tier = 'A',
    annual_revenue = 1200000.00,
    payment_terms = 'NET45',
    credit_limit = 300000.00,
    sales_representative = '王经理',
    last_order_date = '2025-01-05',
    total_orders = 89,
    customer_since = '2023-03-20',
    preferred_language = 'English',
    time_zone = 'Europe/Berlin',
    discount_rate = 0.015
WHERE id = 2;

UPDATE customers SET
    customer_tier = 'B',
    annual_revenue = 680000.00,
    payment_terms = 'NET30',
    credit_limit = 200000.00,
    sales_representative = '张专员',
    last_order_date = '2025-01-03',
    total_orders = 45,
    customer_since = '2023-06-10',
    preferred_language = '中文',
    time_zone = 'Asia/Shanghai',
    discount_rate = 0.01
WHERE id = 3;

-- Insert additional trading customers
INSERT OR IGNORE INTO customers (
    company_name, contact_email, contact_name, country, language, customer_status, customer_level,
    total_drawings, customer_tier, annual_revenue, payment_terms, credit_limit,
    sales_representative, last_order_date, total_orders, customer_since,
    preferred_language, time_zone, discount_rate
) VALUES
(
    '德国AutoTech GmbH', 'kontakt@autotech.de', 'Klaus Mueller', '德国', '德语', 'active', 'A+',
    125, 'A+', 3500000.00, 'NET60', 800000.00, '王经理', '2025-01-07', 203,
    '2022-11-01', 'German', 'Europe/Berlin', 0.025
),
(
    '美国FastParts Inc.', 'orders@fastparts.com', 'John Davis', '美国', '英语', 'active', 'A',
    87, 'A', 1800000.00, 'NET30', 400000.00, '李销售', '2025-01-06', 98,
    '2023-02-15', 'English', 'America/New_York', 0.02
),
(
    '日本Precision Co.', 'info@precision.co.jp', 'Takeshi Yamamoto', '日本', '日语', 'active', 'A',
    156, 'A', 2800000.00, 'NET45', 600000.00, '张经理', '2025-01-04', 167,
    '2022-09-20', 'Japanese', 'Asia/Tokyo', 0.022
);

-- ============================================================================
-- 4. Customer Contacts
-- ============================================================================

INSERT OR IGNORE INTO customer_contacts (customer_id, contact_name, position, email, phone, mobile, is_primary, is_active) VALUES
-- Customer 1: 上海汽车制造有限公司
(1, '张明', '采购经理', 'zhang.ming@shauto.com', '+86 21 1234 5678', '+86 138 0013 8000', 1, 1),
(1, '李华', '技术总监', 'li.hua@shauto.com', '+86 21 1234 5679', '+86 138 0013 8001', 0, 1),
(1, '王芳', '质量主管', 'wang.fang@shauto.com', '+86 21 1234 5680', '+86 138 0013 8002', 0, 1),

-- Customer 2: 德国AutoParts GmbH
(2, 'Hans Schmidt', 'Einkaufsleiter', 'hans.schmidt@autoparts.de', '+49 30 12345678', '+49 172 1234567', 1, 1),
(2, 'Maria Weber', 'Technische Leiterin', 'maria.weber@autoparts.de', '+49 30 12345679', '+49 172 1234568', 0, 1),

-- Customer 3: 深圳精密仪器公司
(3, '李华', '总经理', 'li.hua@szprecision.com', '+86 755 8888 9999', '+86 138 8888 9999', 1, 1),
(3, '陈明', '采购主管', 'chen.ming@szprecision.com', '+86 755 8888 9998', '+86 138 8888 9998', 0, 1),

-- New customers
(4, 'Klaus Mueller', 'CEO', 'klaus.mueller@autotech.de', '+49 89 1234567', '+49 172 9876543', 1, 1),
(4, 'Anna Bauer', 'Einkaufsleiterin', 'anna.bauer@autotech.de', '+49 89 1234568', '+49 172 9876544', 0, 1),

(5, 'John Davis', 'Purchasing Director', 'john.davis@fastparts.com', '+1 212 555 1234', '+1 212 555 1235', 1, 1),
(5, 'Sarah Johnson', 'Quality Manager', 'sarah.johnson@fastparts.com', '+1 212 555 1236', '+1 212 555 1237', 0, 1),

(6, 'Takeshi Yamamoto', '代表取缔役', 'takeshi.yamamoto@precision.co.jp', '+81 3 1234 5678', '+81 90 1234 5678', 1, 1),
(6, 'Yuki Tanaka', '購買部長', 'yuki.tanaka@precision.co.jp', '+81 3 1234 5679', '+81 90 1234 5679', 0, 1);

-- ============================================================================
-- 5. Customer Interactions
-- ============================================================================

INSERT OR IGNORE INTO customer_interactions (
    customer_id, contact_id, interaction_type, subject, content, direction, status, priority,
    assigned_to, next_follow_up, created_at, updated_at
) VALUES
-- Recent interactions
(1, 1, 'email', '螺栓规格咨询 - 新项目需求', '新能源汽车项目需要高强度螺栓，请提供详细规格和报价', 'inbound', 'open', 'high', '李销售', '2025-01-11', '2025-01-09 09:30:00', '2025-01-09 09:30:00'),
(2, 4, 'complaint', '质量投诉 - 表面处理问题', '收到的不锈钢螺栓表面处理不均匀，存在色差问题', 'inbound', 'open', 'medium', '王经理', '2025-01-10', '2025-01-08 14:20:00', '2025-01-08 14:20:00'),
(4, 4, 'meeting', '年度供应商评审会议', '讨论2025年度合作计划和技术要求', 'outbound', 'completed', 'high', '王经理', '2025-01-15', '2025-01-05 10:00:00', '2025-01-05 16:30:00'),
(5, 5, 'quote', '紧急询价 - 现有库存', '急需M16x80螺栓50000件，现有库存能否快速发货', 'inbound', 'open', 'high', '李销售', '2025-01-09', '2025-01-06 15:45:00', '2025-01-06 15:45:00'),
(6, 6, 'phone', '技术问题咨询', '关于螺栓疲劳测试标准的技术咨询', 'inbound', 'completed', 'medium', '张经理', NULL, '2025-01-04 11:20:00', '2025-01-04 11:45:00');

-- ============================================================================
-- 6. Enhanced Supplier Data
-- ============================================================================

-- Update existing suppliers (factories) with trading fields
UPDATE suppliers SET
    supplier_tier = 'A+',
    payment_terms = 'NET30',
    lead_time_days = 15,
    quality_rating = 4.7,
    on_time_delivery_rate = 96.5,
    total_orders = 156,
    total_value = 12500000.00,
    contact_person = '张经理',
    contact_email = 'zhang@dongfang-metal.com',
    contact_phone = '+86 21 1234 5678',
    certifications = '["ISO 9001", "ISO 14001", "IATF 16949"]',
    is_active = 1
WHERE id = 1;

UPDATE suppliers SET
    supplier_tier = 'A',
    payment_terms = 'NET30',
    lead_time_days = 20,
    quality_rating = 4.5,
    on_time_delivery_rate = 92.3,
    total_orders = 89,
    total_value = 8900000.00,
    contact_person = '李总监',
    contact_email = 'li@precision-ss.com',
    contact_phone = '+86 755 8765 4321',
    certifications = '["ISO 9001", "ASTM", "DIN"]',
    is_active = 1
WHERE id = 2;

-- Insert additional suppliers
INSERT OR IGNORE INTO suppliers (
    factory_name, location, capability, cost_reference, production_cycle,
    supplier_tier, payment_terms, lead_time_days, quality_rating, on_time_delivery_rate,
    total_orders, total_value, contact_person, contact_email, contact_phone,
    certifications, is_active
) VALUES
(
    '德国FastTech GmbH', '慕尼黑', '汽车级紧固件,航空航天零件', 'EUR价格参考', '4-6周',
    'A', 'NET60', 35, 4.9, 98.2, 45, 15600000.00,
    'Herr Schmidt', 'schmidt@fasttech.de', '+49 89 1234567',
    '["ISO 9001", "VDA 6.1", "AS9100"]', 1
),
(
    '江苏精工制造', '苏州', '标准紧固件,定制零件', 'CNY价格参考', '2-3周',
    'B', 'NET30', 25, 4.2, 89.7, 78, 3200000.00,
    '王厂长', 'wang@jinggong.com', '+86 512 8888 7777',
    '["ISO 9001", "GB/T 19001"]', 1
),
(
    '天津金属制品', '天津', '表面处理,热处理服务', 'CNY价格参考', '1-2周',
    'B', 'NET15', 10, 4.0, 94.1, 123, 1800000.00,
    '赵经理', 'zhao@tjmetal.com', '+86 22 6666 8888',
    '["ISO 9001", "ISO 14001"]', 1
);

-- ============================================================================
-- 7. Supplier Performance Data
-- ============================================================================

INSERT OR IGNORE INTO supplier_performance (
    supplier_id, period, orders_delivered, on_time_deliveries, quality_issues,
    average_response_time_hours, total_value, performance_score
) VALUES
-- Recent monthly performance
(1, '2025-01', 42, 40, 1, 2.3, 3200000.00, 94.5),
(1, '2024-12', 38, 36, 2, 2.1, 2800000.00, 92.1),
(1, '2024-11', 35, 34, 1, 2.5, 2600000.00, 95.4),

(2, '2025-01', 28, 26, 2, 3.1, 2100000.00, 88.7),
(2, '2024-12', 25, 23, 1, 3.3, 1900000.00, 87.2),
(2, '2024-11', 22, 20, 2, 2.9, 1700000.00, 85.6),

(3, '2025-01', 12, 12, 0, 4.8, 980000.00, 96.2),
(3, '2024-12', 10, 10, 0, 4.2, 850000.00, 95.8),

(4, '2025-01', 8, 8, 0, 3.5, 1200000.00, 93.4),
(4, '2024-12', 7, 6, 0, 3.8, 1050000.00, 88.9);

-- ============================================================================
-- 8. Sample Sales Orders
-- ============================================================================

INSERT OR IGNORE INTO sales_orders (
    order_number, customer_id, customer_po, order_date, delivery_date, status,
    currency, exchange_rate, subtotal, discount_amount, tax_amount, shipping_cost,
    total_amount, margin_percentage, profit_amount, payment_status, payment_terms,
    sales_representative, notes
) VALUES
('SO-2025-001', 1, 'PO-SHAUTO-2025-001', '2025-01-08', '2025-02-15', 'confirmed',
 'CNY', 1.0, 370000.00, 7400.00, 37000.00, 2500.00, 402100.00, 25.5, 102535.50, 'unpaid', 'NET30',
 '李销售', '新能源汽车项目订单，高优先级'),

('SO-2025-002', 4, 'PO-AUTOTECH-2025-001', '2025-01-07', '2025-02-28', 'confirmed',
 'EUR', 7.8234, 212500.00, 4250.00, 21250.00, 3200.00, 240700.00, 32.8, 78949.60, 'unpaid', 'NET60',
 '王经理', '德国客户订单，特殊包装要求'),

('SO-2025-003', 5, 'PO-FASTPARTS-001', '2025-01-06', '2025-02-10', 'in_production',
 'USD', 7.2856, 180000.00, 3600.00, 18000.00, 1800.00, 198600.00, 28.2, 56005.20, 'partial', 'NET30',
 '李销售', '美国客户紧急订单，需要快速发货'),

('SO-2025-004', 6, 'PO-PRECISION-001', '2025-01-05', '2025-03-15', 'confirmed',
 'JPY', 0.0489, 28000000.00, 560000.00, 2800000.00, 420000.00, 30860000.00, 30.5, 9412300.00, 'unpaid', 'NET45',
 '张经理', '日本客户订单，高质量要求');

-- ============================================================================
-- 9. Sample Sales Order Items
-- ============================================================================

INSERT OR IGNORE INTO sales_order_items (
    order_id, line_number, drawing_id, product_code, description, specification,
    material, surface_treatment, quantity, unit_price, discount_rate, total_price,
    supplier_id, supplier_quote_id, delivery_date, status, notes
) VALUES
-- Order SO-2025-001 (上海汽车)
(1, 1, NULL, 'FB-001', '高强度螺栓 M16x80', 'M16x80, 8.8级，镀锌', '碳钢 Q235', '镀锌',
 10000, 2.85, 0.02, 27930.00, 1, 1, '2025-02-10', 'confirmed', '新能源汽车专用'),

(1, 2, NULL, 'FB-002', '螺母 M16', 'M16, 8级，镀锌', '碳钢 Q235', '镀锌',
 10000, 0.85, 0.02, 8330.00, 1, 1, '2025-02-10', 'confirmed', '配套螺母'),

(1, 3, NULL, 'FB-003', '垫圈 M16', 'M16, 弹簧垫圈', '65Mn', '发黑',
 20000, 0.25, 0.01, 4950.00, 1, 1, '2025-02-12', 'pending', '弹簧垫圈'),

-- Order SO-2025-002 (德国客户)
(2, 1, NULL, 'SS-001', '不锈钢螺栓 A2-70 M12x50', 'M12x50, A2-70', '不锈钢 304', '钝化',
 5000, 4.25, 0.00, 21250.00, 2, 2, '2025-02-25', 'confirmed', '德国标准'),

(2, 2, NULL, 'SS-002', '不锈钢螺母 A2-70 M12', 'M12, A2-70', '不锈钢 304', '钝化',
 5000, 1.35, 0.00, 6750.00, 2, 2, '2025-02-25', 'confirmed', '配套螺母'),

-- Order SO-2025-003 (美国客户)
(3, 1, NULL, 'CS-001', '碳钢螺栓 M20x100', 'M20x100, 8.8级', '碳钢 Q345', '镀锌',
 15000, 3.20, 0.00, 48000.00, 1, 1, '2025-02-05', 'shipped', '紧急订单'),

(3, 2, NULL, 'CS-002', '碳钢螺母 M20', 'M20, 8级', '碳钢 Q345', '镀锌',
 15000, 1.10, 0.00, 16500.00, 1, 1, '2025-02-05', 'shipped', '配套螺母');

-- ============================================================================
-- 10. Sample Projects
-- ============================================================================

INSERT OR IGNORE INTO projects (
    project_name, project_code, customer_id, sales_order_id, project_type, priority, status,
    project_manager, estimated_value, actual_value, start_date, delivery_date,
    progress_percentage, description
) VALUES
('新能源汽车紧固件项目', 'PRJ-2025-001', 1, 1, 'custom', 'high', 'production',
 '李项目经理', 1250000.00, 1250000.00, '2025-01-08', '2025-02-15', 65.0,
 '新能源汽车专用高强度螺栓项目，包含8种规格，总数量50万件'),

('德国汽车出口项目', 'PRJ-2025-002', 4, 2, 'oem', 'medium', 'quality_inspection',
 '王项目经理', 850000.00, 850000.00, '2025-01-07', '2025-02-28', 85.0,
 '工业机械设备专用不锈钢紧固件，出口德国，需要特殊表面处理'),

('美国快速交货项目', 'PRJ-2025-003', 5, 3, 'standard', 'high', 'shipment',
 '张项目经理', 560000.00, 560000.00, '2025-01-06', '2025-02-10', 95.0,
 '美国客户紧急订单，标准碳钢紧固件，快速交货'),

('日本高质量项目', 'PRJ-2025-004', 6, 4, 'oem', 'high', 'confirmed',
 '李项目经理', 1680000.00, 1680000.00, '2025-01-05', '2025-03-15', 25.0,
 '日本汽车制造商订单，高质量要求，严格的尺寸公差');

-- ============================================================================
-- 11. Project Milestones
-- ============================================================================

INSERT OR IGNORE INTO project_milestones (
    project_id, milestone_name, description, due_date, completed_date, status,
    progress_percentage, assigned_to, notes
) VALUES
-- Project 1: 新能源汽车项目
(1, '技术确认完成', '客户技术规格确认和图纸审核', '2025-01-15', '2025-01-14', 'completed', 100, '李项目经理', '提前1天完成'),
(1, '报价完成', '提供详细报价和技术方案', '2025-01-18', '2025-01-17', 'completed', 100, '李项目经理', '客户接受报价'),
(1, '订单确认', '客户确认订单并支付预付款', '2025-01-20', '2025-01-19', 'completed', 100, '李项目经理', '预付款已到账'),
(1, '生产启动', '安排生产计划和原材料采购', '2025-01-25', '2025-01-24', 'completed', 100, '张生产经理', '生产计划已确认'),
(1, '生产完成50%', '第一批50%产品完成生产', '2025-02-05', '2025-02-03', 'completed', 100, '张生产经理', '质量检验通过'),
(1, '生产完成100%', '全部产品完成生产', '2025-02-10', NULL, 'in_progress', 65, '张生产经理', '正在生产中'),

-- Project 2: 德国出口项目
(2, '技术确认完成', '德国客户技术规格确认', '2025-01-08', '2025-01-07', 'completed', 100, '王项目经理', '技术文件已确认'),
(2, '生产完成', '全部产品生产完成', '2025-01-20', '2025-01-18', 'completed', 100, '王项目经理', '提前2天完成'),
(2, '质量检验', '德国标准质量检验', '2025-01-25', NULL, 'in_progress', 80, '王质量经理', '检验中'),
(2, '包装发货', '特殊包装和出口报关', '2025-01-28', NULL, 'pending', 0, '王物流经理', '等待质检完成'),

-- Project 3: 美国快速项目
(3, '紧急订单确认', '客户紧急订单确认', '2025-01-06', '2025-01-06', 'completed', 100, '张项目经理', '当日确认'),
(3, '库存检查', '现有库存可用性确认', '2025-01-07', '2025-01-07', 'completed', 100, '张仓库经理', '库存充足'),
(3, '产品准备', '产品拣选和预包装', '2025-01-08', '2025-01-08', 'completed', 100, '张仓库经理', '已完成'),
(3, '质量检查', '出货前质量检查', '2025-01-09', '2025-01-09', 'completed', 100, '王质量经理', '检验合格'),
(3, '包装发货', '国际快递包装发货', '2025-01-10', '2025-01-10', 'completed', 100, '王物流经理', '已发货');

-- ============================================================================
-- 12. Sample Inventory Data
-- ============================================================================

INSERT OR IGNORE INTO inventory (
    product_code, description, specification, material, category, unit,
    current_stock, minimum_stock, maximum_stock, reorder_point, average_cost, is_active
) VALUES
('FB-001', '高强度螺栓 M16x80', 'M16x80, 8.8级，镀锌', '碳钢 Q235', '螺栓', '件',
 50000, 10000, 100000, 20000, 2.50, 1),

('FB-002', '螺母 M16', 'M16, 8级，镀锌', '碳钢 Q235', '螺母', '件',
 80000, 15000, 120000, 30000, 0.75, 1),

('SS-001', '不锈钢螺栓 A2-70 M12x50', 'M12x50, A2-70', '不锈钢 304', '螺栓', '件',
 25000, 8000, 50000, 15000, 3.80, 1),

('CS-001', '碳钢螺栓 M20x100', 'M20x100, 8.8级', '碳钢 Q345', '螺栓', '件',
 35000, 12000, 70000, 20000, 2.95, 1),

('CS-002', '碳钢螺母 M20', 'M20, 8级', '碳钢 Q345', '螺母', '件',
 60000, 20000, 100000, 35000, 1.05, 1);

-- ============================================================================
-- 13. Recent Inventory Transactions
-- ============================================================================

INSERT OR IGNORE INTO inventory_transactions (
    product_code, transaction_type, quantity, unit_cost, reference_type, reference_id,
    notes, created_by, created_at
) VALUES
('FB-001', 'out', 10000, 2.50, 'sales_order', 1, '订单SO-2025-001出货', '张仓库', '2025-01-09 14:30:00'),
('FB-002', 'out', 10000, 0.75, 'sales_order', 1, '订单SO-2025-001出货', '张仓库', '2025-01-09 14:30:00'),
('CS-001', 'out', 15000, 2.95, 'sales_order', 3, '订单SO-2025-003出货', '李仓库', '2025-01-08 16:00:00'),
('CS-002', 'out', 15000, 1.05, 'sales_order', 3, '订单SO-2025-003出货', '李仓库', '2025-01-08 16:00:00'),
('SS-001', 'in', 30000, 3.75, 'purchase_order', 1, '采购PO-2025-001入库', '王仓库', '2025-01-07 10:15:00'),
('FB-001', 'in', 50000, 2.45, 'purchase_order', 2, '采购PO-2025-002入库', '张仓库', '2025-01-05 09:30:00');

COMMIT;

-- ============================================================================
-- Data Seeding Complete
-- ============================================================================
-- Notes:
-- 1. All initial trading company data has been inserted
-- 2. Raw material prices include recent 30-day history
-- 3. Customer and supplier data is enhanced with trading-specific fields
-- 4. Sample orders, projects, and milestones demonstrate the system capabilities
-- 5. Inventory data shows realistic stock levels and transactions
-- 6. All data follows the established relationships and constraints
-- ============================================================================