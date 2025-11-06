#!/usr/bin/env python3
"""
API服务器模拟脚本
提供简单的API服务器用于前端开发和测试
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 数据库路径
DB_PATH = "./data/db.sqlite"

# 管理员账户配置 (从环境变量读取，如果不存在则使用默认值)
ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'Gavin')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'Wcy1223')

def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 返回字典形式的结果
    return conn

@app.route('/api/v1/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/v1/auth/login', methods=['POST'])
def authenticate():
    """用户认证"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()

        if not username or not password:
            return jsonify({
                'success': False,
                'message': '用户名和密码不能为空'
            }), 400

        # 验证管理员账户
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            # 认证成功
            return jsonify({
                'success': True,
                'data': {
                    'username': username,
                    'role': 'admin',
                    'name': '系统管理员',
                    'permissions': {
                        'view_dashboard': True,
                        'manage_reminders': True,
                        'manage_rules': True,
                        'view_analytics': True,
                        'system_settings': True
                    }
                }
            })
        else:
            # 认证失败
            return jsonify({
                'success': False,
                'message': '用户名或密码错误'
            }), 401

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'认证失败: {str(e)}'
        }), 500

@app.route('/api/v1/statistics/drawings/by_category')
def get_drawing_category_stats():
    """获取图纸分类统计"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 基础统计
        cursor.execute("""
            SELECT
                COUNT(*) as total_drawings,
                COUNT(CASE WHEN is_classified = 1 THEN 1 END) as classified_drawings
            FROM drawings
        """)
        basic_stats = cursor.fetchone()

        # 分类分布
        cursor.execute("""
            SELECT
                product_category,
                COUNT(*) as count,
                COUNT(CASE WHEN standard_or_custom = 1 THEN 1 END) as custom_count,
                COUNT(CASE WHEN standard_or_custom = 0 THEN 1 END) as standard_count,
                AVG(classification_confidence) as avg_confidence
            FROM drawings
            GROUP BY product_category
            ORDER BY count DESC
        """)
        categories = cursor.fetchall()

        # 计算百分比
        total = basic_stats['total_drawings']
        category_list = []
        for cat in categories:
            category_list.append({
                'product_category': cat['product_category'],
                'count': cat['count'],
                'percentage': round(cat['count'] / total * 100, 1) if total > 0 else 0,
                'standard_count': cat['standard_count'],
                'custom_count': cat['custom_count'],
                'avg_confidence': round(cat['avg_confidence'], 2) if cat['avg_confidence'] else 0
            })

        # 月度趋势
        cursor.execute("""
            SELECT
                strftime('%Y-%m', classification_date) as month,
                COUNT(CASE WHEN is_classified = 1 THEN 1 END) as classified_count,
                COUNT(*) as total_count
            FROM drawings
            WHERE classification_date IS NOT NULL
            GROUP BY strftime('%Y-%m', classification_date)
            ORDER BY month DESC
            LIMIT 6
        """)
        monthly_data = cursor.fetchall()

        monthly_trend = []
        for month in monthly_data:
            monthly_trend.append({
                'month': month['month'],
                'classified_count': month['classified_count'],
                'total_count': month['total_count']
            })

        conn.close()

        return jsonify({
            'success': True,
            'data': {
                'summary': {
                    'total_drawings': total,
                    'classified_drawings': basic_stats['classified_drawings'],
                    'classification_rate': round(basic_stats['classified_drawings'] / total * 100, 1) if total > 0 else 0,
                    'date_range': {
                        'from': '2025-01-01',
                        'to': datetime.now().strftime('%Y-%m-%d')
                    }
                },
                'categories': category_list,
                'trend': {
                    'monthly_data': monthly_trend
                }
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

@app.route('/api/v1/statistics/customers/by_status')
def get_customer_status_stats():
    """获取客户状态统计"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 基础统计
        cursor.execute("SELECT COUNT(*) as total_customers FROM customers")
        total_customers = cursor.fetchone()['total_customers']

        # 状态分布
        cursor.execute("""
            SELECT
                customer_status,
                COUNT(*) as count,
                AVG(total_drawings) as avg_drawings,
                COUNT(DISTINCT country) as countries
            FROM customers
            GROUP BY customer_status
            ORDER BY count DESC
        """)
        status_data = cursor.fetchall()

        status_list = []
        for status in status_data:
            status_list.append({
                'customer_status': status['customer_status'],
                'count': status['count'],
                'percentage': round(status['count'] / total_customers * 100, 1) if total_customers > 0 else 0,
                'avg_drawings': round(status['avg_drawings'], 1) if status['avg_drawings'] else 0,
                'countries': status['countries']
            })

        # 地理分布
        cursor.execute("""
            SELECT
                country,
                COUNT(*) as count
            FROM customers
            GROUP BY country
            ORDER BY count DESC
        """)
        geo_data = cursor.fetchall()

        geo_distribution = []
        for geo in geo_data:
            # 获取该国家的客户详情
            cursor.execute("""
                SELECT company_name, customer_status, total_drawings
                FROM customers
                WHERE country = ?
                ORDER BY total_drawings DESC
            """, (geo['country'],))
            customers_in_country = cursor.fetchall()

            geo_distribution.append({
                'country': geo['country'],
                'count': geo['count'],
                'customers': [
                    {
                        'company_name': cust['company_name'],
                        'customer_status': cust['customer_status'],
                        'total_drawings': cust['total_drawings']
                    }
                    for cust in customers_in_country[:5]  # 限制显示前5个
                ]
            })

        conn.close()

        return jsonify({
            'success': True,
            'data': {
                'summary': {
                    'total_customers': total_customers,
                    'active_customers': sum(1 for s in status_list if s['customer_status'] == 'active'),
                    'potential_customers': sum(1 for s in status_list if s['customer_status'] == 'potential'),
                    'avg_drawings_per_customer': round(sum(s['avg_drawings'] * s['count'] for s in status_list) / total_customers, 1) if total_customers > 0 else 0
                },
                'status_distribution': status_list,
                'geographic_distribution': geo_distribution
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

@app.route('/api/v1/trends/factory_quotes')
def get_factory_quote_trends():
    """获取工厂报价趋势"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 基础统计
        cursor.execute("""
            SELECT
                COUNT(*) as total_quotes,
                COUNT(DISTINCT factory_id) as total_factories
            FROM factory_quotes
            WHERE price IS NOT NULL AND price > 0
        """)
        basic_stats = cursor.fetchone()

        # 月度趋势
        cursor.execute("""
            SELECT
                fq.quote_month,
                fq.factory_id,
                f.factory_name,
                fq.product_category,
                AVG(fq.price) as avg_price,
                MIN(fq.price) as min_price,
                MAX(fq.price) as max_price,
                COUNT(*) as quote_count,
                AVG(fq.moq) as moq_avg
            FROM factory_quotes fq
            LEFT JOIN factories f ON fq.factory_id = f.id
            WHERE fq.price IS NOT NULL AND fq.price > 0
            GROUP BY fq.quote_month, fq.factory_id, f.factory_name, fq.product_category
            ORDER BY fq.quote_month DESC
            LIMIT 50
        """)
        monthly_data = cursor.fetchall()

        monthly_trends = []
        for data in monthly_data:
            monthly_trends.append({
                'period': data['quote_month'],
                'factory_id': data['factory_id'],
                'factory_name': data['factory_name'],
                'product_category': data['product_category'],
                'avg_price': round(data['avg_price'], 2),
                'min_price': round(data['min_price'], 2),
                'max_price': round(data['max_price'], 2),
                'quote_count': data['quote_count'],
                'moq_avg': round(data['moq_avg']) if data['moq_avg'] else 0,
                'price_change_pct': 0  # 简化实现
            })

        # 工厂表现
        cursor.execute("""
            SELECT
                fq.factory_id,
                f.factory_name,
                COUNT(*) as total_quotes,
                AVG(fq.price) as avg_price,
                COUNT(DISTINCT fq.product_category) as category_count
            FROM factory_quotes fq
            LEFT JOIN factories f ON fq.factory_id = f.id
            GROUP BY fq.factory_id, f.factory_name
            ORDER BY total_quotes DESC
        """)
        factory_data = cursor.fetchall()

        factory_performance = []
        for factory in factory_data:
            factory_performance.append({
                'factory_id': factory['factory_id'],
                'factory_name': factory['factory_name'],
                'total_quotes': factory['total_quotes'],
                'avg_price': round(factory['avg_price'], 2),
                'product_categories': [],  # 简化实现
                'price_volatility': 0,  # 简化实现
                'quote_frequency': round(factory['total_quotes'] / 12, 1)  # 假设12个月
            })

        conn.close()

        return jsonify({
            'success': True,
            'data': {
                'summary': {
                    'total_quotes': basic_stats['total_quotes'],
                    'total_factories': basic_stats['total_factories'],
                    'date_range': {
                        'from': '2025-01-01',
                        'to': datetime.now().strftime('%Y-%m-%d')
                    },
                    'overall_price_trend': 0  # 简化实现
                },
                'monthly_trends': monthly_trends,
                'factory_performance': factory_performance,
                'price_anomalies': []  # 简化实现
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

@app.route('/api/v1/quality/overview')
def get_quality_overview():
    """获取数据质量概览"""
    try:
        # 模拟数据质量数据
        quality_data = {
            'overall_score': 86.8,
            'overall_grade': 'B',
            'last_updated': datetime.now().isoformat(),
            'component_scores': {
                'customer_data': {
                    'score': 92.7,
                    'grade': 'A',
                    'completeness': 100.0,
                    'accuracy': 85.0,
                    'issues': [
                        {
                            'type': 'invalid_email',
                            'count': 4,
                            'severity': 'medium'
                        }
                    ]
                },
                'drawing_data': {
                    'score': 95.2,
                    'grade': 'A',
                    'completeness': 99.5,
                    'classification_rate': 9.6,
                    'issues': []
                },
                'factory_data': {
                    'score': 88.5,
                    'grade': 'B',
                    'completeness': 100.0,
                    'issues': []
                }
            },
            'recommendations': [
                '修复4个无效邮箱格式',
                '提高图纸分类覆盖率',
                '定期验证数据完整性'
            ]
        }

        return jsonify({
            'success': True,
            'data': quality_data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

@app.route('/api/v1/dashboard/overview')
def get_dashboard_overview():
    """获取仪表板概览数据"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # KPI数据
        cursor.execute("SELECT COUNT(*) FROM customers")
        total_customers = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM drawings")
        total_drawings = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM factories")
        total_factories = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(CASE WHEN is_classified = 1 THEN 1 END) * 100.0 / COUNT(*)
            FROM drawings
        """)
        classification_rate = cursor.fetchone()[0] or 0

        kpi_data = {
            'total_customers': total_customers,
            'total_drawings': total_drawings,
            'total_factories': total_factories,
            'classification_rate': round(classification_rate, 1),
            'data_quality_score': 86.8
        }

        # 最近活动
        recent_activity = [
            {
                'type': 'classification',
                'message': '完成25个图纸分类',
                'timestamp': '2025-11-05T17:30:00Z'
            },
            {
                'type': 'analysis',
                'message': '生成工厂报价分析报告',
                'timestamp': '2025-11-05T17:25:00Z'
            },
            {
                'type': 'import',
                'message': '导入5个新客户数据',
                'timestamp': '2025-11-05T17:20:00Z'
            }
        ]

        # 图表数据
        cursor.execute("""
            SELECT product_category, COUNT(*) as count
            FROM drawings
            GROUP BY product_category
            ORDER BY count DESC
            LIMIT 5
        """)
        category_data = cursor.fetchall()

        drawing_distribution = [
            {'category': cat[0], 'value': cat[1]} for cat in category_data
        ]

        monthly_trends = [
            {'month': '2025-10', 'quotes': 15, 'drawings': 25},
            {'month': '2025-09', 'quotes': 12, 'drawings': 18}
        ]

        # 警报
        alerts = [
            {
                'level': 'warning',
                'message': '发现1个价格异常',
                'timestamp': '2025-11-05T17:00:00Z'
            },
            {
                'level': 'info',
                'message': '数据质量评分下降至B级',
                'timestamp': '2025-11-05T16:00:00Z'
            }
        ]

        conn.close()

        return jsonify({
            'success': True,
            'data': {
                'kpi': kpi_data,
                'recent_activity': recent_activity,
                'charts': {
                    'drawing_category_distribution': {
                        'data': drawing_distribution
                    },
                    'monthly_trends': {
                        'data': monthly_trends
                    }
                },
                'alerts': alerts
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

@app.route('/api/v1/search')
def global_search():
    """全局搜索"""
    query = request.args.get('q', '').strip()
    search_type = request.args.get('type', 'all')
    limit = int(request.args.get('limit', 20))
    offset = int(request.args.get('offset', 0))

    if not query:
        return jsonify({
            'success': False,
            'error': {
                'code': 'VALIDATION_ERROR',
                'message': '搜索关键词不能为空'
            }
        }), 400

    try:
        results = []

        if search_type in ['all', 'drawing']:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, drawing_name, product_category, customer_id
                FROM drawings
                WHERE drawing_name LIKE ?
                ORDER BY id DESC
                LIMIT ?
            """, (f'%{query}%', limit))

            for row in cursor.fetchall():
                # 获取客户名称
                customer_name = 'Unknown'
                if row['customer_id']:
                    cursor2 = conn.cursor()
                    cursor2.execute("SELECT company_name FROM customers WHERE id = ?", (row['customer_id'],))
                    customer_result = cursor2.fetchone()
                    if customer_result:
                        customer_name = customer_result[0]

                results.append({
                    'type': 'drawing',
                    'id': row['id'],
                    'title': row['drawing_name'],
                    'product_category': row['product_category'],
                    'customer_name': customer_name,
                    'relevance_score': 0.9
                })

        conn.close()

        return jsonify({
            'success': True,
            'data': {
                'query': query,
                'total_results': len(results),
                'results': results[:limit],
                'facets': {
                    'types': {
                        'drawing': len(results)
                    }
                }
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': {
            'code': 'NOT_FOUND',
            'message': '请求的资源不存在'
        }
    }), 404

# ==================== 提醒系统API接口 ====================

@app.route('/api/v1/reminders/rules')
def get_reminder_rules():
    """获取提醒规则列表"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 获取查询参数
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        priority = request.args.get('priority')
        category = request.args.get('category')
        is_active = request.args.get('is_active')

        # 构建查询条件
        where_conditions = []
        params = []

        if priority:
            where_conditions.append("priority = ?")
            params.append(priority)
        if category:
            where_conditions.append("category = ?")
            params.append(category)
        if is_active is not None:
            where_conditions.append("is_active = ?")
            params.append(is_active.lower() == 'true')

        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""

        # 查询规则列表
        offset = (page - 1) * limit
        cursor.execute(f"""
            SELECT * FROM reminder_rules
            {where_clause}
            ORDER BY priority ASC, created_at DESC
            LIMIT ? OFFSET ?
        """, params + [limit, offset])

        rules = []
        for row in cursor.fetchall():
            rule = dict(row)
            # 解析JSON字段
            rule['trigger_config'] = json.loads(rule['trigger_config']) if rule['trigger_config'] else {}
            rule['schedule_config'] = json.loads(rule['schedule_config']) if rule['schedule_config'] else {}
            rule['notification_config'] = json.loads(rule['notification_config']) if rule['notification_config'] else []
            rule['action_config'] = json.loads(rule['action_config']) if rule['action_config'] else []
            rules.append(rule)

        # 获取总数
        cursor.execute(f"SELECT COUNT(*) as total FROM reminder_rules {where_clause}", params)
        total = cursor.fetchone()['total']

        conn.close()

        return jsonify({
            'success': True,
            'data': {
                'rules': rules,
                'pagination': {
                    'page': page,
                    'limit': limit,
                    'total': total,
                    'total_pages': (total + limit - 1) // limit,
                    'has_next': offset + limit < total,
                    'has_prev': page > 1
                }
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

@app.route('/api/v1/reminders/rules/<rule_id>')
def get_reminder_rule(rule_id):
    """获取单个提醒规则详情"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM reminder_rules WHERE rule_id = ?", (rule_id,))
        row = cursor.fetchone()

        if not row:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'NOT_FOUND',
                    'message': '提醒规则不存在'
                }
            }), 404

        rule = dict(row)
        # 解析JSON字段
        rule['trigger_config'] = json.loads(rule['trigger_config']) if rule['trigger_config'] else {}
        rule['schedule_config'] = json.loads(rule['schedule_config']) if rule['schedule_config'] else {}
        rule['notification_config'] = json.loads(rule['notification_config']) if rule['notification_config'] else []
        rule['action_config'] = json.loads(rule['action_config']) if rule['action_config'] else []

        # 获取最近的执行记录
        cursor.execute("""
            SELECT * FROM reminder_records
            WHERE rule_id = ?
            ORDER BY triggered_at DESC
            LIMIT 10
        """, (rule_id,))

        recent_executions = [dict(row) for row in cursor.fetchall()]
        rule['recent_executions'] = recent_executions

        conn.close()

        return jsonify({
            'success': True,
            'data': rule
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

@app.route('/api/v1/reminders/records')
def get_reminder_records():
    """获取提醒记录列表"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 获取查询参数
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        status = request.args.get('status')
        rule_id = request.args.get('rule_id')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')

        # 构建查询条件
        where_conditions = []
        params = []

        if status:
            where_conditions.append("rr.status = ?")
            params.append(status)
        if rule_id:
            where_conditions.append("rr.rule_id = ?")
            params.append(rule_id)
        if date_from:
            where_conditions.append("DATE(rr.triggered_at) >= ?")
            params.append(date_from)
        if date_to:
            where_conditions.append("DATE(rr.triggered_at) <= ?")
            params.append(date_to)

        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""

        # 查询提醒记录
        offset = (page - 1) * limit
        cursor.execute(f"""
            SELECT
                rr.*,
                rru.name as rule_name,
                rru.description as rule_description,
                rru.priority as rule_priority
            FROM reminder_records rr
            LEFT JOIN reminder_rules rru ON rr.rule_id = rru.rule_id
            {where_clause}
            ORDER BY rr.triggered_at DESC
            LIMIT ? OFFSET ?
        """, params + [limit, offset])

        records = []
        for row in cursor.fetchall():
            record = dict(row)
            # 解析JSON字段
            record['trigger_data'] = json.loads(record['trigger_data']) if record['trigger_data'] else {}
            record['result_data'] = json.loads(record['result_data']) if record['result_data'] else {}
            records.append(record)

        # 获取总数
        cursor.execute(f"SELECT COUNT(*) as total FROM reminder_records rr {where_clause}", params)
        total = cursor.fetchone()['total']

        conn.close()

        return jsonify({
            'success': True,
            'data': {
                'records': records,
                'pagination': {
                    'page': page,
                    'limit': limit,
                    'total': total,
                    'total_pages': (total + limit - 1) // limit,
                    'has_next': offset + limit < total,
                    'has_prev': page > 1
                }
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

@app.route('/api/v1/reminders/notifications')
def get_reminder_notifications():
    """获取通知历史列表"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 获取查询参数
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        notification_type = request.args.get('type')
        status = request.args.get('status')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')

        # 构建查询条件
        where_conditions = []
        params = []

        if notification_type:
            where_conditions.append("nh.notification_type = ?")
            params.append(notification_type)
        if status:
            where_conditions.append("nh.status = ?")
            params.append(status)
        if date_from:
            where_conditions.append("DATE(nh.sent_at) >= ?")
            params.append(date_from)
        if date_to:
            where_conditions.append("DATE(nh.sent_at) <= ?")
            params.append(date_to)

        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""

        # 查询通知历史
        offset = (page - 1) * limit
        cursor.execute(f"""
            SELECT
                nh.*,
                rr.rule_id,
                rru.name as rule_name,
                rr.triggered_at
            FROM notification_history nh
            LEFT JOIN reminder_records rr ON nh.reminder_record_id = rr.id
            LEFT JOIN reminder_rules rru ON rr.rule_id = rru.rule_id
            {where_clause}
            ORDER BY nh.sent_at DESC, nh.created_at DESC
            LIMIT ? OFFSET ?
        """, params + [limit, offset])

        notifications = []
        for row in cursor.fetchall():
            notification = dict(row)
            notifications.append(notification)

        # 获取总数
        cursor.execute(f"SELECT COUNT(*) as total FROM notification_history nh {where_clause}", params)
        total = cursor.fetchone()['total']

        conn.close()

        return jsonify({
            'success': True,
            'data': {
                'notifications': notifications,
                'pagination': {
                    'page': page,
                    'limit': limit,
                    'total': total,
                    'total_pages': (total + limit - 1) // limit,
                    'has_next': offset + limit < total,
                    'has_prev': page > 1
                }
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

@app.route('/api/v1/reminders/dashboard')
def get_reminders_dashboard():
    """获取提醒系统仪表板数据"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 统计数据
        dashboard_data = {}

        # 1. 规则统计
        cursor.execute("""
            SELECT
                COUNT(*) as total_rules,
                COUNT(CASE WHEN is_active = 1 THEN 1 END) as active_rules,
                COUNT(CASE WHEN priority = 1 THEN 1 END) as high_priority_rules,
                COUNT(CASE WHEN priority = 2 THEN 1 END) as medium_priority_rules,
                COUNT(CASE WHEN priority = 3 THEN 1 END) as low_priority_rules
            FROM reminder_rules
        """)
        rule_stats = cursor.fetchone()
        dashboard_data['rules'] = dict(rule_stats)

        # 2. 今日提醒统计
        today = datetime.now().date()
        cursor.execute("""
            SELECT
                COUNT(*) as total_reminders,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_reminders,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_reminders,
                COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_reminders
            FROM reminder_records
            WHERE DATE(triggered_at) = ?
        """, (today,))
        today_stats = cursor.fetchone()
        dashboard_data['today'] = dict(today_stats)

        # 3. 本周趋势
        week_ago = (datetime.now() - timedelta(days=7)).date()
        cursor.execute("""
            SELECT
                DATE(triggered_at) as date,
                COUNT(*) as count,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as success_count
            FROM reminder_records
            WHERE DATE(triggered_at) >= ?
            GROUP BY DATE(triggered_at)
            ORDER BY date
        """, (week_ago,))
        weekly_trend = [dict(row) for row in cursor.fetchall()]
        dashboard_data['weekly_trend'] = weekly_trend

        # 4. 通知统计
        cursor.execute("""
            SELECT
                notification_type,
                COUNT(*) as total_count,
                COUNT(CASE WHEN status = 'sent' THEN 1 END) as sent_count,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_count
            FROM notification_history
            WHERE sent_at >= date('now', '-7 days')
            GROUP BY notification_type
        """)
        notification_stats = [dict(row) for row in cursor.fetchall()]
        dashboard_data['notifications'] = notification_stats

        # 5. 最近活动
        cursor.execute("""
            SELECT
                rr.execution_id,
                rr.rule_id,
                rru.name as rule_name,
                rr.triggered_at,
                rr.status,
                rr.notification_count
            FROM reminder_records rr
            LEFT JOIN reminder_rules rru ON rr.rule_id = rru.rule_id
            ORDER BY rr.triggered_at DESC
            LIMIT 10
        """)
        recent_activity = [dict(row) for row in cursor.fetchall()]
        dashboard_data['recent_activity'] = recent_activity

        # 6. 系统健康状态
        cursor.execute("""
            SELECT
                COUNT(CASE WHEN status = 'failed' THEN 1 END) * 100.0 / COUNT(*) as failure_rate
            FROM reminder_records
            WHERE triggered_at >= date('now', '-1 day')
        """)
        health_result = cursor.fetchone()
        failure_rate = health_result['failure_rate'] if health_result['failure_rate'] else 0

        dashboard_data['system_health'] = {
            'status': 'healthy' if failure_rate < 5 else 'warning' if failure_rate < 15 else 'critical',
            'failure_rate': round(failure_rate, 2),
            'uptime': '99.9%',  # 模拟数据
            'last_check': datetime.now().isoformat()
        }

        conn.close()

        return jsonify({
            'success': True,
            'data': dashboard_data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

@app.route('/api/v1/reminders/trigger', methods=['POST'])
def trigger_reminder_manually():
    """手动触发提醒规则"""
    try:
        data = request.get_json()
        rule_id = data.get('rule_id')

        if not rule_id:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'VALIDATION_ERROR',
                    'message': '缺少规则ID'
                }
            }), 400

        # 这里应该调用实际的提醒检查逻辑
        # 为了演示，我们返回成功响应
        execution_id = f"MANUAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{rule_id}"

        return jsonify({
            'success': True,
            'data': {
                'execution_id': execution_id,
                'message': f'规则 {rule_id} 已手动触发',
                'triggered_at': datetime.now().isoformat()
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

@app.route('/api/v1/reminders/rules/<rule_id>/toggle', methods=['POST'])
def toggle_reminder_rule(rule_id):
    """启用/禁用提醒规则"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 检查规则是否存在
        cursor.execute("SELECT is_active FROM reminder_rules WHERE rule_id = ?", (rule_id,))
        row = cursor.fetchone()

        if not row:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'NOT_FOUND',
                    'message': '提醒规则不存在'
                }
            }), 404

        # 切换状态
        new_status = not row['is_active']
        cursor.execute("""
            UPDATE reminder_rules
            SET is_active = ?, updated_at = ?
            WHERE rule_id = ?
        """, (new_status, datetime.now().isoformat(), rule_id))

        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'data': {
                'rule_id': rule_id,
                'is_active': new_status,
                'message': f'规则已{"启用" if new_status else "禁用"}'
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': {
            'code': 'NOT_FOUND',
            'message': '请求的资源不存在'
        }
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': {
            'code': 'INTERNAL_ERROR',
            'message': '服务器内部错误'
        }
    }), 500

if __name__ == '__main__':
    print("🚀 启动API服务器...")
    print("📡 服务地址: http://localhost:8000")
    print("📚 API文档: http://localhost:8000/api/v1/health")
    print("🔍 测试搜索: http://localhost:8000/api/v1/search?q=螺栓")
    print("🔔 提醒系统: http://localhost:8000/api/v1/reminders/dashboard")

    app.run(host='0.0.0.0', port=8000, debug=True)