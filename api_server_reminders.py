#!/usr/bin/env python3
"""
Reminder System API Server
独立的提醒系统API服务器，处理提醒规则、记录和通知
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ReminderAPI')

app = Flask(__name__)
CORS(app)

# 数据库路径
DB_PATH = "knowledge_base.db"

def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def convert_for_json(obj):
    """转换对象为JSON可序列化格式"""
    if hasattr(obj, 'keys'):
        return {key: convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_for_json(vars(obj))
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

# ==================== Health Check ====================
@app.route('/api/v1/health')
def health_check():
    """健康检查"""
    try:
        # 测试数据库连接
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM reminder_rules WHERE enabled = TRUE")
        rules_count = cursor.fetchone()[0]
        conn.close()

        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0',
            'database': {
                'connected': True,
                'enabled_rules_count': rules_count
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500

# ==================== Reminder Records API ====================
@app.route('/api/v1/reminders/records')
def get_reminder_records():
    """获取提醒记录列表（支持分页和筛选）"""
    try:
        # 获取查询参数
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        status = request.args.get('status')
        rule_id = request.args.get('rule_id')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        entity_type = request.args.get('entity_type')

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
        if entity_type:
            where_conditions.append("rr.business_entity_type = ?")
            params.append(entity_type)

        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""

        # 查询提醒记录
        offset = (page - 1) * limit
        query = f"""
            SELECT
                rr.id,
                rr.rule_id,
                rr.rule_name,
                rr.business_entity_type,
                rr.business_entity_id,
                rr.trigger_time,
                rr.trigger_condition,
                rr.priority,
                rr.status,
                rr.assigned_to,
                rr.due_time,
                rr.completed_time,
                rr.notification_methods,
                rr.auto_processed,
                rr.processing_result,
                rr.error_message,
                rr.retry_count,
                rr.metadata,
                rr.created_at,
                rr.updated_at,
                rru.name as rule_name_db,
                rru.priority as rule_priority_db,
                rru.description as rule_description
            FROM reminder_records rr
            LEFT JOIN reminder_rules rru ON rr.rule_id = rru.id
            {where_clause}
            ORDER BY rr.trigger_time DESC
            LIMIT ? OFFSET ?
        """

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, params + [limit, offset])

        records = []
        for row in cursor.fetchall():
            record = dict(row)
            # 解析JSON字段
            try:
                record['metadata'] = json.loads(record['metadata']) if record['metadata'] else {}
            except json.JSONDecodeError:
                record['metadata'] = {}

            # 格式化时间字段
            if record['trigger_time']:
                record['trigger_time'] = datetime.fromisoformat(record['trigger_time']).isoformat()
            if record['due_time']:
                record['due_time'] = datetime.fromisoformat(record['due_time']).isoformat()
            if record['completed_time']:
                record['completed_time'] = datetime.fromisoformat(record['completed_time']).isoformat()
            if record['created_at']:
                record['created_at'] = datetime.fromisoformat(record['created_at']).isoformat()
            if record['updated_at']:
                record['updated_at'] = datetime.fromisoformat(record['updated_at']).isoformat()

            records.append(record)

        # 获取总数
        count_query = f"SELECT COUNT(*) as total FROM reminder_records rr {where_clause}"
        cursor.execute(count_query, params)
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
                },
                'filters': {
                    'status': status,
                    'rule_id': rule_id,
                    'date_from': date_from,
                    'date_to': date_to,
                    'entity_type': entity_type
                }
            }
        })

    except Exception as e:
        logger.error(f"获取提醒记录失败: {e}")
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

@app.route('/api/v1/reminders/records/<int:record_id>/handle', methods=['POST'])
def handle_reminder_record(record_id):
    """标记提醒记录为已处理"""
    try:
        data = request.get_json() or {}
        handled_by = data.get('handled_by', 'unknown')
        notes = data.get('notes', '')

        conn = get_db_connection()
        cursor = conn.cursor()

        # 检查记录是否存在
        cursor.execute("SELECT id, status FROM reminder_records WHERE id = ?", (record_id,))
        record = cursor.fetchone()

        if not record:
            conn.close()
            return jsonify({
                'success': False,
                'error': {
                    'code': 'NOT_FOUND',
                    'message': '提醒记录不存在'
                }
            }), 404

        if record['status'] == 'completed':
            conn.close()
            return jsonify({
                'success': False,
                'error': {
                    'code': 'ALREADY_HANDLED',
                    'message': '提醒记录已经被处理'
                }
            }), 400

        # 更新记录状态
        cursor.execute('''
            UPDATE reminder_records
            SET status = ?,
                completed_time = ?,
                processing_result = ?,
                error_message = ?,
                updated_at = ?
            WHERE id = ?
        ''', (
            'completed',
            datetime.now().isoformat(),
            json.dumps({
                'handled_by': handled_by,
                'handled_at': datetime.now().isoformat(),
                'notes': notes
            }),
            None,
            datetime.now().isoformat(),
            record_id
        ))

        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'data': {
                'record_id': record_id,
                'status': 'completed',
                'handled_by': handled_by,
                'handled_at': datetime.now().isoformat(),
                'message': '提醒记录已标记为已处理'
            }
        })

    except Exception as e:
        logger.error(f"处理提醒记录失败: {e}")
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

# ==================== Reminder Rules API ====================
@app.route('/api/v1/reminders/rules')
def get_reminder_rules():
    """获取提醒规则列表"""
    try:
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
        if is_active is not None:
            where_conditions.append("enabled = ?")
            params.append(is_active.lower() == 'true')

        where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""

        # 查询规则列表
        offset = (page - 1) * limit
        query = f"""
            SELECT * FROM reminder_rules
            {where_clause}
            ORDER BY priority DESC, updated_at DESC
            LIMIT ? OFFSET ?
        """

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query, params + [limit, offset])

        rules = []
        for row in cursor.fetchall():
            rule = dict(row)
            # 解析JSON字段
            try:
                rule['notification_methods'] = json.loads(rule['notification_methods']) if rule['notification_methods'] else []
            except json.JSONDecodeError:
                rule['notification_methods'] = []

            try:
                rule['escalation_rules'] = json.loads(rule['escalation_rules']) if rule['escalation_rules'] else {}
            except json.JSONDecodeError:
                rule['escalation_rules'] = {}

            try:
                rule['config_parameters'] = json.loads(rule['config_parameters']) if rule['config_parameters'] else {}
            except json.JSONDecodeError:
                rule['config_parameters'] = {}

            rules.append(rule)

        # 获取总数
        count_query = f"SELECT COUNT(*) as total FROM reminder_rules {where_clause}"
        cursor.execute(count_query, params)
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
        logger.error(f"获取提醒规则失败: {e}")
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
        cursor.execute("SELECT enabled FROM reminder_rules WHERE id = ?", (rule_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return jsonify({
                'success': False,
                'error': {
                    'code': 'NOT_FOUND',
                    'message': '提醒规则不存在'
                }
            }), 404

        # 切换状态
        new_status = not row['enabled']
        cursor.execute('''
            UPDATE reminder_rules
            SET enabled = ?, updated_at = ?
            WHERE id = ?
        ''', (new_status, datetime.now().isoformat(), rule_id))

        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'data': {
                'rule_id': rule_id,
                'enabled': new_status,
                'message': f'规则已{"启用" if new_status else "禁用"}'
            }
        })

    except Exception as e:
        logger.error(f"切换规则状态失败: {e}")
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

# ==================== Dashboard API ====================
@app.route('/api/v1/reminders/dashboard')
def get_reminders_dashboard():
    """获取提醒系统仪表板数据"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        dashboard_data = {}

        # 1. 规则统计
        cursor.execute("""
            SELECT
                COUNT(*) as total_rules,
                COUNT(CASE WHEN enabled = 1 THEN 1 END) as enabled_rules,
                COUNT(CASE WHEN priority = '高' THEN 1 END) as high_priority_rules,
                COUNT(CASE WHEN priority = '中' THEN 1 END) as medium_priority_rules,
                COUNT(CASE WHEN priority = '低' THEN 1 END) as low_priority_rules
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
                COUNT(CASE WHEN status = 'processing' THEN 1 END) as processing_reminders,
                COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_reminders,
                COUNT(CASE WHEN status = 'cancelled' THEN 1 END) as cancelled_reminders
            FROM reminder_records
            WHERE DATE(trigger_time) = ?
        """, (today,))
        today_stats = cursor.fetchone()
        dashboard_data['today'] = dict(today_stats)

        # 3. 本周趋势
        week_ago = (datetime.now() - timedelta(days=7)).date()
        cursor.execute("""
            SELECT
                DATE(trigger_time) as date,
                COUNT(*) as count,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as success_count,
                COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_count
            FROM reminder_records
            WHERE DATE(trigger_time) >= ?
            GROUP BY DATE(trigger_time)
            ORDER BY date
        """, (week_ago,))
        weekly_trend = [dict(row) for row in cursor.fetchall()]
        dashboard_data['weekly_trend'] = weekly_trend

        # 4. 优先级统计
        cursor.execute("""
            SELECT
                priority,
                COUNT(*) as reminder_count,
                COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_count,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_count
            FROM reminder_records
            WHERE DATE(trigger_time) = DATE('now')
            GROUP BY priority
            ORDER BY reminder_count DESC
        """)
        priority_stats = [dict(row) for row in cursor.fetchall()]
        dashboard_data['priorities'] = priority_stats

        # 5. 最近活动
        cursor.execute("""
            SELECT
                rr.id,
                rr.rule_id,
                rr.rule_name,
                rr.trigger_time,
                rr.status,
                rr.priority,
                rr.trigger_condition
            FROM reminder_records rr
            ORDER BY rr.trigger_time DESC
            LIMIT 10
        """)
        recent_activity = [dict(row) for row in cursor.fetchall()]
        dashboard_data['recent_activity'] = recent_activity

        # 6. 系统健康状态
        cursor.execute("""
            SELECT
                COUNT(CASE WHEN status = 'cancelled' THEN 1 END) * 100.0 / COUNT(*) as failure_rate
            FROM reminder_records
            WHERE DATE(trigger_time) >= DATE('now', '-1 day')
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
        logger.error(f"获取仪表板数据失败: {e}")
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

# ==================== Manual Trigger API ====================
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
        # 为了演示，我们创建一个手动触发记录
        conn = get_db_connection()
        cursor = conn.cursor()

        # 获取规则信息
        cursor.execute("SELECT name, priority FROM reminder_rules WHERE id = ?", (rule_id,))
        rule_info = cursor.fetchone()

        if not rule_info:
            return jsonify({
                'success': False,
                'error': {
                    'code': 'NOT_FOUND',
                    'message': '提醒规则不存在'
                }
            }), 404

        cursor.execute('''
            INSERT INTO reminder_records
            (rule_id, rule_name, business_entity_type, business_entity_id,
             trigger_time, trigger_condition, priority, status, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            rule_id,
            rule_info['name'],
            'system',
            0,
            datetime.now().isoformat(),
            f"手动触发的提醒 - {rule_info['name']}",
            rule_info['priority'],
            'pending',
            json.dumps({
                'type': 'manual_trigger',
                'triggered_by': 'api_user',
                'timestamp': datetime.now().isoformat()
            }),
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'data': {
                'message': f'规则 {rule_id} ({rule_info["name"]}) 已手动触发',
                'triggered_at': datetime.now().isoformat(),
                'rule_name': rule_info['name'],
                'rule_priority': rule_info['priority']
            }
        })

    except Exception as e:
        logger.error(f"手动触发提醒失败: {e}")
        return jsonify({
            'success': False,
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': str(e)
            }
        }), 500

# ==================== Error Handlers ====================
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

# ==================== Server Start ====================
if __name__ == '__main__':
    print("🚀 启动提醒系统API服务器...")
    print("📡 服务地址: http://localhost:8001")
    print("📚 API文档:")
    print("   - GET  /api/v1/health")
    print("   - GET  /api/v1/reminders/records")
    print("   - POST /api/v1/reminders/records/<id>/handle")
    print("   - GET  /api/v1/reminders/rules")
    print("   - POST /api/v1/reminders/rules/<rule_id>/toggle")
    print("   - GET  /api/v1/reminders/dashboard")
    print("   - POST /api/v1/reminders/trigger")
    print("🔗 数据库: ./data/db.sqlite")

    app.run(host='0.0.0.0', port=8001, debug=True)