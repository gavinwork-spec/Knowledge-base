#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提醒系统检查脚本
实现17条提醒规则的自动化检查和执行
"""

import sqlite3
import logging
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
import time
import os
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/check_reminders.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ReminderChecker:
    """提醒系统检查器"""

    def __init__(self, db_path: str = 'knowledge_base.db'):
        self.db_path = db_path
        self.conn = None
        self._initialize_connection()

        # 通知配置
        self.email_config = {
            'smtp_server': 'smtp.company.com',
            'smtp_port': 587,
            'use_tls': True,
            'sender_email': 'system@company.com',
            'sender_name': '知识库提醒系统'
        }

        # 系统配置
        self.config = {
            'max_retries': 3,
            'batch_size': 100,
            'notification_timeout': 30,
            'dry_run': False  # 设为True则不实际发送通知
        }

    def _initialize_connection(self):
        """初始化数据库连接"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"数据库连接已建立: {self.db_path}")
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise

    def get_enabled_rules(self) -> List[Dict]:
        """获取启用的提醒规则"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM reminder_rules
                WHERE enabled = TRUE
                ORDER BY priority DESC, check_frequency
            """)

            rules = []
            for row in cursor.fetchall():
                rule = dict(row)
                rule['notification_methods'] = json.loads(rule['notification_methods'])
                rule['escalation_rules'] = json.loads(rule['escalation_rules']) if rule['escalation_rules'] else {}
                rules.append(rule)

            logger.info(f"获取到 {len(rules)} 条启用的规则")
            return rules

        except Exception as e:
            logger.error(f"获取启用规则失败: {e}")
            return []

    def check_rule_R001(self, rule: Dict) -> List[Dict]:
        """R001: 客户图纸状态变更提醒"""
        reminders = []

        try:
            cursor = self.conn.cursor()

            # 检查新图纸上传（最近1小时）
            cursor.execute("""
                SELECT d.id, d.drawing_name, d.created_at, d.status, c.company_name,
                       '新图纸上传' as trigger_type
                FROM drawings d
                LEFT JOIN customers c ON d.customer_id = c.id
                WHERE d.created_at > datetime('now', '-1 hour')
                AND d.id NOT IN (
                    SELECT rr.business_entity_id
                    FROM reminder_records rr
                    WHERE rr.rule_id = 'R001'
                    AND rr.business_entity_type = 'drawing'
                    AND rr.trigger_condition LIKE '%新图纸上传%'
                    AND rr.status IN ('pending', 'processing')
                )
            """)

            new_drawings = cursor.fetchall()

            # 检查图纸状态变更（最近1小时）
            cursor.execute("""
                SELECT d.id, d.drawing_name, d.status_updated_at, d.status, c.company_name,
                       '状态变更' as trigger_type
                FROM drawings d
                LEFT JOIN customers c ON d.customer_id = c.id
                WHERE d.status_updated_at > datetime('now', '-1 hour')
                AND d.id NOT IN (
                    SELECT rr.business_entity_id
                    FROM reminder_records rr
                    WHERE rr.rule_id = 'R001'
                    AND rr.business_entity_type = 'drawing'
                    AND rr.trigger_condition LIKE '%状态变更%'
                    AND rr.status IN ('pending', 'processing')
                )
            """)

            status_changes = cursor.fetchall()

            # 合并结果
            all_changes = list(new_drawings) + list(status_changes)

            for change in all_changes:
                reminder = {
                    'rule_id': rule['id'],
                    'rule_name': rule['name'],
                    'business_entity_type': 'drawing',
                    'business_entity_id': change['id'],
                    'trigger_time': datetime.now(),
                    'trigger_condition': f"{change['trigger_type']}: {change['drawing_name']}",
                    'priority': rule['priority'],
                    'metadata': json.dumps({
                        'drawing_name': change['drawing_name'],
                        'company_name': change['company_name'],
                        'status': change['status'],
                        'trigger_type': change['trigger_type']
                    }, ensure_ascii=False)
                }
                reminders.append(reminder)

            logger.info(f"R001检查完成，发现 {len(reminders)} 个图纸变更")

        except Exception as e:
            logger.error(f"R001规则检查失败: {e}")

        return reminders

    def check_rule_R002(self, rule: Dict) -> List[Dict]:
        """R002: 报价超时预警"""
        reminders = []

        try:
            cursor = self.conn.cursor()

            # 正常订单报价超时24小时
            cursor.execute("""
                SELECT fq.id, fq.quote_date, f.factory_name, c.company_name, fq.total_amount,
                       'normal_timeout' as timeout_type
                FROM factory_quotes fq
                JOIN factories f ON fq.factory_id = f.id
                LEFT JOIN customers c ON fq.customer_id = c.id
                WHERE fq.status = 'processing'
                AND fq.quote_date < datetime('now', '-24 hours')
                AND fq.urgent_flag = FALSE
                AND fq.id NOT IN (
                    SELECT rr.business_entity_id
                    FROM reminder_records rr
                    WHERE rr.rule_id = 'R002'
                    AND rr.business_entity_type = 'quote'
                    AND rr.trigger_condition LIKE '%normal_timeout%'
                    AND rr.status IN ('pending', 'processing')
                )
            """)

            normal_timeouts = cursor.fetchall()

            # 紧急订单报价超时12小时
            cursor.execute("""
                SELECT fq.id, fq.quote_date, f.factory_name, c.company_name, fq.total_amount,
                       'urgent_timeout' as timeout_type
                FROM factory_quotes fq
                JOIN factories f ON fq.factory_id = f.id
                LEFT JOIN customers c ON fq.customer_id = c.id
                WHERE fq.status = 'processing'
                AND fq.quote_date < datetime('now', '-12 hours')
                AND fq.urgent_flag = TRUE
                AND fq.id NOT IN (
                    SELECT rr.business_entity_id
                    FROM reminder_records rr
                    WHERE rr.rule_id = 'R002'
                    AND rr.business_entity_type = 'quote'
                    AND rr.trigger_condition LIKE '%urgent_timeout%'
                    AND rr.status IN ('pending', 'processing')
                )
            """)

            urgent_timeouts = cursor.fetchall()

            # 大额订单报价超时48小时
            cursor.execute("""
                SELECT fq.id, fq.quote_date, f.factory_name, c.company_name, fq.total_amount,
                       'large_order_timeout' as timeout_type
                FROM factory_quotes fq
                JOIN factories f ON fq.factory_id = f.id
                LEFT JOIN customers c ON fq.customer_id = c.id
                WHERE fq.status = 'processing'
                AND fq.quote_date < datetime('now', '-48 hours')
                AND fq.total_amount > 100000
                AND fq.id NOT IN (
                    SELECT rr.business_entity_id
                    FROM reminder_records rr
                    WHERE rr.rule_id = 'R002'
                    AND rr.business_entity_type = 'quote'
                    AND rr.trigger_condition LIKE '%large_order_timeout%'
                    AND rr.status IN ('pending', 'processing')
                )
            """)

            large_order_timeouts = cursor.fetchall()

            # 合并所有超时情况
            all_timeouts = list(normal_timeouts) + list(urgent_timeouts) + list(large_order_timeouts)

            for timeout in all_timeouts:
                reminder = {
                    'rule_id': rule['id'],
                    'rule_name': rule['name'],
                    'business_entity_type': 'quote',
                    'business_entity_id': timeout['id'],
                    'trigger_time': datetime.now(),
                    'trigger_condition': f"{timeout['timeout_type']}: {timeout['factory_name']}",
                    'priority': rule['priority'],
                    'due_time': datetime.now() + timedelta(hours=2),  # 2小时内处理
                    'metadata': json.dumps({
                        'factory_name': timeout['factory_name'],
                        'company_name': timeout['company_name'],
                        'total_amount': float(timeout['total_amount']) if timeout['total_amount'] else 0,
                        'timeout_type': timeout['timeout_type']
                    }, ensure_ascii=False)
                }
                reminders.append(reminder)

            logger.info(f"R002检查完成，发现 {len(reminders)} 个报价超时")

        except Exception as e:
            logger.error(f"R002规则检查失败: {e}")

        return reminders

    def check_middle_priority_rules(self, rules: List[Dict]) -> List[Dict]:
        """检查中优先级规则 R006-R010"""
        all_reminders = []

        for rule in rules:
            if rule['id'] in ['R006', 'R007', 'R008', 'R009', 'R010']:
                try:
                    if rule['id'] == 'R008':  # 客户跟进提醒
                        reminders = self._check_customer_follow_up(rule)
                    elif rule['id'] == 'R009':  # 报价分析报告提醒
                        reminders = self._check_quote_analysis_report(rule)
                    else:
                        # 其他中优先级规则的简化实现
                        reminders = self._check_generic_rule(rule)

                    all_reminders.extend(reminders)
                    logger.info(f"{rule['id']}检查完成，发现 {len(reminders)} 个提醒")

                except Exception as e:
                    logger.error(f"{rule['id']}规则检查失败: {e}")

        return all_reminders

    def check_low_priority_rules(self, rules: List[Dict]) -> List[Dict]:
        """检查低优先级规则 R011-R017"""
        all_reminders = []

        for rule in rules:
            if rule['id'] in ['R011', 'R012', 'R013', 'R014', 'R015', 'R016', 'R017']:
                try:
                    # 低优先级规则的简化实现
                    reminders = self._check_generic_rule(rule)
                    all_reminders.extend(reminders)
                    logger.info(f"{rule['id']}检查完成，发现 {len(reminders)} 个提醒")

                except Exception as e:
                    logger.error(f"{rule['id']}规则检查失败: {e}")

        return all_reminders

    def _check_customer_follow_up(self, rule: Dict) -> List[Dict]:
        """R008: 客户跟进提醒"""
        reminders = []

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT c.id, c.company_name, c.last_contact_date, c.contact_email
                FROM customers c
                WHERE c.last_contact_date < date('now', '-7 days')
                AND c.status = 'active'
                AND c.id NOT IN (
                    SELECT rr.business_entity_id
                    FROM reminder_records rr
                    WHERE rr.rule_id = 'R008'
                    AND rr.business_entity_type = 'customer'
                    AND rr.status IN ('pending', 'processing')
                )
            """)

            customers = cursor.fetchall()

            for customer in customers:
                reminder = {
                    'rule_id': rule['id'],
                    'rule_name': rule['name'],
                    'business_entity_type': 'customer',
                    'business_entity_id': customer['id'],
                    'trigger_time': datetime.now(),
                    'trigger_condition': f"客户7天未跟进: {customer['company_name']}",
                    'priority': rule['priority'],
                    'due_time': datetime.now() + timedelta(days=1),
                    'metadata': json.dumps({
                        'company_name': customer['company_name'],
                        'contact_email': customer['contact_email'],
                        'last_contact_date': customer['last_contact_date']
                    }, ensure_ascii=False)
                }
                reminders.append(reminder)

        except Exception as e:
            logger.error(f"客户跟进检查失败: {e}")

        return reminders

    def _check_quote_analysis_report(self, rule: Dict) -> List[Dict]:
        """R009: 报价分析报告提醒"""
        reminders = []

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as updated_quotes
                FROM factory_quotes
                WHERE updated_at > date('now', '-1 day')
            """)

            result = cursor.fetchone()

            if result and result['updated_quotes'] > 0:
                reminder = {
                    'rule_id': rule['id'],
                    'rule_name': rule['name'],
                    'business_entity_type': 'system',
                    'business_entity_id': 0,
                    'trigger_time': datetime.now(),
                    'trigger_condition': f"报价数据更新完成: {result['updated_quotes']}条记录",
                    'priority': rule['priority'],
                    'due_time': datetime.now() + timedelta(hours=2),
                    'metadata': json.dumps({
                        'updated_quotes_count': result['updated_quotes'],
                        'report_date': datetime.now().strftime('%Y-%m-%d')
                    }, ensure_ascii=False)
                }
                reminders.append(reminder)

        except Exception as e:
            logger.error(f"报价分析报告检查失败: {e}")

        return reminders

    def _check_generic_rule(self, rule: Dict) -> List[Dict]:
        """通用规则检查（用于低优先级和其他简单规则）"""
        reminders = []

        # 这里是一个简化实现，实际使用中应该根据具体规则来实现
        # 比如检查是否到了特定时间、是否有特定事件等

        reminder = {
            'rule_id': rule['id'],
            'rule_name': rule['name'],
            'business_entity_type': 'system',
            'business_entity_id': 0,
            'trigger_time': datetime.now(),
            'trigger_condition': f"{rule['name']}定时提醒",
            'priority': rule['priority'],
            'due_time': datetime.now() + timedelta(days=1),
            'metadata': json.dumps({
                'rule_description': rule['description'],
                'check_frequency': rule['check_frequency']
            }, ensure_ascii=False)
        }
        reminders.append(reminder)

        return reminders

    def create_reminder_records(self, reminders: List[Dict]) -> int:
        """创建提醒记录"""
        created_count = 0

        try:
            cursor = self.conn.cursor()

            for reminder in reminders:
                cursor.execute("""
                    INSERT INTO reminder_records
                    (rule_id, rule_name, business_entity_type, business_entity_id,
                     trigger_time, trigger_condition, priority, due_time, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    reminder['rule_id'],
                    reminder['rule_name'],
                    reminder['business_entity_type'],
                    reminder['business_entity_id'],
                    reminder['trigger_time'],
                    reminder['trigger_condition'],
                    reminder['priority'],
                    reminder.get('due_time'),
                    reminder.get('metadata', '{}')
                ))
                created_count += 1

            self.conn.commit()
            logger.info(f"成功创建 {created_count} 条提醒记录")

        except Exception as e:
            logger.error(f"创建提醒记录失败: {e}")
            self.conn.rollback()

        return created_count

    def run_check_cycle(self) -> Dict[str, Any]:
        """执行一次完整的提醒检查周期"""
        start_time = datetime.now()

        try:
            logger.info("开始执行提醒检查周期")

            # 1. 获取启用的规则
            rules = self.get_enabled_rules()
            if not rules:
                logger.warning("没有找到启用的规则")
                return {'status': 'no_rules', 'reminders_created': 0}

            # 2. 分类规则并按优先级检查
            all_reminders = []

            # 高优先级规则 (R001-R002，简化实现)
            high_priority_rules = [r for r in rules if r['id'] in ['R001', 'R002']]
            for rule in high_priority_rules:
                try:
                    if rule['id'] == 'R001':
                        reminders = self.check_rule_R001(rule)
                    elif rule['id'] == 'R002':
                        reminders = self.check_rule_R002(rule)

                    all_reminders.extend(reminders)
                    logger.info(f"{rule['id']}检查完成，发现 {len(reminders)} 个提醒")

                except Exception as e:
                    logger.error(f"{rule['id']}规则检查失败: {e}")

            # 中优先级规则 (R006-R010)
            middle_priority_rules = [r for r in rules if r['id'] in ['R006', 'R007', 'R008', 'R009', 'R010']]
            middle_reminders = self.check_middle_priority_rules(middle_priority_rules)
            all_reminders.extend(middle_reminders)

            # 低优先级规则 (R011-R017)
            low_priority_rules = [r for r in rules if r['id'] in ['R011', 'R012', 'R013', 'R014', 'R015', 'R016', 'R017']]
            low_reminders = self.check_low_priority_rules(low_priority_rules)
            all_reminders.extend(low_reminders)

            # 3. 创建提醒记录
            created_count = self.create_reminder_records(all_reminders)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            result = {
                'status': 'success',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'rules_checked': len(rules),
                'reminders_found': len(all_reminders),
                'reminders_created': created_count,
                'high_priority_reminders': len([r for r in all_reminders if r['priority'] == '高']),
                'middle_priority_reminders': len([r for r in all_reminders if r['priority'] == '中']),
                'low_priority_reminders': len([r for r in all_reminders if r['priority'] == '低'])
            }

            logger.info(f"提醒检查周期完成: {result}")
            return result

        except Exception as e:
            logger.error(f"提醒检查周期执行失败: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'start_time': start_time.isoformat(),
                'duration_seconds': (datetime.now() - start_time).total_seconds()
            }

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='提醒系统检查脚本')
    parser.add_argument('--mode', choices=['single', 'daemon'], default='single',
                       help='运行模式: single(单次) 或 daemon(守护进程)')
    parser.add_argument('--interval', type=int, default=15,
                       help='守护进程模式下的检查间隔(分钟)')
    parser.add_argument('--dry-run', action='store_true',
                       help='试运行模式，不实际发送通知')
    parser.add_argument('--db-path', default='knowledge_base.db',
                       help='数据库文件路径')

    args = parser.parse_args()

    try:
        # 创建提醒检查器
        checker = ReminderChecker(args.db_path)

        if args.dry_run:
            checker.config['dry_run'] = True
            logger.info("启用试运行模式")

        if args.mode == 'single':
            # 单次执行
            logger.info("执行单次提醒检查")
            result = checker.run_check_cycle()

            if result['status'] == 'success':
                print(f"✅ 提醒检查完成")
                print(f"📊 检查规则数: {result['rules_checked']}")
                print(f"🔔 发现提醒: {result['reminders_found']}")
                print(f"📝 创建记录: {result['reminders_created']}")
                print(f"⏱️ 执行时间: {result['duration_seconds']:.2f} 秒")
            else:
                print(f"❌ 提醒检查失败: {result.get('error', '未知错误')}")

        else:
            # 守护进程模式
            print("守护进程模式暂未实现，请使用单次模式")
            # checker.run_daemon(args.interval)

        checker.close()

    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print(f"❌ 执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
