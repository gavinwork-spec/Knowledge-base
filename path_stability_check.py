#!/usr/bin/env python3
"""
文件路径稳定性检查脚本
验证symlink路径访问的稳定性
"""

import os
import sqlite3
import time
from pathlib import Path
import logging

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/path_stability.log'),
            logging.StreamHandler()
        ]
    )

def check_symlink_stability():
    """检查symlink路径稳定性"""
    db_path = "./data/db.sqlite"

    # 确保日志目录存在
    os.makedirs("logs", exist_ok=True)
    setup_logging()

    logging.info("🔍 开始文件路径稳定性检查")
    print("🔍 文件路径稳定性检查")
    print("=" * 50)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 获取所有文件路径
    cursor.execute("""
        SELECT id, drawing_name, file_path, upload_date
        FROM drawings
        WHERE file_path IS NOT NULL AND file_path != ''
        ORDER BY upload_date DESC
    """)

    files = cursor.fetchall()

    print(f"📁 文件路径统计:")
    print(f"  总文件数: {len(files)}")

    # 分析路径类型
    symlink_count = 0
    normal_count = 0
    nutstore_count = 0

    path_analysis = {
        'total_files': len(files),
        'accessible_files': 0,
        'inaccessible_files': 0,
        'symlink_files': 0,
        'normal_files': 0,
        'nutstore_files': 0,
        'path_issues': []
    }

    for file_id, drawing_name, file_path, upload_date in files:
        if not file_path:
            continue

        # 检查路径特征
        is_symlink = '.symlinks' in file_path
        is_nutstore = 'Nutstore' in file_path or '坚果云' in file_path

        if is_symlink:
            symlink_count += 1
        elif is_nutstore:
            nutstore_count += 1
        else:
            normal_count += 1

        # 检查文件可访问性
        path = Path(file_path)
        is_accessible = path.exists() and path.is_file()

        if is_accessible:
            path_analysis['accessible_files'] += 1
        else:
            path_analysis['inaccessible_files'] += 1
            path_analysis['path_issues'].append({
                'id': file_id,
                'name': drawing_name[:50] + '...' if len(drawing_name) > 50 else drawing_name,
                'path': file_path,
                'issue': '文件不可访问'
            })
            logging.warning(f"文件不可访问: {file_path}")

    print(f"  可访问文件: {path_analysis['accessible_files']}")
    print(f"  不可访问文件: {path_analysis['inaccessible_files']}")
    print()

    print(f"🔗 路径类型分析:")
    print(f"  Symlink路径: {symlink_count} 个")
    print(f"  坚果云路径: {nutstore_count} 个")
    print(f"  普通路径: {normal_count} 个")
    print()

    # 详细的symlink检查
    print(f"🔍 Symlink详细检查:")
    symlink_files = [f for f in files if f[2] and '.symlinks' in f[2]]

    if symlink_files:
        print(f"  检查 {len(symlink_files)} 个symlink文件...")

        symlink_errors = []
        for file_id, drawing_name, file_path, upload_date in symlink_files[:10]:  # 只检查前10个
            path = Path(file_path)

            # 检查路径解析
            try:
                resolved_path = path.resolve()
                is_resolved = resolved_path.exists()

                if not is_resolved:
                    symlink_errors.append({
                        'id': file_id,
                        'name': drawing_name[:30] + '...',
                        'original_path': file_path,
                        'resolved_path': str(resolved_path),
                        'issue': 'Symlink解析失败'
                    })
                    logging.error(f"Symlink解析失败: {file_path} -> {resolved_path}")

            except Exception as e:
                symlink_errors.append({
                    'id': file_id,
                    'name': drawing_name[:30] + '...',
                    'original_path': file_path,
                    'issue': f'路径解析错误: {str(e)}'
                })
                logging.error(f"路径解析错误: {file_path} - {e}")

        if symlink_errors:
            print(f"  ❌ 发现 {len(symlink_errors)} 个symlink问题:")
            for error in symlink_errors[:5]:  # 只显示前5个
                print(f"    {error['name']}: {error['issue']}")
        else:
            print(f"  ✅ Symlink检查通过")
    else:
        print(f"  ℹ️  没有发现symlink文件")

    print()

    # 坚果云路径检查
    print(f"🌰 坚果云路径检查:")
    nutstore_files = [f for f in files if f[2] and ('Nutstore' in f[2] or '坚果云' in f[2])]

    if nutstore_files:
        print(f"  检查 {len(nutstore_files)} 个坚果云文件...")

        nutstore_errors = []
        for file_id, drawing_name, file_path, upload_date in nutstore_files[:10]:  # 只检查前10个
            path = Path(file_path)

            # 检查坚果云同步状态
            if path.exists():
                # 检查是否为同步中状态
                try:
                    stat = path.stat()
                    # 这里可以添加更多坚果云特定的检查
                    file_size = stat.st_size

                    if file_size == 0:
                        nutstore_errors.append({
                            'id': file_id,
                            'name': drawing_name[:30] + '...',
                            'path': file_path,
                            'issue': '文件大小为0，可能同步未完成'
                        })
                except Exception as e:
                    nutstore_errors.append({
                        'id': file_id,
                        'name': drawing_name[:30] + '...',
                        'path': file_path,
                        'issue': f'文件状态检查失败: {str(e)}'
                    })
            else:
                nutstore_errors.append({
                    'id': file_id,
                    'name': drawing_name[:30] + '...',
                    'path': file_path,
                    'issue': '文件不存在'
                })

        if nutstore_errors:
            print(f"  ❌ 发现 {len(nutstore_errors)} 个坚果云问题:")
            for error in nutstore_errors[:5]:
                print(f"    {error['name']}: {error['issue']}")
        else:
            print(f"  ✅ 坚果云文件检查通过")
    else:
        print(f"  ℹ️  没有发现坚果云文件")

    print()

    # 路径稳定性建议
    print(f"💡 路径稳定性建议:")

    if path_analysis['inaccessible_files'] > 0:
        print(f"  - 有 {path_analysis['inaccessible_files']} 个文件不可访问，建议检查路径")
        print(f"  - 考虑运行文件修复脚本更新路径")

    if symlink_errors:
        print(f"  - 发现 {len(symlink_errors)} 个symlink问题，建议:")
        print(f"    * 检查symlink源文件是否存在")
        print(f"    * 重新创建损坏的symlink")
        print(f"    * 考虑使用绝对路径")

    if nutstore_errors:
        print(f"  - 发现 {len(nutstore_errors)} 个坚果云问题，建议:")
        print(f"    * 检查坚果云同步状态")
        print(f"    * 确认云同步已完成")
        print(f"    * 检查网络连接")

    if symlink_count > 0:
        print(f"  - 建议：为symlink路径创建监控机制")
        print(f"  - 建议：定期验证symlink有效性")

    # 生成报告
    report = {
        'check_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'path_analysis': path_analysis,
        'symlink_errors': len(symlink_errors) if 'symlink_errors' in locals() else 0,
        'nutstore_errors': len(nutstore_errors) if 'nutstore_errors' in locals() else 0,
        'recommendations': [
            '定期检查文件路径有效性',
            '监控symlink状态',
            '验证坚果云同步完成度'
        ]
    }

    import json
    with open('data/processed/path_stability_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📄 详细报告已保存到: data/processed/path_stability_report.json")

    conn.close()
    logging.info("文件路径稳定性检查完成")

def test_path_resolution():
    """测试路径解析性能"""
    print(f"\n⚡ 路径解析性能测试:")

    db_path = "./data/db.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 获取一些示例路径
    cursor.execute("SELECT file_path FROM drawings WHERE file_path IS NOT NULL LIMIT 100")
    paths = [row[0] for row in cursor.fetchall() if row[0]]

    if paths:
        # 测试路径解析时间
        start_time = time.time()
        resolved_paths = []

        for path in paths:
            try:
                resolved = Path(path).resolve()
                resolved_paths.append(str(resolved))
            except Exception:
                pass

        end_time = time.time()
        avg_time = (end_time - start_time) / len(paths) * 1000

        print(f"  解析 {len(paths)} 个路径")
        print(f"  总耗时: {end_time - start_time:.3f} 秒")
        print(f"  平均耗时: {avg_time:.2f} 毫秒/路径")
        print(f"  成功解析: {len(resolved_paths)} 个")

    conn.close()

if __name__ == "__main__":
    check_symlink_stability()
    test_path_resolution()