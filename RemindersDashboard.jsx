import React, { useState, useEffect } from 'react';

const RemindersDashboard = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [reminderRecords, setReminderRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    status: '',
    page: 1,
    limit: 20
  });

  // API基础URL
  const API_BASE_URL = 'http://localhost:8001/api/v1';

  // 获取仪表板数据
  const fetchDashboardData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/reminders/dashboard`);
      const result = await response.json();

      if (result.success) {
        setDashboardData(result.data);
      } else {
        setError(result.error?.message || '获取仪表板数据失败');
      }
    } catch (err) {
      setError(`网络错误: ${err.message}`);
    }
  };

  // 获取提醒记录
  const fetchReminderRecords = async () => {
    try {
      const params = new URLSearchParams({
        page: filters.page,
        limit: filters.limit,
        ...(filters.status && { status: filters.status })
      });

      const response = await fetch(`${API_BASE_URL}/reminders/records?${params}`);
      const result = await response.json();

      if (result.success) {
        setReminderRecords(result.data);
      } else {
        setError(result.error?.message || '获取提醒记录失败');
      }
    } catch (err) {
      setError(`网络错误: ${err.message}`);
    }
  };

  // 标记提醒为已处理
  const handleReminder = async (recordId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/reminders/records/${recordId}/handle`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          handled_by: 'user',
          notes: '通过前端界面手动处理'
        })
      });

      const result = await response.json();

      if (result.success) {
        // 刷新数据
        fetchReminderRecords();
        fetchDashboardData();
        alert('提醒已标记为已处理');
      } else {
        alert(`处理失败: ${result.error?.message || '未知错误'}`);
      }
    } catch (err) {
      alert(`网络错误: ${err.message}`);
    }
  };

  // 手动触发提醒
  const triggerReminder = async (ruleId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/reminders/trigger`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ rule_id: ruleId })
      });

      const result = await response.json();

      if (result.success) {
        alert('提醒已手动触发');
        fetchReminderRecords();
        fetchDashboardData();
      } else {
        alert(`触发失败: ${result.error?.message || '未知错误'}`);
      }
    } catch (err) {
      alert(`网络错误: ${err.message}`);
    }
  };

  // 切换规则状态
  const toggleRule = async (ruleId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/reminders/rules/${ruleId}/toggle`, {
        method: 'POST'
      });

      const result = await response.json();

      if (result.success) {
        alert(`规则已${result.data.is_active ? '启用' : '禁用'}`);
        fetchDashboardData();
      } else {
        alert(`操作失败: ${result.error?.message || '未知错误'}`);
      }
    } catch (err) {
      alert(`网络错误: ${err.message}`);
    }
  };

  // 刷新数据
  const refreshData = () => {
    setLoading(true);
    Promise.all([fetchDashboardData(), fetchReminderRecords()])
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    refreshData();
  }, [filters]);

  // 格式化日期时间
  const formatDateTime = (dateTimeStr) => {
    if (!dateTimeStr) return '-';
    return new Date(dateTimeStr).toLocaleString('zh-CN');
  };

  // 获取状态颜色
  const getStatusColor = (status) => {
    const colors = {
      'pending': '#ff9800',
      'handled': '#4caf50',
      'failed': '#f44336',
      'processing': '#2196f3'
    };
    return colors[status] || '#757575';
  };

  // 获取优先级颜色
  const getPriorityColor = (priority) => {
    const colors = {
      1: '#f44336', // 高优先级 - 红色
      2: '#ff9800', // 中优先级 - 橙色
      3: '#4caf50'  // 低优先级 - 绿色
    };
    return colors[priority] || '#757575';
  };

  // 分页处理
  const handlePageChange = (newPage) => {
    setFilters(prev => ({ ...prev, page: newPage }));
  };

  // 筛选处理
  const handleFilterChange = (newFilters) => {
    setFilters(prev => ({ ...prev, ...newFilters, page: 1 }));
  };

  if (loading) {
    return (
      <div style={{ padding: '20px', textAlign: 'center' }}>
        <div>加载中...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '20px', textAlign: 'center', color: 'red' }}>
        <div>错误: {error}</div>
        <button onClick={refreshData} style={{ marginTop: '10px' }}>
          重试
        </button>
      </div>
    );
  }

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>🔔 提醒中心仪表盘</h1>

      {/* 顶部操作栏 */}
      <div style={{
        marginBottom: '20px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <button
            onClick={refreshData}
            style={{
              padding: '8px 16px',
              backgroundColor: '#2196f3',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              marginRight: '10px'
            }}
          >
            🔄 刷新数据
          </button>
          <span style={{ color: '#666' }}>
            最后更新: {formatDateTime(dashboardData?.system_health?.last_check)}
          </span>
        </div>

        <div style={{
          padding: '8px 16px',
          backgroundColor: dashboardData?.system_health?.status === 'healthy' ? '#4caf50' : '#ff9800',
          color: 'white',
          borderRadius: '4px',
          fontSize: '14px'
        }}>
          系统状态: {dashboardData?.system_health?.status === 'healthy' ? '✅ 健康' : '⚠️ 警告'}
        </div>
      </div>

      {/* 统计卡片 */}
      {dashboardData && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '20px',
          marginBottom: '30px'
        }}>
          {/* 规则统计 */}
          <div style={{
            border: '1px solid #ddd',
            borderRadius: '8px',
            padding: '20px',
            backgroundColor: '#f9f9f9'
          }}>
            <h3 style={{ margin: '0 0 15px 0', color: '#333' }}>📋 提醒规则</h3>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#2196f3' }}>
              {dashboardData.rules?.active_rules || 0}
            </div>
            <div style={{ color: '#666', fontSize: '14px' }}>
              活跃规则 / 总计 {dashboardData.rules?.total_rules || 0}
            </div>
            <div style={{ marginTop: '10px', fontSize: '12px' }}>
              <div>🔴 高优先级: {dashboardData.rules?.high_priority_rules || 0}</div>
              <div>🟡 中优先级: {dashboardData.rules?.medium_priority_rules || 0}</div>
              <div>🟢 低优先级: {dashboardData.rules?.low_priority_rules || 0}</div>
            </div>
          </div>

          {/* 今日统计 */}
          <div style={{
            border: '1px solid #ddd',
            borderRadius: '8px',
            padding: '20px',
            backgroundColor: '#f9f9f9'
          }}>
            <h3 style={{ margin: '0 0 15px 0', color: '#333' }}>📊 今日提醒</h3>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#ff9800' }}>
              {dashboardData.today?.total_reminders || 0}
            </div>
            <div style={{ color: '#666', fontSize: '14px' }}>
              总提醒数
            </div>
            <div style={{ marginTop: '10px', fontSize: '12px' }}>
              <div>✅ 已处理: {dashboardData.today?.handled_reminders || 0}</div>
              <div>⏳ 待处理: {dashboardData.today?.pending_reminders || 0}</div>
              <div>❌ 失败: {dashboardData.today?.failed_reminders || 0}</div>
            </div>
          </div>

          {/* 系统健康 */}
          <div style={{
            border: '1px solid #ddd',
            borderRadius: '8px',
            padding: '20px',
            backgroundColor: '#f9f9f9'
          }}>
            <h3 style={{ margin: '0 0 15px 0', color: '#333' }}>🏥 系统健康</h3>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#4caf50' }}>
              {dashboardData.system_health?.failure_rate || 0}%
            </div>
            <div style={{ color: '#666', fontSize: '14px' }}>
              失败率
            </div>
            <div style={{ marginTop: '10px', fontSize: '12px' }}>
              <div>⏱️ 运行时间: {dashboardData.system_health?.uptime || 'N/A'}</div>
              <div>🔧 服务状态: 正常</div>
            </div>
          </div>
        </div>
      )}

      {/* 筛选器 */}
      <div style={{
        marginBottom: '20px',
        padding: '15px',
        backgroundColor: '#f5f5f5',
        borderRadius: '8px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <label>
            状态筛选:
            <select
              value={filters.status}
              onChange={(e) => handleFilterChange({ status: e.target.value })}
              style={{ marginLeft: '5px', padding: '5px' }}
            >
              <option value="">全部</option>
              <option value="pending">待处理</option>
              <option value="handled">已处理</option>
              <option value="failed">失败</option>
              <option value="processing">处理中</option>
            </select>
          </label>
        </div>
      </div>

      {/* 提醒记录表格 */}
      <div style={{
        border: '1px solid #ddd',
        borderRadius: '8px',
        overflow: 'hidden'
      }}>
        <div style={{
          backgroundColor: '#2196f3',
          color: 'white',
          padding: '15px',
          fontWeight: 'bold'
        }}>
          📝 提醒记录
        </div>

        {reminderRecords.records?.length > 0 ? (
          <>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ backgroundColor: '#f5f5f5' }}>
                    <th style={{ padding: '12px', textAlign: 'left', borderBottom: '1px solid #ddd' }}>规则名称</th>
                    <th style={{ padding: '12px', textAlign: 'left', borderBottom: '1px solid #ddd' }}>触发时间</th>
                    <th style={{ padding: '12px', textAlign: 'left', borderBottom: '1px solid #ddd' }}>实体类型</th>
                    <th style={{ padding: '12px', textAlign: 'left', borderBottom: '1px solid #ddd' }}>触发原因</th>
                    <th style={{ padding: '12px', textAlign: 'left', borderBottom: '1px solid #ddd' }}>状态</th>
                    <th style={{ padding: '12px', textAlign: 'left', borderBottom: '1px solid #ddd' }}>操作</th>
                  </tr>
                </thead>
                <tbody>
                  {reminderRecords.records.map((record) => (
                    <tr key={record.id} style={{ borderBottom: '1px solid #ddd' }}>
                      <td style={{ padding: '12px' }}>
                        <div style={{ fontWeight: 'bold' }}>{record.rule_name}</div>
                        <div style={{ fontSize: '12px', color: '#666' }}>
                          优先级: <span style={{ color: getPriorityColor(record.rule_priority) }}>
                            {record.rule_priority === 1 ? '高' : record.rule_priority === 2 ? '中' : '低'}
                          </span>
                        </div>
                      </td>
                      <td style={{ padding: '12px' }}>
                        {formatDateTime(record.triggered_at)}
                      </td>
                      <td style={{ padding: '12px' }}>
                        {record.business_entity_type}
                        {record.business_entity_id && ` #${record.business_entity_id}`}
                      </td>
                      <td style={{ padding: '12px', maxWidth: '300px' }}>
                        <div style={{
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}>
                          {record.trigger_reason}
                        </div>
                      </td>
                      <td style={{ padding: '12px' }}>
                        <span style={{
                          padding: '4px 8px',
                          borderRadius: '4px',
                          fontSize: '12px',
                          backgroundColor: getStatusColor(record.status),
                          color: 'white'
                        }}>
                          {record.status === 'pending' ? '待处理' :
                           record.status === 'handled' ? '已处理' :
                           record.status === 'failed' ? '失败' :
                           record.status === 'processing' ? '处理中' : record.status}
                        </span>
                      </td>
                      <td style={{ padding: '12px' }}>
                        {record.status === 'pending' && (
                          <button
                            onClick={() => handleReminder(record.id)}
                            style={{
                              padding: '6px 12px',
                              backgroundColor: '#4caf50',
                              color: 'white',
                              border: 'none',
                              borderRadius: '4px',
                              cursor: 'pointer',
                              fontSize: '12px'
                            }}
                          >
                            标记已处理
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* 分页 */}
            {reminderRecords.pagination && (
              <div style={{
                padding: '15px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                backgroundColor: '#f9f9f9'
              }}>
                <div>
                  显示 {((reminderRecords.pagination.page - 1) * reminderRecords.pagination.limit) + 1} -
                  {Math.min(reminderRecords.pagination.page * reminderRecords.pagination.limit, reminderRecords.pagination.total)}
                  共 {reminderRecords.pagination.total} 条记录
                </div>
                <div>
                  <button
                    onClick={() => handlePageChange(reminderRecords.pagination.page - 1)}
                    disabled={!reminderRecords.pagination.has_prev}
                    style={{
                      padding: '6px 12px',
                      marginRight: '5px',
                      backgroundColor: reminderRecords.pagination.has_prev ? '#2196f3' : '#ccc',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: reminderRecords.pagination.has_prev ? 'pointer' : 'not-allowed'
                    }}
                  >
                    上一页
                  </button>
                  <span style={{ margin: '0 10px' }}>
                    第 {reminderRecords.pagination.page} / {reminderRecords.pagination.total_pages} 页
                  </span>
                  <button
                    onClick={() => handlePageChange(reminderRecords.pagination.page + 1)}
                    disabled={!reminderRecords.pagination.has_next}
                    style={{
                      padding: '6px 12px',
                      backgroundColor: reminderRecords.pagination.has_next ? '#2196f3' : '#ccc',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: reminderRecords.pagination.has_next ? 'pointer' : 'not-allowed'
                    }}
                  >
                    下一页
                  </button>
                </div>
              </div>
            )}
          </>
        ) : (
          <div style={{ padding: '40px', textAlign: 'center', color: '#666' }}>
            📭 暂无提醒记录
          </div>
        )}
      </div>

      {/* 最近活动 */}
      {dashboardData?.recent_activity?.length > 0 && (
        <div style={{ marginTop: '30px' }}>
          <h3>🕐 最近活动</h3>
          <div style={{
            border: '1px solid #ddd',
            borderRadius: '8px',
            overflow: 'hidden'
          }}>
            {dashboardData.recent_activity.map((activity, index) => (
              <div key={activity.id} style={{
                padding: '15px',
                borderBottom: index < dashboardData.recent_activity.length - 1 ? '1px solid #eee' : 'none',
                backgroundColor: index % 2 === 0 ? '#f9f9f9' : 'white'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <div>
                    <strong>{activity.rule_name}</strong>
                    <div style={{ color: '#666', fontSize: '14px' }}>
                      {activity.trigger_reason}
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{
                      padding: '4px 8px',
                      borderRadius: '4px',
                      fontSize: '12px',
                      backgroundColor: getStatusColor(activity.status),
                      color: 'white',
                      marginBottom: '5px'
                    }}>
                      {activity.status === 'pending' ? '待处理' :
                       activity.status === 'handled' ? '已处理' :
                       activity.status === 'failed' ? '失败' : activity.status}
                    </div>
                    <div style={{ fontSize: '12px', color: '#666' }}>
                      {formatDateTime(activity.triggered_at)}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default RemindersDashboard;