import React, { useState, useEffect } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';

const ComparisonMatrix = () => {
  const [selectedEntries, setSelectedEntries] = useState([]);
  const [availableEntries, setAvailableEntries] = useState([]);
  const [comparisonData, setComparisonData] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  // 商业色彩方案
  const businessColors = {
    primary: '#003366',    // 深蓝色
    secondary: '#FF8800',  // 橙色强调色
    success: '#28a745',    // 绿色
    danger: '#dc3545',     // 红色
    warning: '#ffc107',    // 黄色
    info: '#17a2b8',       // 信息蓝
    light: '#f8f9fa',      // 浅色背景
    dark: '#343a40'        // 深色文字
  };

  useEffect(() => {
    loadAvailableEntries();
  }, []);

  const loadAvailableEntries = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/knowledge/entries?limit=50');
      const data = await response.json();
      if (data.success) {
        setAvailableEntries(data.data.entries);
      }
    } catch (error) {
      console.error('Failed to load entries:', error);
    }
  };

  const handleEntrySelection = (entry) => {
    if (selectedEntries.find(e => e.id === entry.id)) {
      setSelectedEntries(selectedEntries.filter(e => e.id !== entry.id));
    } else if (selectedEntries.length < 3) {
      setSelectedEntries([...selectedEntries, entry]);
    } else {
      alert('最多只能选择3个条目进行对比');
    }
  };

  const performComparison = async () => {
    if (selectedEntries.length < 2) {
      alert('请至少选择2个条目进行对比');
      return;
    }

    setIsLoading(true);
    try {
      // 准备对比数据
      const comparisonFields = [
        { key: 'name', label: '名称', type: 'text' },
        { key: 'entity_type', label: '类型', type: 'badge' },
        { key: 'description', label: '描述', type: 'text' },
        { key: 'created_at', label: '创建时间', type: 'date' }
      ];

      // 动态添加属性字段
      const allAttributes = new Set();
      selectedEntries.forEach(entry => {
        if (entry.attributes_json) {
          try {
            const attrs = JSON.parse(entry.attributes_json);
            Object.keys(attrs).forEach(key => allAttributes.add(key));
          } catch (e) {
            console.error('Failed to parse attributes:', e);
          }
        }
      });

      // 添加属性字段到对比列表
      Array.from(allAttributes).forEach(attr => {
        comparisonFields.push({
          key: `attributes.${attr}`,
          label: attr,
          type: 'text'
        });
      });

      // 生成对比数据
      const comparison = comparisonFields.map(field => {
        const row = { field: field.label, type: field.type };

        selectedEntries.forEach((entry, index) => {
          let value = '';

          if (field.key.startsWith('attributes.')) {
            const attrKey = field.key.replace('attributes.', '');
            if (entry.attributes_json) {
              try {
                const attrs = JSON.parse(entry.attributes_json);
                value = attrs[attrKey] || '';
              } catch (e) {
                value = '';
              }
            }
          } else {
            value = entry[field.key] || '';
          }

          row[`entry_${index}`] = {
            value: value,
            original: value
          };
        });

        // 计算差异并高亮
        if (selectedEntries.length === 2) {
          const val1 = row.entry_0.value;
          const val2 = row.entry_1.value;

          if (val1 && val2 && val1 !== val2) {
            // 计算数值差异
            const num1 = parseFloat(val1.replace(/[^0-9.-]/g, ''));
            const num2 = parseFloat(val2.replace(/[^0-9.-]/g, ''));

            if (!isNaN(num1) && !isNaN(num2)) {
              const diff = Math.abs((num1 - num2) / num1 * 100);
              row.entry_0.highlight = diff > 10 ? (diff > 20 ? 'danger' : 'warning') : '';
              row.entry_1.highlight = diff > 10 ? (diff > 20 ? 'danger' : 'warning') : '';
              row.entry_0.deviation = diff.toFixed(1) + '%';
              row.entry_1.deviation = diff.toFixed(1) + '%';
            } else {
              row.entry_0.highlight = 'info';
              row.entry_1.highlight = 'info';
            }
          }
        } else if (selectedEntries.length === 3) {
          // 三方对比逻辑
          const values = [row.entry_0.value, row.entry_1.value, row.entry_2.value].filter(v => v);
          const allSame = values.every(v => v === values[0]);

          if (!allSame && values.length > 1) {
            [0, 1, 2].forEach(i => {
              row[`entry_${i}`].highlight = 'warning';
            });
          }
        }

        return row;
      });

      setComparisonData(comparison);
    } catch (error) {
      console.error('Comparison failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const exportComparison = () => {
    const csvContent = generateCSV();
    downloadCSV(csvContent, 'knowledge_comparison.csv');
  };

  const generateCSV = () => {
    const headers = ['字段', ...selectedEntries.map((_, i) => `条目 ${i + 1}`)];
    const rows = comparisonData.map(row => {
      const rowData = [row.field];
      selectedEntries.forEach((_, i) => {
        rowData.push(row[`entry_${i}`]?.value || '');
      });
      return rowData;
    });

    return [headers, ...rows].map(row => row.join(',')).join('\n');
  };

  const downloadCSV = (content, filename) => {
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const renderValue = (value, highlight, deviation) => {
    const cellClass = highlight ? `table-${highlight}` : '';
    const deviationText = deviation ? ` <span class="badge bg-secondary">${deviation}</span>` : '';

    return (
      <td className={cellClass}>
        <div>
          {value || '-'}
          {deviationText}
        </div>
      </td>
    );
  };

  const getBadgeColor = (entityType) => {
    const colors = {
      'product': 'primary',
      'customer': 'success',
      'factory': 'warning',
      'quote': 'danger',
      'inquiry': 'info'
    };
    return colors[entityType] || 'secondary';
  };

  return (
    <div className="container-fluid py-4" style={{ fontFamily: 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif' }}>
      {/* 页面标题 */}
      <div className="row mb-4">
        <div className="col-12">
          <div className="d-flex justify-content-between align-items-center">
            <div>
              <h2 className="mb-0" style={{ color: businessColors.primary }}>
                <i className="bi bi-columns-gap me-2"></i>
                知识对比矩阵
              </h2>
              <p className="text-muted mb-0">选择2-3个知识条目进行详细对比分析</p>
            </div>
            <div>
              <button
                className="btn btn-outline-secondary me-2"
                onClick={exportComparison}
                disabled={comparisonData.length === 0}
              >
                <i className="bi bi-download me-1"></i>
                导出CSV
              </button>
              <button
                className="btn btn-primary"
                onClick={performComparison}
                disabled={selectedEntries.length < 2 || isLoading}
              >
                {isLoading ? (
                  <>
                    <span className="spinner-border spinner-border-sm me-1" role="status"></span>
                    分析中...
                  </>
                ) : (
                  <>
                    <i className="bi bi-play-circle me-1"></i>
                    开始对比
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 选择区域 */}
      <div className="row mb-4">
        <div className="col-12">
          <div className="card" style={{ border: 'none', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
            <div className="card-header" style={{ backgroundColor: businessColors.primary, color: 'white' }}>
              <h5 className="mb-0">
                <i className="bi bi-check-square me-2"></i>
                选择对比条目 (已选择: {selectedEntries.length}/3)
              </h5>
            </div>
            <div className="card-body">
              <div className="row">
                {availableEntries.map(entry => (
                  <div key={entry.id} className="col-md-4 col-lg-3 mb-3">
                    <div
                      className={`card h-100 cursor-pointer ${
                        selectedEntries.find(e => e.id === entry.id)
                          ? 'border-primary'
                          : 'border-light'
                      }`}
                      style={{
                        cursor: 'pointer',
                        border: selectedEntries.find(e => e.id === entry.id)
                          ? `2px solid ${businessColors.primary}`
                          : '1px solid #dee2e6',
                        borderRadius: '8px',
                        transition: 'all 0.2s ease'
                      }}
                      onClick={() => handleEntrySelection(entry)}
                    >
                      <div className="card-body p-3">
                        <div className="d-flex justify-content-between align-items-start mb-2">
                          <h6 className="card-title mb-1 text-truncate" style={{ color: businessColors.primary }}>
                            {entry.name}
                          </h6>
                          <span className={`badge bg-${getBadgeColor(entry.entity_type)}`}>
                            {entry.entity_type_display}
                          </span>
                        </div>
                        <p className="card-text small text-muted mb-2" style={{
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical'
                        }}>
                          {entry.description}
                        </p>
                        <div className="d-flex justify-content-between align-items-center">
                          <small className="text-muted">
                            {new Date(entry.created_at).toLocaleDateString('zh-CN')}
                          </small>
                          {selectedEntries.find(e => e.id === entry.id) && (
                            <i className="bi bi-check-circle-fill text-primary"></i>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 对比结果 */}
      {comparisonData.length > 0 && (
        <div className="row">
          <div className="col-12">
            <div className="card" style={{ border: 'none', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
              <div className="card-header" style={{ backgroundColor: businessColors.secondary, color: 'white' }}>
                <h5 className="mb-0">
                  <i className="bi bi-bar-chart me-2"></i>
                  对比分析结果
                  <div className="float-end">
                    <span className="badge bg-light text-dark me-2">
                      差异 >20%: <span className="text-danger">红色高亮</span>
                    </span>
                    <span className="badge bg-light text-dark">
                      差异 10-20%: <span className="text-warning">黄色高亮</span>
                    </span>
                  </div>
                </h5>
              </div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-hover mb-0">
                    <thead style={{ backgroundColor: businessColors.light }}>
                      <tr>
                        <th style={{ borderRight: `2px solid ${businessColors.primary}` }}>字段</th>
                        {selectedEntries.map((entry, index) => (
                          <th key={index} style={{ borderRight: index < selectedEntries.length - 1 ? `1px solid ${businessColors.primary}` : 'none' }}>
                            <div className="text-center">
                              <div className="fw-bold">{entry.name}</div>
                              <small className="text-muted">{entry.entity_type_display}</small>
                            </div>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {comparisonData.map((row, index) => (
                        <tr key={index} className={index % 2 === 0 ? 'table-light' : ''}>
                          <td style={{
                            fontWeight: 'bold',
                            borderRight: `2px solid ${businessColors.primary}`,
                            backgroundColor: businessColors.light
                          }}>
                            {row.field}
                          </td>
                          {selectedEntries.map((_, entryIndex) => {
                            const cellData = row[`entry_${entryIndex}`];
                            return renderValue(
                              cellData?.value,
                              cellData?.highlight,
                              cellData?.deviation
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 使用说明 */}
      <div className="row mt-4">
        <div className="col-12">
          <div className="alert alert-info" style={{ border: 'none', backgroundColor: '#e3f2fd' }}>
            <h6 className="alert-heading">
              <i className="bi bi-info-circle me-2"></i>
              使用说明
            </h6>
            <ul className="mb-0">
              <li>选择2-3个知识条目进行对比分析</li>
              <li>系统会自动识别数值差异并高亮显示</li>
              <li>差异超过20%显示为红色，10-20%显示为黄色</li>
              <li>支持导出CSV格式进行进一步分析</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ComparisonMatrix;