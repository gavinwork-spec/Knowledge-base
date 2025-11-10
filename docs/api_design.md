# 知识库系统 API 接口设计文档

## 概述
本文档定义了知识库系统的 RESTful API 接口规范，用于前端可视化数据展示和交互。

## 基础信息

- **Base URL**: `http://localhost:8000/api/v1`
- **认证方式**: API Key (待实现)
- **数据格式**: JSON
- **字符编码**: UTF-8
- **时间格式**: ISO 8601 (YYYY-MM-DDTHH:MM:SSZ)

## 通用响应格式

### 成功响应
```json
{
  "success": true,
  "data": { ... },
  "message": "操作成功",
  "timestamp": "2025-11-05T17:30:00Z"
}
```

### 错误响应
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "参数验证失败",
    "details": { ... }
  },
  "timestamp": "2025-11-05T17:30:00Z"
}
```

## API 端点

### 1. 图纸统计接口

#### 1.1 图纸分类统计
- **端点**: `GET /statistics/drawings/by_category`
- **描述**: 获取按产品类别统计的图纸数据

**查询参数**:
- `date_from`: 开始日期 (YYYY-MM-DD)
- `date_to`: 结束日期 (YYYY-MM-DD)
- `include_unclassified`: 是否包含未分类 (true/false, 默认false)

**响应示例**:
```json
{
  "success": true,
  "data": {
    "summary": {
      "total_drawings": 408,
      "classified_drawings": 39,
      "classification_rate": 9.6,
      "date_range": {
        "from": "2025-01-01",
        "to": "2025-11-05"
      }
    },
    "categories": [
      {
        "product_category": "紧固件-螺栓螺钉",
        "count": 31,
        "percentage": 7.6,
        "standard_count": 28,
        "custom_count": 3,
        "avg_confidence": 0.75
      },
      {
        "product_category": "紧固件-垫圈垫片",
        "count": 4,
        "percentage": 1.0,
        "standard_count": 4,
        "custom_count": 0,
        "avg_confidence": 0.80
      }
    ],
    "trend": {
      "monthly_data": [
        {
          "month": "2025-10",
          "classified_count": 25,
          "total_count": 380
        }
      ]
    }
  }
}
```

#### 1.2 图纸数据源统计
- **端点**: `GET /statistics/drawings/by_source`
- **描述**: 获取按数据源统计的图纸数据

**响应示例**:
```json
{
  "success": true,
  "data": {
    "sources": [
      {
        "data_source": "email",
        "count": 150,
        "percentage": 36.8
      },
      {
        "data_source": "scan",
        "count": 200,
        "percentage": 49.0
      }
    ]
  }
}
```

### 2. 客户统计接口

#### 2.1 客户状态统计
- **端点**: `GET /statistics/customers/by_status`
- **描述**: 获取按客户状态统计的数据

**查询参数**:
- `status_filter`: 状态过滤 (potential/active/inactive/silent)
- `country_filter`: 国家过滤
- `date_from`: 注册日期开始
- `date_to`: 注册日期结束

**响应示例**:
```json
{
  "success": true,
  "data": {
    "summary": {
      "total_customers": 11,
      "active_customers": 8,
      "potential_customers": 3,
      "avg_drawings_per_customer": 37.1
    },
    "status_distribution": [
      {
        "customer_status": "potential",
        "count": 8,
        "percentage": 72.7,
        "avg_drawings": 2.5,
        "countries": 3
      },
      {
        "customer_status": "active",
        "count": 3,
        "percentage": 27.3,
        "avg_drawings": 130.0,
        "countries": 2
      }
    ],
    "geographic_distribution": [
      {
        "country": "中国",
        "count": 8,
        "customers": [
          {
            "company_name": "ABC Company",
            "customer_status": "potential",
            "total_drawings": 5
          }
        ]
      }
    ]
  }
}
```

#### 2.2 客户活跃度分析
- **端点**: `GET /statistics/customers/activity`
- **描述**: 获取客户活跃度分析数据

**响应示例**:
```json
{
  "success": true,
  "data": {
    "activity_levels": [
      {
        "level": "高活跃",
        "min_drawings": 50,
        "count": 2,
        "customers": ["Customer A", "Customer B"]
      },
      {
        "level": "中活跃",
        "min_drawings": 10,
        "max_drawings": 49,
        "count": 3,
        "customers": ["Customer C", "Customer D", "Customer E"]
      },
      {
        "level": "低活跃",
        "max_drawings": 9,
        "count": 6,
        "customers": ["Customer F", "Customer G"]
      }
    ],
    "trend": {
      "new_customers_per_month": [
        {
          "month": "2025-10",
          "new_customers": 2
        }
      ]
    }
  }
}
```

### 3. 工厂报价趋势接口

#### 3.1 工厂报价趋势
- **端点**: `GET /trends/factory_quotes`
- **描述**: 获取工厂报价趋势数据

**查询参数**:
- `period`: 时间周期 (month/quarter/year, 默认month)
- `factory_id`: 工厂ID过滤
- `product_category`: 产品类别过滤
- `date_from`: 开始日期
- `date_to`: 结束日期

**响应示例**:
```json
{
  "success": true,
  "data": {
    "summary": {
      "total_quotes": 156,
      "total_factories": 2,
      "date_range": {
        "from": "2025-01-01",
        "to": "2025-11-05"
      },
      "overall_price_trend": 2.5
    },
    "monthly_trends": [
      {
        "period": "2025-10",
        "factory_id": 1,
        "factory_name": "Factory A",
        "product_category": "紧固件-螺栓螺钉",
        "avg_price": 12.50,
        "min_price": 10.00,
        "max_price": 15.00,
        "quote_count": 15,
        "price_change_pct": 2.1,
        "moq_avg": 1000
      }
    ],
    "factory_performance": [
      {
        "factory_id": 1,
        "factory_name": "Factory A",
        "total_quotes": 89,
        "avg_price": 12.80,
        "product_categories": ["紧固件-螺栓螺钉", "紧固件-螺母"],
        "price_volatility": 8.5,
        "quote_frequency": 8.9
      }
    ],
    "price_anomalies": [
      {
        "factory_id": 1,
        "factory_name": "Factory A",
        "product_category": "紧固件-螺栓螺钉",
        "quote_date": "2025-10-15",
        "price": 25.00,
        "expected_range": "10.00 - 16.00",
        "anomaly_type": "price_outlier",
        "deviation_pct": 95.2
      }
    ]
  }
}
```

#### 3.2 价格波动分析
- **端点**: `GET /trends/price_volatility`
- **描述**: 获取价格波动性分析数据

**响应示例**:
```json
{
  "success": true,
  "data": {
    "overall_volatility": 12.5,
    "category_volatility": [
      {
        "product_category": "紧固件-螺栓螺钉",
        "avg_monthly_change": 2.1,
        "price_volatility": 8.5,
        "max_increase": 15.2,
        "max_decrease": -8.7,
        "change_frequency": 3,
        "total_periods": 12
      }
    ],
    "factory_volatility": [
      {
        "factory_id": 1,
        "factory_name": "Factory A",
        "overall_volatility": 10.2,
        "most_volatile_category": "紧固件-螺栓螺钉"
      }
    ],
    "volatility_trend": {
      "monthly_volatility": [
        {
          "month": "2025-10",
          "volatility_score": 8.5
        }
      ]
    }
  }
}
```

### 4. 数据质量接口

#### 4.1 数据质量概览
- **端点**: `GET /quality/overview`
- **描述**: 获取数据质量概览信息

**响应示例**:
```json
{
  "success": true,
  "data": {
    "overall_score": 86.8,
    "overall_grade": "B",
    "last_updated": "2025-11-05T17:30:00Z",
    "component_scores": {
      "customer_data": {
        "score": 92.7,
        "grade": "A",
        "completeness": 100.0,
        "accuracy": 85.0,
        "issues": [
          {
            "type": "invalid_email",
            "count": 4,
            "severity": "medium"
          }
        ]
      },
      "drawing_data": {
        "score": 95.2,
        "grade": "A",
        "completeness": 99.5,
        "classification_rate": 9.6,
        "issues": []
      },
      "factory_data": {
        "score": 88.5,
        "grade": "B",
        "completeness": 100.0,
        "issues": []
      }
    },
    "recommendations": [
      "修复4个无效邮箱格式",
      "提高图纸分类覆盖率",
      "定期验证数据完整性"
    ]
  }
}
```

#### 4.2 数据质量详情
- **端点**: `GET /quality/details`
- **描述**: 获取详细的数据质量信息

**查询参数**:
- `component`: 组件 (customer/drawing/factory/all)
- `issue_type`: 问题类型过滤

**响应示例**:
```json
{
  "success": true,
  "data": {
    "customer_issues": [
      {
        "company_name": "巴林客户",
        "contact_email": "contact@巴林客户.com",
        "issue_type": "invalid_email",
        "severity": "medium",
        "suggested_fix": "更新为有效邮箱格式"
      }
    ],
    "drawing_issues": [
      {
        "drawing_id": 123,
        "drawing_name": "未知文件",
        "issue_type": "missing_customer",
        "severity": "high",
        "suggested_fix": "关联到正确客户"
      }
    ],
    "quality_metrics": {
      "email_completeness": 100.0,
      "classification_coverage": 9.6,
      "relationship_integrity": 98.5
    }
  }
}
```

### 5. 搜索接口

#### 5.1 全局搜索
- **端点**: `GET /search`
- **描述**: 全局搜索接口

**查询参数**:
- `q`: 搜索关键词 (必需)
- `type`: 搜索类型 (customer/drawing/factory/all, 默认all)
- `limit`: 结果数量限制 (默认20)
- `offset`: 偏移量 (默认0)

**响应示例**:
```json
{
  "success": true,
  "data": {
    "query": "螺栓",
    "total_results": 35,
    "results": [
      {
        "type": "drawing",
        "id": 123,
        "title": "六角螺栓图纸",
        "product_category": "紧固件-螺栓螺钉",
        "customer_name": "ABC Company",
        "highlight": "六角<mark>螺栓</mark>图纸",
        "relevance_score": 0.95
      },
      {
        "type": "customer",
        "id": 456,
        "title": "螺栓制造有限公司",
        "contact_email": "info@bolt.com",
        "highlight": "<mark>螺栓</mark>制造有限公司",
        "relevance_score": 0.88
      }
    ],
    "facets": {
      "types": {
        "drawing": 30,
        "customer": 3,
        "factory": 2
      },
      "categories": {
        "紧固件-螺栓螺钉": 25,
        "紧固件-螺母": 5
      }
    }
  }
}
```

### 6. 仪表板数据接口

#### 6.1 仪表板概览
- **端点**: `GET /dashboard/overview`
- **描述**: 获取仪表板首页数据

**响应示例**:
```json
{
  "success": true,
  "data": {
    "kpi": {
      "total_customers": 11,
      "total_drawings": 408,
      "total_factories": 2,
      "classification_rate": 9.6,
      "data_quality_score": 86.8
    },
    "recent_activity": [
      {
        "type": "classification",
        "message": "完成25个图纸分类",
        "timestamp": "2025-11-05T17:30:00Z"
      },
      {
        "type": "analysis",
        "message": "生成工厂报价分析报告",
        "timestamp": "2025-11-05T17:25:00Z"
      }
    ],
    "charts": {
      "drawing_category_distribution": {
        "data": [
          {"category": "紧固件-螺栓螺钉", "value": 31},
          {"category": "未分类", "value": 369}
        ]
      },
      "monthly_trends": {
        "data": [
          {"month": "2025-10", "quotes": 15, "drawings": 25}
        ]
      }
    },
    "alerts": [
      {
        "level": "warning",
        "message": "发现1个价格异常",
        "timestamp": "2025-11-05T17:00:00Z"
      }
    ]
  }
}
```

#### 6.2 实时更新
- **端点**: `GET /dashboard/realtime`
- **描述**: 获取实时更新的数据 (WebSocket)

**WebSocket 消息格式**:
```json
{
  "type": "data_update",
  "channel": "statistics",
  "data": {
    "component": "drawing_classification",
    "value": 42,
    "timestamp": "2025-11-05T17:35:00Z"
  }
}
```

## 错误代码

| 错误代码 | HTTP状态码 | 描述 |
|---------|-----------|------|
| VALIDATION_ERROR | 400 | 请求参数验证失败 |
| UNAUTHORIZED | 401 | 认证失败 |
| FORBIDDEN | 403 | 权限不足 |
| NOT_FOUND | 404 | 资源不存在 |
| RATE_LIMIT_EXCEEDED | 429 | 请求频率超限 |
| INTERNAL_ERROR | 500 | 服务器内部错误 |
| DATABASE_ERROR | 500 | 数据库错误 |
| DATA_NOT_AVAILABLE | 404 | 数据不可用 |

## 分页

支持分页的接口使用以下参数：
- `page`: 页码 (默认1)
- `limit`: 每页数量 (默认20，最大100)
- `sort`: 排序字段
- `order`: 排序方向 (asc/desc, 默认desc)

**分页响应格式**:
```json
{
  "success": true,
  "data": {
    "items": [ ... ],
    "pagination": {
      "page": 1,
      "limit": 20,
      "total": 156,
      "total_pages": 8,
      "has_next": true,
      "has_prev": false
    }
  }
}
```

## 缓存策略

- **统计数据**: 缓存5分钟
- **分析数据**: 缓存1小时
- **实时数据**: 不缓存

## 版本控制

API版本通过URL路径控制：
- v1: 当前版本
- v2: 未来版本 (向后兼容)

## 安全考虑

1. **认证**: 所有API需要有效的API Key
2. **授权**: 基于角色的访问控制
3. **限流**: 每分钟最多100个请求
4. **日志**: 记录所有API访问日志
5. **HTTPS**: 生产环境强制使用HTTPS

## 开发工具

### API 测试
- **Swagger UI**: `/api/docs`
- **Postman Collection**: 提供API测试集合
- **示例代码**: Python/JavaScript/React 示例

### 监控
- **健康检查**: `/api/health`
- **指标监控**: `/api/metrics`
- **日志查看**: `/api/logs`

## 前端集成示例

### React + Axios
```javascript
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
  headers: {
    'X-API-Key': process.env.REACT_APP_API_KEY
  }
});

// 获取图纸分类统计
const getDrawingCategoryStats = async (params) => {
  const response = await api.get('/statistics/drawings/by_category', { params });
  return response.data;
};
```

### Vue.js + Fetch
```javascript
const API_BASE = 'http://localhost:8000/api/v1';

const getDrawingCategoryStats = async (params) => {
  const url = new URL(`${API_BASE}/statistics/drawings/by_category`);
  Object.keys(params).forEach(key => url.searchParams.append(key, params[key]));

  const response = await fetch(url, {
    headers: {
      'X-API-Key': process.env.VUE_APP_API_KEY
    }
  });

  return response.json();
};
```

---

**文档版本**: 1.0
**创建日期**: 2025-11-05
**更新周期**: 按需更新
**维护者**: API 开发团队