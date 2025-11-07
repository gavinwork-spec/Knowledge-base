-- PostgreSQL 初始化脚本
-- 为知识库微服务架构创建数据库结构

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";

-- 创建用户表
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    full_name TEXT,
    is_active BOOLEAN DEFAULT true,
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    preferences_json JSONB DEFAULT '{}',
    role VARCHAR(50) DEFAULT 'user',
    auth_provider VARCHAR(50) DEFAULT 'local'
);

-- 创建知识条目表
CREATE TABLE IF NOT EXISTS knowledge_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    description TEXT,
    entity_type TEXT NOT NULL,
    attributes_json JSONB,
    embedding_vector vector(384),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID REFERENCES users(id),
    updated_by UUID REFERENCES users(id),
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    file_path TEXT,
    file_hash TEXT,
    confidence_score DECIMAL(3,2) DEFAULT 0.8,
    source_service TEXT DEFAULT 'legacy',
    tags TEXT[],
    category VARCHAR(100),
    language VARCHAR(10) DEFAULT 'zh'
);

-- 创建向量索引表
CREATE TABLE IF NOT EXISTS vector_index (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_entry_id UUID REFERENCES knowledge_entries(id) ON DELETE CASCADE,
    vector vector(384),
    embedding_model TEXT DEFAULT 'all-MiniLM-L6-v2',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    model_version TEXT DEFAULT 'v1.0',
    dimensions INTEGER DEFAULT 384
);

-- 创建事件表
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(100) NOT NULL,
    source_service VARCHAR(100) NOT NULL,
    correlation_id UUID,
    causation_id UUID,
    user_id UUID REFERENCES users(id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    data JSONB,
    priority INTEGER DEFAULT 5,
    processed BOOLEAN DEFAULT FALSE,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT
);

-- 创建推荐表
CREATE TABLE IF NOT EXISTS recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    inquiry_id UUID,
    recommended_products JSONB,
    recommended_suppliers JSONB,
    recommended_price_range NUMRANGE,
    confidence_score DECIMAL(5,2),
    recommendation_type TEXT,
    recommendation_reason TEXT,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID REFERENCES knowledge_entries(id),
    algorithm_version TEXT DEFAULT 'v1.0',
    is_active BOOLEAN DEFAULT true,
    feedback_score INTEGER CHECK (feedback_score >= 1 AND feedback_score <= 5)
);

-- 创建聊天历史表
CREATE TABLE IF NOT EXISTS chat_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(100) NOT NULL,
    user_query TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    query_type VARCHAR(50),
    context_used JSONB,
    feedback_score INTEGER CHECK (feedback_score >= 1 AND feedback_score <= 5),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    session_started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_id UUID REFERENCES users(id),
    intent VARCHAR(100),
    entities_detected JSONB,
    response_time_ms INTEGER
);

-- 创建搜索历史表
CREATE TABLE IF NOT EXISTS search_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query TEXT NOT NULL,
    query_type VARCHAR(50),
    results_count INTEGER DEFAULT 0,
    search_time_ms INTEGER,
    clicked_result_id UUID,
    user_id UUID REFERENCES users(id),
    session_id VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    search_algorithm VARCHAR(50) DEFAULT 'hybrid',
    filters_applied JSONB,
    top_result_score DECIMAL(5,4)
);

-- 创建工作流状态表
CREATE TABLE IF NOT EXISTS workflow_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_type VARCHAR(100) NOT NULL,
    workflow_id VARCHAR(100) NOT NULL,
    current_step VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'running',
    data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    triggered_by_event_id UUID REFERENCES events(id)
);

-- 创建文档处理任务表
CREATE TABLE IF NOT EXISTS document_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path TEXT NOT NULL,
    file_hash TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    file_size BIGINT,
    mime_type VARCHAR(100),
    processing_results JSONB,
    created_by UUID REFERENCES users(id)
);

-- 创建系统配置表
CREATE TABLE IF NOT EXISTS system_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key VARCHAR(200) UNIQUE NOT NULL,
    value JSONB,
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_by UUID REFERENCES users(id),
    category VARCHAR(100),
    is_public BOOLEAN DEFAULT false
);

-- 创建API密钥表
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    permissions JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    usage_count INTEGER DEFAULT 0,
    rate_limit INTEGER DEFAULT 1000
);

-- 创建会话表
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT true
);

-- 创建审计日志表
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    session_id UUID REFERENCES user_sessions(id)
);

-- 创建通知表
CREATE TABLE IF NOT EXISTS notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    type VARCHAR(50) DEFAULT 'info',
    priority VARCHAR(20) DEFAULT 'normal',
    data JSONB,
    is_read BOOLEAN DEFAULT false,
    read_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    action_url TEXT,
    action_text TEXT
);

-- 创建文件存储表
CREATE TABLE IF NOT EXISTS file_storage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    mime_type VARCHAR(100),
    file_hash VARCHAR(64) NOT NULL,
    storage_provider VARCHAR(50) DEFAULT 'local',
    storage_path TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID REFERENCES users(id),
    is_public BOOLEAN DEFAULT false,
    download_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP WITH TIME ZONE
);

-- 创建标签表
CREATE TABLE IF NOT EXISTS tags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    color VARCHAR(7) DEFAULT '#007bff',
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID REFERENCES users(id),
    usage_count INTEGER DEFAULT 0
);

-- 创建知识条目标签关联表
CREATE TABLE IF NOT EXISTS knowledge_entry_tags (
    knowledge_entry_id UUID REFERENCES knowledge_entries(id) ON DELETE CASCADE,
    tag_id UUID REFERENCES tags(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (knowledge_entry_id, tag_id)
);

-- 创建触发器函数：更新时间戳
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 创建触发器
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_knowledge_entries_updated_at BEFORE UPDATE ON knowledge_entries FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_vector_index_updated_at BEFORE UPDATE ON vector_index FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_workflow_states_updated_at BEFORE UPDATE ON workflow_states FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_document_tasks_updated_at BEFORE UPDATE ON document_tasks FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_system_config_updated_at BEFORE UPDATE ON system_config FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_sessions_last_accessed_at BEFORE UPDATE ON user_sessions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_file_storage_last_accessed_at BEFORE UPDATE ON file_storage FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 创建触发器函数：更新标签使用计数
CREATE OR REPLACE FUNCTION update_tag_usage_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE tags SET usage_count = usage_count + 1 WHERE id = NEW.tag_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE tags SET usage_count = usage_count - 1 WHERE id = OLD.tag_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ language 'plpgsql';

-- 创建标签使用计数触发器
CREATE TRIGGER update_tag_count_on_insert AFTER INSERT ON knowledge_entry_tags FOR EACH ROW EXECUTE FUNCTION update_tag_usage_count();
CREATE TRIGGER update_tag_count_on_delete AFTER DELETE ON knowledge_entry_tags FOR EACH ROW EXECUTE FUNCTION update_tag_usage_count();

-- 插入默认系统配置
INSERT INTO system_config (key, value, description, category, is_public) VALUES
('app_name', '"知识库管理系统"', '应用名称', 'general', true),
('app_version', '"1.0.0"', '应用版本', 'general', true),
('max_upload_size', '104857600', '最大上传文件大小（字节）', 'upload', true),
('allowed_file_types', '["pdf","docx","xlsx","txt","csv","jpg","png","tiff"]', '允许的文件类型', 'upload', true),
('embedding_model', '"all-MiniLM-L6-v2"', '默认嵌入模型', 'ai', false),
('similarity_threshold', '0.7', '相似度阈值', 'search', false),
('max_search_results', '50', '最大搜索结果数', 'search', true),
('chat_session_timeout', '3600', '聊天会话超时时间（秒）', 'chat', false),
('rate_limit_requests', '100', 'API速率限制请求数', 'api', false),
('rate_limit_window', '60', 'API速率限制时间窗口（秒）', 'api', false)
ON CONFLICT (key) DO NOTHING;

-- 插入默认标签
INSERT INTO tags (name, color, description) VALUES
('紧固件', '#ff6b6b', '紧固件相关产品'),
('报价', '#4ecdc4', '报价和价格信息'),
('询盘', '#45b7d1', '客户询盘信息'),
('供应商', '#96ceb4', '供应商信息'),
('客户', '#ffeaa7', '客户信息'),
('规格', '#dfe6e9', '产品规格参数'),
('材料', '#74b9ff', '材料信息'),
('标准', '#a29bfe', '标准和规范')
ON CONFLICT (name) DO NOTHING;

-- 创建默认管理员用户（密码: admin123，生产环境需要修改）
INSERT INTO users (username, email, full_name, role, is_active) VALUES
('admin', 'admin@knowledge-base.com', '系统管理员', 'admin', true)
ON CONFLICT (username) DO NOTHING;

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_type ON knowledge_entries(entity_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_created ON knowledge_entries(created_at);
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_active ON knowledge_entries(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_source ON knowledge_entries(source_service);
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_tags ON knowledge_entries USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_category ON knowledge_entries(category);
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_language ON knowledge_entries(language);
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_attributes ON knowledge_entries USING GIN(attributes_json);

CREATE INDEX IF NOT EXISTS idx_vector_index_vector ON vector_index USING ivfflat (vector vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_vector_index_entry ON vector_index(knowledge_entry_id);
CREATE INDEX IF NOT EXISTS idx_vector_index_model ON vector_index(embedding_model);

CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_processed ON events(processed) WHERE processed = false;
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source_service);
CREATE INDEX IF NOT EXISTS idx_events_correlation ON events(correlation_id);
CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id);
CREATE INDEX IF NOT EXISTS idx_events_priority ON events(priority);
CREATE INDEX IF NOT EXISTS idx_events_data ON events USING GIN(data);

CREATE INDEX IF NOT EXISTS idx_recommendations_inquiry ON recommendations(inquiry_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_created ON recommendations(created_at);
CREATE INDEX IF NOT EXISTS idx_recommendations_type ON recommendations(recommendation_type);
CREATE INDEX IF NOT EXISTS idx_recommendations_active ON recommendations(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_recommendations_expires ON recommendations(expires_at) WHERE expires_at > NOW();

CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_created ON chat_history(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_history_query_type ON chat_history(query_type);
CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_intent ON chat_history(intent);
CREATE INDEX IF NOT EXISTS idx_chat_history_context ON chat_history USING GIN(context_used);

CREATE INDEX IF NOT EXISTS idx_search_history_query ON search_history(query);
CREATE INDEX IF NOT EXISTS idx_search_history_created ON search_history(created_at);
CREATE INDEX IF NOT EXISTS idx_search_history_session ON search_history(session_id);
CREATE INDEX IF NOT EXISTS idx_search_history_user ON search_history(user_id);
CREATE INDEX IF NOT EXISTS idx_search_history_type ON search_history(query_type);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);

CREATE INDEX IF NOT EXISTS idx_workflow_states_type ON workflow_states(workflow_type);
CREATE INDEX IF NOT EXISTS idx_workflow_states_status ON workflow_states(status);
CREATE INDEX IF NOT EXISTS idx_workflow_states_updated ON workflow_states(updated_at);
CREATE INDEX IF NOT EXISTS idx_workflow_states_workflow ON workflow_states(workflow_id);

CREATE INDEX IF NOT EXISTS idx_document_tasks_status ON document_tasks(status);
CREATE INDEX IF NOT EXISTS idx_document_tasks_hash ON document_tasks(file_hash);
CREATE INDEX IF NOT EXISTS idx_document_tasks_created ON document_tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_document_tasks_user ON document_tasks(created_by);

CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active) WHERE is_active = true;

CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires ON user_sessions(expires_at) WHERE expires_at > NOW();

CREATE INDEX IF NOT EXISTS idx_audit_logs_user ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id);

CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_read ON notifications(is_read) WHERE is_read = false;
CREATE INDEX IF NOT EXISTS idx_notifications_created ON notifications(created_at);
CREATE INDEX IF NOT EXISTS idx_notifications_type ON notifications(type);
CREATE INDEX IF NOT EXISTS idx_notifications_expires ON notifications(expires_at) WHERE expires_at > NOW();

CREATE INDEX IF NOT EXISTS idx_file_storage_hash ON file_storage(file_hash);
CREATE INDEX IF NOT EXISTS idx_file_storage_user ON file_storage(created_by);
CREATE INDEX IF NOT EXISTS idx_file_storage_public ON file_storage(is_public);
CREATE INDEX IF NOT EXISTS idx_file_storage_created ON file_storage(created_at);

CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
CREATE INDEX IF NOT EXISTS idx_tags_usage ON tags(usage_count);

COMMIT;