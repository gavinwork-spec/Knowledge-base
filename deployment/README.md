# Manufacturing Knowledge Base - Cloud Native Deployment
# ===============================================
# This repository contains the complete cloud-native deployment setup
# for the Manufacturing Knowledge Base System including:
# - Multi-service API containers
# - Kubernetes manifests
# - Helm charts
# - Monitoring and observability
# - Security and compliance configurations

## Quick Start

### Prerequisites
- Kubernetes cluster (v1.25+)
- Helm 3.0+
- kubectl configured
- Docker registry access

### Deploy with Helm
```bash
# Add the repository
helm repo add knowledge-base https://github.com/gavinwork-spec/Knowledge-base/charts
helm repo update

# Install the complete system
helm install knowledge-base knowledge-base/knowledge-base \
  --namespace knowledge-base \
  --create-namespace \
  --set ingress.enabled=true \
  --set monitoring.enabled=true

# Or install from local source
helm install knowledge-base ./deployment/helm/knowledge-base \
  --namespace knowledge-base \
  --create-namespace
```

### Deploy with Kubernetes manifests
```bash
# Deploy the core system
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/kubernetes/
```

## Architecture Overview

```
Internet/Ingress
       ↓
┌─────────────────────────────────────────────────────────┐
│                   Load Balancer                         │
├─────────────────────────────────────────────────────────┤
│                      Ingress                            │
├─────────────────────────────────────────────────────────┤
│  Frontend (React)  │  API Gateway  │  Observability     │
├────────────────────┼───────────────┼─────────────────────┤
│                    │               │                     │
│  ┌────────────────┼───────────────┼─────────────────────┤
│  │ API Services                                         │
│  │ ├── Knowledge API (8001)                            │
│  │ ├── Chat Interface API (8002)                       │
│  │ └── Reminders API (8000)                            │
│  └─────────────────────────────────────────────────────┤
│                    │               │                     │
├────────────────────┼───────────────┼─────────────────────┤
│  Persistent Storage  │   Monitoring   │    Security         │
├─────────────────────┼────────────────┼─────────────────────┤
│  - SQLite Databases │  - Prometheus  │  - Authentication   │
│  - File Storage     │  - Grafana     │  - Authorization    │
│  - Backups          │  - Jaeger      │  - GDPR Compliance  │
└─────────────────────┴────────────────┴─────────────────────┘
```

## Components

### API Services
- **Knowledge API** (`:8001`): Knowledge base search and retrieval
- **Chat Interface API** (`:8002`): AI-powered chat interface
- **Reminders API** (`:8000`): Reminder and notification management

### Frontend
- **React Application**: Modern web interface with TypeScript
- **Responsive Design**: Mobile and desktop optimized
- **Real-time Updates**: WebSocket integration

### Observability Stack
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Jaeger**: Distributed tracing
- **Loki**: Log aggregation

### Storage
- **SQLite**: Primary database with persistent storage
- **Redis**: Caching and session management
- **MinIO**: Object storage for backups and files

## Configuration

### Environment Variables
Key environment variables for configuration:

```bash
# Database Configuration
DATABASE_URL=sqlite:///knowledge_base.db
DATABASE_BACKUP_ENABLED=true
DATABASE_BACKUP_SCHEDULE="0 2 * * *"

# API Configuration
API_RATE_LIMIT=1000
API_TIMEOUT=30
API_RETRIES=3

# Security
JWT_SECRET=your-secret-key
ENCRYPTION_KEY=your-encryption-key
GDPR_COMPLIANCE=true

# Monitoring
PROMETHEUS_ENABLED=true
JAEGER_ENABLED=true
LOG_LEVEL=INFO
```

### Scaling Configuration
```yaml
# Horizontal Pod Autoscaler
resources:
  limits:
    cpu: 500m
    memory: 1Gi
  requests:
    cpu: 250m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

## Security & Compliance

### Data Protection
- **Encryption at Rest**: All data encrypted with AES-256
- **Encryption in Transit**: TLS 1.3 for all communications
- **Key Management**: Kubernetes secrets with rotation

### GDPR Compliance
- **Data Minimization**: Only collect necessary data
- **Right to Erasure**: Automated data deletion capabilities
- **Data Portability**: Export functionality for user data
- **Consent Management**: Granular consent controls

### Access Control
- **Authentication**: JWT-based with refresh tokens
- **Authorization**: RBAC with granular permissions
- **Audit Logging**: Comprehensive access logging
- **Session Management**: Secure session handling

## Monitoring & Alerting

### Key Metrics
- **API Response Time**: Average latency per endpoint
- **Error Rate**: 4xx/5xx error percentages
- **Database Performance**: Query execution times
- **Resource Usage**: CPU, memory, and storage metrics
- **Business Metrics**: User engagement, satisfaction scores

### Alerting Rules
- **High Error Rate**: Alert if error rate > 5%
- **Response Time**: Alert if P95 latency > 2s
- **Resource Usage**: Alert if CPU > 80% or Memory > 85%
- **Database Issues**: Alert on connection failures or slow queries

## Backup & Disaster Recovery

### Automated Backups
- **Database Backups**: Daily full backups with 30-day retention
- **Incremental Backups**: Hourly incremental backups
- **Cross-region Replication**: Backups replicated to DR region
- **Backup Verification**: Automated backup restoration testing

### Disaster Recovery
- **RTO**: 4 hours (Recovery Time Objective)
- **RPO**: 1 hour (Recovery Point Objective)
- **Multi-region Deployment**: Active-passive setup
- **Failover Automation**: Automatic failover on critical failures

## Development

### Local Development
```bash
# Build and run locally
docker-compose -f deployment/docker/docker-compose.yml up -d

# Run tests
pytest tests/
pytest tests/integration/

# Build Docker images
docker build -f deployment/docker/Dockerfile.api -t knowledge-base-api .
docker build -f deployment/docker/Dockerfile.frontend -t knowledge-base-frontend .
```

### CI/CD Pipeline
- **GitHub Actions**: Automated testing and deployment
- **Image Scanning**: Security vulnerability scanning
- **Helm Testing**: Automated chart testing
- **Integration Tests**: End-to-end testing pipeline

## Support

### Documentation
- [API Documentation](./docs/api/)
- [Kubernetes Deployment Guide](./docs/kubernetes-deployment.md)
- [Monitoring Guide](./docs/monitoring.md)
- [Security Compliance](./docs/security.md)

### Troubleshooting
- [Common Issues](./docs/troubleshooting.md)
- [Performance Tuning](./docs/performance.md)
- [Backup Recovery](./docs/backup-recovery.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for our code of conduct and the process for submitting pull requests.

---

**Manufacturing Knowledge Base System** - Cloud Native Deployment
Built for scalability, security, and manufacturing excellence.