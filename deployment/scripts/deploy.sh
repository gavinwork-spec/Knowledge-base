#!/bin/bash
# Manufacturing Knowledge Base - Cloud Native Deployment Script
# Complete automated deployment with validation and rollback

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="knowledge-base"
CHART_PATH="deployment/helm/knowledge-base"
VALUES_FILE="deployment/helm/knowledge-base/values.yaml"
RELEASE_NAME="knowledge-base"
TIMEOUT=600

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check for required tools
    local tools=("helm" "kubectl" "docker")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done

    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check Helm version
    helm_version=$(helm version --short | grep -o 'v[0-9]\+\.[0-9]\+\.[0-9]\+')
    log_info "Helm version: $helm_version"

    # Check Kubernetes version
    k8s_version=$(kubectl version --short | grep 'Server Version' | grep -o 'v[0-9]\+\.[0-9]\+')
    log_info "Kubernetes version: $k8s_version"

    log_success "Prerequisites check completed"
}

# Function to create namespace
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"

    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace "$NAMESPACE"
        log_success "Namespace $NAMESPACE created"
    fi
}

# Function to add Helm repositories
add_helm_repos() {
    log_info "Adding Helm repositories..."

    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update

    log_success "Helm repositories added and updated"
}

# Function to create secrets
create_secrets() {
    log_info "Creating secrets..."

    # Check if secrets file exists
    local secrets_file="deployment/kubernetes/secrets.yaml"
    if [[ -f "$secrets_file" ]]; then
        kubectl apply -f "$secrets_file"
        log_success "Secrets created from $secrets_file"
    else
        log_warning "Secrets file not found: $secrets_file"
        log_info "Creating sample secrets..."

        # Create JWT secret
        kubectl create secret generic knowledge-base-secrets \
            --from-literal=JWT_SECRET="$(openssl rand -base64 32)" \
            --from-literal=ENCRYPTION_KEY="$(openssl rand -hex 16)" \
            --dry-run=client -o yaml | kubectl apply -f -
    fi
}

# Function to validate Helm chart
validate_chart() {
    log_info "Validating Helm chart..."

    if [[ ! -d "$CHART_PATH" ]]; then
        log_error "Helm chart not found at $CHART_PATH"
        exit 1
    fi

    # Lint the chart
    helm lint "$CHART_PATH"
    log_success "Helm chart validation passed"
}

# Function to deploy with Helm
deploy_helm() {
    log_info "Deploying Knowledge Base with Helm..."

    local deploy_cmd="helm upgrade --install $RELEASE_NAME $CHART_PATH"

    # Add common flags
    deploy_cmd="$deploy_cmd --namespace $NAMESPACE"
    deploy_cmd="$deploy_cmd --values $VALUES_FILE"
    deploy_cmd="$deploy_cmd --timeout ${TIMEOUT}s"
    deploy_cmd="$deploy_cmd --wait"

    # Add create namespace flag if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        deploy_cmd="$deploy_cmd --create-namespace"
    fi

    # Execute deployment
    log_info "Running: $deploy_cmd"
    if eval "$deploy_cmd"; then
        log_success "Helm deployment completed successfully"
    else
        log_error "Helm deployment failed"
        exit 1
    fi
}

# Function to validate deployment
validate_deployment() {
    log_info "Validating deployment..."

    # Wait for pods to be ready
    local max_attempts=60
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        local ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/part-of=knowledge-base --field-selector=status.phase=Running --no-headers | wc -l)
        local total_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/part-of=knowledge-base --no-headers | wc -l)

        if [[ $ready_pods -eq $total_pods ]] && [[ $total_pods -gt 0 ]]; then
            log_success "All $total_pods pods are ready"
            break
        fi

        log_info "Waiting for pods to be ready... ($ready_pods/$total_pods ready, attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done

    if [[ $attempt -gt $max_attempts ]]; then
        log_error "Pods did not become ready within the timeout period"
        kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/part-of=knowledge-base
        exit 1
    fi

    # Check service status
    log_info "Checking service status..."
    kubectl get services -n "$NAMESPACE"

    # Check ingress status
    if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
        log_info "Ingress status:"
        kubectl get ingress -n "$NAMESPACE"
    fi
}

# Function to run health checks
run_health_checks() {
    log_info "Running health checks..."

    # Check frontend
    local frontend_url=$(kubectl get ingress -n "$NAMESPACE" -o jsonpath='{.items[?(@.metadata.name=="knowledge-base-ingress")].spec.rules[0].host}')
    if [[ -n "$frontend_url" ]]; then
        log_info "Checking frontend health at https://$frontend_url"
        if curl -f -s "https://$frontend_url/health" > /dev/null; then
            log_success "Frontend health check passed"
        else
            log_warning "Frontend health check failed"
        fi
    fi

    # Check API endpoints
    local services=("knowledge-api" "chat-api" "reminders-api")
    for service in "${services[@]}"; do
        local service_ip=$(kubectl get service "$service-service" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
        if [[ -n "$service_ip" ]]; then
            log_info "Checking $service health"
            if kubectl port-forward -n "$NAMESPACE" "service/$service-service" 8000:8000 &
            then
                sleep 5
                if curl -f -s "http://localhost:8000/health" > /dev/null; then
                    log_success "$service health check passed"
                else
                    log_warning "$service health check failed"
                fi
                pkill -f "kubectl port-forward.*$service" || true
            fi
        fi
    done
}

# Function to show deployment summary
show_deployment_summary() {
    log_info "Deployment Summary"
    echo "=================="

    echo "Namespace: $NAMESPACE"
    echo "Release: $RELEASE_NAME"
    echo ""

    echo "Services:"
    kubectl get services -n "$NAMESPACE"
    echo ""

    echo "Ingress:"
    kubectl get ingress -n "$NAMESPACE" 2>/dev/null || echo "No ingress configured"
    echo ""

    echo "Pods:"
    kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/part-of=knowledge-base
    echo ""

    echo "Access URLs:"
    local frontend_url=$(kubectl get ingress -n "$NAMESPACE" -o jsonpath='{.items[0].spec.rules[0].host}' 2>/dev/null)
    if [[ -n "$frontend_url" ]]; then
        echo "Frontend: https://$frontend_url"
        echo "API: https://api.$frontend_url"
        echo "Chat API: https://chat-api.$frontend_url"
        echo "Reminders API: https://reminders-api.$frontend_url"
    fi

    echo ""
    echo "To check logs:"
    echo "kubectl logs -n $NAMESPACE deployment/knowledge-api -f"
    echo ""
    echo "To scale services:"
    echo "kubectl scale deployment knowledge-api -n $NAMESPACE --replicas=3"
    echo ""
    echo "To upgrade:"
    echo "helm upgrade $RELEASE_NAME $CHART_PATH --namespace $NAMESPACE"
    echo ""
    echo "To rollback:"
    echo "helm rollback $RELEASE_NAME -n $NAMESPACE"
}

# Function to rollback on failure
rollback_deployment() {
    log_error "Deployment failed, initiating rollback..."

    if helm history -n "$NAMESPACE" "$RELEASE_NAME" | grep -q "deployed"; then
        helm rollback "$RELEASE_NAME" -n "$NAMESPACE"
        log_info "Rollback initiated"
    else
        log_warning "No previous deployment found for rollback"
    fi
}

# Function to cleanup on exit
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        rollback_deployment
    fi
    # Kill any background processes
    pkill -f "kubectl port-forward" || true
}

# Main function
main() {
    log_info "Starting Manufacturing Knowledge Base deployment"
    echo "================================================="

    # Set up cleanup trap
    trap cleanup EXIT

    # Run deployment steps
    check_prerequisites
    add_helm_repos
    create_namespace
    create_secrets
    validate_chart
    deploy_helm
    validate_deployment
    run_health_checks
    show_deployment_summary

    log_success "Deployment completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Configure your DNS to point to the load balancer IP"
    echo "2. Update your environment variables and secrets"
    echo "3. Monitor the deployment with the provided Grafana dashboard"
    echo "4. Set up automated backups and monitoring alerts"
}

# Script options
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --values)
            VALUES_FILE="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --namespace NAMESPACE    Kubernetes namespace (default: knowledge-base)"
            echo "  --values VALUES_FILE     Helm values file (default: deployment/helm/knowledge-base/values.yaml)"
            echo "  --timeout TIMEOUT        Deployment timeout in seconds (default: 600)"
            echo "  --dry-run               Show what would be deployed without actually deploying"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Handle dry run
if [[ "${DRY_RUN:-}" == "true" ]]; then
    log_info "DRY RUN: Showing what would be deployed"
    helm template "$RELEASE_NAME" "$CHART_PATH" \
        --namespace "$NAMESPACE" \
        --values "$VALUES_FILE"
    exit 0
fi

# Execute main function
main