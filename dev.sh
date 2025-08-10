#!/bin/bash

# Crawlerr Development Server Manager
# Safely manages MCP server and Docker services for development

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MCP_SERVER_NAME="crawlerr-mcp-server"
PIDFILE="./.crawlerr-dev.pid"
LOGFILE="./logs/dev-server.log"

# Ensure logs directory exists
mkdir -p logs

echo -e "${BLUE}🚀 Crawlerr Development Server Manager${NC}"
echo "=================================================="

# Function to print colored output
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if our MCP server is running
check_mcp_server() {
    if [ -f "$PIDFILE" ]; then
        local pid=$(cat "$PIDFILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            # Check if it's actually our server process
            if ps -p "$pid" -o command= | grep -q "crawlerr"; then
                return 0  # Server is running
            else
                # PID file exists but process is not our server, clean it up
                rm -f "$PIDFILE"
                return 1
            fi
        else
            # PID file exists but process is dead, clean it up
            rm -f "$PIDFILE"
            return 1
        fi
    fi
    return 1  # Not running
}

# Function to stop our MCP server safely
stop_mcp_server() {
    if check_mcp_server; then
        local pid=$(cat "$PIDFILE")
        log_info "Stopping existing Crawlerr MCP server (PID: $pid)..."
        
        # Send SIGTERM first (graceful shutdown)
        if kill "$pid" 2>/dev/null; then
            # Wait up to 10 seconds for graceful shutdown
            local count=0
            while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            # If still running, force kill
            if ps -p "$pid" > /dev/null 2>&1; then
                log_warning "Graceful shutdown timed out, force killing..."
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
        
        # Clean up PID file
        rm -f "$PIDFILE"
        log_success "Previous server stopped"
    else
        log_info "No existing Crawlerr MCP server found"
    fi
}

# Function to check Docker Compose services
check_docker_services() {
    log_info "Checking Docker Compose services..."
    
    # Check if docker-compose.yml exists
    if [ ! -f "docker-compose.yml" ]; then
        log_error "docker-compose.yml not found in current directory"
        exit 1
    fi
    
    # Check if services are running
    local qdrant_status=$(docker compose ps -q qdrant 2>/dev/null | wc -l)
    local tei_status=$(docker compose ps -q text-embeddings-inference 2>/dev/null | wc -l)
    
    if [ "$qdrant_status" -eq 0 ] || [ "$tei_status" -eq 0 ]; then
        log_warning "Docker services not running, starting them..."
        docker compose up -d
        
        # Wait for services to be healthy
        log_info "Waiting for services to be ready..."
        local max_attempts=60
        local attempt=0
        
        while [ $attempt -lt $max_attempts ]; do
            local qdrant_health=$(docker compose ps qdrant --format json 2>/dev/null | grep -o '"Health":"[^"]*"' | cut -d'"' -f4)
            local tei_health=$(docker compose ps text-embeddings-inference --format json 2>/dev/null | grep -o '"Health":"[^"]*"' | cut -d'"' -f4)
            
            if [[ "$qdrant_health" == "healthy" || "$qdrant_health" == "" ]] && [[ "$tei_health" == "healthy" ]]; then
                log_success "All services are healthy"
                break
            fi
            
            echo -n "."
            sleep 2
            attempt=$((attempt + 1))
        done
        
        if [ $attempt -eq $max_attempts ]; then
            log_error "Services did not become healthy within 2 minutes"
            log_info "Current service status:"
            docker compose ps
            exit 1
        fi
    else
        log_success "Docker services are already running"
    fi
    
    # Verify service connectivity
    log_info "Verifying service connectivity..."
    
    # Test Qdrant
    if curl -sf http://localhost:6333/ >/dev/null; then
        log_success "Qdrant is accessible at http://localhost:6333"
    else
        log_error "Qdrant is not accessible at http://localhost:6333"
        exit 1
    fi
    
    # Test TEI
    if curl -sf http://localhost:8080/info >/dev/null; then
        log_success "TEI is accessible at http://localhost:8080"
    else
        log_error "TEI is not accessible at http://localhost:8080"
        exit 1
    fi
}

# Function to start the MCP server
start_mcp_server() {
    log_info "Starting Crawlerr MCP server..."
    
    # Check if virtual environment exists
    if [ ! -d ".venv" ]; then
        log_error "Virtual environment not found. Run 'uv sync' first."
        exit 1
    fi
    
    # Check if server file exists
    if [ ! -f "crawlerr/server.py" ]; then
        log_error "Server file 'crawlerr/server.py' not found"
        exit 1
    fi
    
    # Start the server in background and capture PID
    log_info "Starting Crawlerr MCP server directly..."
    
    # Run the server directly with Python to respect .env configuration
    nohup uv run python -m crawlerr.server \
        > "$LOGFILE" 2>&1 &
    
    local server_pid=$!
    
    # Save PID to file
    echo "$server_pid" > "$PIDFILE"
    
    # Wait a moment to check if server started successfully
    sleep 3
    
    if ps -p "$server_pid" > /dev/null 2>&1; then
        log_success "MCP server started successfully (PID: $server_pid)"
        log_info "Server logs: $LOGFILE"
        log_info "MCP Inspector UI should be available shortly..."
        log_info ""
        log_info "Following server logs (press Ctrl+C to exit):"
        log_info "To stop the server, run: $0 stop"
        echo ""
        
        # Tail the logs by default
        tail -f "$LOGFILE"
    else
        log_error "Failed to start MCP server"
        log_info "Check the logs: $LOGFILE"
        rm -f "$PIDFILE"
        exit 1
    fi
}

# Function to show server status
show_status() {
    echo -e "${BLUE}📊 Server Status${NC}"
    echo "=================="
    
    # MCP Server status
    if check_mcp_server; then
        local pid=$(cat "$PIDFILE")
        log_success "MCP Server: Running (PID: $pid)"
    else
        log_info "MCP Server: Stopped"
    fi
    
    # Docker services status
    echo ""
    log_info "Docker Services:"
    docker compose ps 2>/dev/null || log_warning "Docker Compose not available"
    
    # Service health checks
    echo ""
    log_info "Service Health Checks:"
    
    if curl -sf http://localhost:6333/ >/dev/null 2>&1; then
        log_success "Qdrant: Healthy (http://localhost:6333)"
    else
        log_warning "Qdrant: Not accessible"
    fi
    
    if curl -sf http://localhost:8080/info >/dev/null 2>&1; then
        log_success "TEI: Healthy (http://localhost:8080)"
    else
        log_warning "TEI: Not accessible"
    fi
    
    # Show recent logs if server is running
    if check_mcp_server && [ -f "$LOGFILE" ]; then
        echo ""
        log_info "Recent server logs (last 10 lines):"
        tail -n 10 "$LOGFILE" 2>/dev/null || log_warning "No logs available"
    fi
}

# Function to show logs
show_logs() {
    if [ -f "$LOGFILE" ]; then
        log_info "Showing server logs (press Ctrl+C to exit):"
        tail -f "$LOGFILE"
    else
        log_warning "No log file found at $LOGFILE"
    fi
}

# Main script logic
case "${1:-start}" in
    "start")
        stop_mcp_server
        check_docker_services
        start_mcp_server
        ;;
    "stop")
        stop_mcp_server
        log_success "Development server stopped"
        ;;
    "restart")
        stop_mcp_server
        check_docker_services
        start_mcp_server
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs
        ;;
    "docker-up")
        log_info "Starting Docker services only..."
        docker compose up -d
        ;;
    "docker-down")
        log_info "Stopping Docker services only..."
        docker compose down
        ;;
    "help"|"-h"|"--help")
        echo "Crawlerr Development Server Manager"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start       Start the development server (default)"
        echo "  stop        Stop the development server"
        echo "  restart     Restart the development server"
        echo "  status      Show server and service status"
        echo "  logs        Show and follow server logs"
        echo "  docker-up   Start Docker services only"
        echo "  docker-down Stop Docker services only"
        echo "  help        Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0              # Start the server"
        echo "  $0 start        # Start the server"
        echo "  $0 stop         # Stop the server"
        echo "  $0 status       # Check status"
        echo "  $0 logs         # Follow logs"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac