#!/bin/bash

# Docker Integration Gallery Test Script
# Tests all Docker configurations in the project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Docker binary path
DOCKER="/usr/local/bin/docker"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to cleanup containers and images
cleanup() {
    local project_name=$1
    local container_name=$2
    
    print_status "Cleaning up $project_name..."
    
    # Stop and remove containers
    if [ -n "$container_name" ]; then
        $DOCKER stop "$container_name" 2>/dev/null || true
        $DOCKER rm "$container_name" 2>/dev/null || true
    fi
    
    # Remove images
    $DOCKER rmi "$project_name" 2>/dev/null || true
    
    # Clean up any dangling images
    $DOCKER image prune -f >/dev/null 2>&1 || true
}

# Function to test Dockerfile
test_dockerfile() {
    local project_dir=$1
    local project_name=$2
    local container_name=$3
    local port=$4
    
    print_status "Testing Dockerfile in $project_dir..."
    
    # Build the image
    print_status "Building image for $project_name..."
    if ! $DOCKER build -t "$project_name" "$project_dir" >/dev/null 2>&1; then
        print_error "Failed to build $project_name"
        return 1
    fi
    
    # Run the container
    print_status "Running container for $project_name..."
    if [ -n "$port" ]; then
        # Run with port mapping
        if ! $DOCKER run -d --name "$container_name" -p "$port:$port" "$project_name" >/dev/null 2>&1; then
            print_error "Failed to run $project_name container"
            cleanup "$project_name" "$container_name"
            return 1
        fi
    else
        # Run without port mapping
        if ! $DOCKER run -d --name "$container_name" "$project_name" >/dev/null 2>&1; then
            print_error "Failed to run $project_name container"
            cleanup "$project_name" "$container_name"
            return 1
        fi
    fi
    
    # Wait a moment for container to start
    sleep 3
    
    # Check if container ran successfully (either still running or exited with success)
    if $DOCKER ps | grep -q "$container_name"; then
        print_success "$project_name container is running"
    elif $DOCKER ps -a | grep "$container_name" | grep -q "Exited (0)"; then
        print_success "$project_name container completed successfully"
    else
        print_error "$project_name container failed to start or exited with error"
        $DOCKER logs "$container_name" 2>/dev/null || true
        cleanup "$project_name" "$container_name"
        return 1
    fi
    
    # Cleanup
    cleanup "$project_name" "$container_name"
    print_success "$project_name test completed successfully"
    return 0
}

# Function to test docker-compose
test_docker_compose() {
    local project_dir=$1
    local project_name=$2
    
    print_status "Testing docker-compose in $project_dir..."
    
    # Build and start services
    print_status "Building and starting services for $project_name..."
    if ! $DOCKER compose -f "$project_dir/docker-compose.yaml" up -d --build >/dev/null 2>&1; then
        print_error "Failed to start $project_name services"
        $DOCKER compose -f "$project_dir/docker-compose.yaml" logs 2>/dev/null || true
        $DOCKER compose -f "$project_dir/docker-compose.yaml" down 2>/dev/null || true
        return 1
    fi
    
    # Wait for services to be ready
    sleep 15
    
    # Check if services are running
    if $DOCKER compose -f "$project_dir/docker-compose.yaml" ps | grep -q "Up"; then
        print_success "$project_name services are running"
    else
        print_error "$project_name services failed to start"
        $DOCKER compose -f "$project_dir/docker-compose.yaml" logs 2>/dev/null || true
        $DOCKER compose -f "$project_dir/docker-compose.yaml" down 2>/dev/null || true
        return 1
    fi
    
    # Stop and cleanup
    print_status "Stopping and cleaning up $project_name services..."
    $DOCKER compose -f "$project_dir/docker-compose.yaml" down -v >/dev/null 2>&1 || true
    
    print_success "$project_name test completed successfully"
    return 0
}

# Main execution
main() {
    print_status "Starting Docker Integration Gallery Tests..."
    print_status "Docker binary: $DOCKER"
    
    # Check if Docker is available
    if ! command -v "$DOCKER" >/dev/null 2>&1; then
        print_error "Docker not found at $DOCKER"
        exit 1
    fi
    
    # Test results
    local passed=0
    local failed=0
    local total=0
    
    # Test configurations
    local tests=(
        "00.HelloDocker|dockerfile|hello-docker|"
        "01.Multi-Stage-Build|dockerfile|multi-stage-build|5000"
        "02.Multi-Container-Data-Pipelines|compose|multi-container-pipeline|"
        "03.Volume-Management-Data-Persistence|compose|volume-management|"
        "04.Security-Run-with-Non-Root-Users|dockerfile|security-non-root|"
        "05.Heart-Disease-Predictions|compose|heart-disease|5000"
        "06.Jupyter|dockerfile|jupyter|8888"
    )
    
    # Run tests
    for test in "${tests[@]}"; do
        IFS='|' read -r project_dir test_type project_name port <<< "$test"
        total=$((total + 1))
        
        print_status "=========================================="
        print_status "Testing: $project_dir ($test_type)"
        print_status "=========================================="
        
        if [ "$test_type" = "dockerfile" ]; then
            if test_dockerfile "$project_dir" "$project_name" "$project_name" "$port"; then
                passed=$((passed + 1))
            else
                failed=$((failed + 1))
            fi
        elif [ "$test_type" = "compose" ]; then
            if test_docker_compose "$project_dir" "$project_name"; then
                passed=$((passed + 1))
            else
                failed=$((failed + 1))
            fi
        fi
        
        print_status "=========================================="
        echo
    done
    
    # Final cleanup
    print_status "Performing final cleanup..."
    $DOCKER system prune -f >/dev/null 2>&1 || true
    
    # Summary
    print_status "=========================================="
    print_status "TEST SUMMARY"
    print_status "=========================================="
    print_status "Total tests: $total"
    print_success "Passed: $passed"
    if [ $failed -gt 0 ]; then
        print_error "Failed: $failed"
    else
        print_success "Failed: $failed"
    fi
    
    if [ $failed -eq 0 ]; then
        print_success "All tests passed! 🎉"
        exit 0
    else
        print_error "Some tests failed. Please check the output above."
        exit 1
    fi
}

# Run main function
main "$@"
