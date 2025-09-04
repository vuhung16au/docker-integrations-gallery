#!/bin/bash

# Heart Disease Prediction Service - Endpoint Testing Script
# This script tests all available endpoints of the Flask service

# Configuration
BASE_URL="http://localhost:5000"
SERVICE_NAME="Heart Disease Prediction Service"
TIMEOUT=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test data for predictions (sample patient data)
SAMPLE_DATA='{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}'

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    
    case $status in
        "SUCCESS")
            echo -e "${GREEN}✓ SUCCESS${NC}: $message"
            ;;
        "FAILED")
            echo -e "${RED}✗ FAILED${NC}: $message"
            ;;
        "WARNING")
            echo -e "${YELLOW}⚠ WARNING${NC}: $message"
            ;;
        "INFO")
            echo -e "${BLUE}ℹ INFO${NC}: $message"
            ;;
    esac
}

# Function to test an endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    local description=$4
    
    echo -e "\n${BLUE}Testing: $description${NC}"
    echo "Endpoint: $method $endpoint"
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT "$BASE_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$BASE_URL$endpoint")
    fi
    
    # Extract response body and status code
    http_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" -eq 200 ]; then
        print_status "SUCCESS" "HTTP $http_code - Endpoint working correctly"
        # Truncate response if too long
        if [ ${#response_body} -gt 200 ]; then
            echo "Response: ${response_body:0:200}..."
        else
            echo "Response: $response_body"
        fi
    else
        print_status "FAILED" "HTTP $http_code - Endpoint failed"
        if [ -n "$response_body" ]; then
            echo "Response: $response_body"
        fi
    fi
}

# Function to test prediction endpoints
test_prediction_endpoint() {
    local model_name=$1
    local endpoint="/predict/$model_name"
    local description="Prediction using $model_name"
    
    test_endpoint "POST" "$endpoint" "$SAMPLE_DATA" "$description"
}

# Function to check if service is running
check_service_status() {
    echo -e "${BLUE}Checking if $SERVICE_NAME is running...${NC}"
    
    if curl -s --max-time 5 "$BASE_URL/health" > /dev/null 2>&1; then
        print_status "SUCCESS" "Service is running and accessible"
        return 0
    else
        print_status "FAILED" "Service is not accessible at $BASE_URL"
        print_status "INFO" "Make sure the Docker container is running:"
        echo "  docker-compose up -d"
        echo "  or"
        echo "  docker run -p 5000:5000 heart-disease-prediction"
        return 1
    fi
}

# Function to display test summary
display_summary() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}        TEST SUMMARY${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}✓ Health & Status Endpoints${NC}"
    echo "  - /health - Service health check"
    echo "  - /status - Model status information"
    echo "  - /sample - Dataset sample data"
    echo ""
    echo -e "${GREEN}✓ Scikit-learn Model Endpoints${NC}"
    echo "  - /predict/logistic_regression - Logistic Regression predictions"
    echo "  - /predict/random_forest - Random Forest predictions"
    echo "  - /predict/svm - Support Vector Machine predictions"
    echo "  - /predict/decision_tree - Decision Tree predictions"
    echo ""
    echo -e "${GREEN}✓ Gradient Boosting Model Endpoints${NC}"
    echo "  - /predict/catboost - CatBoost predictions"
    echo "  - /predict/lightgbm - LightGBM predictions"
    echo "  - /predict/xgboost - XGBoost predictions"
    echo ""
    echo -e "${GREEN}✓ Multi-model Endpoint${NC}"
    echo "  - /predict/all - Predictions from all available models"
    echo ""
    echo -e "${YELLOW}Note:${NC} Prediction endpoints will return errors until models are trained."
    echo "Run the training pipeline first: python 80-run_pipeline.py"
}

# Main execution
main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  $SERVICE_NAME - Endpoint Testing${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "Base URL: $BASE_URL"
    echo "Timeout: ${TIMEOUT}s per request"
    echo ""
    
    # Check if service is running
    if ! check_service_status; then
        exit 1
    fi
    
    echo -e "\n${BLUE}Starting endpoint tests...${NC}"
    
    # Test Health & Status endpoints
    echo -e "\n${YELLOW}=== Health & Status Endpoints ===${NC}"
    test_endpoint "GET" "/health" "" "Health Check"
    test_endpoint "GET" "/status" "" "Model Status"
    test_endpoint "GET" "/sample" "" "Dataset Sample"
    
    # Test Scikit-learn Model endpoints
    echo -e "\n${YELLOW}=== Scikit-learn Model Endpoints ===${NC}"
    test_prediction_endpoint "logistic_regression"
    test_prediction_endpoint "random_forest"
    test_prediction_endpoint "svm"
    test_prediction_endpoint "decision_tree"
    
    # Test Gradient Boosting Model endpoints
    echo -e "\n${YELLOW}=== Gradient Boosting Model Endpoints ===${NC}"
    test_prediction_endpoint "catboost"
    test_prediction_endpoint "lightgbm"
    test_prediction_endpoint "xgboost"
    
    # Test Multi-model endpoint
    echo -e "\n${YELLOW}=== Multi-model Endpoint ===${NC}"
    test_endpoint "POST" "/predict/all" "$SAMPLE_DATA" "Predictions from all models"
    
    # Display summary
    display_summary
    
    echo -e "\n${BLUE}Endpoint testing completed!${NC}"
}

# Check if curl is available
if ! command -v curl &> /dev/null; then
    echo -e "${RED}Error: curl is not installed${NC}"
    echo "Please install curl to run this script:"
    echo "  macOS: brew install curl"
    echo "  Ubuntu/Debian: sudo apt-get install curl"
    echo "  CentOS/RHEL: sudo yum install curl"
    exit 1
fi

# Run main function
main "$@"
