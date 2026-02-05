#!/bin/bash
# Run All InteractComp Integration Tests
# This script runs all tests in sequence without requiring GPU

set -e  # Exit on error

echo "========================================"
echo "InteractComp Integration Test Suite"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run a test
run_test() {
    local test_name=$1
    local test_script=$2

    echo ""
    echo "========================================"
    echo "Running: $test_name"
    echo "========================================"

    if python "$test_script"; then
        echo -e "${GREEN}✓ $test_name PASSED${NC}"
        return 0
    else
        echo -e "${RED}✗ $test_name FAILED${NC}"
        return 1
    fi
}

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0

# Test 1: Config and Dry Run (fastest, most important)
((TOTAL_TESTS++))
if run_test "Configuration & Dry Run" "test_config_dryrun.py"; then
    ((PASSED_TESTS++))
fi

# Test 2: Data Flow (medium speed, no API needed)
((TOTAL_TESTS++))
if run_test "Data Flow & Format" "test_data_flow.py"; then
    ((PASSED_TESTS++))
fi

# Test 3: Environment Integration (slowest, needs API)
# Check if API keys are set
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo -e "${YELLOW}⚠ Skipping Environment Integration test (OPENAI_API_KEY not set)${NC}"
    echo "  To run this test, set the following environment variables:"
    echo "    export OPENAI_API_KEY=\"your_key\""
    echo "    export OPENAI_BASE_URL=\"\""
    echo "    export MULTITURN_MODEL_NAME=\"gpt-4o-mini-2024-07-18\""
    echo "    export SERPER_API_KEY=\"your_serper_key\""
else
    ((TOTAL_TESTS++))
    if run_test "Environment Integration" "test_interactcomp_env.py"; then
        ((PASSED_TESTS++))
    fi
fi

# Summary
echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Total tests run: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $((TOTAL_TESTS - PASSED_TESTS))"

if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "Your InteractComp integration is ready for training!"
    echo "Next steps:"
    echo "  1. Wait for GPU resources to become available"
    echo "  2. Run: bash examples/interactcomp/train.sh"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo ""
    echo "Please fix the failing tests before running training."
    echo "See TESTING_GUIDE.md for troubleshooting tips."
    exit 1
fi
