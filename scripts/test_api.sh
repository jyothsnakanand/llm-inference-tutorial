#!/bin/bash
# Script to test the LLM Inference API

set -e

BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "Testing LLM Inference API at $BASE_URL"
echo "========================================"

# Test 1: Health Check
echo -e "\n1. Testing health endpoint..."
curl -s "$BASE_URL/health" | jq '.'

# Test 2: Root endpoint
echo -e "\n2. Testing root endpoint..."
curl -s "$BASE_URL/" | jq '.'

# Test 3: Generate text with default parameters
echo -e "\n3. Testing text generation (default parameters)..."
curl -s -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time"
  }' | jq '.'

# Test 4: Generate text with custom parameters
echo -e "\n4. Testing text generation (custom parameters)..."
curl -s -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of artificial intelligence is",
    "max_length": 80,
    "temperature": 0.8,
    "top_p": 0.95,
    "num_return_sequences": 2
  }' | jq '.'

# Test 5: Invalid request (empty prompt)
echo -e "\n5. Testing validation (should fail)..."
curl -s -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": ""
  }' | jq '.'

# Test 6: Metrics endpoint
echo -e "\n6. Testing metrics endpoint..."
curl -s "$BASE_URL/metrics" | head -n 20

echo -e "\n========================================"
echo "API tests completed!"
