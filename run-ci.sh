#!/bin/bash
set -e

echo "================================"
echo "Running CI checks"
echo "================================"

# Run linting
echo "Running linting..."
if command -v flake8 &> /dev/null; then
    flake8 src/
else
    echo "flake8 not found, skipping linting"
fi

# Run type checking
echo "Running type checking..."
if command -v mypy &> /dev/null; then
    mypy src/
else
    echo "mypy not found, skipping type checking"
fi

# Run tests
echo "Running tests..."
if command -v pytest &> /dev/null; then
    python -m pytest -xvs
else
    echo "pytest not found, skipping tests"
fi

echo "CI checks completed!"
exit 0
