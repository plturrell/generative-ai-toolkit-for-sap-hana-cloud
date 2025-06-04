#!/bin/bash
set -e

# Test script for NVIDIA deployment on Brev lab instances
# Usage: ./test-brevlab.sh [username] [ssh_key_path]

# Default values
HOST="jupyter0-ipzl7zn0p.brevlab.com"
USERNAME=${1:-"ubuntu"}
SSH_KEY=${2:-"$HOME/.ssh/id_rsa"}

echo "==========================================="
echo "NVIDIA T4 Deployment Tester for Brev Lab"
echo "==========================================="
echo "Host: $HOST"
echo "Username: $USERNAME"
echo "SSH Key: $SSH_KEY"

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
  echo "Error: SSH key not found at $SSH_KEY"
  echo "Please provide a valid SSH key path as the second argument"
  exit 1
fi

# Verify SSH connectivity
echo "Verifying SSH connectivity..."
if ! ssh -i "$SSH_KEY" -o ConnectTimeout=5 -o BatchMode=yes "$USERNAME@$HOST" "echo SSH connection successful"; then
  echo "Error: Unable to connect to $HOST via SSH"
  exit 1
fi

# Execute test script
echo "Running deployment tests..."
./test-deployment.py --host "$HOST" --username "$USERNAME" --ssh-key "$SSH_KEY" --output "test-results-$(date +%Y%m%d-%H%M%S).json"

echo "Test completed!"