#!/bin/bash
# Deploy the Generative AI Toolkit to T4 GPU VM

set -e

# Default settings
T4_SERVER=${T4_SERVER:-"jupyter0-4ckg1m6x0.brevlab.com"}
SSH_USER=${SSH_USER:-"ubuntu"}
SSH_KEY=${SSH_KEY:-"~/.ssh/id_rsa"}
REMOTE_DIR=${REMOTE_DIR:-"/home/ubuntu/generative-ai-toolkit"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display help
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                 Show this help message and exit"
    echo "  -s, --server SERVER        T4 GPU server address (default: $T4_SERVER)"
    echo "  -u, --user USER            SSH username (default: $SSH_USER)"
    echo "  -k, --key KEY              SSH private key path (default: $SSH_KEY)"
    echo "  -d, --dir DIRECTORY        Remote directory (default: $REMOTE_DIR)"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--server)
            T4_SERVER="$2"
            shift 2
            ;;
        -u|--user)
            SSH_USER="$2"
            shift 2
            ;;
        -k|--key)
            SSH_KEY="$2"
            shift 2
            ;;
        -d|--dir)
            REMOTE_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Print script header
echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}  SAP HANA Cloud Generative AI Toolkit - T4 GPU Deployment${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo ""

# Check if we can connect to the server
echo -e "${BLUE}Checking connection to T4 GPU server: ${T4_SERVER}...${NC}"
if ! ssh -q -o BatchMode=yes -o ConnectTimeout=5 -i "$SSH_KEY" "$SSH_USER@$T4_SERVER" exit; then
    echo -e "${RED}Cannot connect to the T4 GPU server. Please check your SSH credentials and server address.${NC}"
    exit 1
else
    echo -e "${GREEN}Connection to T4 GPU server successful.${NC}"
fi

# Create remote directory if it doesn't exist
echo -e "${BLUE}Creating remote directory: ${REMOTE_DIR}...${NC}"
ssh -i "$SSH_KEY" "$SSH_USER@$T4_SERVER" "mkdir -p $REMOTE_DIR"

# Rsync the project files to the remote server
echo -e "${BLUE}Syncing project files to T4 GPU server...${NC}"
rsync -avz --progress --exclude ".git" --exclude "node_modules" --exclude "venv" \
    -e "ssh -i $SSH_KEY" \
    ./ "$SSH_USER@$T4_SERVER:$REMOTE_DIR/"

# Check if Docker is installed on the remote server
echo -e "${BLUE}Checking Docker installation on T4 GPU server...${NC}"
if ! ssh -i "$SSH_KEY" "$SSH_USER@$T4_SERVER" "which docker > /dev/null"; then
    echo -e "${YELLOW}Docker not found on the T4 GPU server. Installing Docker...${NC}"
    ssh -i "$SSH_KEY" "$SSH_USER@$T4_SERVER" "curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh"
    ssh -i "$SSH_KEY" "$SSH_USER@$T4_SERVER" "sudo usermod -aG docker $SSH_USER"
    echo -e "${GREEN}Docker installed successfully.${NC}"
else
    echo -e "${GREEN}Docker is already installed on the T4 GPU server.${NC}"
fi

# Check if Docker Compose is installed on the remote server
echo -e "${BLUE}Checking Docker Compose installation on T4 GPU server...${NC}"
if ! ssh -i "$SSH_KEY" "$SSH_USER@$T4_SERVER" "which docker-compose > /dev/null"; then
    echo -e "${YELLOW}Docker Compose not found on the T4 GPU server. Installing Docker Compose...${NC}"
    ssh -i "$SSH_KEY" "$SSH_USER@$T4_SERVER" "sudo curl -L \"https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)\" -o /usr/local/bin/docker-compose"
    ssh -i "$SSH_KEY" "$SSH_USER@$T4_SERVER" "sudo chmod +x /usr/local/bin/docker-compose"
    echo -e "${GREEN}Docker Compose installed successfully.${NC}"
else
    echo -e "${GREEN}Docker Compose is already installed on the T4 GPU server.${NC}"
fi

# Check if NVIDIA Container Toolkit is installed on the remote server
echo -e "${BLUE}Checking NVIDIA Container Toolkit installation on T4 GPU server...${NC}"
if ! ssh -i "$SSH_KEY" "$SSH_USER@$T4_SERVER" "which nvidia-container-toolkit > /dev/null"; then
    echo -e "${YELLOW}NVIDIA Container Toolkit not found on the T4 GPU server. Installing NVIDIA Container Toolkit...${NC}"
    ssh -i "$SSH_KEY" "$SSH_USER@$T4_SERVER" "distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID) && \
        curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add - && \
        curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
        sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker"
    echo -e "${GREEN}NVIDIA Container Toolkit installed successfully.${NC}"
else
    echo -e "${GREEN}NVIDIA Container Toolkit is already installed on the T4 GPU server.${NC}"
fi

# Deploy using Docker Compose
echo -e "${BLUE}Deploying with Docker Compose on T4 GPU server...${NC}"
ssh -i "$SSH_KEY" "$SSH_USER@$T4_SERVER" "cd $REMOTE_DIR && docker-compose down && docker-compose up -d"

# Check deployment status
echo -e "${BLUE}Checking deployment status...${NC}"
ssh -i "$SSH_KEY" "$SSH_USER@$T4_SERVER" "cd $REMOTE_DIR && docker-compose ps"

# Print success message
echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${BLUE}You can access the application at: ${NC}"
echo -e "${GREEN}http://${T4_SERVER}${NC}"
echo ""
echo -e "${BLUE}API is available at: ${NC}"
echo -e "${GREEN}http://${T4_SERVER}/api${NC}"
echo ""
echo -e "${BLUE}Metrics are available at: ${NC}"
echo -e "${GREEN}http://${T4_SERVER}/metrics${NC}"
echo ""
echo -e "${BLUE}Grafana dashboard is available at: ${NC}"
echo -e "${GREEN}http://${T4_SERVER}/grafana${NC}"
echo ""
echo -e "${BLUE}To check logs: ${NC}"
echo -e "${YELLOW}ssh -i $SSH_KEY $SSH_USER@$T4_SERVER 'cd $REMOTE_DIR && docker-compose logs -f'${NC}"
echo ""
echo -e "${BLUE}To run tests: ${NC}"
echo -e "${YELLOW}ssh -i $SSH_KEY $SSH_USER@$T4_SERVER 'cd $REMOTE_DIR && ./run_tests.sh --all'${NC}"
echo ""
echo -e "${BLUE}=========================================================${NC}"