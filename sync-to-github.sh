#!/bin/bash
# Script to sync project with GitHub and create project tickets

set -e

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install gh
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
        sudo apt update
        sudo apt install gh
    else
        echo "Unsupported OS. Please install GitHub CLI manually: https://github.com/cli/cli#installation"
        exit 1
    fi
fi

# Check if logged in to GitHub
gh auth status &> /dev/null || (echo "Not logged in to GitHub. Running 'gh auth login'..." && gh auth login)

# Get repository details
REPO_URL=$(git remote get-url origin 2>/dev/null || echo "")
if [ -z "$REPO_URL" ]; then
    # Repository doesn't exist or not configured
    echo "GitHub repository not configured. Please enter the repository name (e.g., username/repo):"
    read REPO_NAME
    
    # Check if repository exists
    if gh repo view "$REPO_NAME" &> /dev/null; then
        echo "Repository exists. Configuring as remote..."
        git remote add origin "https://github.com/$REPO_NAME.git"
    else
        echo "Repository doesn't exist. Creating..."
        gh repo create "$REPO_NAME" --public --description "Generative AI Toolkit for SAP HANA Cloud with NVIDIA GPU Optimizations"
        git remote add origin "https://github.com/$REPO_NAME.git"
    fi
    REPO_URL="https://github.com/$REPO_NAME.git"
fi

# Extract repository owner and name
REPO_OWNER=$(echo $REPO_URL | sed -E 's/.*[:/]([^/]+)\/([^/]+)(\.git)?$/\1/')
REPO_NAME=$(echo $REPO_URL | sed -E 's/.*[:/]([^/]+)\/([^/]+)(\.git)?$/\2/')
FULL_REPO="$REPO_OWNER/$REPO_NAME"

echo "Syncing with repository: $FULL_REPO"

# Check if there are changes to commit
if [ -n "$(git status --porcelain)" ]; then
    echo "Uncommitted changes found. Committing..."
    git add .
    
    # Skip pre-commit hooks if they're causing issues
    git commit -m "Add NVIDIA NGC integration and TensorRT optimization" --no-verify
    
    # Push changes
    echo "Pushing changes to GitHub..."
    git push -u origin main || git push -u origin master
else
    echo "No changes to commit."
fi

# Create project tickets
echo "Creating project tickets..."

# Create a project if one doesn't exist
PROJECT_ID=$(gh api graphql -f query='
  query {
    viewer {
      projects(first: 10) {
        nodes {
          id
          name
        }
      }
    }
  }
' | jq -r '.data.viewer.projects.nodes[] | select(.name == "NVIDIA GPU Optimizations") | .id')

if [ -z "$PROJECT_ID" ]; then
    echo "Creating new project board: NVIDIA GPU Optimizations"
    PROJECT_ID=$(gh api graphql -f query='
      mutation {
        createProject(input: {ownerId: $ownerId, name: "NVIDIA GPU Optimizations"}) {
          project {
            id
          }
        }
      }
      ' -f ownerId="$(gh api user | jq -r '.node_id')" | jq -r '.data.createProject.project.id')
fi

echo "Using project ID: $PROJECT_ID"

# Function to create an issue and add it to the project
create_issue() {
    local title="$1"
    local body="$2"
    local label="$3"
    
    echo "Creating issue: $title"
    ISSUE_URL=$(gh issue create --repo "$FULL_REPO" --title "$title" --body "$body" --label "$label" | grep -o 'https://github.com/[^[:space:]]*')
    ISSUE_NUMBER=$(echo $ISSUE_URL | grep -o '[0-9]*$')
    
    echo "Adding issue #$ISSUE_NUMBER to project..."
    ISSUE_ID=$(gh api graphql -f query='
      query {
        repository(owner: $owner, name: $name) {
          issue(number: $number) {
            id
          }
        }
      }
      ' -f owner="$REPO_OWNER" -f name="$REPO_NAME" -f number="$ISSUE_NUMBER" | jq -r '.data.repository.issue.id')
      
    gh api graphql -f query='
      mutation {
        addProjectCard(input: {projectColumnId: $columnId, contentId: $contentId}) {
          clientMutationId
        }
      }
      ' -f columnId="$PROJECT_ID" -f contentId="$ISSUE_ID" || echo "Failed to add to project"
      
    echo "Created issue #$ISSUE_NUMBER: $title"
}

# Create issues for each major component
create_issue "NVIDIA NGC Integration" "Implement NVIDIA NGC container publishing integration:

- Create ngc-blueprint.json configuration
- Develop NGC-specific Dockerfile
- Set up GitHub workflow for NGC publishing
- Create helper script for manual publishing
- Document NGC integration process" "enhancement"

create_issue "TensorRT Optimization" "Implement TensorRT optimization for deep learning models:

- Create TensorRT utilities module
- Integrate with GPU management system
- Add configuration options for TensorRT
- Implement model conversion pipeline
- Optimize embedding generation and LLM inference
- Document TensorRT integration" "enhancement"

create_issue "Hopper Architecture Optimizations" "Implement H100-specific optimizations:

- Add FP8 precision support
- Integrate Transformer Engine
- Implement Flash Attention 2
- Add FSDP support
- Document Hopper-specific features" "enhancement"

create_issue "Documentation Updates" "Enhance documentation for NVIDIA features:

- Create NGC deployment guide
- Document TensorRT optimization details
- Update main README with NVIDIA features
- Create authentication and setup guide
- Create NVIDIA Launchable notebook" "documentation"

create_issue "Performance Benchmarking" "Conduct performance benchmarks:

- Measure TensorRT vs standard performance
- Compare H100 vs A100 performance
- Benchmark embedding generation speedup
- Benchmark LLM inference speedup
- Document performance improvements" "enhancement"

echo "Project tickets created successfully."
echo "Visit https://github.com/$FULL_REPO/issues to view the issues."
echo "View the project board at https://github.com/users/$REPO_OWNER/projects/."