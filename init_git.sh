#!/usr/bin/env bash
set -euo pipefail

# Variables
REPO_URL="git@github.com:rag2024-ai/multi-agent-anomaly-system.git"

# Initialize git repo
git init

# Create .gitignore if missing
if [ ! -f .gitignore ]; then
  cat > .gitignore <<EOF
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
*.db
*.log
*.sqlite3
.DS_Store
.ipynb_checkpoints/
data/outputs/topics/*
data/outputs/reports/*
EOF
fi

# Stage and commit
git add .
git commit -m "Initial commit: Multi-Agent Anomaly Correlation & Impact Analysis"

# Add remote and push
git branch -M main
git remote add origin "$REPO_URL"
git push -u origin main

