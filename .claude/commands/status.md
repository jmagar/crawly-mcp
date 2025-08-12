---
description: Show comprehensive project status
allowed-tools: Bash(git:*), Bash(ls:*), Bash(find:*), Bash(wc:*)
---

# Project Status Report

## Git Status
Current git status: !`git status --porcelain`

## Branch Information
Current branch: !`git branch --show-current`
Recent commits: !`git log --oneline -5`

## Modified Files (if any)
!`git diff --name-only HEAD`

## Project Structure
Total files: !`find . -type f -not -path "./.git/*" | wc -l`
Python files: !`find . -name "*.py" -not -path "./.git/*" | wc -l`
Markdown files: !`find . -name "*.md" -not -path "./.git/*" | wc -l`

## Recent Activity
Files modified in last 24h: !`find . -type f -not -path "./.git/*" -mtime -1 | head -10`
