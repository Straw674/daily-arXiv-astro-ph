#!/usr/bin/env bash
set -euo pipefail

# Ensure essential tools (uv, git, homebrew) are available in launchd/cron environments
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORKTREE_DIR="$REPO_ROOT/dist"

# Target date and force regen can be supplied via environment variables
TARGET_DATE="${TARGET_DATE:-}"
FORCE_REGEN="${FORCE_REGEN:-false}"

if [ -n "$TARGET_DATE" ]; then
    RUN_DATE="$TARGET_DATE"
else
    RUN_DATE="$(date -u +%Y-%m-%d)"
fi

# Format YYYYMMDD for git commit message
COMMIT_DATE="${RUN_DATE//-/}"

echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting arXiv Daily Pipeline"
echo "Target Date: $RUN_DATE | Force Regen: $FORCE_REGEN"
echo "Repo Root: $REPO_ROOT"
echo "============================================================"

# Ensure dist worktree exists and tracks branch 'data'
if [ ! -d "$WORKTREE_DIR/.git" ] && [ ! -f "$WORKTREE_DIR/.git" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Initializing dist worktree on data branch..."
    rm -rf "$WORKTREE_DIR"
    git worktree add "$WORKTREE_DIR" data
fi

# Pull latest commits from origin/data
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Syncing data branch with remote..."
git -C "$WORKTREE_DIR" pull --rebase origin data

# Execute Python pipeline
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Executing src/main.py..."
TARGET_DATE="$RUN_DATE" FORCE_REGEN="$FORCE_REGEN" uv run python src/main.py

# Detect changes in worktree
CHANGES=$(git -C "$WORKTREE_DIR" status --porcelain)

if [ -n "$CHANGES" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] New data generated. Staging and committing..."
    git -C "$WORKTREE_DIR" add README.md data/
    git -C "$WORKTREE_DIR" commit -m "update ${COMMIT_DATE}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pushing commit to origin data..."
    git -C "$WORKTREE_DIR" push origin data

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Successfully updated and pushed data for $RUN_DATE."
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No changes detected in data branch (feed empty or already generated)."
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline finished."
