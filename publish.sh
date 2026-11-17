#!/usr/bin/env bash
# Publish this Principia competition project to a public GitHub repository.
#
# Usage:   bash publish.sh [repo-name]
# Default repo name: principia-354674-moving-spikes
#
# Needs git and the GitHub CLI (gh), logged in. If gh is missing, the script
# says how to install it and stops without changing anything.
set -euo pipefail

REPO_NAME="${1:-principia-354674-moving-spikes}"
DESCRIPTION="Principia Fall Competition 2026 (MATHDB 354674): moving spikes and measure-valued limits of backward recurrences, with an application to backpropagation through ResNets."
cd "$(dirname "$0")"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# 1. Tools ------------------------------------------------------------------
command -v git >/dev/null || { echo "git is not installed: sudo apt install git"; exit 1; }
if ! command -v gh >/dev/null; then
  echo "The GitHub CLI (gh) is not installed. Install it, log in, and run this script again:"
  echo "  sudo apt install gh        # or see https://cli.github.com"
  echo "  gh auth login"
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  say "Logging in to GitHub (follow the prompts)"
  gh auth login
fi
GH_USER="$(gh api user --jq .login)"

if [ -z "$(git config user.name || true)" ] || [ -z "$(git config user.email || true)" ]; then
  echo "Set your git identity first, then rerun:"
  echo '  git config --global user.name  "Your Name"'
  echo '  git config --global user.email "you@example.com"'
  exit 1
fi

# 2. Local repository ---------------------------------------------------------
say "Preparing the local repository"
[ -d .git ] || git init -q -b main
git add -A
if git diff --cached --quiet; then
  echo "Nothing new to commit."
else
  git commit -q -m "Principia Fall Competition 2026: MATHDB 354674 project files"
  echo "Committed."
fi

# 3. GitHub repository ----------------------------------------------------------
if gh repo view "$GH_USER/$REPO_NAME" >/dev/null 2>&1; then
  say "Pushing to existing repository $GH_USER/$REPO_NAME"
  git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/$GH_USER/$REPO_NAME.git"
  git push -u origin main
else
  say "Creating public repository $GH_USER/$REPO_NAME"
  gh repo create "$REPO_NAME" --public --description "$DESCRIPTION" --source . --remote origin --push
fi
gh repo edit "$GH_USER/$REPO_NAME" \
  --add-topic principia --add-topic mathematics --add-topic scientific-machine-learning \
  --add-topic neural-ode --add-topic numerical-analysis >/dev/null 2>&1 || true

# 4. Links for the Principia stages --------------------------------------------
BASE="https://github.com/$GH_USER/$REPO_NAME"
say "Done. Links for the Principia project stages:"
echo "  Repository:   $BASE"
echo "  Paper:        $BASE/blob/main/paper/paper.pdf"
echo "  Exposition:   $BASE/tree/main/exposition"
echo "  Applications: $BASE/blob/main/applications/applications.pdf"
