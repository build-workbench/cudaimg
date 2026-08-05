#!/usr/bin/env bash
# Create a GitHub Release for mini-opencv.
#
# Usage:
#   GH_TOKEN=<token> ./scripts/create_release.sh <version> [title]
#   e.g. GH_TOKEN=*** ./scripts/create_release.sh v3.1.0
#
# Requires: curl, jq, and a GH_TOKEN env var with release scope.
# The release notes point to CHANGELOG.md; edit the BODY below to customize.

set -euo pipefail

REPO="AICL-Lab/mini-opencv"

VERSION="${1:-}"
TITLE="${2:-$VERSION}"

if [[ -z "$VERSION" ]]; then
  echo "error: version argument required (e.g. v3.1.0)" >&2
  echo "usage: GH_TOKEN=<token> $0 <version> [title]" >&2
  exit 1
fi

for dep in curl jq; do
  if ! command -v "$dep" >/dev/null 2>&1; then
    echo "error: required dependency '$dep' not found on PATH" >&2
    exit 1
  fi
done

if [[ -z "${GH_TOKEN:-}" ]]; then
  echo "error: GH_TOKEN environment variable is not set" >&2
  exit 1
fi

BODY=$(cat << 'EOF'
## What's New

See [CHANGELOG.md](https://github.com/AICL-Lab/mini-opencv/blob/main/CHANGELOG.md)
for the full list of changes in this release.

### Quick Start
```bash
git clone https://github.com/AICL-Lab/mini-opencv.git
cd mini-opencv
cmake -S . -B build
cmake --build build -j$(nproc)
```

**Full Documentation**: https://aicl-lab.github.io/mini-opencv/
EOF
)

# -f: fail on HTTP errors instead of printing the error body as if successful.
curl -fsS -X POST \
  -H "Authorization: token $GH_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/repos/$REPO/releases" \
  -d "{
    \"tag_name\": \"$VERSION\",
    \"name\": \"$TITLE\",
    \"body\": $(echo "$BODY" | jq -Rs .),
    \"draft\": false,
    \"prerelease\": false
  }"
