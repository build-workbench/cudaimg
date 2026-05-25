#!/bin/bash
# Create GitHub Release Script

VERSION="v3.0.0"
TITLE="v3.0.0 - AI Framework Removal & Repository Simplification"

BODY=$(cat << 'EOF'
## What's New

This release removes AI workflow frameworks and simplifies the repository back to a minimal, maintainable structure.

### Cleanup
- Removed OpenSpec specs, Claude skills, and AI-only control files
- Reduced repository complexity and dead documentation paths
- Consolidated changelog ownership to the root `CHANGELOG.md`

### Files Changed
- Simplified contributing and Copilot guidance
- Removed GitHub Pages changelog pages and sync script
- Updated README.md and README.zh-CN.md to point to live VitePress docs

### Quick Start
```bash
git clone https://github.com/LessUp/mini-opencv.git
cd mini-opencv
cmake -S . -B build
cmake --build build -j$(nproc)
```

**Full Documentation**: https://lessup.github.io/mini-opencv/
EOF
)

# Create release using curl
curl -X POST \
  -H "Authorization: token $GH_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/LessUp/mini-opencv/releases \
  -d "{
    \"tag_name\": \"$VERSION\",
    \"name\": \"$TITLE\",
    \"body\": $(echo "$BODY" | jq -Rs .),
    \"draft\": false,
    \"prerelease\": false
  }" 2>&1
