#!/usr/bin/env bash
# Publish the current commit to a Hugging Face Docker Space.
# Usage: scripts/push_to_space.sh <hf-username>/<space-name>
# Needs `git` and a Hugging Face login (run `hf auth login` or use a token as the git password).
set -euo pipefail

SPACE="${1:?usage: $0 <hf-username>/<space-name>}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git archive HEAD | tar -x -C "$TMP"

# Spaces read their config from README front matter; keep it out of the GitHub README.
{
  printf -- '---\ntitle: SAABI Cattle Breed Identifier\nemoji: 🐄\ncolorFrom: green\ncolorTo: yellow\nsdk: docker\napp_port: 7860\npinned: false\n---\n\n'
  cat "$TMP/README.md"
} > "$TMP/README.tmp" && mv "$TMP/README.tmp" "$TMP/README.md"

cd "$TMP"
git init -q -b main
git add -A
git -c user.name="SAABI deploy" -c user.email="deploy@localhost" commit -qm "Deploy SAABI"
git push --force "https://huggingface.co/spaces/${SPACE}" main
echo "Pushed. Build logs: https://huggingface.co/spaces/${SPACE}"
