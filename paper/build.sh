#!/usr/bin/env bash
# GridForge preprint build helper (Linux / macOS / WSL / Git Bash).
#
# Stages figures from scripts/output/, then runs Pandoc to produce
# both PDF (via XeLaTeX) and standalone HTML versions of the paper.
#
# Usage:
#   ./paper/build.sh              # full build, both outputs
#   ./paper/build.sh --pdf        # PDF only
#   ./paper/build.sh --html       # HTML only

set -euo pipefail

PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$PAPER_DIR/.." && pwd)"
OUT_DIR="$REPO_ROOT/scripts/output/07_paper"
mkdir -p "$OUT_DIR"

echo
echo "Staging figures..."
python "$PAPER_DIR/build_figures.py" || {
  echo "Figure staging failed — regenerate the source PNGs first." >&2
  exit 1
}

want_pdf=true
want_html=true
case "${1:-}" in
  --pdf)  want_html=false ;;
  --html) want_pdf=false ;;
esac

if ! command -v pandoc >/dev/null 2>&1; then
  echo "Pandoc not found in PATH; install it first." >&2
  exit 2
fi

if $want_html; then
  echo
  echo "Building HTML..."
  pandoc \
    --standalone \
    --bibliography="$PAPER_DIR/references.bib" \
    --citeproc \
    --resource-path="$PAPER_DIR" \
    --metadata=link-citations:true \
    -o "$OUT_DIR/paper.html" \
    "$PAPER_DIR/paper.md"
  echo "  HTML : $OUT_DIR/paper.html"
fi

if $want_pdf; then
  echo
  echo "Building PDF (requires XeLaTeX)..."
  pandoc \
    --pdf-engine=xelatex \
    --bibliography="$PAPER_DIR/references.bib" \
    --citeproc \
    --resource-path="$PAPER_DIR" \
    --metadata=link-citations:true \
    -o "$OUT_DIR/paper.pdf" \
    "$PAPER_DIR/paper.md" \
    || echo "PDF build failed (XeLaTeX missing?). HTML output remains usable."
  if [ -f "$OUT_DIR/paper.pdf" ]; then
    echo "  PDF  : $OUT_DIR/paper.pdf"
  fi
fi

echo
echo "Done."
