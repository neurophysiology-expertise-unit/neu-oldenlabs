#!/usr/bin/env bash
# Render an experiment report.md to HTML and PDF.
#
# The report that reaches a collaborator is a PDF, so the command that builds it
# belongs in the repository rather than in one person's shell history: without
# this, report.pdf and report.md can drift apart with nothing to show which
# version of the figures a delivered PDF contains.
#
# Usage: scripts/render_report.sh <experiment-dir> [header.tex]
#   <experiment-dir>  directory holding report.md (figures are referenced from it)
#   [header.tex]      optional LaTeX header; defaults to ../pdf-header.tex if present
set -euo pipefail

experiment="${1:?usage: render_report.sh <experiment-dir> [header.tex]}"
report="$experiment/report.md"
[ -f "$report" ] || { echo "no report.md in $experiment" >&2; exit 1; }

header="${2:-}"
if [ -z "$header" ] && [ -f "$experiment/../pdf-header.tex" ]; then
  header="$experiment/../pdf-header.tex"
fi

# --resource-path lets pandoc resolve the relative figure paths in report.md.
# --lua-filter forces every table to explicit, content-proportional column
# widths: a plain markdown pipe table parses with width 0 ("auto") per
# column, which pandoc's LaTeX writer renders as an unwrapped `l` column --
# fine for a narrow table, but the wide results tables here (measure names
# up to 31 characters, headers like `cages_contributing`) then run straight
# off the page margin instead of wrapping. See table-widths.lua for why
# pandoc's own width heuristic isn't a substitute for this.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
common=(--standalone --toc --resource-path="$experiment"
        --lua-filter="$script_dir/table-widths.lua"
        --metadata title="Oldenlabs home-cage report")

pandoc "$report" "${common[@]}" --self-contained -o "$experiment/report.html"
echo "wrote $experiment/report.html"

pdf=(--pdf-engine=xelatex -V geometry:margin=2.5cm -V linkcolor:blue)
[ -n "$header" ] && pdf+=(--include-in-header="$header")
pandoc "$report" "${common[@]}" "${pdf[@]}" -o "$experiment/report.pdf"
echo "wrote $experiment/report.pdf"
