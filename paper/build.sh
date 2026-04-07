#!/usr/bin/env bash
# paper/build.sh — compile LaTeX paper
set -euo pipefail
cd "$(dirname "$0")"

echo "Checking figures..."
shopt -s nullglob
for fig in figures/fig{1..9}_*.png; do
    if [ ! -f "$fig" ]; then
        echo "WARNING: Missing $fig — run python3 -m benchmark.plot first"
    fi
done

echo "Compiling LaTeX..."
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "Done: main.pdf"
