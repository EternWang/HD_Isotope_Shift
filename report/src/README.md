# Report source

`main.tex` is a LaTeX source that matches the structure/numbers of the final report and is suitable for GitHub version control.

To compile (from `report/src/`):

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Figures are expected in `report/figures/` (see `\graphicspath` in `main.tex`).
