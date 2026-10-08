# Build the dataset paper. Requires a LaTeX toolchain (pdflatex + bibtex),
# e.g. TeX Live or MiKTeX. `make` produces main.pdf.

MAIN = main

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

# Alternative one-shot builders if available:
#   latexmk -pdf $(MAIN).tex
#   tectonic $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc
