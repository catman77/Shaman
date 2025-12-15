# Makefile for Natural Observation-Based Computing Paper

# Main document name (without .tex extension)
MAIN = main

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Viewer
VIEWER = evince

# Flags
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

# Default target
all: $(MAIN).pdf

# Full build with bibliography
$(MAIN).pdf: $(MAIN).tex sections/*.tex references.bib
	@echo "=== First LaTeX pass ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo ""
	@echo "=== BibTeX pass ==="
	$(BIBTEX) $(MAIN)
	@echo ""
	@echo "=== Second LaTeX pass ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo ""
	@echo "=== Third LaTeX pass ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo ""
	@echo "=== Build complete: $(MAIN).pdf ==="

# Quick build (without bibliography, for drafts)
quick: $(MAIN).tex sections/*.tex
	@echo "=== Quick build (no bibliography) ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "=== Quick build complete ==="

# Build with latexmk (recommended if available)
latexmk: $(MAIN).tex sections/*.tex references.bib
	@echo "=== Building with latexmk ==="
	latexmk -pdf -pdflatex="$(LATEX) $(LATEX_FLAGS)" $(MAIN).tex
	@echo "=== Build complete ==="

# View PDF
view: $(MAIN).pdf
	@echo "=== Opening PDF viewer ==="
	$(VIEWER) $(MAIN).pdf &

# Clean auxiliary files
clean:
	@echo "=== Cleaning auxiliary files ==="
	rm -f $(MAIN).aux
	rm -f $(MAIN).log
	rm -f $(MAIN).out
	rm -f $(MAIN).toc
	rm -f $(MAIN).bbl
	rm -f $(MAIN).blg
	rm -f $(MAIN).lof
	rm -f $(MAIN).lot
	rm -f $(MAIN).fls
	rm -f $(MAIN).fdb_latexmk
	rm -f $(MAIN).synctex.gz
	rm -f sections/*.aux
	@echo "=== Auxiliary files removed ==="

# Clean everything including PDF
distclean: clean
	@echo "=== Cleaning PDF ==="
	rm -f $(MAIN).pdf
	@echo "=== All generated files removed ==="

# Word count (approximate)
wordcount:
	@echo "=== Word count by section ==="
	@echo "Abstract:"
	@texcount -brief sections/00_abstract.tex
	@echo "Introduction:"
	@texcount -brief sections/01_introduction.tex
	@echo "Related Work:"
	@texcount -brief sections/02_related_work.tex
	@echo "Theoretical Framework:"
	@texcount -brief sections/03_theoretical_framework.tex
	@echo "Market Theory:"
	@texcount -brief sections/04_market_theory.tex
	@echo "Algorithm:"
	@texcount -brief sections/05_algorithm.tex
	@echo "Experiments:"
	@texcount -brief sections/06_experiments.tex
	@echo "Implications:"
	@texcount -brief sections/07_implications.tex
	@echo "Discussion:"
	@texcount -brief sections/08_discussion.tex
	@echo "Future Work:"
	@texcount -brief sections/09_future_work.tex
	@echo "Conclusion:"
	@texcount -brief sections/10_conclusion.tex
	@echo ""
	@echo "=== Total word count ==="
	@texcount -brief -merge -sum $(MAIN).tex

# Check for common LaTeX errors
check:
	@echo "=== Checking for common LaTeX errors ==="
	@echo "Checking for undefined references..."
	@grep -n "ref{" sections/*.tex | grep -v "\\label" || echo "OK"
	@echo "Checking for bad boxes..."
	@grep -i "overfull\\|underfull" $(MAIN).log || echo "OK"
	@echo "Checking for missing citations..."
	@grep "Citation.*undefined" $(MAIN).log || echo "OK"
	@echo "=== Check complete ==="

# Spell check (requires aspell)
spell:
	@echo "=== Spell checking ==="
	@for file in sections/*.tex; do \
		echo "Checking $$file..."; \
		aspell --lang=en --mode=tex check $$file; \
	done
	@echo "=== Spell check complete ==="

# Create submission archive
archive: $(MAIN).pdf
	@echo "=== Creating submission archive ==="
	@mkdir -p submission
	@cp $(MAIN).tex submission/
	@cp -r sections submission/
	@cp references.bib submission/
	@cp $(MAIN).pdf submission/
	@cp README_LATEX.md submission/README.md
	@tar -czf natural_observation_computing_submission.tar.gz submission/
	@rm -rf submission/
	@echo "=== Archive created: natural_observation_computing_submission.tar.gz ==="

# Help
help:
	@echo "Available targets:"
	@echo "  make             - Build PDF with bibliography (full build)"
	@echo "  make quick       - Quick build without bibliography"
	@echo "  make latexmk     - Build using latexmk (if available)"
	@echo "  make view        - Open PDF in viewer"
	@echo "  make clean       - Remove auxiliary files"
	@echo "  make distclean   - Remove all generated files including PDF"
	@echo "  make wordcount   - Count words in document"
	@echo "  make check       - Check for common LaTeX errors"
	@echo "  make spell       - Spell check all sections (requires aspell)"
	@echo "  make archive     - Create submission archive"
	@echo "  make help        - Show this help message"

.PHONY: all quick latexmk view clean distclean wordcount check spell archive help
