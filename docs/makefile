# Tools
TYPST    ?= typst
MERMAIDX ?= mermaidx

# Files
MMD_SRC     := architecture.mmd
PNG_DIAGRAM := architecture.png
SVG_DIAGRAM := architecture.svg
TYP_SRC     := abstract.typ
PDF_TARGET  := abstract.pdf

.PHONY: all pdf diagram clean watch

# Default target
all: pdf

# Build abstract PDF
pdf: $(PDF_TARGET)

# Compile Typst document into PDF
$(PDF_TARGET): $(TYP_SRC) $(PNG_DIAGRAM) $(SVG_DIAGRAM)
	$(TYPST) compile $(TYP_SRC) $(PDF_TARGET)

# Compile Mermaid source into high-resolution PNG and SVG
$(PNG_DIAGRAM): $(MMD_SRC)
	$(MERMAIDX) -i $(MMD_SRC) -o $(PNG_DIAGRAM) --scale 3.0 -b white

$(SVG_DIAGRAM): $(MMD_SRC)
	$(MERMAIDX) -i $(MMD_SRC) -o $(SVG_DIAGRAM) -t neutral -b transparent

diagram: $(PNG_DIAGRAM) $(SVG_DIAGRAM)

# Watch mode for live editing
watch: $(PNG_DIAGRAM) $(SVG_DIAGRAM)
	$(TYPST) watch $(TYP_SRC) $(PDF_TARGET)

# Clean generated artifacts
clean:
	rm -f $(PDF_TARGET) $(PNG_DIAGRAM) $(SVG_DIAGRAM)
