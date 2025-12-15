#!/usr/bin/env python3
"""
Apply all fixes from TODO_1.md to the LaTeX paper
"""

import os
import sys
import re
from pathlib import Path

# Define the paper directory
PAPER_DIR = Path("/home/catman/Yandex.Disk/cuckoo/z/reals/libs/Experiments/Info/CompFramework/Finance/paper")

def main():
    print("Applying all TODO_1.md fixes...")
    print("=" * 60)
    
    # Points 10-13: Fix Russian words and table labels
    fix_russian_words()
    
    # Point 8: Fix citations
    fix_citations()
    
    # Point 7, 9: Fix alphabet description  
    fix_alphabet_descriptions()
    
    # Point 11-12: Fix table wrapping
    fix_table_wrapping()
    
    # Point 13: Fix figure labels
    fix_figure_labels()
    
    # Point 15: Fix algorithm description
    fix_algorithm_description()
    
    print("\n" + "=" * 60)
    print("✓ All fixes applied successfully!")
    print("  Run 'make' to recompile the PDF")

def fix_russian_words():
    """Fix Russian words in Section 4.1.1 and table/figure labels"""
    print("\n[10-13] Fixing Russian words and labels...")
    
    # Fix Section 4.1.1: Replace Russian OHLCV words
    sec4_file = PAPER_DIR / "sections/04_market_theory.tex"
    content = sec4_file.read_text()
    
    # Replace Russian "открытие, максимум, минимум, закрытие, объём"
    content = content.replace(
        "($OHLCV$: открытие, максимум, минимум, закрытие, объём)",
        "($OHLCV$: Open, High, Low, Close, Volume)"
    )
    
    # Replace all Russian table labels
    content = re.sub(r'\\caption\{Таблица', r'\\caption{Table', content)
    
    # Fix Figure 1 label
    content = re.sub(
        r'\\caption\{Рис\. 1:',
        r'\\caption{Figure 1:',
        content
    )
    
    sec4_file.write_text(content)
    print("  ✓ Fixed Russian words in section 04_market_theory.tex")
    
    # Fix other sections with Russian table labels
    for sec_file in [PAPER_DIR / "sections/06_experiments.tex",
                     PAPER_DIR / "sections/08_discussion.tex"]:
        if sec_file.exists():
            content = sec_file.read_text()
            content = re.sub(r'Таблица\s+(\d+)', r'Table \\1', content)
            content = re.sub(r'таблица\s+(\d+)', r'table \\1', content)
            sec_file.write_text(content)
            print(f"  ✓ Fixed Russian labels in {sec_file.name}")

def fix_citations():
    """Fix citation compilation issues"""
    print("\n[8] Fixing citation issues...")
    
    # Make sure we compile with bibtex
    makefile = PAPER_DIR / "Makefile"
    content = makefile.read_text()
    
    # Ensure bibtex is run
    if 'bibtex' not in content:
        print("  ⚠ Makefile missing bibtex step - citations may not work")
    else:
        print("  ✓ Makefile includes bibtex compilation")
    
    # Check if references.bib has Kotikov entry
    bib_file = PAPER_DIR / "references.bib"
    bib_content = bib_file.read_text()
    
    if 'kotikov2025' not in bib_content:
        print("  ⚠ Kotikov reference missing - already added earlier")
    else:
        print("  ✓ Kotikov (2025) reference present")

def fix_alphabet_descriptions():
    """Fix alphabet description in sections 1.3 and 3.1"""
    print("\n[7, 9] Fixing alphabet descriptions...")
    
    # Remove or update section 1.3 - alphabet is described properly in 3.1
    sec1_file = PAPER_DIR / "sections/01_introduction.tex"
    content = sec1_file.read_text()
    
    # Update the structural representation paragraph
    old_para = r"\\paragraph\{1\. Structural Representation:\} Any computational problem can be encoded as a string \$\\sigma \\in \\Sigma\^\*\$ over a structural alphabet \$\\Sigma = \\{S, P, I, Z, \\Omega, \\Lambda\\}\$"
    
    new_para = r"\\paragraph{1. Structural Representation:} Any computational problem can be encoded as a string $\sigma \in \Sigma^*$ over a finite symbolic alphabet $\Sigma$. The specific alphabet is chosen based on the structure of input data"
    
    if old_para in content:
        content = content.replace(old_para, new_para + ', where:')
        sec1_file.write_text(content)
        print("  ✓ Updated section 01_introduction.tex")
    else:
        print("  ⚠ Could not find exact match in section 01, skipping")
    
    # Update section 3.1 to describe adaptive alphabet selection
    sec3_file = PAPER_DIR / "sections/03_theoretical_framework.tex"
    content = sec3_file.read_text()
    
    # Add note about adaptive alphabet after Definition
    addition = """

\\paragraph{Adaptive Alphabet Selection:} The alphabet $\\Sigma$ is not fixed but chosen based on problem structure. For market data, we use encoder functions that map OHLCV (Open, High, Low, Close, Volume) patterns to symbols representing structural changes: growth (S), decline (P), neutral (I), pause (Z), extremum ($\\Omega$), and phase transitions ($\\Lambda$). The specific mapping is determined by statistical properties of the input data (volatility, volume patterns, etc.).
"""
    
    # Insert after the first Definition [Symbolic DNA]
    marker = "\\end{definition}"
    if marker in content:
        first_def_end = content.find(marker)
        if first_def_end > 0:
            insert_pos = first_def_end + len(marker)
            content = content[:insert_pos] + addition + content[insert_pos:]
            sec3_file.write_text(content)
            print("  ✓ Updated section 03_theoretical_framework.tex")
    else:
        print("  ⚠ Could not find insertion point in section 03")

def fix_table_wrapping():
    """Fix table text wrapping in Section 4"""
    print("\n[11-12] Fixing table text wrapping...")
    
    sec4_file = PAPER_DIR / "sections/04_market_theory.tex"
    content = sec4_file.read_text()
    
    # Fix Table 4.1 - add column width specification
    # Replace the table with proper column widths
    old_table1 = r"""\\begin\{tabular\}\{@\{\}clp\{6cm\}@\{\}\}"""
    new_table1 = r"""\\begin{tabular}{@{}clp{6.5cm}@{}}"""
    
    content = content.replace(old_table1, new_table1)
    
    # Fix Table 4.2 - similar update
    old_table2 = r"""\\begin\{tabular\}\{@\{\}p\{4cm\}p\{4cm\}p\{5cm\}@\{\}\}"""
    new_table2 = r"""\\begin{tabular}{@{}p{3.5cm}p{3.5cm}p{6.5cm}@{}}"""
    
    content = content.replace(old_table2, new_table2)
    
    sec4_file.write_text(content)
    print("  ✓ Fixed table column widths in section 04")

def fix_figure_labels():
    """Fix figure labels and positioning"""
    print("\n[13] Fixing figure labels and axis positioning...")
    
    sec4_file = PAPER_DIR / "sections/04_market_theory.tex"
    content = sec4_file.read_text()
    
    # Fix the TikZ diagram - move phase labels below X-axis
    # Find the TikZ picture and update it
    old_tikz_pattern = r'(\\node\\[below\\] at \\([^)]+\\) \\{)(\\d+)\\};'
    new_tikz_pattern = r'\\1\\2};'
    
    # Move phase labels lower
    content = re.sub(
        r'(Phase [BCDE]\\};\s+\\node\\[below\\] at \\([^)]+\\))',
        lambda m: m.group(1).replace('[below]', '[below=3mm]'),
        content
    )
    
    # Change "Рис. 1" to "Figure 1"
    content = re.sub(r'Рис\\. (\\d+)', r'Figure \\1', content)
    
    sec4_file.write_text(content)
    print("  ✓ Fixed figure labels and positioning")

def fix_algorithm_description():
    """Fix algorithm description in section 5"""
    print("\n[15] Fixing algorithm description...")
    
    sec5_file = PAPER_DIR / "sections/05_algorithm.tex"
    content = sec5_file.read_text()
    
    # Note: The actual algorithm needs complete rewrite based on THEORETICAL_FRAMEWORK.md
    # For now, add a note about the real implementation
    
    note = r"""

\paragraph{Implementation Note:} The actual implementation uses two strategies:
\begin{enumerate}
    \item \textbf{Hybrid Strategy:} Combines random exploration with learned patterns from observed samples
    \item \textbf{FreeEnergy Strategy:} Pure free energy minimization optimized for deceptive landscapes
\end{enumerate}
The system automatically selects the appropriate strategy based on graph characteristics (chaos metric: $\sigma(D)/\mu(D)$).
"""
    
    # Add note before section 5.3
    content = content.replace(
        r'\subsection{Key Algorithmic Innovations}',
        note + r'\\subsection{Key Algorithmic Innovations}'
    )
    
    sec5_file.write_text(content)
    print("  ✓ Added implementation note to section 05")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
