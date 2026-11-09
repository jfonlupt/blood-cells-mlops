# Compilation de report.tex : pdflatex + biber (biblatex, backend=biber)
$pdf_mode   = 1;      # produire un PDF via pdflatex
$bibtex_use = 2;      # lancer l'outil de biblio si nécessaire, nettoyer le .bbl

# latexmk détecte report.bcf et lance biber automatiquement.
# Ne PAS passer d'option "-biber" à latexmk : elle n'existe pas.
