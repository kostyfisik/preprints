#!/bin/bash
echo Clean up...
#rm -f 2014-Ladutenko-Mie-optimization.bbl
for filename in `ls *.tex`; do
file=${filename::-4}
echo $file

rm -f *.bbl
rm -f *Notes.bib
rm -f *.aux
rm -f *.blg
rm -f *.log
rm -f *.out
rm -f $file.pdf


pdflatex  -interaction=nonstopmode "\input" $file.tex
bibtex $file
pdflatex  -interaction=nonstopmode "\input" $file.tex
pdflatex  -interaction=nonstopmode "\input" $file.tex
echo
echo
echo Check result for problems - they may be skiped during compilation

rm -f *Notes.bib
rm -f *.aux
rm -f *.blg
rm -f *.log
rm -f *.out
#rm -f 2014-Ladutenko-Mie-optimization.bbl
done
