#!/bin/bash
# apt-get install librsvg2-bin
for file in `ls *.svg`; do
    rsvg-convert -f pdf -o ${file%.svg}.pdf $file
    pdfcrop ${file%.svg}.pdf ${file%.svg}.pdf
    echo ${file%.svg}.pdf
done
