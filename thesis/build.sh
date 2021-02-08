#!/bin/bash
xelatex thesis.tex &&
biber thesis &&
xelatex thesis.tex
# xelatex thesis.tex
