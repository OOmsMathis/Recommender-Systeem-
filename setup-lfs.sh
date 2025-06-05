#!/bin/bash

# Initialise Git LFS une seule fois sur la machine
git lfs install

# Suit les fichiers lourds
git lfs track "*.csv"
git lfs track "*.p"
git lfs track "*..jpg"
# Ajoute les modifications au suivi Git
git add .gitattributes

echo "✅ Git LFS configuré. Ajoute maintenant les fichiers lourds avec git add."
git add "C:\Users\Ooms Mathis\Documents\mlsmm2156"
