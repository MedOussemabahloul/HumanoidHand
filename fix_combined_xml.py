#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour corriger les chemins dans le fichier XML combiné
"""

import os
import re

def fix_combined_xml():
    """Corrige les chemins dans le fichier XML combiné"""
    print("🔧 Correction du fichier XML combiné...")
    
    combined_xml = "results/g1_combined.xml"
    
    if not os.path.exists(combined_xml):
        print(f"❌ Fichier {combined_xml} non trouvé!")
        return False
    
    # Lire le fichier
    with open(combined_xml, 'r') as f:
        content = f.read()
    
    print("   📄 Fichier XML combiné trouvé")
    
    # Remplacer les chemins absolus par des chemins relatifs
    # Chercher les patterns comme file="/workspace/assets/hands/g1_body.xml"
    content = re.sub(
        r'file="[^"]*assets/hands/g1_body\.xml"',
        'file="assets/hands/g1_body.xml"',
        content
    )
    
    content = re.sub(
        r'file="[^"]*assets/hands/g1_fingers\.xml"',
        'file="assets/hands/g1_fingers.xml"',
        content
    )
    
    # Sauvegarder
    with open(combined_xml, 'w') as f:
        f.write(content)
    
    print("   ✅ Chemins corrigés dans le fichier XML combiné")
    return True

if __name__ == "__main__":
    fix_combined_xml()