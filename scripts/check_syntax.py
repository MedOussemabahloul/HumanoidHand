#!/usr/bin/env python3
"""
Script pour vérifier la syntaxe de tous les scripts Python
"""

import os
import py_compile
import sys
from pathlib import Path

def check_python_syntax(file_path):
    """Vérifie la syntaxe d'un fichier Python"""
    try:
        py_compile.compile(file_path, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)

def main():
    """Fonction principale"""
    print("🔍 VÉRIFICATION DE LA SYNTAXE DES SCRIPTS PYTHON")
    print("=" * 60)
    
    workspace_path = Path(__file__).parent.parent
    scripts_dir = workspace_path / "scripts"
    
    # Trouver tous les fichiers Python
    python_files = list(scripts_dir.glob("*.py"))
    
    if not python_files:
        print("❌ Aucun fichier Python trouvé dans scripts/")
        return False
    
    print(f"📁 Vérification de {len(python_files)} fichiers Python...")
    print()
    
    all_valid = True
    
    for py_file in sorted(python_files):
        print(f"🔍 {py_file.name}...", end=" ")
        
        is_valid, error = check_python_syntax(py_file)
        
        if is_valid:
            print("✅ OK")
        else:
            print("❌ ERREUR")
            print(f"   {error}")
            all_valid = False
    
    print()
    print("=" * 60)
    
    if all_valid:
        print("🎉 TOUS LES FICHIERS SONT SYNTAXIQUEMENT CORRECTS!")
        print("✅ Aucune erreur de syntaxe détectée")
    else:
        print("❌ ERREURS DE SYNTAXE DÉTECTÉES")
        print("💡 Corrigez les erreurs ci-dessus avant de continuer")
    
    return all_valid

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)