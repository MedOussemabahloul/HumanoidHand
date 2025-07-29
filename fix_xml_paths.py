#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour corriger les chemins dans les fichiers XML MuJoCo
"""

import os
import re
import shutil

def fix_xml_paths():
    """Corrige les chemins dans les fichiers XML"""
    print("🔧 Correction des chemins XML...")
    
    # Chemins à corriger
    old_path = "/content/HumanoidHand/assets/hands/meshes/"
    new_path = "assets/hands/meshes/"
    
    xml_files = [
        "assets/hands/g1_body.xml",
        "assets/hands/g1_fingers.xml"
    ]
    
    for xml_file in xml_files:
        if os.path.exists(xml_file):
            print(f"   🔧 Correction de {xml_file}...")
            
            # Lire le fichier
            with open(xml_file, 'r') as f:
                content = f.read()
            
            # Remplacer les chemins
            content = content.replace(old_path, new_path)
            
            # Sauvegarder
            with open(xml_file, 'w') as f:
                f.write(content)
            
            print(f"   ✅ {xml_file} corrigé")
        else:
            print(f"   ❌ {xml_file} non trouvé")
    
    # Vérifier si le dossier meshes existe
    meshes_dir = "assets/hands/meshes"
    if not os.path.exists(meshes_dir):
        print(f"   ⚠️  Dossier {meshes_dir} non trouvé")
        print("   📁 Création d'un dossier meshes vide...")
        os.makedirs(meshes_dir, exist_ok=True)
        
        # Créer un fichier STL simple pour éviter les erreurs
        simple_stl = """solid simple_mesh
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid simple_mesh"""
        
        # Créer des fichiers STL simples pour tous les meshes référencés
        mesh_names = [
            "waist_yaw_link", "waist_roll_link", "torso_link", "logo_link", 
            "head_link", "waist_support_link", "left_shoulder_pitch_link",
            "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
            "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
            "left_rubber_hand", "right_shoulder_pitch_link", "right_shoulder_roll_link",
            "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
            "right_wrist_pitch_link", "right_wrist_yaw_link", "right_rubber_hand"
        ]
        
        for mesh_name in mesh_names:
            stl_file = os.path.join(meshes_dir, f"{mesh_name}.STL")
            with open(stl_file, 'w') as f:
                f.write(simple_stl)
            print(f"   ✅ Créé {stl_file}")
    
    print("✅ Correction des chemins terminée")

if __name__ == "__main__":
    fix_xml_paths()