#!/usr/bin/env python3
"""
Script pour corriger les chemins des meshes dans les fichiers XML G1
Corrige les chemins absolus vers des chemins relatifs corrects
"""

import xml.etree.ElementTree as ET
import os
import sys
from pathlib import Path
import shutil

def fix_mesh_paths_in_file(xml_file):
    """Corrige les chemins des meshes dans un fichier XML"""
    
    print(f"🔧 Correction des chemins dans: {xml_file}")
    
    # Créer une sauvegarde
    backup_file = xml_file.with_suffix(xml_file.suffix + '.mesh_backup')
    if not backup_file.exists():
        shutil.copy2(xml_file, backup_file)
        print(f"💾 Sauvegarde créée: {backup_file}")
    
    try:
        # Parser le fichier XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        corrections_count = 0
        
        # Trouver tous les éléments mesh dans la section asset
        for mesh in root.findall(".//mesh"):
            file_attr = mesh.get("file")
            if file_attr:
                # Vérifier si c'est un chemin absolu problématique
                if file_attr.startswith("/content/HumanoidHand/assets/hands/meshes/"):
                    # Extraire juste le nom du fichier
                    mesh_filename = Path(file_attr).name
                    
                    # Nouveau chemin relatif depuis le fichier XML
                    new_path = f"meshes/{mesh_filename}"
                    
                    print(f"  🔄 {mesh.get('name', 'sans_nom')}: {file_attr} → {new_path}")
                    
                    # Mettre à jour le chemin
                    mesh.set("file", new_path)
                    corrections_count += 1
                
                elif file_attr.startswith("/content/") or file_attr.startswith("/home/"):
                    # Autres chemins absolus problématiques
                    mesh_filename = Path(file_attr).name
                    new_path = f"meshes/{mesh_filename}"
                    
                    print(f"  🔄 {mesh.get('name', 'sans_nom')}: {file_attr} → {new_path}")
                    mesh.set("file", new_path)
                    corrections_count += 1
        
        if corrections_count > 0:
            # Sauvegarder le fichier corrigé
            ET.indent(tree, space="  ", level=0)
            tree.write(xml_file, encoding="utf-8", xml_declaration=True)
            print(f"  ✅ {corrections_count} chemins corrigés")
        else:
            print(f"  ℹ️  Aucun chemin à corriger")
        
        return corrections_count > 0
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def verify_mesh_files_exist():
    """Vérifie que les fichiers mesh existent"""
    
    print("\n🔍 VÉRIFICATION DES FICHIERS MESH")
    print("=" * 50)
    
    project_root = Path.cwd()
    meshes_dir = project_root / "assets" / "hands" / "meshes"
    
    print(f"📁 Répertoire des meshes: {meshes_dir}")
    
    if not meshes_dir.exists():
        print("❌ Répertoire des meshes non trouvé!")
        print("💡 Vérifiez que vos meshes sont dans assets/hands/meshes/")
        return False
    
    # Lister les fichiers mesh
    mesh_files = list(meshes_dir.glob("*.STL")) + list(meshes_dir.glob("*.stl"))
    
    if mesh_files:
        print(f"✅ {len(mesh_files)} fichiers mesh trouvés:")
        for mesh_file in sorted(mesh_files)[:5]:  # Afficher les 5 premiers
            print(f"   - {mesh_file.name}")
        if len(mesh_files) > 5:
            print(f"   ... et {len(mesh_files) - 5} autres")
        return True
    else:
        print("❌ Aucun fichier mesh (.STL) trouvé!")
        return False

def create_corrected_combined_model():
    """Crée un nouveau modèle combiné avec les chemins corrigés"""
    
    print("\n🚀 CRÉATION DU MODÈLE COMBINÉ AVEC CHEMINS CORRIGÉS")
    print("=" * 60)
    
    project_root = Path.cwd()
    body_file = project_root / "assets" / "hands" / "g1_body.xml"
    fingers_file = project_root / "assets" / "hands" / "g1_fingers.xml"
    
    # Créer le dossier results s'il n'existe pas
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / "g1_combined_corrected.xml"
    
    try:
        # Créer le modèle combiné
        root = ET.Element("mujoco", model="g1_combined_corrected")
        
        # Ajouter les includes avec chemins relatifs
        include_body = ET.SubElement(root, "include", file="../assets/hands/g1_body.xml")
        include_fingers = ET.SubElement(root, "include", file="../assets/hands/g1_fingers.xml")
        
        # Lire les fichiers pour extraire les joints
        body_tree = ET.parse(body_file)
        fingers_tree = ET.parse(fingers_file)
        
        # Créer la section actuator
        actuator_section = ET.SubElement(root, "actuator")
        
        # Collecter tous les joints
        joints = []
        
        for joint in body_tree.findall(".//joint"):
            joint_name = joint.get("name")
            if joint_name:
                joints.append(joint_name)
        
        for joint in fingers_tree.findall(".//joint"):
            joint_name = joint.get("name")
            if joint_name:
                joints.append(joint_name)
        
        # Créer les actuateurs
        for joint_name in joints:
            actuator = ET.SubElement(actuator_section, "position")
            actuator.set("name", f"act_{joint_name}")
            actuator.set("joint", joint_name)
            actuator.set("gear", "1")
        
        # Sauvegarder
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        
        print(f"✅ Modèle combiné créé: {output_file}")
        print(f"   - Joints: {len(joints)}")
        print(f"   - Actuateurs: {len(joints)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la création: {e}")
        return False

def main():
    """Fonction principale"""
    
    print("🚀 CORRECTION DES CHEMINS DES MESHES G1")
    print("=" * 60)
    
    project_root = Path.cwd()
    
    # Fichiers à corriger
    files_to_fix = [
        project_root / "assets" / "hands" / "g1_body.xml",
        project_root / "assets" / "hands" / "g1_fingers.xml"
    ]
    
    # Étape 1: Vérifier les meshes
    if not verify_mesh_files_exist():
        print("\n❌ Problème avec les fichiers mesh")
        print("💡 Assurez-vous que vos fichiers .STL sont dans assets/hands/meshes/")
        return False
    
    # Étape 2: Corriger les chemins dans les fichiers XML
    print(f"\n🔧 CORRECTION DES FICHIERS XML")
    print("=" * 50)
    
    corrections_made = False
    for xml_file in files_to_fix:
        if xml_file.exists():
            if fix_mesh_paths_in_file(xml_file):
                corrections_made = True
        else:
            print(f"⚠️  Fichier non trouvé: {xml_file}")
    
    # Étape 3: Créer le modèle combiné corrigé
    if not create_corrected_combined_model():
        print("❌ Échec de la création du modèle combiné")
        return False
    
    # Résumé
    print(f"\n🎉 CORRECTION TERMINÉE!")
    print("=" * 50)
    
    if corrections_made:
        print("✅ Chemins des meshes corrigés")
    
    print("✅ Modèle combiné créé avec chemins corrects")
    print("✅ Fichiers de sauvegarde créés (.mesh_backup)")
    
    print(f"\n💡 Testez maintenant avec:")
    print("   python scripts/test_stability.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)