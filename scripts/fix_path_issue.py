#!/usr/bin/env python3
"""
Script pour corriger le problème de chemin du fichier g1_combined_corrected.xml
"""

import xml.etree.ElementTree as ET
import os
import sys
from pathlib import Path
import shutil

def fix_path_and_create_corrected_model():
    """Corrige le problème de chemin et crée le modèle au bon endroit"""
    
    print("🔧 CORRECTION DU PROBLÈME DE CHEMIN")
    print("=" * 50)
    
    # Déterminer le répertoire de travail
    if os.getcwd().endswith('project'):
        project_root = Path.cwd()
    else:
        project_root = Path.cwd()
    
    print(f"📁 Répertoire de travail: {project_root}")
    
    # Chemins des fichiers
    body_file = project_root / "assets" / "hands" / "g1_body.xml"
    fingers_file = project_root / "assets" / "hands" / "g1_fingers.xml"
    
    # Créer le dossier results s'il n'existe pas
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / "g1_combined_corrected.xml"
    
    print(f"📂 Corps: {body_file}")
    print(f"🖐  Doigts: {fingers_file}")
    print(f"📄 Sortie: {output_file}")
    
    # Vérifier que les fichiers sources existent
    if not body_file.exists():
        print(f"❌ Fichier non trouvé: {body_file}")
        return False
        
    if not fingers_file.exists():
        print(f"❌ Fichier non trouvé: {fingers_file}")
        return False
    
    try:
        # Créer le modèle combiné avec chemins relatifs corrects
        root = ET.Element("mujoco", model="g1_combined_corrected")
        
        # Ajouter les includes avec chemins relatifs depuis results/
        include_body = ET.SubElement(root, "include", file="../assets/hands/g1_body.xml")
        include_fingers = ET.SubElement(root, "include", file="../assets/hands/g1_fingers.xml")
        
        # Lire les fichiers pour extraire les joints
        body_tree = ET.parse(body_file)
        fingers_tree = ET.parse(fingers_file)
        
        # Créer la section actuator
        actuator_section = ET.SubElement(root, "actuator")
        
        # Collecter tous les joints des deux modèles
        joints = []
        
        # Joints du corps
        for joint in body_tree.findall(".//joint"):
            joint_name = joint.get("name")
            if joint_name:
                joints.append(joint_name)
        
        # Joints des doigts
        for joint in fingers_tree.findall(".//joint"):
            joint_name = joint.get("name")
            if joint_name:
                joints.append(joint_name)
        
        # Créer les actuateurs pour tous les joints
        for joint_name in joints:
            actuator = ET.SubElement(actuator_section, "position")
            actuator.set("name", f"act_{joint_name}")
            actuator.set("joint", joint_name)
            actuator.set("gear", "1")
        
        # Sauvegarder le fichier au bon endroit
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        
        print(f"✅ Modèle combiné corrigé créé: {output_file}")
        print(f"   - Joints trouvés: {len(joints)}")
        print(f"   - Actuateurs créés: {len(joints)}")
        print(f"   - Fichier placé dans: {output_file.parent}")
        
        # Vérifier que le fichier a été créé
        if output_file.exists():
            file_size = output_file.stat().st_size
            print(f"   - Taille du fichier: {file_size} bytes")
            print(f"✅ Fichier créé avec succès!")
        else:
            print(f"❌ Échec de la création du fichier")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la création: {e}")
        return False

def verify_file_location():
    """Vérifie que le fichier est au bon endroit"""
    
    print("\n🔍 VÉRIFICATION DE L'EMPLACEMENT DU FICHIER")
    print("=" * 50)
    
    project_root = Path.cwd()
    expected_file = project_root / "results" / "g1_combined_corrected.xml"
    
    print(f"📍 Fichier attendu: {expected_file}")
    
    if expected_file.exists():
        print("✅ Fichier trouvé au bon endroit!")
        return True
    else:
        print("❌ Fichier non trouvé au bon endroit")
        
        # Chercher le fichier ailleurs
        possible_locations = [
            project_root / "assets" / "results" / "g1_combined_corrected.xml",
            project_root / "g1_combined_corrected.xml"
        ]
        
        for location in possible_locations:
            if location.exists():
                print(f"📍 Fichier trouvé à: {location}")
                print("💡 Déplacement du fichier...")
                try:
                    shutil.move(str(location), str(expected_file))
                    print("✅ Fichier déplacé avec succès!")
                    return True
                except Exception as e:
                    print(f"❌ Erreur lors du déplacement: {e}")
        
        return False

def main():
    """Fonction principale"""
    
    print("🚀 CORRECTION DU PROBLÈME DE CHEMIN G1_COMBINED_CORRECTED.XML")
    print("=" * 70)
    
    # Étape 1: Créer le fichier au bon endroit
    if not fix_path_and_create_corrected_model():
        print("❌ Échec de la création du modèle")
        return False
    
    # Étape 2: Vérifier l'emplacement
    if not verify_file_location():
        print("❌ Problème avec l'emplacement du fichier")
        return False
    
    print("\n🎉 PROBLÈME DE CHEMIN RÉSOLU!")
    print("=" * 50)
    print("✅ Le fichier g1_combined_corrected.xml est maintenant dans results/")
    print("✅ Votre script test_stability.py devrait maintenant fonctionner")
    
    print("\n💡 Testez maintenant avec:")
    print("   python scripts/test_stability.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)