#!/usr/bin/env python3
"""
Script pour corriger le problème frictionloss dans g1_combined.xml
Résout l'erreur: Schema violation: unrecognized attribute: 'frictionloss'
"""

import xml.etree.ElementTree as ET
import os
import sys
from pathlib import Path
import shutil

def fix_frictionloss_in_included_files():
    """
    Corrige les attributs frictionloss dans les fichiers inclus
    """
    print("🔧 CORRECTION DES FICHIERS INCLUS")
    print("=" * 40)
    
    workspace_path = Path(__file__).parent.parent
    body_file = workspace_path / "assets" / "hands" / "g1_body.xml"
    fingers_file = workspace_path / "assets" / "hands" / "g1_fingers.xml"
    
    files_to_fix = [body_file, fingers_file]
    
    for file_path in files_to_fix:
        if not file_path.exists():
            print(f"⚠️  Fichier non trouvé: {file_path}")
            continue
            
        print(f"🔍 Analyse: {file_path}")
        
        # Créer une sauvegarde
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
        if not backup_path.exists():
            shutil.copy2(file_path, backup_path)
            print(f"💾 Sauvegarde créée: {backup_path}")
        
        try:
            # Parser le fichier XML
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Statistiques
            removed_count = 0
            kept_count = 0
            added_count = 0
            
            # Éléments où frictionloss est autorisé
            allowed_elements = {'joint', 'tendon', 'spatial', 'fixed'}
            
            # Parcourir tous les éléments
            for element in root.iter():
                if 'frictionloss' in element.attrib:
                    if element.tag in allowed_elements:
                        kept_count += 1
                        print(f"  ✅ Conservé frictionloss dans <{element.tag}> {element.get('name', '')}")
                    else:
                        del element.attrib['frictionloss']
                        removed_count += 1
                        print(f"  ❌ Supprimé frictionloss de <{element.tag}> {element.get('name', '')}")
            
            # Ajouter frictionloss aux joints qui n'en ont pas
            for joint in root.findall('.//joint'):
                if 'frictionloss' not in joint.attrib:
                    joint.set('frictionloss', '0.01')
                    added_count += 1
                    print(f"  ➕ Ajouté frictionloss='0.01' au joint: {joint.get('name', 'sans_nom')}")
            
            # Sauvegarder le fichier corrigé
            ET.indent(tree, space="  ", level=0)
            tree.write(file_path, encoding="utf-8", xml_declaration=True)
            
            print(f"  📊 Résultats:")
            print(f"     - Supprimés: {removed_count}")
            print(f"     - Conservés: {kept_count}")
            print(f"     - Ajoutés: {added_count}")
            print(f"  ✅ Fichier corrigé: {file_path}")
            
        except ET.ParseError as e:
            print(f"  ❌ Erreur de parsing: {e}")
            return False
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            return False
    
    return True

def create_corrected_combined_model():
    """
    Crée un modèle combiné avec chemins relatifs et corrections
    """
    print("\n🚀 CRÉATION DU MODÈLE COMBINÉ CORRIGÉ")
    print("=" * 50)
    
    workspace_path = Path(__file__).parent.parent
    output_file = workspace_path / "results" / "g1_combined_corrected.xml"
    
    # Créer le dossier results s'il n'existe pas
    output_file.parent.mkdir(exist_ok=True)
    
    # Créer le modèle combiné avec chemins relatifs
    root = ET.Element("mujoco", model="g1_combined_corrected")
    
    # Ajouter les includes avec chemins relatifs
    include_body = ET.SubElement(root, "include", file="../assets/hands/g1_body.xml")
    include_fingers = ET.SubElement(root, "include", file="../assets/hands/g1_fingers.xml")
    
    # Lire les fichiers pour extraire les joints
    body_file = workspace_path / "assets" / "hands" / "g1_body.xml"
    fingers_file = workspace_path / "assets" / "hands" / "g1_fingers.xml"
    
    try:
        body_tree = ET.parse(body_file)
        fingers_tree = ET.parse(fingers_file)
    except ET.ParseError as e:
        print(f"❌ Erreur de parsing XML: {e}")
        return False
    
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
    
    # Sauvegarder le fichier
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    
    try:
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        print(f"✅ Modèle combiné corrigé créé: {output_file}")
        print(f"   - Joints trouvés: {len(joints)}")
        print(f"   - Actuateurs créés: {len(joints)}")
        print(f"   - Chemins relatifs utilisés")
        return True
        
    except Exception as e:
        print(f"❌ Erreur d'écriture: {e}")
        return False

def main():
    """Fonction principale"""
    print("🚀 CORRECTEUR FRICTIONLOSS POUR G1 COMBINED")
    print("=" * 60)
    
    # Étape 1: Corriger les fichiers inclus
    if not fix_frictionloss_in_included_files():
        print("❌ Échec de la correction des fichiers inclus")
        return False
    
    # Étape 2: Créer le modèle combiné corrigé
    if not create_corrected_combined_model():
        print("❌ Échec de la création du modèle combiné")
        return False
    
    print("\n🎉 CORRECTION TERMINÉE AVEC SUCCÈS!")
    print("=" * 50)
    print("✅ Fichiers corrigés:")
    print("   - assets/hands/g1_body.xml")
    print("   - assets/hands/g1_fingers.xml")
    print("   - results/g1_combined_corrected.xml")
    print("\n💡 Testez le modèle avec:")
    print("   python scripts/test_stability.py")
    print("   (modifiez le chemin vers g1_combined_corrected.xml)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)