#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_xml.py

Script simple pour valider la structure XML sans dépendances MuJoCo
"""

import xml.etree.ElementTree as ET
import os
import sys

def validate_xml_structure(xml_path):
    """Valide la structure XML basique"""
    print(f"🔍 Validation de: {xml_path}")
    
    try:
        # Parser le XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        if root.tag != "mujoco":
            print(f"❌ Root tag incorrect: {root.tag} (attendu: mujoco)")
            return False
            
        print(f"✅ XML bien formé, root: {root.tag}")
        
        # Compter les éléments principaux
        elements = {
            'joint': len(root.findall(".//joint")),
            'geom': len(root.findall(".//geom")),
            'body': len(root.findall(".//body")),
            'site': len(root.findall(".//site")),
            'sensor': len(root.findall(".//sensor/*")),
            'actuator': len(root.findall(".//actuator/*")),
        }
        
        print("📊 Éléments trouvés:")
        for name, count in elements.items():
            print(f"   {name}: {count}")
            
        # Vérifier les références à des matériaux inexistants
        problematic_refs = []
        for geom in root.findall(".//geom"):
            material = geom.get("material")
            if material and material not in ["groundplane"]:  # matériaux connus OK
                # Vérifier si le matériau est défini
                if not root.find(f".//material[@name='{material}']"):
                    problematic_refs.append(material)
        
        if problematic_refs:
            print(f"⚠️  Références matériaux manquants: {set(problematic_refs)}")
        else:
            print("✅ Pas de références matériaux manquants")
            
        # Vérifier les plages de joints
        joint_ranges = []
        for joint in root.findall(".//joint"):
            range_attr = joint.get("range")
            name = joint.get("name", "unnamed")
            if range_attr:
                try:
                    min_val, max_val = map(float, range_attr.split())
                    joint_ranges.append((name, min_val, max_val))
                    
                    # Vérifier si les plages sont raisonnables
                    if max_val > 2.0:  # > 114 degrés
                        print(f"⚠️  {name}: plage excessive {max_val:.2f} rad ({max_val*57.3:.0f}°)")
                    elif max_val > 1.6:  # > 91 degrés
                        print(f"⚠️  {name}: plage importante {max_val:.2f} rad ({max_val*57.3:.0f}°)")
                        
                except ValueError:
                    print(f"❌ {name}: plage invalide '{range_attr}'")
        
        if joint_ranges:
            print(f"✅ {len(joint_ranges)} joints avec plages validées")
        
        return True
        
    except ET.ParseError as e:
        print(f"❌ Erreur de parsing XML: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def compare_files(original_path, optimized_path):
    """Compare la structure entre fichier original et optimisé"""
    print(f"\n🔄 Comparaison {os.path.basename(original_path)} vs {os.path.basename(optimized_path)}")
    
    try:
        orig_tree = ET.parse(original_path)
        opt_tree = ET.parse(optimized_path)
        
        # Compter les éléments
        for file_name, tree in [("Original", orig_tree), ("Optimisé", opt_tree)]:
            elements = {
                'joints': len(tree.findall(".//joint")),
                'capteurs': len(tree.findall(".//sensor/*")),
                'actuateurs': len(tree.findall(".//actuator/*")),
            }
            print(f"  {file_name}: {elements}")
            
        return True
        
    except Exception as e:
        print(f"❌ Erreur de comparaison: {e}")
        return False

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Fichiers à valider
    files_to_check = [
        "assets/hands/g1_fingers.xml",
        "assets/hands/g1_fingers_optimized.xml",
        "assets/hands/g1_body.xml"
    ]
    
    print("=" * 60)
    print("🚀 VALIDATION XML STRUCTURE")
    print("=" * 60)
    
    all_valid = True
    
    for file_path in files_to_check:
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            valid = validate_xml_structure(full_path)
            all_valid = all_valid and valid
            print()
        else:
            print(f"⚠️  Fichier non trouvé: {file_path}")
            print()
    
    # Comparaison entre original et optimisé
    original = os.path.join(project_root, "assets/hands/g1_fingers.xml")
    optimized = os.path.join(project_root, "assets/hands/g1_fingers_optimized.xml")
    
    if os.path.exists(original) and os.path.exists(optimized):
        compare_files(original, optimized)
    
    print("=" * 60)
    if all_valid:
        print("🎉 VALIDATION RÉUSSIE - Structure XML correcte!")
        print("✅ Prêt pour utilisation avec train_rl.py")
    else:
        print("❌ PROBLÈMES DÉTECTÉS - Vérifiez les erreurs ci-dessus")
    print("=" * 60)
    
    return 0 if all_valid else 1

if __name__ == "__main__":
    sys.exit(main())