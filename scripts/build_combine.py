#!/usr/bin/env python3
"""
Script pour construire un modèle MuJoCo combiné G1 (corps + doigts)
Version originale qui peut causer le problème frictionloss
"""

import xml.etree.ElementTree as ET
import os
import sys
from pathlib import Path

def create_combined_model():
    """Crée un modèle combiné G1 corps + doigts"""
    
    print("🚀 CONSTRUCTION DU MODÈLE G1 COMBINÉ")
    print("=" * 50)
    
    # Chemins des fichiers
    workspace_path = Path(__file__).parent.parent
    body_file = workspace_path / "assets" / "hands" / "g1_body.xml"
    fingers_file = workspace_path / "assets" / "hands" / "g1_fingers.xml"
    output_file = workspace_path / "results" / "g1_combined.xml"
    
    # Vérifier que les fichiers existent
    if not body_file.exists():
        print(f"❌ Fichier non trouvé: {body_file}")
        return False
        
    if not fingers_file.exists():
        print(f"❌ Fichier non trouvé: {fingers_file}")
        return False
    
    # Créer le dossier results s'il n'existe pas
    output_file.parent.mkdir(exist_ok=True)
    
    print(f"📂 Corps: {body_file}")
    print(f"🖐  Doigts: {fingers_file}")
    print(f"📄 Sortie: {output_file}")
    
    # Créer le modèle combiné
    root = ET.Element("mujoco", model="g1_combined")
    
    # Ajouter les includes avec chemins absolus (peut causer des problèmes)
    include_body = ET.SubElement(root, "include", file=str(body_file.absolute()))
    include_fingers = ET.SubElement(root, "include", file=str(fingers_file.absolute()))
    
    # Lire les fichiers pour extraire les joints
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
        print(f"✅ Modèle combiné créé: {output_file}")
        print(f"   - Joints trouvés: {len(joints)}")
        print(f"   - Actuateurs créés: {len(joints)}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur d'écriture: {e}")
        return False

if __name__ == "__main__":
    success = create_combined_model()
    
    if success:
        print("\n💡 Testez le modèle avec:")
        print("   python scripts/test_stability.py")
    
    sys.exit(0 if success else 1)