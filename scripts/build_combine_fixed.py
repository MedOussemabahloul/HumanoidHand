#!/usr/bin/env python3
"""
Script pour construire un modèle MuJoCo combiné G1 (corps + doigts)
Version corrigée pour gérer correctement l'attribut frictionloss
"""

import xml.etree.ElementTree as ET
import os
import sys
from pathlib import Path

def clean_frictionloss_from_element(element):
    """
    Nettoie l'attribut frictionloss des éléments où il n'est pas autorisé
    
    Args:
        element: Élément XML à nettoyer
    """
    # Éléments où frictionloss est autorisé
    allowed_elements = ['joint', 'tendon', 'spatial', 'fixed']
    
    # Supprimer frictionloss si l'élément ne l'autorise pas
    if element.tag not in allowed_elements and 'frictionloss' in element.attrib:
        del element.attrib['frictionloss']
    
    # Traiter récursivement les enfants
    for child in element:
        clean_frictionloss_from_element(child)

def extract_joints_from_xml(xml_file):
    """
    Extrait tous les joints d'un fichier XML
    
    Args:
        xml_file (str): Chemin vers le fichier XML
        
    Returns:
        list: Liste des noms de joints
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        joints = []
        for joint in root.iter('joint'):
            joint_name = joint.get('name')
            if joint_name:
                joints.append(joint_name)
        
        return joints
        
    except Exception as e:
        print(f"⚠️ Erreur lors de l'extraction des joints de {xml_file}: {e}")
        return []

def create_combined_model(body_xml, fingers_xml, output_dir="results", output_name="g1_combined_fixed.xml"):
    """
    Crée un modèle MuJoCo combiné à partir des fichiers corps et doigts
    
    Args:
        body_xml (str): Chemin vers le fichier XML du corps
        fingers_xml (str): Chemin vers le fichier XML des doigts
        output_dir (str): Répertoire de sortie
        output_name (str): Nom du fichier de sortie
        
    Returns:
        str: Chemin vers le fichier créé
    """
    # Vérifier que les fichiers existent
    if not os.path.exists(body_xml):
        raise FileNotFoundError(f"Fichier corps non trouvé: {body_xml}")
    
    if not os.path.exists(fingers_xml):
        raise FileNotFoundError(f"Fichier doigts non trouvé: {fingers_xml}")
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    
    # Obtenir les chemins absolus
    abs_body = os.path.abspath(body_xml)
    abs_fingers = os.path.abspath(fingers_xml)
    
    print(f"📁 Création du modèle combiné:")
    print(f"   - Corps: {abs_body}")
    print(f"   - Doigts: {abs_fingers}")
    print(f"   - Sortie: {os.path.abspath(output_path)}")
    
    # Extraire les joints des deux fichiers
    body_joints = extract_joints_from_xml(body_xml)
    fingers_joints = extract_joints_from_xml(fingers_xml)
    all_joints = body_joints + fingers_joints
    
    print(f"🔧 Joints détectés:")
    print(f"   - Corps: {len(body_joints)} joints")
    print(f"   - Doigts: {len(fingers_joints)} joints")
    print(f"   - Total: {len(all_joints)} joints")
    
    # Créer le contenu XML combiné
    xml_content = f'''<?xml version="1.0"?>
<mujoco model="g1_combined_fixed">
  <!-- Modèle G1 combiné (corps + doigts) avec correction frictionloss -->
  
  <!-- Options de simulation optimisées -->
  <option timestep="0.002" iterations="50" tolerance="1e-10" solver="Newton" jacobian="auto">
    <flag warmstart="enable" energy="enable"/>
  </option>
  
  <!-- Paramètres de taille -->
  <size nconmax="100" njmax="1000" nstack="600000"/>
  
  <!-- Inclure les modèles de base -->
  <include file="{abs_body}"/>
  <include file="{abs_fingers}"/>

  <!-- Actuateurs automatiques pour tous les joints -->
  <actuator>'''
    
    # Ajouter les actuateurs pour tous les joints trouvés
    for joint_name in all_joints:
        xml_content += f'''
    <position name="act_{joint_name}" joint="{joint_name}" gear="1" kp="100" dampratio="1"/>'''
    
    xml_content += '''
  </actuator>
  
  <!-- Capteurs pour surveillance -->
  <sensor>'''
    
    # Ajouter des capteurs pour les joints principaux
    for joint_name in all_joints[:10]:  # Limiter à 10 capteurs pour éviter la surcharge
        xml_content += f'''
    <jointpos name="pos_{joint_name}" joint="{joint_name}"/>
    <jointvel name="vel_{joint_name}" joint="{joint_name}"/>'''
    
    xml_content += '''
  </sensor>
</mujoco>'''
    
    # Écrire le fichier
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)
    
    print(f"✅ Modèle combiné créé: {output_path}")
    
    # Vérifier et nettoyer le fichier créé
    try:
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        # Nettoyer les attributs frictionloss incorrects
        clean_frictionloss_from_element(root)
        
        # Réécrire le fichier nettoyé
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        print(f"🧹 Fichier nettoyé et optimisé")
        
    except Exception as e:
        print(f"⚠️ Avertissement lors du nettoyage: {e}")
    
    return output_path

def main():
    """Fonction principale"""
    print("🚀 GÉNÉRATEUR DE MODÈLE G1 COMBINÉ (VERSION CORRIGÉE)")
    print("=" * 60)
    
    # Paramètres par défaut
    default_body = "assets/hands/g1_body.xml"
    default_fingers = "assets/hands/g1_fingers.xml"
    
    # Permettre de spécifier des fichiers personnalisés
    if len(sys.argv) >= 3:
        body_xml = sys.argv[1]
        fingers_xml = sys.argv[2]
    else:
        body_xml = default_body
        fingers_xml = default_fingers
    
    try:
        # Créer le modèle combiné
        output_path = create_combined_model(body_xml, fingers_xml)
        
        print(f"\n🎉 Modèle combiné créé avec succès!")
        print(f"📄 Fichier: {output_path}")
        print(f"\n💡 Étapes suivantes:")
        print(f"   1. Testez le modèle: python scripts/test_stability_fixed.py")
        print(f"   2. Visualisez avec: python -c \"import mujoco.viewer; mujoco.viewer.launch_passive(mujoco.MjModel.from_xml_path('{output_path}'))\"")
        
    except Exception as e:
        print(f"❌ Erreur lors de la création du modèle: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()