#!/usr/bin/env python3
"""
Script simple pour tester la validité XML des modèles MuJoCo
Ne nécessite pas l'installation de MuJoCo
"""

import xml.etree.ElementTree as ET
import os
import sys

def test_xml_validity(xml_file):
    """
    Teste la validité XML d'un fichier
    
    Args:
        xml_file (str): Chemin vers le fichier XML
        
    Returns:
        tuple: (success, error_message)
    """
    try:
        print(f"🔍 Test de validité XML: {xml_file}")
        
        if not os.path.exists(xml_file):
            return False, f"Fichier non trouvé: {xml_file}"
        
        # Parser le fichier XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Statistiques du modèle
        stats = {
            'root_element': root.tag,
            'bodies': len(list(root.iter('body'))),
            'joints': len(list(root.iter('joint'))),
            'geoms': len(list(root.iter('geom'))),
            'actuators': len(list(root.iter('actuator'))),
            'sensors': len(list(root.iter('sensor'))),
            'includes': len(list(root.iter('include')))
        }
        
        print(f"✅ XML valide!")
        print(f"   - Élément racine: {stats['root_element']}")
        print(f"   - Corps: {stats['bodies']}")
        print(f"   - Joints: {stats['joints']}")
        print(f"   - Géométries: {stats['geoms']}")
        print(f"   - Actuateurs: {stats['actuators']}")
        print(f"   - Capteurs: {stats['sensors']}")
        print(f"   - Inclusions: {stats['includes']}")
        
        # Vérifier les attributs frictionloss problématiques
        problematic_elements = []
        for element in root.iter():
            if 'frictionloss' in element.attrib:
                allowed_elements = ['joint', 'tendon', 'spatial', 'fixed']
                if element.tag not in allowed_elements:
                    problematic_elements.append(f"{element.tag} (ligne inconnue)")
        
        if problematic_elements:
            print(f"⚠️  Attributs frictionloss problématiques trouvés:")
            for elem in problematic_elements:
                print(f"     - {elem}")
            return False, f"Attributs frictionloss dans des éléments non autorisés: {problematic_elements}"
        else:
            print(f"✅ Aucun attribut frictionloss problématique")
        
        return True, None
        
    except ET.ParseError as e:
        return False, f"Erreur de parsing XML: {e}"
    except Exception as e:
        return False, f"Erreur inattendue: {e}"

def main():
    """Fonction principale"""
    print("=" * 60)
    print("🚀 TEST DE VALIDITÉ XML POUR MODÈLES MUJOCO")
    print("=" * 60)
    
    # Fichiers à tester
    test_files = [
        "assets/hands/g1_fingers.xml",
        "assets/hands/g1_body.xml",
        "results/g1_combined.xml"
    ]
    
    # Si un fichier est spécifié en argument
    if len(sys.argv) > 1:
        test_files = sys.argv[1:]
    
    all_valid = True
    
    for xml_file in test_files:
        print(f"\n{'='*40}")
        print(f"FICHIER: {xml_file}")
        print(f"{'='*40}")
        
        success, error = test_xml_validity(xml_file)
        
        if not success:
            print(f"❌ Erreur: {error}")
            all_valid = False
        else:
            print(f"✅ Fichier valide")
    
    print(f"\n{'='*60}")
    if all_valid:
        print("🎉 Tous les fichiers XML sont valides!")
        print("💡 Le problème frictionloss semble être résolu.")
        print("💡 Vous pouvez maintenant essayer de charger les modèles dans MuJoCo.")
    else:
        print("❌ Certains fichiers ont des problèmes.")
        print("💡 Utilisez le script fix_frictionloss.py pour corriger les problèmes.")

if __name__ == "__main__":
    main()