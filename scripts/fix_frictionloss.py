#!/usr/bin/env python3
"""
Script pour corriger le problème avec l'attribut frictionloss dans g1_fingers.xml
L'attribut frictionloss n'est supporté que dans les éléments <joint> et <tendon>,
pas dans <general> ou d'autres éléments.
"""

import xml.etree.ElementTree as ET
import os
import sys

def fix_frictionloss_in_xml(input_file, output_file=None):
    """
    Corrige le problème avec l'attribut frictionloss dans un fichier XML MuJoCo
    
    Args:
        input_file (str): Chemin vers le fichier XML d'entrée
        output_file (str): Chemin vers le fichier XML de sortie (optionnel)
    """
    if not os.path.exists(input_file):
        print(f"Erreur: Le fichier {input_file} n'existe pas")
        return False
    
    try:
        # Parser le fichier XML
        tree = ET.parse(input_file)
        root = tree.getroot()
        
        print(f"🔍 Analyse du fichier: {input_file}")
        
        # Compteur des modifications
        modifications = 0
        
        # Fonction récursive pour parcourir tous les éléments
        def process_element(element):
            nonlocal modifications
            
            # Vérifier si l'élément a l'attribut frictionloss
            if 'frictionloss' in element.attrib:
                # Éléments où frictionloss est autorisé
                allowed_elements = ['joint', 'tendon', 'spatial', 'fixed']
                
                if element.tag not in allowed_elements:
                    print(f"⚠️  Suppression de frictionloss dans <{element.tag}>")
                    del element.attrib['frictionloss']
                    modifications += 1
                else:
                    print(f"✅ Attribut frictionloss conservé dans <{element.tag}>")
            
            # Traiter récursivement les enfants
            for child in element:
                process_element(child)
        
        # Traiter tous les éléments
        process_element(root)
        
        # Ajouter frictionloss aux joints s'ils n'en ont pas
        for joint in root.iter('joint'):
            if 'frictionloss' not in joint.attrib:
                # Ajouter une valeur par défaut raisonnable
                joint.set('frictionloss', '0.01')
                print(f"➕ Ajout de frictionloss='0.01' au joint: {joint.get('name', 'sans_nom')}")
                modifications += 1
        
        if modifications == 0:
            print("✅ Aucune modification nécessaire")
            return True
        
        # Déterminer le fichier de sortie
        if output_file is None:
            # Créer une sauvegarde
            backup_file = input_file + '.backup'
            if not os.path.exists(backup_file):
                os.rename(input_file, backup_file)
                print(f"💾 Sauvegarde créée: {backup_file}")
            output_file = input_file
        
        # Sauvegarder le fichier modifié
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        print(f"✅ Fichier corrigé sauvegardé: {output_file}")
        print(f"📊 Total des modifications: {modifications}")
        
        return True
        
    except ET.ParseError as e:
        print(f"❌ Erreur de parsing XML: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return False

def main():
    """Fonction principale"""
    print("🚀 CORRECTEUR D'ATTRIBUT FRICTIONLOSS POUR MUJOCO")
    print("=" * 60)
    
    # Vérifier les arguments
    if len(sys.argv) < 2:
        print("Usage: python fix_frictionloss.py <fichier_xml> [fichier_sortie]")
        print("Exemple: python fix_frictionloss.py assets/hands/g1_fingers.xml")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Corriger le fichier
    success = fix_frictionloss_in_xml(input_file, output_file)
    
    if success:
        print("\n🎉 Correction terminée avec succès!")
        print("\nMaintenant, vous pouvez tester votre modèle avec:")
        print("python test_stability.py")
    else:
        print("\n❌ Échec de la correction")
        sys.exit(1)

if __name__ == "__main__":
    main()