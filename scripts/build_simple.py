#!/usr/bin/env python3
"""
Script simple pour construire le fichier MJCF combiné sans dépendances
"""

import os
import xml.etree.ElementTree as ET

def build_combined_xml_simple(body_xml, fingers_xml, out_dir):
    """Version simplifiée sans dépendances"""
    os.makedirs(out_dir, exist_ok=True)
    combined_path = os.path.join(out_dir, "g1_combined.xml")
    
    # Vérifier si la version optimisée existe
    fingers_dir = os.path.dirname(fingers_xml)
    optimized_fingers = os.path.join(fingers_dir, "g1_fingers_optimized.xml")
    if os.path.exists(optimized_fingers):
        print(f"[INFO] Utilisation de la version optimisée: {optimized_fingers}")
        fingers_xml = optimized_fingers

    abs_body = os.path.abspath(body_xml)
    abs_finger = os.path.abspath(fingers_xml)

    # Parser les deux XML pour extraire la liste de joints
    tree_b = ET.parse(abs_body)
    tree_f = ET.parse(abs_finger)
    joints = {j.attrib["name"] for j in tree_b.findall(".//joint") if "name" in j.attrib}
    joints.update({j.attrib["name"] for j in tree_f.findall(".//joint") if "name" in j.attrib})

    # Construire le fichier combiné
    with open(combined_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<mujoco model="g1_combined">\n')

        # inclure body + fingers
        f.write(f'  <include file="{abs_body}"/>\n')
        f.write(f'  <include file="{abs_finger}"/>\n\n')

        # générer automatiquement les actuators
        f.write('  <actuator>\n')
        for jn in sorted(joints):
            f.write(f'    <position name="act_{jn}" joint="{jn}" gear="1"/>\n')
        f.write('  </actuator>\n')

        f.write('</mujoco>\n')

    print(f"[INFO] Combined MJCF written to {combined_path}")
    return combined_path

if __name__ == "__main__":
    body_xml = "assets/hands/g1_body.xml"
    fingers_xml = "assets/hands/g1_fingers.xml"
    out_dir = "results"
    
    model_path = build_combined_xml_simple(body_xml, fingers_xml, out_dir)
    print(f"✅ Modèle construit: {model_path}")