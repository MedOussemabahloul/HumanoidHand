#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Force Sensors to G1 Model
Script pour ajouter des force sensors au modèle G1 pour améliorer la détection de contact
"""

import xml.etree.ElementTree as ET
import copy
import os

def add_force_sensors_to_model(input_path="results/g1_combined.xml", output_path="results/g1_combined_with_force_sensors.xml"):
    """Ajouter des force sensors au modèle G1"""
    
    print(f"Lecture du modèle: {input_path}")
    
    # Parser le XML
    tree = ET.parse(input_path)
    root = tree.getroot()
    
    # Trouver la section sensor
    sensor_section = root.find('sensor')
    if sensor_section is None:
        # Créer la section sensor si elle n'existe pas
        sensor_section = ET.SubElement(root, 'sensor')
    
    # Force sensors pour chaque doigt
    force_sensors = [
        # Main gauche
        {"name": "left_thumb_force_sensor_0", "site": "left_thumb_tip_site", "objtype": "site"},
        {"name": "left_thumb_force_sensor_1", "site": "left_thumb_tip_site", "objtype": "site"},
        {"name": "left_thumb_force_sensor_2", "site": "left_thumb_tip_site", "objtype": "site"},
        
        {"name": "left_index_force_sensor_0", "site": "left_index_tip_site", "objtype": "site"},
        {"name": "left_index_force_sensor_1", "site": "left_index_tip_site", "objtype": "site"},
        {"name": "left_index_force_sensor_2", "site": "left_index_tip_site", "objtype": "site"},
        
        {"name": "left_middle_force_sensor_0", "site": "left_middle_tip_site", "objtype": "site"},
        {"name": "left_middle_force_sensor_1", "site": "left_middle_tip_site", "objtype": "site"},
        {"name": "left_middle_force_sensor_2", "site": "left_middle_tip_site", "objtype": "site"},
        
        {"name": "left_ring_force_sensor_0", "site": "left_ring_tip_site", "objtype": "site"},
        {"name": "left_ring_force_sensor_1", "site": "left_ring_tip_site", "objtype": "site"},
        {"name": "left_ring_force_sensor_2", "site": "left_ring_tip_site", "objtype": "site"},
        
        # Main droite
        {"name": "right_thumb_force_sensor_0", "site": "right_thumb_tip_site", "objtype": "site"},
        {"name": "right_thumb_force_sensor_1", "site": "right_thumb_tip_site", "objtype": "site"},
        {"name": "right_thumb_force_sensor_2", "site": "right_thumb_tip_site", "objtype": "site"},
        
        {"name": "right_index_force_sensor_0", "site": "right_index_tip_site", "objtype": "site"},
        {"name": "right_index_force_sensor_1", "site": "right_index_tip_site", "objtype": "site"},
        {"name": "right_index_force_sensor_2", "site": "right_index_tip_site", "objtype": "site"},
        
        {"name": "right_middle_force_sensor_0", "site": "right_middle_tip_site", "objtype": "site"},
        {"name": "right_middle_force_sensor_1", "site": "right_middle_tip_site", "objtype": "site"},
        {"name": "right_middle_force_sensor_2", "site": "right_middle_tip_site", "objtype": "site"},
        
        {"name": "right_ring_force_sensor_0", "site": "right_ring_tip_site", "objtype": "site"},
        {"name": "right_ring_force_sensor_1", "site": "right_ring_tip_site", "objtype": "site"},
        {"name": "right_ring_force_sensor_2", "site": "right_ring_tip_site", "objtype": "site"},
    ]
    
    # Ajouter les force sensors
    sensors_added = 0
    for sensor_config in force_sensors:
        # Vérifier si le sensor existe déjà
        existing_sensor = sensor_section.find(f"force[@name='{sensor_config['name']}']")
        if existing_sensor is None:
            # Créer le force sensor
            force_sensor = ET.SubElement(sensor_section, 'force')
            force_sensor.set('name', sensor_config['name'])
            force_sensor.set('site', sensor_config['site'])
            sensors_added += 1
            print(f"Ajouté: {sensor_config['name']}")
        else:
            print(f"Déjà présent: {sensor_config['name']}")
    
    # Ajouter des contact sensors pour les doigts
    contact_sensors = [
        {"name": "left_thumb_contact", "geom1": "left_thumb_1_geom", "geom2": "cube_geom"},
        {"name": "left_index_contact", "geom1": "left_index_1_geom", "geom2": "cube_geom"},
        {"name": "left_middle_contact", "geom1": "left_middle_1_geom", "geom2": "cube_geom"},
        {"name": "left_ring_contact", "geom1": "left_ring_1_geom", "geom2": "cube_geom"},
        {"name": "right_thumb_contact", "geom1": "right_thumb_1_geom", "geom2": "cube_geom"},
        {"name": "right_index_contact", "geom1": "right_index_1_geom", "geom2": "cube_geom"},
        {"name": "right_middle_contact", "geom1": "right_middle_1_geom", "geom2": "cube_geom"},
        {"name": "right_ring_contact", "geom1": "right_ring_1_geom", "geom2": "cube_geom"},
    ]
    
    for sensor_config in contact_sensors:
        # Vérifier si le sensor existe déjà
        existing_sensor = sensor_section.find(f"contact[@name='{sensor_config['name']}']")
        if existing_sensor is None:
            # Créer le contact sensor
            contact_sensor = ET.SubElement(sensor_section, 'contact')
            contact_sensor.set('name', sensor_config['name'])
            contact_sensor.set('geom1', sensor_config['geom1'])
            contact_sensor.set('geom2', sensor_config['geom2'])
            sensors_added += 1
            print(f"Ajouté: {sensor_config['name']}")
        else:
            print(f"Déjà présent: {sensor_config['name']}")
    
    # Sauvegarder le modèle modifié
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    print(f"\nModèle sauvegardé: {output_path}")
    print(f"Sensors ajoutés: {sensors_added}")
    
    return output_path

def create_enhanced_grasp_model():
    """Créer un modèle amélioré pour le grasping"""
    
    # Ajouter les force sensors
    enhanced_model_path = add_force_sensors_to_model()
    
    print(f"\nModèle amélioré créé: {enhanced_model_path}")
    print("Ce modèle inclut maintenant des force sensors pour une meilleure détection de contact.")
    
    return enhanced_model_path

if __name__ == "__main__":
    create_enhanced_grasp_model()