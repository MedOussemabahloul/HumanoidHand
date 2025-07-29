#!/usr/bin/env python3
"""
Script pour créer un modèle MuJoCo combiné avec des masses appropriées
pour éviter l'erreur "body mass is too small, cannot compute center of mass"
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

def create_combined_model():
    """Crée un modèle MuJoCo combiné avec des masses appropriées"""
    
    # Définition du modèle avec des masses appropriées
    model_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="G1_Manipulation_Model">
  <compiler angle="radian" coordinate="local" meshdir="assets/meshes/"/>
  
  <!-- Définition des matériaux -->
  <asset>
    <material name="table_material" rgba="0.7 0.5 0.3 1"/>
    <material name="robot_material" rgba="0.2 0.2 0.8 1"/>
    <material name="object_material" rgba="0.8 0.2 0.2 1"/>
  </asset>
  
  <!-- Définition des corps avec des masses appropriées -->
  <worldbody>
    <!-- Table de base -->
    <body name="table" pos="0 0 0">
      <inertial pos="0 0 0.01" mass="10.0" diaginertia="0.1 0.1 0.1"/>
      <geom name="table_geom" type="box" size="0.5 0.5 0.02" 
            pos="0 0 0" material="table_material"/>
    </body>
    
    <!-- Robot G1 (simplifié) -->
    <body name="robot_base" pos="0 0 0.02">
      <inertial pos="0 0 0.1" mass="5.0" diaginertia="0.05 0.05 0.05"/>
      <geom name="base_geom" type="cylinder" size="0.1 0.05" 
            pos="0 0 0.1" material="robot_material"/>
      
      <!-- Bras principal -->
      <body name="arm_base" pos="0 0 0.2">
        <inertial pos="0 0 0.1" mass="2.0" diaginertia="0.02 0.02 0.02"/>
        <geom name="arm_geom" type="cylinder" size="0.05 0.2" 
              pos="0 0 0.1" material="robot_material"/>
        <joint name="arm_joint" type="hinge" axis="0 0 1" pos="0 0 0" range="-3.14 3.14"/>
        
        <!-- Avant-bras -->
        <body name="forearm" pos="0 0 0.4">
          <inertial pos="0 0 0.1" mass="1.5" diaginertia="0.015 0.015 0.015"/>
          <geom name="forearm_geom" type="cylinder" size="0.04 0.15" 
                pos="0 0 0.1" material="robot_material"/>
          <joint name="forearm_joint" type="hinge" axis="0 0 1" pos="0 0 0" range="-3.14 3.14"/>
          
          <!-- Main avec capteurs de force -->
          <body name="hand" pos="0 0 0.3">
            <inertial pos="0 0 0.05" mass="1.0" diaginertia="0.01 0.01 0.01"/>
            <geom name="hand_geom" type="box" size="0.03 0.03 0.05" 
                  pos="0 0 0.05" material="robot_material"/>
            <joint name="hand_joint" type="hinge" axis="0 0 1" pos="0 0 0" range="-3.14 3.14"/>
            
            <!-- Capteurs de force sur les doigts -->
            <body name="left_thumb_force_sensors" pos="0.02 0 0.05">
              <inertial pos="0 0 0.01" mass="0.1" diaginertia="0.001 0.001 0.001"/>
              <geom name="thumb_geom" type="sphere" size="0.01" 
                    pos="0 0 0.01" material="robot_material"/>
              <site name="thumb_force_site" pos="0 0 0.01" size="0.005"/>
            </body>
            
            <body name="right_thumb_force_sensors" pos="-0.02 0 0.05">
              <inertial pos="0 0 0.01" mass="0.1" diaginertia="0.001 0.001 0.001"/>
              <geom name="thumb2_geom" type="sphere" size="0.01" 
                    pos="0 0 0.01" material="robot_material"/>
              <site name="thumb2_force_site" pos="0 0 0.01" size="0.005"/>
            </body>
          </body>
        </body>
      </body>
    </body>
    
    <!-- Objet à manipuler -->
    <body name="target_object" pos="0.3 0 0.02">
      <inertial pos="0 0 0.02" mass="0.5" diaginertia="0.005 0.005 0.005"/>
      <geom name="object_geom" type="box" size="0.03 0.03 0.04" 
            pos="0 0 0.02" material="object_material"/>
      <joint name="object_joint" type="free"/>
    </body>
  </worldbody>
  
  <!-- Définition des actuateurs -->
  <actuator>
    <motor name="arm_motor" joint="arm_joint" gear="100"/>
    <motor name="forearm_motor" joint="forearm_joint" gear="100"/>
    <motor name="hand_motor" joint="hand_joint" gear="50"/>
  </actuator>
  
  <!-- Définition des capteurs -->
  <sensor>
    <force name="thumb_force" site="thumb_force_site"/>
    <force name="thumb2_force" site="thumb2_force_site"/>
  </sensor>
  
  <!-- Définition des contacts -->
  <contact>
    <pair name="hand_object" geom1="hand_geom" geom2="object_geom"/>
    <pair name="table_object" geom1="table_geom" geom2="object_geom"/>
  </contact>
</mujoco>'''
    
    # Créer le dossier assets/meshes s'il n'existe pas
    assets_dir = Path("assets/meshes")
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Écrire le modèle dans un fichier
    model_path = "g1_manipulation_model.xml"
    with open(model_path, 'w') as f:
        f.write(model_xml)
    
    print(f"✅ Modèle créé avec succès: {model_path}")
    print("📋 Caractéristiques du modèle:")
    print("   - Table: masse 10.0 kg")
    print("   - Robot base: masse 5.0 kg")
    print("   - Bras: masse 2.0 kg")
    print("   - Avant-bras: masse 1.5 kg")
    print("   - Main: masse 1.0 kg")
    print("   - Capteurs: masse 0.1 kg chacun")
    print("   - Objet: masse 0.5 kg")
    print("   - Toutes les masses sont > 0.01 kg pour éviter l'erreur de centre de masse")
    
    return model_path

if __name__ == "__main__":
    create_combined_model()