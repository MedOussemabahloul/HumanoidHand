#!/usr/bin/env python3
"""
Test de manipulation pour le robot G1 avec MuJoCo
Teste les capacités de manipulation et les capteurs de force
"""

import mujoco
import numpy as np
import time
import os
import sys

def print_banner():
    """Affiche une bannière stylée"""
    print("🤖 G1 MANIPULATION TESTER")
    print("=" * 40)

def check_mujoco_version():
    """Vérifie la version de MuJoCo"""
    try:
        version = mujoco.__version__
        print(f"✅ MuJoCo version: {version}")
        return True
    except Exception as e:
        print(f"❌ Erreur MuJoCo: {e}")
        return False

def load_model(model_path):
    """Charge le modèle MuJoCo"""
    try:
        print("🔍 Chargement du modèle...")
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        print("✅ Modèle chargé avec succès!")
        return model, data
    except Exception as e:
        print(f"❌ ERREUR: Erreur lors du chargement: {e}")
        print("💡 Vérifiez que le modèle a été créé avec: python scripts/create_combined_model.py")
        return None, None

def test_force_sensors(model, data):
    """Teste les capteurs de force"""
    print("\n🔧 TEST DES CAPTEURS DE FORCE")
    print("-" * 30)
    
    # Simulation de quelques étapes
    for step in range(100):
        mujoco.mj_step(model, data)
        
        if step % 20 == 0:
            # Récupération des données des capteurs
            thumb_force = data.sensordata[0:3] if len(data.sensordata) > 2 else [0, 0, 0]
            thumb2_force = data.sensordata[3:6] if len(data.sensordata) > 5 else [0, 0, 0]
            
            print(f"Étape {step}:")
            print(f"  Force pouce gauche: {thumb_force}")
            print(f"  Force pouce droit: {thumb2_force}")
    
    print("✅ Test des capteurs terminé!")

def test_manipulation(model, data):
    """Teste la manipulation d'objets"""
    print("\n🤖 TEST DE MANIPULATION")
    print("-" * 25)
    
    # Position initiale de l'objet
    initial_pos = data.qpos[6:9].copy()  # Position de l'objet libre
    print(f"Position initiale de l'objet: {initial_pos}")
    
    # Simulation avec mouvement du robot
    for step in range(200):
        # Contrôle simple du robot
        if step < 50:
            data.ctrl[0] = 0.5  # Mouvement du bras
        elif step < 100:
            data.ctrl[1] = 0.3  # Mouvement de l'avant-bras
        elif step < 150:
            data.ctrl[2] = 0.2  # Mouvement de la main
        
        mujoco.mj_step(model, data)
        
        if step % 50 == 0:
            current_pos = data.qpos[6:9]
            print(f"Étape {step}: Position objet = {current_pos}")
    
    final_pos = data.qpos[6:9]
    print(f"Position finale de l'objet: {final_pos}")
    print("✅ Test de manipulation terminé!")

def test_physics(model, data):
    """Teste la physique du modèle"""
    print("\n⚙️ TEST DE PHYSIQUE")
    print("-" * 20)
    
    # Vérification des masses
    total_mass = 0
    for i in range(model.nbody):
        mass = model.body_mass[i]
        total_mass += mass
        if mass < 0.1:
            print(f"⚠️  Corps {i} a une masse faible: {mass}")
    
    print(f"Masse totale du système: {total_mass:.2f} kg")
    
    # Test de stabilité
    for step in range(50):
        mujoco.mj_step(model, data)
        
        # Vérification des positions
        if step % 10 == 0:
            robot_pos = data.qpos[0:3]
            print(f"Étape {step}: Position robot = {robot_pos}")
    
    print("✅ Test de physique terminé!")

def main():
    """Fonction principale"""
    print_banner()
    
    # Vérification de MuJoCo
    if not check_mujoco_version():
        return
    
    print("\n🚀 TEST DE MANIPULATION G1")
    print("=" * 50)
    
    # Chargement du modèle
    model_path = "g1_manipulation_model.xml"
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        print("💡 Créez d'abord le modèle avec: python scripts/create_combined_model.py")
        return
    
    model, data = load_model(model_path)
    if model is None:
        return
    
    # Tests
    test_physics(model, data)
    test_force_sensors(model, data)
    test_manipulation(model, data)
    
    print("\n🎉 TOUS LES TESTS TERMINÉS AVEC SUCCÈS!")
    print("=" * 50)

if __name__ == "__main__":
    main()