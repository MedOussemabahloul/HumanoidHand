#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de lancement sécurisé pour l'entraînement SAC
Version corrigée pour éviter les segmentation faults
"""

import os
import sys
import subprocess
import argparse
import yaml
from pathlib import Path

def print_banner():
    """Affiche la bannière du système"""
    print("🚀 ULTRA-ROBUST SAC PER TRAINING SYSTEM (VERSION SÉCURISÉE)")
    print("=" * 60)
    print("🔧 Corrections apportées:")
    print("   • Gestion mémoire améliorée")
    print("   • Validation des données")
    print("   • Gestion des erreurs robuste")
    print("   • Configuration sécurisée")
    print("   • Gradient clipping")
    print("   • Nettoyage mémoire automatique")
    print("=" * 60)

def check_dependencies():
    """Vérifie les dépendances requises"""
    print("🔍 Vérification des dépendances...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch non installé")
        return False
    
    try:
        import mujoco
        print(f"✅ MuJoCo: {mujoco.__version__}")
    except ImportError:
        print("❌ MuJoCo non installé")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy non installé")
        return False
    
    try:
        import yaml
        print("✅ PyYAML")
    except ImportError:
        print("❌ PyYAML non installé")
        return False
    
    return True

def create_safe_config(config_name):
    """Crée une configuration sécurisée"""
    safe_configs = {
        "quick": {
            "task": {
                "cube_body_name": "cube",
                "max_steps_per_episode": 100,
                "touch_sensors": [],
                "force_sensors": [],
                "include_orientation_reward": False,
                "force_reward_weight_normal": 0.0,
                "force_reward_weight_tangential": 0.0,
                "translation_penalty_weight": 0.0,
                "output_dir": "results",
                "save_freq_steps": 1000
            },
            "rl": {
                "gamma": 0.99,
                "alpha": 0.2,
                "learning_rate": 3e-4,
                "hidden_size": 256,
                "batch_size": 32,
                "replay_size": 10000,
                "start_steps": 100,
                "update_after": 100,
                "update_every": 1,
                "num_updates": 1,
                "total_steps": 1000,
                "tau": 0.005,
                "act_limit": 1.0
            }
        },
        "medium": {
            "task": {
                "cube_body_name": "cube",
                "max_steps_per_episode": 200,
                "touch_sensors": [],
                "force_sensors": [],
                "include_orientation_reward": False,
                "force_reward_weight_normal": 0.0,
                "force_reward_weight_tangential": 0.0,
                "translation_penalty_weight": 0.0,
                "output_dir": "results",
                "save_freq_steps": 2000
            },
            "rl": {
                "gamma": 0.99,
                "alpha": 0.2,
                "learning_rate": 3e-4,
                "hidden_size": 512,
                "batch_size": 64,
                "replay_size": 50000,
                "start_steps": 200,
                "update_after": 200,
                "update_every": 1,
                "num_updates": 2,
                "total_steps": 5000,
                "tau": 0.005,
                "act_limit": 1.0
            }
        },
        "full": {
            "task": {
                "cube_body_name": "cube",
                "max_steps_per_episode": 500,
                "touch_sensors": [],
                "force_sensors": [],
                "include_orientation_reward": True,
                "force_reward_weight_normal": 0.1,
                "force_reward_weight_tangential": 0.05,
                "translation_penalty_weight": 0.01,
                "output_dir": "results",
                "save_freq_steps": 10000
            },
            "rl": {
                "gamma": 0.99,
                "alpha": 0.2,
                "learning_rate": 3e-4,
                "hidden_size": 1024,
                "batch_size": 128,
                "replay_size": 100000,
                "start_steps": 1000,
                "update_after": 1000,
                "update_every": 1,
                "num_updates": 4,
                "total_steps": 100000,
                "tau": 0.005,
                "act_limit": 1.0
            }
        }
    }
    
    if config_name not in safe_configs:
        print(f"❌ Configuration '{config_name}' non reconnue")
        print(f"   Configurations disponibles: {list(safe_configs.keys())}")
        return None
    
    return safe_configs[config_name]

def save_config(config, config_path):
    """Sauvegarde la configuration dans un fichier YAML"""
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"✅ Configuration sauvegardée: {config_path}")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde de la config: {e}")
        return False

def find_xml_files():
    """Trouve les fichiers XML nécessaires"""
    xml_files = {
        "body_xml": None,
        "fingers_xml": None
    }
    
    # Chercher dans les dossiers typiques
    search_dirs = ["assets", "models", "xml", "urdf", "."]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        for file in os.listdir(search_dir):
            file_path = os.path.join(search_dir, file)
            if file.endswith('.xml'):
                if 'body' in file.lower() or 'g1' in file.lower():
                    xml_files["body_xml"] = file_path
                elif 'finger' in file.lower() or 'hand' in file.lower():
                    xml_files["fingers_xml"] = file_path
    
    # Si pas trouvé, créer des fichiers XML minimaux
    if xml_files["body_xml"] is None:
        xml_files["body_xml"] = create_minimal_body_xml()
    
    if xml_files["fingers_xml"] is None:
        xml_files["fingers_xml"] = create_minimal_fingers_xml()
    
    return xml_files

def create_minimal_body_xml():
    """Crée un fichier XML minimal pour le body"""
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="g1_body">
  <worldbody>
    <!-- Base du robot -->
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.1 0.05" rgba="0.5 0.5 0.5 1"/>
    </body>
    
    <!-- Bras gauche -->
    <body name="left_arm" pos="0 0.1 0" parent="base">
      <joint name="left_shoulder" type="hinge" axis="0 0 1" range="-180 180"/>
      <geom type="capsule" size="0.02 0.1" rgba="0.8 0.2 0.2 1"/>
    </body>
    
    <!-- Bras droit -->
    <body name="right_arm" pos="0 -0.1 0" parent="base">
      <joint name="right_shoulder" type="hinge" axis="0 0 1" range="-180 180"/>
      <geom type="capsule" size="0.02 0.1" rgba="0.2 0.2 0.8 1"/>
    </body>
    
    <!-- Cube à saisir -->
    <body name="cube" pos="0.3 0 0.1">
      <geom type="box" size="0.05 0.05 0.05" rgba="1 1 0 1"/>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="left_motor" joint="left_shoulder" gear="100"/>
    <motor name="right_motor" joint="right_shoulder" gear="100"/>
  </actuator>
</mujoco>'''
    
    xml_path = "assets/g1_body_minimal.xml"
    os.makedirs("assets", exist_ok=True)
    
    with open(xml_path, 'w') as f:
        f.write(xml_content)
    
    print(f"✅ Fichier XML body créé: {xml_path}")
    return xml_path

def create_minimal_fingers_xml():
    """Crée un fichier XML minimal pour les doigts"""
    xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="g1_fingers">
  <worldbody>
    <!-- Doigts de la main gauche -->
    <body name="left_finger1" pos="0.2 0.15 0">
      <joint name="left_finger1_joint" type="hinge" axis="0 1 0" range="0 90"/>
      <geom type="capsule" size="0.01 0.05" rgba="0.9 0.1 0.1 1"/>
    </body>
    
    <body name="left_finger2" pos="0.2 0.05 0">
      <joint name="left_finger2_joint" type="hinge" axis="0 1 0" range="0 90"/>
      <geom type="capsule" size="0.01 0.05" rgba="0.9 0.1 0.1 1"/>
    </body>
    
    <!-- Doigts de la main droite -->
    <body name="right_finger1" pos="0.2 -0.05 0">
      <joint name="right_finger1_joint" type="hinge" axis="0 1 0" range="0 90"/>
      <geom type="capsule" size="0.01 0.05" rgba="0.1 0.1 0.9 1"/>
    </body>
    
    <body name="right_finger2" pos="0.2 -0.15 0">
      <joint name="right_finger2_joint" type="hinge" axis="0 1 0" range="0 90"/>
      <geom type="capsule" size="0.01 0.05" rgba="0.1 0.1 0.9 1"/>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="left_finger1_motor" joint="left_finger1_joint" gear="50"/>
    <motor name="left_finger2_motor" joint="left_finger2_joint" gear="50"/>
    <motor name="right_finger1_motor" joint="right_finger1_joint" gear="50"/>
    <motor name="right_finger2_motor" joint="right_finger2_joint" gear="50"/>
  </actuator>
</mujoco>'''
    
    xml_path = "assets/g1_fingers_minimal.xml"
    os.makedirs("assets", exist_ok=True)
    
    with open(xml_path, 'w') as f:
        f.write(xml_content)
    
    print(f"✅ Fichier XML fingers créé: {xml_path}")
    return xml_path

def run_training(config_path, body_xml, fingers_xml, output_dir):
    """Lance l'entraînement avec le script sécurisé"""
    print("🚀 Lancement entraînement...")
    
    # Commande à exécuter
    cmd = [
        sys.executable,  # Python executable
        "scripts/train_rl_safe.py",
        "--config", config_path,
        "--body_xml", body_xml,
        "--fingers_xml", fingers_xml,
        "--output_dir", output_dir
    ]
    
    print(f"📜 Commande: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        # Exécuter la commande
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Entraînement terminé avec succès!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Entraînement interrompu par l'utilisateur")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        return False

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Lancement sécurisé de l'entraînement SAC")
    parser.add_argument("--config", default="quick", choices=["quick", "medium", "full"],
                       help="Configuration d'entraînement (default: quick)")
    parser.add_argument("--output_dir", default="results",
                       help="Répertoire de sortie (default: results)")
    parser.add_argument("--debug", action="store_true",
                       help="Mode debug")
    
    args = parser.parse_args()
    
    # Affichage de la bannière
    print_banner()
    
    # Vérification des dépendances
    if not check_dependencies():
        print("❌ Dépendances manquantes. Veuillez les installer.")
        sys.exit(1)
    
    # Création de la configuration sécurisée
    print(f"📝 Configuration: {args.config}")
    config = create_safe_config(args.config)
    if config is None:
        sys.exit(1)
    
    # Sauvegarde de la configuration
    config_path = f"config/train_config_{args.config}_safe.yaml"
    if not save_config(config, config_path):
        sys.exit(1)
    
    # Recherche des fichiers XML
    print("🔍 Recherche des fichiers XML...")
    xml_files = find_xml_files()
    print(f"📁 Body XML: {xml_files['body_xml']}")
    print(f"📁 Fingers XML: {xml_files['fingers_xml']}")
    
    # Création du répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Lancement de l'entraînement
    success = run_training(config_path, xml_files["body_xml"], xml_files["fingers_xml"], args.output_dir)
    
    if success:
        print("🎉 Entraînement terminé avec succès!")
        print(f"📁 Résultats dans: {args.output_dir}")
    else:
        print("💥 Échec de l'entraînement")
        sys.exit(1)

if __name__ == "__main__":
    main()