#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de lancement pour l'entraînement SAC
"""

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Lancement entraînement SAC")
    parser.add_argument("--config", default="quick", help="Configuration à utiliser")
    parser.add_argument("--debug", action="store_true", help="Mode debug")
    
    args = parser.parse_args()
    
    print("🚀 ULTRA-ROBUST SAC PER TRAINING SYSTEM")
    print("=" * 50)
    print(f"📝 Configuration: {args.config}")
    
    # Déterminer le fichier de config
    if args.config == "quick":
        config_file = "config/train_config_quick.yaml"
    else:
        config_file = f"config/train_config_{args.config}.yaml"
    
    print(f"📁 Config file: {config_file}")
    print(f"🔧 Debug mode: {'ON' if args.debug else 'OFF'}")
    print("=" * 50)
    
    # Vérifier que le fichier de config existe
    if not os.path.exists(config_file):
        print(f"❌ Fichier de configuration {config_file} non trouvé!")
        print("📁 Fichiers disponibles dans config/:")
        config_dir = "config"
        if os.path.exists(config_dir):
            for f in os.listdir(config_dir):
                if f.endswith('.yaml') or f.endswith('.yml'):
                    print(f"   - {f}")
        else:
            print("   Aucun fichier de configuration trouvé")
        return 1
    
    # Construire la commande
    cmd = [
        "python", "scripts/train_rl.py",
        "--config", config_file,
        "--body_xml", "assets/hands/g1_body.xml",
        "--fingers_xml", "assets/hands/g1_fingers.xml",
        "--output_dir", "results"
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    print("🚀 Lancement entraînement...")
    print(f"📜 Commande: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        # Exécuter la commande
        result = subprocess.run(cmd, check=True)
        print("✅ Entraînement terminé avec succès!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'exécution: {e}")
        return e.returncode
    except FileNotFoundError:
        print("❌ Script train_rl.py non trouvé dans scripts/")
        return 1

if __name__ == "__main__":
    sys.exit(main())