#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de lancement sûr pour l'entraînement SAC
"""

import os
import sys
import subprocess
import argparse

def setup_environment():
    """Configure l'environnement pour éviter les segfaults"""
    env_vars = {
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"   {var}={value}")

def main():
    parser = argparse.ArgumentParser(description="Lancement entraînement SAC sûr")
    parser.add_argument("--config", default="quick", help="Configuration à utiliser")
    parser.add_argument("--debug", action="store_true", help="Mode debug")
    
    args = parser.parse_args()
    
    print("🚀 ULTRA-ROBUST SAC PER TRAINING SYSTEM (SAFE)")
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
        return 1
    
    # Configuration de l'environnement
    print("🔧 Configuration de l'environnement...")
    setup_environment()
    
    # Utiliser le fichier XML simple
    simple_xml = "assets/simple_robot.xml"
    
    if not os.path.exists(simple_xml):
        print(f"❌ Fichier {simple_xml} non trouvé!")
        return 1
    
    print("✅ Fichier XML simple trouvé")
    
    # Construire la commande avec le fichier XML simple
    cmd = [
        sys.executable, "scripts/train_rl.py",
        "--config", config_file,
        "--body_xml", simple_xml,
        "--fingers_xml", simple_xml,
        "--output_dir", "results"
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    print("🚀 Lancement entraînement...")
    print(f"📜 Commande: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        # Exécuter la commande
        result = subprocess.run(cmd, check=True, cwd=os.getcwd())
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