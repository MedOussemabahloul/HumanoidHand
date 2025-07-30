
#!/usr/bin/env python3
"""
🚀 LANCEUR ULTRA-ROBUST SAC  PER TRAINING
=========================================

Script de lancement simple pour l'entraînement G1 grasp & lift
Utilise les configurations optimales et le système ultra-robuste
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="🚀 Ultra-Robust SACPER Training pour G1 Grasp & Lift",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c', 
        choices=['standard', 'quick', 'production', 'cpu', 'gpu'],
        default='standard',
        help='Configuration à utiliser'
    )
    
    # Mode
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Activer mode debug avec validation renforcée'
    )
    
    # GPU
    parser.add_argument(
        '--gpu', 
        type=int, 
        default=None,
        help='ID GPU à utiliser (auto-détection si non spécifié)'
    )
    
    # Resume
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Chemin checkpoint pour reprendre entraînement'
    )
    
    args = parser.parse_args()
    
    # Mapping configurations
    config_files = {
        'standard': 'config/sac_grasp_lift.yaml',
        'quick': 'config/train_config_quick.yaml', 
        'production': 'config/train_config_production.yaml',
        'cpu': 'config/train_config_cpu.yaml',
        'gpu': 'config/sac_grasp_lift_gpu.yaml'
    }
    
    config_path = config_files[args.config]
    
    print("🚀 ULTRA-ROBUST SAC  PER TRAINING SYSTEM")
    print("="*50)
    print(f"📝 Configuration: {args.config}")
    print(f"📁 Config file: {config_path}")
    print(f"🔧 Debug mode: {'ON' if args.debug else 'OFF'}")
    
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"🎮 GPU: {args.gpu}")
    
    if args.resume:
        print(f"🔄 Resume: {args.resume}")
    
    print("="*50)
    
    # Construction commande
    cmd_args = [
        'python', 'scripts/train_sac_per_ultra.py',
        '--config', config_path
    ]
    
    if args.debug:
        cmd_args.append('--debug')
    
    if args.resume:
        cmd_args.extend(['--resume', args.resume])
    
    # Vérifications
    if not os.path.exists(config_path):
        print(f"❌ ERREUR: Configuration non trouvée: {config_path}")
        return 1
    
    if not os.path.exists('scripts/train_sac_per_ultra.py'):
        print("❌ ERREUR: Script d'entraînement non trouvé: scripts/train_sac_per_ultra.py")
        return 1
    
    if not os.path.exists('results/g1_combined.xml'):
        print("❌ ERREUR: Modèle G1 non trouvé: results/g1_combined.xml")
        print("💡 Exécutez d'abord: python scripts/create_combined_model.py")
        return 1
    
    # Lancement
    print("🚀 Lancement entraînement...")
    print(f"📜 Commande: {' '.join(cmd_args)}")
    print()
    
    # Execution
    os.execvp('python', cmd_args)

if __name__ == "__main__":
    exit(main())
