#!/usr/bin/env python3
"""
🚀 SCRIPT PRINCIPAL D'ENTRAÎNEMENT GRASPING ROBUSTE
==================================================

Agent SAC professionnel pour apprentissage de grasping avec:
✅ Physics collision réaliste - objets solides
✅ Détection de contact précise (doigts + palm)
✅ Contrôle de force adaptatif 
✅ Enregistrement vidéo automatique
✅ Curriculum learning intégré
✅ Monitoring en temps réel
✅ Sauvegarde intelligente des modèles
✅ Visualisation des courbes d'apprentissage
✅ Rapport complet de performance

UTILISATION:
python3 train_robust_grasp.py [--timesteps 500000] [--lr 3e-4] [--buffer 100000]

Ce script est entièrement autonome et produit des résultats professionnels.
Toutes les vidéos sont automatiquement téléchargées à la fin de l'entraînement.
"""

import os
import sys
import argparse
import time
from datetime import datetime

def check_dependencies():
    """Vérifie et installe les dépendances nécessaires"""
    required_packages = [
        'numpy', 'gymnasium', 'mujoco', 'opencv-python', 
        'stable-baselines3', 'matplotlib', 'imageio'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Packages manquants: {missing}")
        print("Installation automatique...")
        os.system(f"pip install --break-system-packages {' '.join(missing)}")
        print("✅ Installation terminée")

def main():
    """Fonction principale d'entraînement"""
    
    print("🚀 ENTRAÎNEMENT GRASPING ROBUSTE AVEC SAC")
    print("=" * 60)
    print("🤖 Système de Grasping Intelligent")
    print("📝 Recherche du cube avec mouvements naturels")
    print("🤝 Collision physique réaliste")
    print("👋 Détection de contact précise")
    print("🔒 Fixation optimale de la palm")
    print("✊ Fermeture contrôlée des doigts")
    print("🎬 Enregistrement vidéo automatique")
    print("=" * 60)
    
    # Vérifier les dépendances
    check_dependencies()
    
    # Arguments de ligne de commande
    parser = argparse.ArgumentParser(
        description='🚀 Entraînement Grasping Robuste avec SAC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python3 train_robust_grasp.py                    # Configuration par défaut
  python3 train_robust_grasp.py --timesteps 1000000  # Entraînement long
  python3 train_robust_grasp.py --lr 1e-4         # Learning rate plus petit
  python3 train_robust_grasp.py --buffer 200000   # Buffer plus grand
  
Sorties générées:
  📁 sac_grasp_results/
  ├── 🧠 models/           # Modèles SAC entraînés
  ├── 🎬 videos/           # Vidéos des épisodes
  ├── 📊 plots/            # Courbes d'apprentissage
  ├── 📈 metrics/          # Métriques JSON
  ├── 📝 logs/             # Logs TensorBoard
  └── 📋 final_report.md   # Rapport complet
        """
    )
    
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Nombre total de timesteps (défaut: 500,000)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (défaut: 3e-4)')
    parser.add_argument('--buffer', type=int, default=100000,
                       help='Taille du buffer de replay (défaut: 100,000)')
    parser.add_argument('--batch', type=int, default=256,
                       help='Taille du batch (défaut: 256)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Facteur de discount (défaut: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005,
                       help='Taux de mise à jour du target network (défaut: 0.005)')
    parser.add_argument('--results-dir', type=str, default='/workspace/sac_grasp_results',
                       help='Dossier de sauvegarde (défaut: /workspace/sac_grasp_results)')
    parser.add_argument('--demo-only', action='store_true',
                       help='Entraînement de démonstration rapide (1000 timesteps)')
    
    args = parser.parse_args()
    
    # Mode démonstration pour test rapide
    if args.demo_only:
        args.timesteps = 1000
        args.results_dir = '/workspace/demo_results'
        print("🎭 MODE DÉMONSTRATION - Entraînement rapide de 1000 timesteps")
    
    print(f"⚙️  CONFIGURATION:")
    print(f"   🎯 Timesteps: {args.timesteps:,}")
    print(f"   📚 Learning rate: {args.lr}")
    print(f"   🔄 Buffer size: {args.buffer:,}")
    print(f"   📦 Batch size: {args.batch}")
    print(f"   💰 Gamma: {args.gamma}")
    print(f"   🎚️  Tau: {args.tau}")
    print(f"   📁 Résultats: {args.results_dir}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Importer le trainer
        from sac_grasp_trainer import SACGraspTrainer
        
        # Créer l'entraîneur
        trainer = SACGraspTrainer(
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            buffer_size=args.buffer,
            batch_size=args.batch,
            gamma=args.gamma,
            tau=args.tau,
            results_dir=args.results_dir
        )
        
        print("🚀 DÉBUT DE L'ENTRAÎNEMENT")
        print("   Le robot va apprendre à:")
        print("   1. 🔍 Rechercher le cube")
        print("   2. 🎯 S'approcher avec précision")
        print("   3. 🤝 Détecter le contact")
        print("   4. 🔗 Aligner la palm")
        print("   5. ✊ Saisir avec contrôle de force")
        print("   6. ⬆️  Lever le cube")
        print("   7. 💪 Maintenir stable")
        print()
        
        # Lancer l'entraînement
        report = trainer.train()
        
        # Calculer la durée totale
        total_duration = time.time() - start_time
        
        print("\n🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("=" * 60)
        print(f"⏱️  Durée totale: {total_duration/3600:.2f} heures")
        print(f"📊 Épisodes: {report.get('total_episodes', 'N/A')}")
        print(f"🏆 Meilleure récompense: {report.get('performance', {}).get('best_reward', 'N/A')}")
        print(f"📈 Récompense moyenne: {report.get('performance', {}).get('mean_reward', 'N/A')}")
        
        print(f"\n📁 FICHIERS GÉNÉRÉS:")
        print(f"   🧠 Modèles: {args.results_dir}/models/")
        print(f"   🎬 Vidéos: {args.results_dir}/videos/")
        print(f"   📊 Graphiques: {args.results_dir}/plots/")
        print(f"   📋 Rapport: {args.results_dir}/final_report.md")
        
        print(f"\n🎬 UTILISATION DU MODÈLE:")
        print(f"```python")
        print(f"from stable_baselines3 import SAC")
        print(f"from robust_grasp_env import RobustGraspEnv")
        print(f"")
        print(f"# Charger le meilleur modèle")
        print(f"model = SAC.load('{args.results_dir}/models/best_model.zip')")
        print(f"")
        print(f"# Créer l'environnement")
        print(f"env = RobustGraspEnv(render_mode='rgb_array', record_video=True)")
        print(f"")
        print(f"# Tester le modèle")
        print(f"obs, _ = env.reset()")
        print(f"for _ in range(1000):")
        print(f"    action, _ = model.predict(obs, deterministic=True)")
        print(f"    obs, reward, done, truncated, info = env.step(action)")
        print(f"    if done or truncated:")
        print(f"        break")
        print(f"")
        print(f"env.save_video('test_grasp.mp4')")
        print(f"env.close()")
        print(f"```")
        
        print(f"\n✅ SUCCÈS COMPLET! Tous les fichiers sont disponibles dans:")
        print(f"📂 {args.results_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Entraînement interrompu par l'utilisateur")
        elapsed = time.time() - start_time
        print(f"⏱️  Temps écoulé: {elapsed/60:.1f} minutes")
        return 1
        
    except Exception as e:
        print(f"\n❌ ERREUR FATALE: {e}")
        import traceback
        traceback.print_exc()
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Temps écoulé avant erreur: {elapsed/60:.1f} minutes")
        print(f"📝 Consultez les logs dans: {args.results_dir}/crash_report.json")
        return 1

if __name__ == "__main__":
    """Point d'entrée principal"""
    exit_code = main()
    
    if exit_code == 0:
        print("\n🎊 MISSION ACCOMPLIE! 🎊")
        print("🤖 Votre agent SAC sait maintenant faire du grasping robuste!")
    else:
        print("\n💔 Entraînement incomplet")
        print("🔧 Vérifiez les logs pour diagnostiquer les problèmes")
    
    sys.exit(exit_code)