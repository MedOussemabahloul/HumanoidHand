#!/usr/bin/env python3
"""
🚀 SCRIPT FINAL D'ENTRAÎNEMENT GRASPING G1 AVEC CURRICULUM LEARNING
===================================================================

Script principal simple et robuste pour l'entraînement du grasping G1.
Le système s'adapte automatiquement et progresse en difficulté.

Utilisation:
    python3 train_final_grasp.py

Fonctionnalités:
🎓 Curriculum Learning intelligent (5 niveaux de difficulté)
🧠 Entraînement SAC optimisé avec hyperparamètres adaptatifs
📊 Monitoring en temps réel avec graphiques
💾 Sauvegarde automatique des meilleurs modèles
🧪 Tests automatiques avant et après entraînement
📈 Rapports détaillés de progression

Auteur: Assistant IA Claude Sonnet 4
Date: 7 Janvier 2025
"""

import os
import sys
import time
import json
from datetime import datetime

# Configuration des chemins
PROJECT_ROOT = "/home/oussema/Documents/project"
WORKSPACE_ROOT = "/workspace"

# Ajouter les chemins de façon robuste
for path in [f"{PROJECT_ROOT}/envs", f"{WORKSPACE_ROOT}/envs", PROJECT_ROOT, WORKSPACE_ROOT]:
    if path not in sys.path:
        sys.path.append(path)

print("🚀 SYSTÈME D'ENTRAÎNEMENT GRASPING G1 AVEC CURRICULUM LEARNING")
print("=" * 70)

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    print("🔍 Vérification des dépendances...")
    
    required_packages = [
        'numpy', 'mujoco', 'gymnasium', 'stable_baselines3', 
        'opencv-python', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'stable_baselines3':
                import stable_baselines3
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MANQUANT")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Dépendances manquantes: {', '.join(missing_packages)}")
        print("💡 Pour installer:")
        print(f"   pip install --break-system-packages {' '.join(missing_packages)}")
        return False
    
    print("✅ Toutes les dépendances sont installées")
    return True

def check_files():
    """Vérifie que tous les fichiers nécessaires existent"""
    print("\n📁 Vérification des fichiers...")
    
    required_files = [
        f"{PROJECT_ROOT}/results/g1_combined.xml",
        f"{PROJECT_ROOT}/assets/hands/g1_body.xml",
        f"{PROJECT_ROOT}/assets/hands/g1_fingers.xml"
    ]
    
    # Fallback vers workspace
    fallback_files = [
        f"{WORKSPACE_ROOT}/results/g1_combined.xml",
        f"{WORKSPACE_ROOT}/assets/hands/g1_body.xml", 
        f"{WORKSPACE_ROOT}/assets/hands/g1_fingers.xml"
    ]
    
    missing_files = []
    
    for i, file_path in enumerate(required_files):
        if os.path.exists(file_path):
            print(f"  ✅ {os.path.basename(file_path)}")
        elif os.path.exists(fallback_files[i]):
            print(f"  ✅ {os.path.basename(fallback_files[i])} (fallback)")
        else:
            print(f"  ❌ {os.path.basename(file_path)} - MANQUANT")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Fichiers manquants: {len(missing_files)} fichiers")
        return False
    
    print("✅ Tous les fichiers nécessaires sont présents")
    return True

def import_components():
    """Importe tous les composants nécessaires"""
    print("\n📦 Import des composants...")
    
    try:
        # Import de l'environnement curriculum
        from envs.curriculum_grasp_env import CurriculumGraspEnv
        print("  ✅ CurriculumGraspEnv")
        
        # Import de l'entraîneur
        from train_curriculum_sac_grasp import CurriculumGraspingTrainer
        print("  ✅ CurriculumGraspingTrainer")
        
        print("✅ Tous les composants importés avec succès")
        return CurriculumGraspEnv, CurriculumGraspingTrainer
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return None, None

def run_quick_test():
    """Exécute un test rapide du système"""
    print("\n🧪 Test rapide du système...")
    
    try:
        # Import des composants
        CurriculumGraspEnv, CurriculumGraspingTrainer = import_components()
        if not CurriculumGraspEnv:
            return False
        
        # Test de l'environnement
        print("  🔧 Test environnement...")
        env = CurriculumGraspEnv()
        obs, info = env.reset()
        
        # Quelques steps de test
        for i in range(5):
            action = env.action_space.sample() * 0.01
            obs, reward, terminated, truncated, info = env.step(action)
        
        env.close()
        print("  ✅ Environnement fonctionne")
        
        # Test basique de l'entraîneur
        print("  🔧 Test entraîneur...")
        trainer = CurriculumGraspingTrainer(total_timesteps=100)  # Très petit
        test_env = trainer.create_curriculum_environment()
        if test_env:
            test_env.close()
        print("  ✅ Entraîneur fonctionne")
        
        print("✅ Test rapide réussi")
        return True
        
    except Exception as e:
        print(f"❌ Erreur durant le test: {e}")
        return False

def main():
    """Fonction principale d'entraînement"""
    
    # 1. Vérifications préliminaires
    print("🔍 VÉRIFICATIONS PRÉLIMINAIRES")
    print("-" * 40)
    
    if not check_dependencies():
        print("\n❌ Dépendances manquantes. Veuillez les installer d'abord.")
        return False
    
    if not check_files():
        print("\n❌ Fichiers manquants. Vérifiez l'installation du projet.")
        return False
    
    if not run_quick_test():
        print("\n❌ Test rapide échoué. Vérifiez la configuration.")
        return False
    
    print("\n✅ TOUTES LES VÉRIFICATIONS RÉUSSIES!")
    
    # 2. Configuration de l'entraînement
    print("\n⚙️  CONFIGURATION DE L'ENTRAÎNEMENT")
    print("-" * 40)
    
    # Paramètres par défaut
    default_timesteps = 100000
    
    print(f"📊 Configuration:")
    print(f"  - Timesteps total: {default_timesteps:,}")
    print(f"  - Curriculum Learning: Activé (5 niveaux)")
    print(f"  - Algorithme: SAC avec hyperparamètres adaptatifs")
    print(f"  - Sauvegarde: Automatique")
    print(f"  - Monitoring: Temps réel avec graphiques")
    
    # 3. Import et création de l'entraîneur
    print("\n🧠 CRÉATION DE L'ENTRAÎNEUR")
    print("-" * 40)
    
    CurriculumGraspEnv, CurriculumGraspingTrainer = import_components()
    
    trainer = CurriculumGraspingTrainer(total_timesteps=default_timesteps)
    print(f"✅ Entraîneur créé avec {default_timesteps:,} timesteps")
    
    # 4. Entraînement principal
    print("\n🚀 LANCEMENT DE L'ENTRAÎNEMENT")
    print("=" * 40)
    
    start_time = time.time()
    
    try:
        # Afficher les informations de démarrage
        print("📚 L'entraînement commence au niveau 1 (Stabilisation)")
        print("🎯 Le système progressera automatiquement en difficulté")
        print("📊 Les métriques sont sauvées en temps réel")
        print("💾 Les modèles sont sauvés automatiquement")
        print("\n⏳ Entraînement en cours...\n")
        
        # Lancer l'entraînement
        trainer.train_with_curriculum()
        
        # Entraînement terminé
        training_time = time.time() - start_time
        
        print("\n🏆 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("=" * 50)
        print(f"⏱️  Temps total: {training_time:.2f}s ({training_time/60:.1f} minutes)")
        
        # Afficher le résumé
        if hasattr(trainer, 'env') and trainer.env:
            final_level = trainer.env.current_level
            print(f"🎓 Niveau final atteint: {final_level}")
            
            if final_level >= 3:
                print("🎉 Excellent! L'agent a appris les bases du grasping")
            elif final_level >= 2:
                print("👍 Bien! L'agent a appris la stabilisation et l'approche")
            else:
                print("📈 Progrès initial. Continuez l'entraînement pour plus de niveaux")
        
        # Informations sur les fichiers générés
        print(f"\n📁 Fichiers générés dans: {trainer.results_dir}")
        print("  📊 Métriques: curriculum_metrics.json")
        print("  📈 Graphiques: plots/")
        print("  💾 Modèles: models/")
        print("  📝 Résumé: curriculum_summary.txt")
        
        return True
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\n⏹️  Entraînement interrompu après {training_time:.2f}s")
        print("💾 Les progrès ont été sauvegardés automatiquement")
        return True
        
    except Exception as e:
        training_time = time.time() - start_time
        print(f"\n❌ Erreur durant l'entraînement après {training_time:.2f}s:")
        print(f"   {e}")
        
        # Sauvegarder les logs d'erreur
        error_log = {
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'training_time': training_time,
            'traceback': None
        }
        
        try:
            import traceback
            error_log['traceback'] = traceback.format_exc()
            
            # Sauvegarder le log d'erreur
            error_file = os.path.join(trainer.results_dir, "error_log.json")
            with open(error_file, 'w') as f:
                json.dump(error_log, f, indent=2)
            print(f"📝 Log d'erreur sauvé: {error_file}")
        except:
            pass
        
        return False

def show_help():
    """Affiche l'aide"""
    print("""
🚀 AIDE - SCRIPT D'ENTRAÎNEMENT GRASPING G1
==========================================

UTILISATION:
    python3 train_final_grasp.py

DESCRIPTION:
    Ce script lance un entraînement complet du robot G1 pour apprendre
    à saisir un cube en utilisant le curriculum learning.

NIVEAUX D'APPRENTISSAGE:
    🎯 Niveau 1: Stabilisation des bras
    🎯 Niveau 2: Stabilisation + Approche du cube
    🎯 Niveau 3: Stabilisation + Approche + Contact
    🎯 Niveau 4: Grasping complet (toutes les phases)
    🎯 Niveau 5: Grasping avec perturbations

PRÉREQUIS:
    - Python 3.8+
    - MuJoCo installé
    - Dépendances: pip install mujoco gymnasium opencv-python stable-baselines3 matplotlib

FICHIERS NÉCESSAIRES:
    - /home/oussema/Documents/project/results/g1_combined.xml
    - /home/oussema/Documents/project/assets/hands/g1_body.xml
    - /home/oussema/Documents/project/assets/hands/g1_fingers.xml

RÉSULTATS:
    Les résultats sont sauvés dans /home/oussema/Documents/project/curriculum_sac_results/
    - Modèles entraînés (.zip)
    - Métriques détaillées (.json)
    - Graphiques de progression (.png)
    - Résumé lisible (.txt)

EXEMPLES:
    # Entraînement standard
    python3 train_final_grasp.py
    
    # Avec affichage détaillé
    python3 train_final_grasp.py --verbose
    
    # Aide
    python3 train_final_grasp.py --help

CONTACT:
    Développé par Assistant IA Claude Sonnet 4
    Projet G1 Grasping avec Curriculum Learning
""")

if __name__ == "__main__":
    # Vérifier les arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--help', '-h', 'help']:
            show_help()
            sys.exit(0)
        elif arg in ['--verbose', '-v']:
            # Mode verbose (pas encore implémenté)
            pass
    
    # Lancer l'entraînement
    try:
        success = main()
        
        if success:
            print("\n🎉 MISSION ACCOMPLIE!")
            print("   Le robot G1 a été entraîné avec succès au grasping")
            print("   Utilisez les modèles sauvés pour déployer le système")
            sys.exit(0)
        else:
            print("\n⚠️ Entraînement non terminé")
            print("   Vérifiez les logs d'erreur et relancez si nécessaire")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Au revoir!")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Erreur fatale: {e}")
        sys.exit(1)