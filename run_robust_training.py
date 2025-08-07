#!/usr/bin/env python3
"""
🚀 LANCEUR PRINCIPAL POUR L'ENTRAÎNEMENT ROBUSTE
================================================

Script principal qui orchestre tout le processus d'entraînement robuste:
1. Test de l'environnement
2. Entraînement avec curriculum learning
3. Génération de vidéos
4. Ouverture de la simulation Mujoco
5. Sauvegarde des résultats

Version ultra-stable et professionnelle qui corrige tous les problèmes identifiés.
"""
import os
import sys
import subprocess
import time
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    print("🔍 Vérification des dépendances...")
    
    required_packages = [
        'numpy',
        'gymnasium',
        'stable_baselines3',
        'mujoco',
        'cv2',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'mujoco':
                import mujoco
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MANQUANT")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Packages manquants: {', '.join(missing_packages)}")
        print("Installez-les avec: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ Toutes les dépendances sont installées")
    return True

def check_model_file():
    """Vérifie que le fichier modèle existe"""
    model_path = "/workspace/results/g1_combined.xml"
    
    if not os.path.exists(model_path):
        print(f"❌ Fichier modèle non trouvé: {model_path}")
        print("Assurez-vous que le fichier g1_combined.xml existe dans le dossier results/")
        return False
    
    print(f"✅ Fichier modèle trouvé: {model_path}")
    return True

def run_tests():
    """Lance les tests de l'environnement"""
    print("\n🧪 Lancement des tests de l'environnement...")
    
    try:
        # Lancer le script de test
        result = subprocess.run([
            sys.executable, 
            "test_robust_environment.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Tests réussis")
            return True
        else:
            print("❌ Tests échoués")
            print("Sortie d'erreur:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors des tests: {e}")
        return False

def run_training():
    """Lance l'entraînement robuste"""
    print("\n🎯 Lancement de l'entraînement robuste...")
    
    try:
        # Lancer le script d'entraînement
        print("🚀 Démarrage de l'entraînement...")
        print("📊 L'entraînement peut prendre plusieurs heures selon la configuration")
        print("🎥 La simulation Mujoco s'ouvrira automatiquement")
        print("📹 Les vidéos seront générées automatiquement")
        
        # Lancer en arrière-plan pour permettre l'ouverture du viewer
        process = subprocess.Popen([
            sys.executable,
            "train_robust_curriculum_sac.py"
        ])
        
        print(f"✅ Processus d'entraînement lancé (PID: {process.pid})")
        print("⏳ Attente de la fin de l'entraînement...")
        
        # Attendre la fin du processus
        process.wait()
        
        if process.returncode == 0:
            print("✅ Entraînement terminé avec succès")
            return True
        else:
            print("❌ Entraînement échoué")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement: {e}")
        return False

def open_results():
    """Ouvre les résultats de l'entraînement"""
    print("\n📁 Ouverture des résultats...")
    
    results_dir = "/workspace/robust_curriculum_sac_results"
    
    if not os.path.exists(results_dir):
        print(f"❌ Dossier de résultats non trouvé: {results_dir}")
        return False
    
    try:
        # Ouvrir le dossier de résultats
        subprocess.Popen(['xdg-open', results_dir])
        print(f"✅ Dossier de résultats ouvert: {results_dir}")
        
        # Ouvrir la vidéo finale si elle existe
        video_path = os.path.join(results_dir, "videos", "final_demo.mp4")
        if os.path.exists(video_path):
            subprocess.Popen(['xdg-open', video_path])
            print(f"🎬 Vidéo finale ouverte: {video_path}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Impossible d'ouvrir les résultats: {e}")
        return False

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Lanceur principal pour l'entraînement robuste")
    parser.add_argument("--skip-tests", action="store_true", help="Passer les tests")
    parser.add_argument("--skip-training", action="store_true", help="Passer l'entraînement")
    parser.add_argument("--open-results", action="store_true", help="Ouvrir les résultats")
    
    args = parser.parse_args()
    
    print("🚀 LANCEUR PRINCIPAL POUR L'ENTRAÎNEMENT ROBUSTE")
    print("=" * 60)
    print(f"⏰ Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Étape 1: Vérification des dépendances
    if not check_dependencies():
        print("❌ Dépendances manquantes - Arrêt")
        return
    
    # Étape 2: Vérification du fichier modèle
    if not check_model_file():
        print("❌ Fichier modèle manquant - Arrêt")
        return
    
    # Étape 3: Tests de l'environnement (optionnel)
    if not args.skip_tests:
        if not run_tests():
            print("⚠️ Tests échoués - Continuer quand même? (y/n)")
            response = input().lower()
            if response != 'y':
                print("❌ Arrêt demandé par l'utilisateur")
                return
    else:
        print("⏭️ Tests ignorés")
    
    # Étape 4: Entraînement (optionnel)
    if not args.skip_training:
        if not run_training():
            print("❌ Entraînement échoué")
            return
    else:
        print("⏭️ Entraînement ignoré")
    
    # Étape 5: Ouverture des résultats (optionnel)
    if args.open_results:
        open_results()
    
    print("\n🎉 PROCESSUS TERMINÉ AVEC SUCCÈS!")
    print("=" * 60)
    print("📁 Résultats disponibles dans: /workspace/robust_curriculum_sac_results")
    print("🎬 Vidéos disponibles dans: /workspace/robust_curriculum_sac_results/videos")
    print("🤖 Modèles disponibles dans: /workspace/robust_curriculum_sac_results/models")

if __name__ == "__main__":
    main()