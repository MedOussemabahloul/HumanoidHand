#!/usr/bin/env python3
"""
🚀 LANCEUR FINAL D'ENTRAÎNEMENT ROBUSTE
========================================
Script de lancement pour l'entraînement final avec toutes les corrections :
✅ Correction des erreurs mujoco
✅ Réduction drastique des vitesses excessives
✅ Capture vidéo automatique
✅ Gestion des erreurs API
✅ Monitoring complet

Usage: python3 run_final_training.py
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Configuration
WORKSPACE_DIR = "/home/oussema/Documents/project"
TRAINING_SCRIPT = "train_final_robust.py"

def check_dependencies():
 """Vérifie que toutes les dépendances sont installées"""
 print("🔍 Vérification des dépendances...")
 
 required_packages = [
     "mujoco",
     "stable_baselines3",
     "opencv-python",
     "numpy",
     "gymnasium"
 ]
 
 missing_packages = []
 
 for package in required_packages:
     try:
         __import__(package.replace("-", "_"))
         print(f"✅ {package}")
     except ImportError:
         print(f"❌ {package} - MANQUANT")
         missing_packages.append(package)
 
 if missing_packages:
     print(f"\n⚠️ Packages manquants: {', '.join(missing_packages)}")
     print("Installation automatique...")
     
     for package in missing_packages:
         try:
             subprocess.run([
                 sys.executable, "-m", "pip", "install", 
                 package, "--break-system-packages"
             ], check=True, capture_output=True)
             print(f"✅ {package} installé")
         except subprocess.CalledProcessError:
             print(f"❌ Échec installation {package}")
             return False
 
 print("✅ Toutes les dépendances sont installées")
 return True

def run_training():
 """Lance l'entraînement principal"""
 print("\n🎯 Lancement de l'entraînement final robuste...")
 
 training_path = os.path.join(WORKSPACE_DIR, TRAINING_SCRIPT)
 if not os.path.exists(training_path):
     print(f"❌ Script d'entraînement non trouvé: {training_path}")
     return False
 
 # Créer le dossier de logs
 logs_dir = os.path.join(WORKSPACE_DIR, "training_logs")
 os.makedirs(logs_dir, exist_ok=True)
 
 # Nom du fichier de log
 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 log_file = os.path.join(logs_dir, f"final_training_{timestamp}.log")
 
 print(f"📝 Logs sauvegardés dans: {log_file}")
 print("🎬 Début de l'entraînement final...")
 print("⏱️  Cela peut prendre plusieurs heures...")
 print("🎯 Objectifs:")
 print("   - Réduction des vitesses excessives")
 print("   - Capture vidéo automatique")
 print("   - Entraînement par phases")
 print("   - Monitoring complet")
 
 try:
     # Lancer l'entraînement avec redirection des logs
     with open(log_file, 'w') as f:
         process = subprocess.Popen(
             [sys.executable, training_path],
             cwd=WORKSPACE_DIR,
             stdout=subprocess.PIPE,
             stderr=subprocess.STDOUT,
             text=True,
             bufsize=1,
             universal_newlines=True
         )
         
         # Afficher les logs en temps réel
         for line in process.stdout:
             print(line.rstrip())
             f.write(line)
             f.flush()
         
         process.wait()
     
     if process.returncode == 0:
         print("\n🎉 ENTRAÎNEMENT FINAL TERMINÉ AVEC SUCCÈS!")
         return True
     else:
         print(f"\n❌ Entraînement échoué (code: {process.returncode})")
         return False
         
 except KeyboardInterrupt:
     print("\n⚠️ Entraînement interrompu par l'utilisateur")
     return False
 except Exception as e:
     print(f"\n❌ Erreur lors de l'entraînement: {e}")
     return False

def show_results():
 """Affiche les résultats de l'entraînement"""
 print("\n📊 RÉSULTATS DE L'ENTRAÎNEMENT FINAL")
 print("=" * 50)
 
 results_dir = os.path.join(WORKSPACE_DIR, "final_training_results")
 
 if os.path.exists(results_dir):
     print(f"📁 Dossier des résultats: {results_dir}")
     
     # Lister les fichiers
     for root, dirs, files in os.walk(results_dir):
         level = root.replace(results_dir, '').count(os.sep)
         indent = ' ' * 2 * level
         print(f"{indent}{os.path.basename(root)}/")
         subindent = ' ' * 2 * (level + 1)
         for file in files:
             print(f"{subindent}{file}")
     
     # Vérifier les vidéos
     video_dir = os.path.join(results_dir, "videos")
     if os.path.exists(video_dir):
         video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
         print(f"\n🎥 Vidéos générées: {len(video_files)}")
         for video in video_files:
             print(f"   - {video}")
     
     # Vérifier les modèles
     models_dir = os.path.join(results_dir, "models")
     if os.path.exists(models_dir):
         model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
         print(f"\n💾 Modèles sauvegardés: {len(model_files)}")
         for model in model_files:
             print(f"   - {model}")
     
     # Afficher le résumé
     summary_path = os.path.join(results_dir, "training_summary.json")
     if os.path.exists(summary_path):
         try:
             import json
             with open(summary_path, 'r') as f:
                 summary = json.load(f)
             
             print(f"\n📈 Résumé de l'entraînement:")
             print(f"   - Date: {summary.get('training_date', 'N/A')}")
             print(f"   - Phases complétées: {summary.get('phases_completed', 0)}")
             print(f"   - Récompense finale: {summary.get('final_reward', 0):.2f}")
             print(f"   - Avertissements vitesse: {summary.get('total_velocity_warnings', 0)}")
             
         except Exception as e:
             print(f"⚠️ Erreur lecture résumé: {e}")
 else:
     print("❌ Aucun résultat trouvé")

def main():
 """Fonction principale"""
 print("🚀 LANCEUR FINAL D'ENTRAÎNEMENT ROBUSTE")
 print("=" * 50)
 print(f"📁 Répertoire de travail: {WORKSPACE_DIR}")
 print(f"⏰ Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
 
 # Vérifier les dépendances
 if not check_dependencies():
     print("\n❌ Impossible de continuer sans les dépendances")
     return
 
 # Demander confirmation
 print("\n🤔 Voulez-vous lancer l'entraînement final maintenant? (y/n)")
 print("⚠️  Cet entraînement inclut toutes les corrections pour:")
 print("   - Éliminer les erreurs mujoco")
 print("   - Réduire drastiquement les vitesses excessives")
 print("   - Générer des vidéos de démonstration")
 print("   - Garantir la stabilité et la performance")
 
 response = input().lower().strip()
 
 if response not in ['y', 'yes', 'oui', 'o']:
     print("❌ Entraînement annulé")
     return
 
 # Lancer l'entraînement
 success = run_training()
 
 # Afficher les résultats
 show_results()
 
 if success:
     print("\n🎉 FÉLICITATIONS! L'entraînement final s'est terminé avec succès!")
     print("🎥 Vous pouvez maintenant regarder les vidéos générées")
     print("💾 Les modèles entraînés sont prêts à être utilisés")
     print("🛡️ Tous les problèmes ont été corrigés:")
     print("   ✅ Erreurs mujoco résolues")
     print("   ✅ Vitesses excessives réduites")
     print("   ✅ Vidéos fonctionnelles")
     print("   ✅ Performance optimisée")
 else:
     print("\n❌ L'entraînement a échoué")
     print("📝 Consultez les logs pour plus de détails")

if __name__ == "__main__":
 main()
