#!/usr/bin/env python3
"""
🚀 LANCEUR D'ENTRAÎNEMENT ROBUSTE
==================================

Script simple pour lancer l'entraînement robuste avec toutes les corrections :
✅ Correction des erreurs mujoco
✅ Réduction des vitesses excessives
✅ Capture vidéo automatique
✅ Monitoring complet

Usage: python3 run_training.py
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Configuration
WORKSPACE_DIR = "/home/oussema/Documents/project"
TRAINING_SCRIPT = "train_robust_final.py"
TEST_SCRIPT = "test_robust_environment.py"

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
     print("Installez-les avec: pip install " + " ".join(missing_packages))
     return False
 
 print("✅ Toutes les dépendances sont installées")
 return True

def run_test_first():
 """Lance d'abord le test pour vérifier que tout fonctionne"""
 print("\n🧪 Lancement du test de l'environnement...")
 
 test_path = os.path.join(WORKSPACE_DIR, TEST_SCRIPT)
 if not os.path.exists(test_path):
     print(f"❌ Script de test non trouvé: {test_path}")
     return False
 
 try:
     result = subprocess.run(
         [sys.executable, test_path],
         cwd=WORKSPACE_DIR,
         capture_output=True,
         text=True,
         timeout=300  # 5 minutes max
     )
     
     if result.returncode == 0:
         print("✅ Test réussi - L'environnement fonctionne correctement")
         return True
     else:
         print("❌ Test échoué")
         print("Sortie d'erreur:")
         print(result.stderr)
         return False
         
 except subprocess.TimeoutExpired:
     print("❌ Test timeout - L'environnement prend trop de temps")
     return False
 except Exception as e:
     print(f"❌ Erreur lors du test: {e}")
     return False

def run_training():
 """Lance l'entraînement principal"""
 print("\n🎯 Lancement de l'entraînement robuste...")
 
 training_path = os.path.join(WORKSPACE_DIR, TRAINING_SCRIPT)
 if not os.path.exists(training_path):
     print(f"❌ Script d'entraînement non trouvé: {training_path}")
     return False
 
 # Créer le dossier de logs
 logs_dir = os.path.join(WORKSPACE_DIR, "training_logs")
 os.makedirs(logs_dir, exist_ok=True)
 
 # Nom du fichier de log
 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 log_file = os.path.join(logs_dir, f"training_{timestamp}.log")
 
 print(f"📝 Logs sauvegardés dans: {log_file}")
 print("🎬 Début de l'entraînement...")
 print("⏱️  Cela peut prendre plusieurs heures...")
 
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
         print("\n🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
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
 print("\n📊 RÉSULTATS DE L'ENTRAÎNEMENT")
 print("=" * 50)
 
 results_dir = os.path.join(WORKSPACE_DIR, "robust_training_results")
 
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
 else:
     print("❌ Aucun résultat trouvé")

def main():
 """Fonction principale"""
 print("🚀 LANCEUR D'ENTRAÎNEMENT ROBUSTE")
 print("=" * 50)
 print(f"📁 Répertoire de travail: {WORKSPACE_DIR}")
 print(f"⏰ Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
 
 # Vérifier les dépendances
 if not check_dependencies():
     print("\n❌ Impossible de continuer sans les dépendances")
     return
 
 # Lancer le test d'abord
 if not run_test_first():
     print("\n❌ Impossible de continuer sans test réussi")
     return
 
 # Demander confirmation
 print("\n🤔 Voulez-vous lancer l'entraînement maintenant? (y/n)")
 response = input().lower().strip()
 
 if response not in ['y', 'yes', 'oui', 'o']:
     print("❌ Entraînement annulé")
     return
 
 # Lancer l'entraînement
 success = run_training()
 
 # Afficher les résultats
 show_results()
 
 if success:
     print("\n🎉 FÉLICITATIONS! L'entraînement s'est terminé avec succès!")
     print("🎥 Vous pouvez maintenant regarder les vidéos générées")
     print("💾 Les modèles entraînés sont prêts à être utilisés")
 else:
     print("\n❌ L'entraînement a échoué")
     print("📝 Consultez les logs pour plus de détails")

if __name__ == "__main__":
 main()
