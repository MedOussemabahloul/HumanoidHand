#!/usr/bin/env python3
"""
🔍 VÉRIFICATION COMPLÈTE DU SYSTÈME
====================================

Script de vérification finale pour s'assurer que tout le système d'entraînement robuste
est en place et fonctionnel avant le lancement.
"""
import os
import sys
import subprocess
import importlib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def check_python_version():
    """Vérifie la version de Python"""
    print("🐍 Vérification de la version Python...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} détecté")
        print("⚠️ Python 3.8+ requis")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Vérifie toutes les dépendances"""
    print("\n📦 Vérification des dépendances...")
    
    dependencies = {
        'numpy': 'NumPy',
        'gymnasium': 'Gymnasium',
        'stable_baselines3': 'Stable-Baselines3',
        'mujoco': 'Mujoco',
        'cv2': 'OpenCV',
        'matplotlib': 'Matplotlib'
    }
    
    missing = []
    
    for package, name in dependencies.items():
        try:
            if package == 'cv2':
                import cv2
            elif package == 'mujoco':
                import mujoco
            else:
                importlib.import_module(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - MANQUANT")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ Packages manquants: {', '.join(missing)}")
        print("Installez-les avec:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def check_files():
    """Vérifie que tous les fichiers nécessaires existent"""
    print("\n📁 Vérification des fichiers...")
    
    required_files = [
        'envs/robust_curriculum_grasp_env.py',
        'train_robust_curriculum_sac.py',
        'test_robust_environment.py',
        'run_robust_training.py',
        'README_ROBUST_TRAINING.md'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MANQUANT")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ Fichiers manquants: {', '.join(missing_files)}")
        return False
    
    return True

def check_model_file():
    """Vérifie que le fichier modèle existe"""
    print("\n🤖 Vérification du fichier modèle...")
    
    model_path = "/workspace/results/g1_combined.xml"
    
    if os.path.exists(model_path):
        print(f"✅ Modèle trouvé: {model_path}")
        
        # Vérifier la taille du fichier
        size = os.path.getsize(model_path)
        if size > 1000:  # Plus de 1KB
            print(f"✅ Taille du modèle: {size/1024:.1f} KB")
        else:
            print(f"⚠️ Taille du modèle suspecte: {size} bytes")
            return False
        
        return True
    else:
        print(f"❌ Modèle non trouvé: {model_path}")
        print("Assurez-vous que le fichier g1_combined.xml existe dans le dossier results/")
        return False

def check_directories():
    """Vérifie et crée les dossiers nécessaires"""
    print("\n📂 Vérification des dossiers...")
    
    base_dir = "/workspace"
    required_dirs = [
        base_dir,
        os.path.join(base_dir, "envs"),
        os.path.join(base_dir, "results"),
        os.path.join(base_dir, "robust_curriculum_sac_results"),
        os.path.join(base_dir, "robust_curriculum_sac_results", "models"),
        os.path.join(base_dir, "robust_curriculum_sac_results", "videos"),
        os.path.join(base_dir, "robust_curriculum_sac_results", "logs"),
        os.path.join(base_dir, "robust_curriculum_sac_results", "plots")
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"✅ {dir_path} - CRÉÉ")
            except Exception as e:
                print(f"❌ {dir_path} - ERREUR: {e}")
                return False
    
    return True

def check_environment_import():
    """Vérifie que l'environnement peut être importé"""
    print("\n🔧 Vérification de l'import de l'environnement...")
    
    try:
        # Ajouter le chemin
        sys.path.append('/workspace/envs')
        
        # Importer l'environnement
        from envs.robust_curriculum_grasp_env import RobustCurriculumGraspEnv
        print("✅ Import de RobustCurriculumGraspEnv réussi")
        
        # Test de création
        env = RobustCurriculumGraspEnv(
            model_path="/workspace/results/g1_combined.xml",
            render_mode="rgb_array",
            video_capture=False  # Pas de vidéo pour le test
        )
        print("✅ Création de l'environnement réussie")
        
        # Test de reset
        obs, info = env.reset()
        print(f"✅ Reset réussi - Observation shape: {obs.shape}")
        
        # Test de step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Step réussi - Reward: {reward:.2f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test de l'environnement: {e}")
        return False

def check_permissions():
    """Vérifie les permissions d'écriture"""
    print("\n🔐 Vérification des permissions...")
    
    test_dirs = [
        "/workspace",
        "/workspace/robust_curriculum_sac_results"
    ]
    
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            if os.access(dir_path, os.W_OK):
                print(f"✅ Permissions d'écriture: {dir_path}")
            else:
                print(f"❌ Pas de permissions d'écriture: {dir_path}")
                return False
        else:
            print(f"⚠️ Dossier non trouvé: {dir_path}")
    
    return True

def run_quick_test():
    """Lance un test rapide du système avec le test simplifié"""
    print("\n🧪 Test rapide du système...")
    
    try:
        # Lancer le test simplifié de l'environnement
        result = subprocess.run([
            sys.executable,
            "test_quick_training.py"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Test rapide réussi")
            return True
        else:
            print("❌ Test rapide échoué")
            print("Sortie d'erreur:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Test rapide interrompu (timeout)")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du test rapide: {e}")
        return False

def main():
    """Fonction principale de vérification"""
    print("🔍 VÉRIFICATION COMPLÈTE DU SYSTÈME D'ENTRAÎNEMENT ROBUSTE")
    print("=" * 70)
    
    checks = [
        ("Version Python", check_python_version),
        ("Dépendances", check_dependencies),
        ("Fichiers", check_files),
        ("Fichier modèle", check_model_file),
        ("Dossiers", check_directories),
        ("Permissions", check_permissions),
        ("Import environnement", check_environment_import),
        ("Test rapide", run_quick_test)
    ]
    
    results = []
    
    for name, check_func in checks:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Erreur lors de la vérification {name}: {e}")
            results.append((name, False))
    
    # Résumé
    print(f"\n{'='*70}")
    print("📊 RÉSUMÉ DE LA VÉRIFICATION")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSÉ" if result else "❌ ÉCHOUÉ"
        print(f"{name:25} : {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Résultat: {passed}/{total} vérifications réussies")
    
    if passed == total:
        print("\n🎉 SYSTÈME PRÊT POUR L'ENTRAÎNEMENT!")
        print("=" * 70)
        print("🚀 Vous pouvez maintenant lancer:")
        print("   python3 run_robust_training.py")
        print("\n📚 Consultez README_ROBUST_TRAINING.md pour plus d'informations")
    else:
        print(f"\n⚠️ {total - passed} problème(s) détecté(s)")
        print("Corrigez les problèmes avant de lancer l'entraînement")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)