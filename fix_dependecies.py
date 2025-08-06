#!/usr/bin/env python3
"""
Script pour corriger les dépendances et problèmes d'installation
Auteur: Assistant IA
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description=""):
    """Exécute une commande avec gestion d'erreur"""
    print(f"🔧 {description}")
    print(f"   Commande: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Succès")
            if result.stdout:
                print(f"   Sortie: {result.stdout.strip()}")
        else:
            print(f"   ❌ Échec (code {result.returncode})")
            if result.stderr:
                print(f"   Erreur: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def fix_ffmpeg():
    """Corrige le problème FFmpeg pour imageio"""
    print("\n🎬 Correction du problème FFmpeg...")
    
    # Essayer d'installer imageio avec FFmpeg
    commands = [
        "pip install 'imageio[ffmpeg]'",
        "pip install imageio-ffmpeg",
        "python -c 'import imageio; imageio.plugins.ffmpeg.download()'",
    ]
    
    for cmd in commands:
        if run_command(cmd, f"Installation FFmpeg via {cmd.split()[1]}"):
            break
    
    # Vérifier si FFmpeg est installé système
    if not run_command("ffmpeg -version", "Vérification FFmpeg système"):
        print("\n💡 FFmpeg non trouvé. Installations alternatives:")
        print("   Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg")
        print("   macOS: brew install ffmpeg")
        print("   Windows: Télécharger depuis https://ffmpeg.org/")

def fix_mujoco():
    """Corrige les problèmes MuJoCo"""
    print("\n🤖 Vérification de MuJoCo...")
    
    # Vérifier l'installation MuJoCo
    try:
        import mujoco
        print(f"   ✅ MuJoCo {mujoco.__version__} installé")
        
        # Tester un modèle simple
        try:
            from mujoco import MjModel, MjData
            # Créer un modèle minimal pour test
            test_xml = """
            <mujoco>
                <worldbody>
                    <body>
                        <geom type="box" size="0.1 0.1 0.1"/>
                    </body>
                </worldbody>
            </mujoco>
            """
            model = MjModel.from_xml_string(test_xml)
            data = MjData(model)
            print("   ✅ Test MuJoCo réussi")
        except Exception as e:
            print(f"   ⚠️  Problème MuJoCo: {e}")
            
    except ImportError:
        print("   ❌ MuJoCo non installé")
        run_command("pip install mujoco", "Installation MuJoCo")

def fix_torch():
    """Corrige l'installation PyTorch"""
    print("\n🧠 Vérification de PyTorch...")
    
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__} installé")
        print(f"   CUDA disponible: {torch.cuda.is_available()}")
        
        # Test simple
        x = torch.randn(2, 3)
        print(f"   ✅ Test PyTorch réussi: {x.shape}")
        
    except ImportError:
        print("   ❌ PyTorch non installé")
        # Installation CPU par défaut
        run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu", 
                   "Installation PyTorch CPU")

def fix_matplotlib():
    """Corrige l'installation matplotlib"""
    print("\n📊 Vérification de matplotlib...")
    
    try:
        import matplotlib
        print(f"   ✅ matplotlib {matplotlib.__version__} installé")
        
        # Configurer pour mode non-interactif
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        print("   ✅ Backend non-interactif configuré")
        
    except ImportError:
        print("   ❌ matplotlib non installé")
        run_command("pip install matplotlib", "Installation matplotlib")

def install_missing_packages():
    """Installe tous les packages manquants"""
    print("\n📦 Installation des dépendances de base...")
    
    packages = [
        "numpy",
        "gymnasium", 
        "imageio",
        "tqdm",
        "pathlib",
        "json5"
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"   ✅ {package}: Déjà installé")
        except ImportError:
            print(f"   ❌ {package}: Manquant")
            run_command(f"pip install {package}", f"Installation {package}")

def test_environment():
    """Teste l'environnement stabilisé"""
    print("\n🧪 Test de l'environnement stabilisé...")
    
    try:
        # Ajouter au path
        sys.path.append('.')
        sys.path.append('./envs')
        
        # Test d'import
        from envs.stable_grasp_env import StableGraspEnv
        print("   ✅ Import StableGraspEnv réussi")
        
        # Vérifier que le modèle existe
        model_path = "results/g1_combined.xml"
        if Path(model_path).exists():
            print(f"   ✅ Modèle trouvé: {model_path}")
            
            # Test de création d'environnement (sans simulation)
            try:
                env = StableGraspEnv(xml_path=model_path, max_episode_steps=10)
                print("   ✅ Création d'environnement réussie")
                env.close()
            except Exception as e:
                print(f"   ⚠️  Problème environnement: {e}")
        else:
            print(f"   ❌ Modèle manquant: {model_path}")
            print("   💡 Placez votre g1_combined.xml dans results/")
            
    except Exception as e:
        print(f"   ❌ Erreur test environnement: {e}")

def create_alternative_video_recorder():
    """Crée un enregistreur vidéo sans FFmpeg"""
    print("\n🎬 Création d'un enregistreur vidéo alternatif...")
    
    alternative_code = '''#!/usr/bin/env python3
"""
Enregistreur vidéo alternatif sans FFmpeg
"""

import numpy as np
import os
from pathlib import Path
from datetime import datetime
import pickle

class AlternativeVideoRecorder:
    """Enregistreur sans dépendance FFmpeg"""
    
    def __init__(self, output_dir="training_results/videos", fps=30):
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.is_recording = False
        self.frames = []
        self.episode_info = {}
        
        print(f"✅ Enregistreur alternatif initialisé (sans FFmpeg)")
    
    def start_recording(self, episode_info=None):
        """Démarre l'enregistrement"""
        self.is_recording = True
        self.frames = []
        self.episode_info = episode_info or {}
    
    def add_frame(self, frame):
        """Ajoute une frame"""
        if self.is_recording and frame is not None:
            self.frames.append(frame)
    
    def stop_recording(self, filename=None):
        """Arrête et sauvegarde en format pickle"""
        if not self.is_recording or len(self.frames) == 0:
            return None
        
        self.is_recording = False
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode = self.episode_info.get("episode", "unknown")
            reward = self.episode_info.get("total_reward", 0)
            filename = f"episode_{episode}_{timestamp}_reward_{reward:.1f}.pkl"
        
        filepath = self.output_dir / filename
        
        try:
            # Sauvegarder en pickle au lieu de MP4
            data = {
                'frames': self.frames,
                'episode_info': self.episode_info,
                'fps': self.fps
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"✅ Données vidéo sauvegardées: {filepath}")
            
            # Créer un fichier de métadonnées lisible
            meta_path = filepath.with_suffix('.txt')
            with open(meta_path, 'w') as f:
                f.write(f"Métadonnées épisode\\n")
                f.write(f"Fichier: {filepath.name}\\n")
                f.write(f"Frames: {len(self.frames)}\\n")
                f.write(f"FPS: {self.fps}\\n")
                for key, value in self.episode_info.items():
                    f.write(f"{key}: {value}\\n")
            
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
            return None
        finally:
            self.frames = []
    
    def record_episode(self, env, agent, max_steps=500, render_mode="rgb_array"):
        """Enregistre un épisode"""
        print("⚠️  Enregistrement en mode alternatif (sans vidéo MP4)")
        
        obs, info = env.reset()
        self.start_recording()
        
        total_reward = 0
        step = 0
        done = False
        
        while not done and step < max_steps:
            try:
                frame = env.render(mode=render_mode)
                self.add_frame(frame)
            except:
                pass
            
            if agent:
                action = agent.select_action(obs, evaluate=True)
            else:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward = reward
            done = terminated or truncated
            step = 1
        
        episode_info = {
            "total_reward": total_reward,
            "steps": step,
            "success": terminated and total_reward > 0
        }
        
        self.episode_info = episode_info
        video_path = self.stop_recording()
        
        return video_path, episode_info
'''
    
    with open("utils/alternative_video_recorder.py", 'w') as f:
        f.write(alternative_code)
    print("   ✅ Enregistreur alternatif créé")

def main():
    """Fonction principale de correction"""
    print("🔧 CORRECTION DES DÉPENDANCES ET PROBLÈMES")
    print("=" * 60)
    
    # Vérifier Python
    python_version = sys.version_info
    print(f"🐍 Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("⚠️  Python 3.8 recommandé")
    
    # Installer les packages de base
    install_missing_packages()
    
    # Corriger les problèmes spécifiques
    fix_torch()
    fix_mujoco()
    fix_matplotlib()
    fix_ffmpeg()
    
    # Créer un enregistreur alternatif
    create_alternative_video_recorder()
    
    # Tester l'environnement
    test_environment()
    
    print("\n"  "=" * 60)
    print("✅ CORRECTIONS TERMINÉES")
    print("\n🚀 Prochaines étapes:")
    print("1. Testez: python3 test_simple_grasp_basic.py")
    print("2. Entraînement stabilisé: python3 train_stable_grasp.py --episodes 100")
    print("3. Si problèmes persistent: vérifiez votre modèle g1_combined.xml")
    
    print("\n💡 CONSEILS:")
    print("- Utilisez train_stable_grasp.py au lieu de train_simple_grasp.py")
    print("- Commencez avec peu d'épisodes (50-100)")
    print("- Les vidéos seront en format pickle si FFmpeg indisponible")
    print("- Vérifiez les logs pour diagnostiquer les instabilités")

if __name__ == "__main__":
    main()
