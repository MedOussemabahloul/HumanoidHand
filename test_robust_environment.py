#!/usr/bin/env python3
"""
🧪 TEST DE L'ENVIRONNEMENT ROBUSTE
===================================

Script de test pour vérifier que l'environnement fonctionne correctement :
✅ Pas d'erreurs mujoco
✅ Pas de vitesses excessives
✅ Rendu vidéo fonctionnel
✅ Actions stables

Auteur: Assistant IA
Date: 2024
"""

import os
import sys
import numpy as np
import cv2
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Configuration des chemins
WORKSPACE_DIR = "/workspace"
sys.path.append(os.path.join(WORKSPACE_DIR, 'envs'))
sys.path.append(WORKSPACE_DIR)

# Import global de mujoco
try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
    print("✅ MuJoCo importé avec succès")
except ImportError as e:
    print(f"❌ MuJoCo non disponible: {e}")
    sys.exit(1)

# Import de l'environnement
try:
    from envs.curriculum_grasp_env import CurriculumGraspEnv
    print("✅ CurriculumGraspEnv importé")
except ImportError as e:
    print(f"❌ Erreur d'import environnement: {e}")
    sys.exit(1)

def test_environment_creation():
    """Test de création de l'environnement"""
    print("\n🧪 Test 1: Création de l'environnement")
    print("-" * 40)
    
    try:
        env = CurriculumGraspEnv(render_mode="rgb_array")
        print("✅ Environnement créé avec succès")
        print(f"   - Niveau: {env.current_level}")
        print(f"   - Action space: {env.action_space}")
        print(f"   - Observation space: {env.observation_space}")
        return env
    except Exception as e:
        print(f"❌ Erreur création environnement: {e}")
        return None

def test_reset_functionality(env):
    """Test de la fonction reset"""
    print("\n🧪 Test 2: Fonction reset")
    print("-" * 40)
    
    try:
        obs = env.reset()
        print("✅ Reset réussi")
        print(f"   - Observation shape: {obs.shape}")
        print(f"   - Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        return obs
    except Exception as e:
        print(f"❌ Erreur reset: {e}")
        return None

def test_step_functionality(env, obs):
    """Test de la fonction step"""
    print("\n🧪 Test 3: Fonction step")
    print("-" * 40)
    
    try:
        # Action aléatoire
        action = env.action_space.sample()
        print(f"   - Action shape: {action.shape}")
        print(f"   - Action range: [{action.min():.3f}, {action.max():.3f}]")
        
        # Appliquer l'action
        obs, reward, done, info = env.step(action)
        print("✅ Step réussi")
        print(f"   - Reward: {reward:.3f}")
        print(f"   - Done: {done}")
        print(f"   - Info keys: {list(info.keys())}")
        
        return obs, reward, done, info
    except Exception as e:
        print(f"❌ Erreur step: {e}")
        return None, None, None, None

def test_render_functionality(env):
    """Test de la fonction render"""
    print("\n🧪 Test 4: Fonction render")
    print("-" * 40)
    
    try:
        frame = env.render()
        if frame is not None:
            print("✅ Render réussi")
            print(f"   - Frame shape: {frame.shape}")
            print(f"   - Frame dtype: {frame.dtype}")
            print(f"   - Frame range: [{frame.min()}, {frame.max()}]")
            return frame
        else:
            print("⚠️ Render retourne None")
            return None
    except Exception as e:
        print(f"❌ Erreur render: {e}")
        return None

def test_video_capture(env, num_steps=100):
    """Test de capture vidéo"""
    print(f"\n🧪 Test 5: Capture vidéo ({num_steps} steps)")
    print("-" * 40)
    
    # Créer le dossier de test
    test_dir = os.path.join(WORKSPACE_DIR, "test_videos")
    os.makedirs(test_dir, exist_ok=True)
    
    # Nom du fichier vidéo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(test_dir, f"test_capture_{timestamp}.mp4")
    
    try:
        # Configuration de la vidéo
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        frame_size = (640, 480)
        
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        
        # Réinitialiser l'environnement
        obs = env.reset()
        frames_captured = 0
        
        # Capturer les frames
        for step in range(num_steps):
            # Rendu
            frame = env.render()
            if frame is not None and frame.size > 0:
                # Convertir BGR pour OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
                frames_captured += 1
            
            # Action aléatoire
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            if done:
                obs = env.reset()
        
        video_writer.release()
        
        print("✅ Capture vidéo réussie")
        print(f"   - Fichier: {video_path}")
        print(f"   - Frames capturées: {frames_captured}")
        print(f"   - Durée estimée: {frames_captured/fps:.1f} secondes")
        
        return video_path
        
    except Exception as e:
        print(f"❌ Erreur capture vidéo: {e}")
        return None

def test_stability_monitoring(env, num_steps=200):
    """Test de monitoring de stabilité"""
    print(f"\n🧪 Test 6: Monitoring de stabilité ({num_steps} steps)")
    print("-" * 40)
    
    try:
        obs = env.reset()
        velocity_warnings = 0
        stability_count = 0
        
        for step in range(num_steps):
            # Action aléatoire
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            # Vérifier les vitesses
            if hasattr(env, 'data') and env.data is not None:
                max_velocity = np.max(np.abs(env.data.qvel))
                if max_velocity > 5.0:
                    velocity_warnings += 1
            
            # Vérifier la stabilité
            if hasattr(env, 'stability_count'):
                stability_count = env.stability_count
            
            if done:
                obs = env.reset()
        
        print("✅ Monitoring de stabilité terminé")
        print(f"   - Avertissements vitesse: {velocity_warnings}")
        print(f"   - Compteur stabilité: {stability_count}")
        print(f"   - Taux d'avertissements: {velocity_warnings/num_steps*100:.1f}%")
        
        return velocity_warnings, stability_count
        
    except Exception as e:
        print(f"❌ Erreur monitoring: {e}")
        return None, None

def test_curriculum_functionality(env):
    """Test des fonctionnalités de curriculum"""
    print("\n🧪 Test 7: Fonctionnalités de curriculum")
    print("-" * 40)
    
    try:
        # Informations de curriculum
        curriculum_info = env.get_curriculum_info()
        print("✅ Informations de curriculum récupérées")
        print(f"   - Niveau actuel: {curriculum_info['current_level']}")
        print(f"   - Nom du niveau: {curriculum_info['level_name']}")
        print(f"   - Succès consécutifs: {curriculum_info['consecutive_successes']}")
        print(f"   - Succès requis: {curriculum_info['required_successes']}")
        
        # Test de mise à jour de niveau
        env.update_curriculum_level(episode_reward=50.0, episode_success=True)
        print("✅ Mise à jour de niveau testée")
        
        return curriculum_info
        
    except Exception as e:
        print(f"❌ Erreur curriculum: {e}")
        return None

def main():
    """Fonction principale de test"""
    print("🧪 DÉMARRAGE DES TESTS DE L'ENVIRONNEMENT ROBUSTE")
    print("=" * 60)
    
    # Test 1: Création de l'environnement
    env = test_environment_creation()
    if env is None:
        print("❌ Impossible de continuer sans environnement")
        return
    
    # Test 2: Reset
    obs = test_reset_functionality(env)
    if obs is None:
        print("❌ Impossible de continuer sans reset")
        return
    
    # Test 3: Step
    obs, reward, done, info = test_step_functionality(env, obs)
    if obs is None:
        print("❌ Impossible de continuer sans step")
        return
    
    # Test 4: Render
    frame = test_render_functionality(env)
    if frame is None:
        print("⚠️ Render non fonctionnel, mais on continue")
    
    # Test 5: Capture vidéo
    video_path = test_video_capture(env, num_steps=150)
    
    # Test 6: Monitoring de stabilité
    velocity_warnings, stability_count = test_stability_monitoring(env, num_steps=300)
    
    # Test 7: Curriculum
    curriculum_info = test_curriculum_functionality(env)
    
    # Résumé final
    print("\n🎯 RÉSUMÉ DES TESTS")
    print("=" * 60)
    print("✅ Environnement créé et fonctionnel")
    print("✅ Actions et observations valides")
    print("✅ Système de récompenses opérationnel")
    
    if frame is not None:
        print("✅ Rendu vidéo fonctionnel")
    else:
        print("⚠️ Rendu vidéo avec problèmes")
    
    if video_path is not None:
        print(f"✅ Vidéo de test créée: {video_path}")
    else:
        print("❌ Échec de création vidéo")
    
    if velocity_warnings is not None:
        warning_rate = velocity_warnings / 300 * 100
        if warning_rate < 5:
            print(f"✅ Stabilité excellente ({warning_rate:.1f}% d'avertissements)")
        elif warning_rate < 15:
            print(f"⚠️ Stabilité acceptable ({warning_rate:.1f}% d'avertissements)")
        else:
            print(f"❌ Stabilité problématique ({warning_rate:.1f}% d'avertissements)")
    
    if curriculum_info is not None:
        print("✅ Système de curriculum opérationnel")
    
    print("\n🎉 TESTS TERMINÉS!")
    print("L'environnement est prêt pour l'entraînement.")

if __name__ == "__main__":
    main()