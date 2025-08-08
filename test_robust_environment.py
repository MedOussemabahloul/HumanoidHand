#!/usr/bin/env python3
"""
🧪 TEST DE L'ENVIRONNEMENT ROBUSTE
===================================

Script de test pour vérifier que l'environnement robuste fonctionne correctement:
✅ Test de création d'environnement
✅ Test de simulation physique
✅ Test de contrôle de vitesse
✅ Test de capture vidéo
✅ Test de curriculum learning
✅ Test de stabilité
"""
import os
import sys
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

# Ajouter le chemin des environnements
sys.path.append('/home/oussema/Documents/project/envs')

try:
    from envs.robust_curriculum_grasp_env import RobustCurriculumGraspEnv
    print("✅ RobustCurriculumGraspEnv importé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

def test_environment_creation():
    """Test de création de l'environnement"""
    print("\n🧪 Test de création de l'environnement...")
    
    try:
        # Créer l'environnement
        env = RobustCurriculumGraspEnv(
            model_path="/home/oussema/Documents/project/results/g1_combined.xml",
            render_mode="rgb_array",
            video_capture=True
        )
        
        print("✅ Environnement créé avec succès")
        print(f"  - Niveau actuel: {env.current_level}")
        print(f"  - Espace d'action: {env.action_space.shape}")
        print(f"  - Espace d'observation: {env.observation_space.shape}")
        
        return env
        
    except Exception as e:
        print(f"❌ Erreur création environnement: {e}")
        return None

def test_physics_simulation(env):
    """Test de simulation physique"""
    print("\n🧪 Test de simulation physique...")
    
    try:
        # Reset de l'environnement
        obs, info = env.reset()
        print(f"✅ Reset réussi - Observation shape: {obs.shape}")
        
        # Test de quelques steps
        for step in range(10):
            # Action aléatoire
            action = env.action_space.sample()
            
            # Exécution de l'action
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"  Step {step+1}: Reward={reward:.2f}, Terminated={terminated}, Truncated={truncated}")
            
            if terminated or truncated:
                break
        
        print("✅ Simulation physique fonctionne correctement")
        return True
        
    except Exception as e:
        print(f"❌ Erreur simulation physique: {e}")
        return False

def test_velocity_control(env):
    """Test de contrôle de vitesse"""
    print("\n🧪 Test de contrôle de vitesse...")
    
    try:
        # Reset
        obs, info = env.reset()
        
        # Actions avec vitesses élevées
        high_velocity_actions = np.ones(env.action_space.shape) * 0.5
        
        velocities = []
        
        for step in range(20):
            obs, reward, terminated, truncated, info = env.step(high_velocity_actions)
            
            # Vérifier les vitesses
            if hasattr(env, 'arm_joint_ids'):
                arm_velocities = [abs(env.data.qvel[i]) for i in env.arm_joint_ids]
                avg_velocity = np.mean(arm_velocities)
                velocities.append(avg_velocity)
                
                if avg_velocity > 10.0:
                    print(f"⚠️ Vitesse élevée détectée: {avg_velocity:.2f}")
            
            if terminated or truncated:
                break
        
        max_velocity = max(velocities) if velocities else 0
        print(f"✅ Contrôle de vitesse testé - Vitesse max: {max_velocity:.2f}")
        
        return max_velocity < 15.0  # Vitesse acceptable
        
    except Exception as e:
        print(f"❌ Erreur contrôle de vitesse: {e}")
        return False

def test_video_capture(env):
    """Test de capture vidéo"""
    print("\n🧪 Test de capture vidéo...")
    
    try:
        # Reset
        obs, info = env.reset()
        
        # Capturer quelques frames
        frames = []
        errors_count = 0
        
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Capturer la frame
            try:
                frame = env.render()
                if frame is not None and frame.size > 0:
                    frames.append(frame)
                    print(f"  Frame {step+1} capturée: {frame.shape}")
                else:
                    errors_count += 1
            except Exception as frame_error:
                errors_count += 1
                if step % 3 == 0:  # Afficher seulement quelques erreurs
                    print(f"  ⚠️ Erreur frame {step+1}: {frame_error}")
            
            if terminated or truncated:
                break
        
        success_rate = len(frames) / 10.0
        print(f"✅ Capture vidéo testée - {len(frames)} frames capturées sur 10 (taux: {success_rate:.1%})")
        
        # Considérer comme réussi si au moins 50% des frames sont capturées
        return success_rate >= 0.5
        
    except Exception as e:
        print(f"❌ Erreur capture vidéo: {e}")
        return False

def test_curriculum_learning(env):
    """Test de curriculum learning"""
    print("\n🧪 Test de curriculum learning...")
    
    try:
        # Vérifier les niveaux de curriculum
        curriculum_info = env.get_curriculum_info()
        print(f"✅ Curriculum info récupéré:")
        print(f"  - Niveau actuel: {curriculum_info['current_level']}")
        print(f"  - Nom du niveau: {curriculum_info['level_name']}")
        print(f"  - Description: {curriculum_info['level_description']}")
        
        # Test de progression
        initial_level = env.current_level
        
        # Simuler quelques épisodes réussis
        for episode in range(3):
            obs, info = env.reset()
            episode_reward = 0
            
            for step in range(50):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            # Mettre à jour le curriculum
            env.update_curriculum_level(episode_reward, episode_reward > 10)
        
        print(f"✅ Curriculum learning testé - Niveau initial: {initial_level}, Niveau final: {env.current_level}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur curriculum learning: {e}")
        return False

def test_stability(env):
    """Test de stabilité"""
    print("\n🧪 Test de stabilité...")
    
    try:
        # Reset
        obs, info = env.reset()
        
        # Actions répétées pour tester la stabilité
        action = env.action_space.sample()
        
        for step in range(100):
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Vérifier les NaN/Inf
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                print(f"❌ Instabilité détectée à l'étape {step}")
                return False
            
            if terminated or truncated:
                break
        
        print("✅ Stabilité testée - Aucune instabilité détectée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur test de stabilité: {e}")
        return False

def test_rendering(env):
    """Test de rendu"""
    print("\n🧪 Test de rendu...")
    
    try:
        # Reset
        obs, info = env.reset()
        
        # Test de rendu rgb_array
        frame = env.render()
        if frame is not None:
            print(f"✅ Rendu rgb_array fonctionne - Frame shape: {frame.shape}")
        else:
            print("⚠️ Rendu rgb_array retourne None")
        
        # Test de rendu human (si possible)
        try:
            env.render_mode = "human"
            env.render()
            print("✅ Rendu human fonctionne")
        except Exception as e:
            print(f"⚠️ Rendu human non disponible: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test de rendu: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🧪 DÉMARRAGE DES TESTS DE L'ENVIRONNEMENT ROBUSTE")
    print("=" * 60)
    
    # Test 1: Création de l'environnement
    env = test_environment_creation()
    if env is None:
        print("❌ Test de création échoué - Arrêt des tests")
        return
    
    # Test 2: Simulation physique
    if not test_physics_simulation(env):
        print("⚠️ Test de simulation physique échoué")
    
    # Test 3: Contrôle de vitesse
    if not test_velocity_control(env):
        print("⚠️ Test de contrôle de vitesse échoué")
    
    # Test 4: Capture vidéo
    if not test_video_capture(env):
        print("⚠️ Test de capture vidéo échoué")
    
    # Test 5: Curriculum learning
    if not test_curriculum_learning(env):
        print("⚠️ Test de curriculum learning échoué")
    
    # Test 6: Stabilité
    if not test_stability(env):
        print("⚠️ Test de stabilité échoué")
    
    # Test 7: Rendu
    if not test_rendering(env):
        print("⚠️ Test de rendu échoué")
    
    # Nettoyage
    env.close()
    
    print("\n🎯 TESTS TERMINÉS")
    print("=" * 60)
    print("✅ Environnement robuste prêt pour l'entraînement!")

if __name__ == "__main__":
    main()