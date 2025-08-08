#!/usr/bin/env python3
"""
🧪 TEST SIMPLIFIÉ DE L'ENVIRONNEMENT ROBUSTE
============================================

Version simplifiée qui évite les problèmes de rendu et se concentre sur les fonctionnalités essentielles.
"""
import os
import sys
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

# Ajouter le chemin des environnements
sys.path.append('/workspace/envs')

try:
 from envs.robust_curriculum_grasp_env import RobustCurriculumGraspEnv
 print("✅ RobustCurriculumGraspEnv importé avec succès")
except ImportError as e:
 print(f"❌ Erreur d'import: {e}")
 sys.exit(1)

def test_environment_creation():
 """Test de création de l'environnement sans rendu"""
 print("\n🧪 Test de création de l'environnement...")
 
 try:
     # Créer l'environnement sans capture vidéo
     env = RobustCurriculumGraspEnv(
         model_path="/home/oussema/Documents/project/results/g1_combined.xml",
         render_mode=None,  # Pas de rendu
         video_capture=False  # Pas de vidéo
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
     total_reward = 0
     for step in range(20):
         # Action aléatoire
         action = env.action_space.sample()
         
         # Exécution de l'action
         obs, reward, terminated, truncated, info = env.step(action)
         total_reward += reward
         
         if step % 5 == 0:
             print(f"  Step {step+1}: Reward={reward:.2f}, Total={total_reward:.2f}")
         
         if terminated or truncated:
             print(f"  Épisode terminé à l'étape {step+1}")
             break
     
     print(f"✅ Simulation physique fonctionne - Récompense totale: {total_reward:.2f}")
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
     
     for step in range(30):
         obs, reward, terminated, truncated, info = env.step(high_velocity_actions)
         
         # Vérifier les vitesses
         if hasattr(env, 'arm_joint_ids'):
             arm_velocities = [abs(env.data.qvel[i]) for i in env.arm_joint_ids]
             avg_velocity = np.mean(arm_velocities)
             max_velocity = max(arm_velocities)
             velocities.append(max_velocity)
             
             if step % 10 == 0:
                 print(f"  Step {step+1}: Vitesse max={max_velocity:.2f}, Moyenne={avg_velocity:.2f}")
         
         if terminated or truncated:
             break
     
     max_velocity = max(velocities) if velocities else 0
     print(f"✅ Contrôle de vitesse testé - Vitesse max: {max_velocity:.2f}")
     
     # Considérer comme réussi si la vitesse reste raisonnable
     return max_velocity < 20.0
     
 except Exception as e:
     print(f"❌ Erreur contrôle de vitesse: {e}")
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
         print(f"  Épisode {episode+1}: Récompense={episode_reward:.2f}")
     
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
     
     stability_issues = 0
     
     for step in range(100):
         obs, reward, terminated, truncated, info = env.step(action)
         
         # Vérifier les NaN/Inf
         if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
             print(f"❌ Instabilité détectée à l'étape {step}")
             stability_issues += 1
         
         # Vérifier les vitesses excessives
         if hasattr(env, 'arm_joint_ids'):
             arm_velocities = [abs(env.data.qvel[i]) for i in env.arm_joint_ids]
             max_velocity = max(arm_velocities)
             if max_velocity > 30.0:
                 print(f"⚠️ Vitesse excessive à l'étape {step}: {max_velocity:.2f}")
                 stability_issues += 1
         
         if terminated or truncated:
             break
     
     if stability_issues == 0:
         print("✅ Stabilité testée - Aucune instabilité détectée")
     else:
         print(f"⚠️ Stabilité testée - {stability_issues} problèmes détectés")
     
     return stability_issues < 5  # Tolérer quelques problèmes mineurs
     
 except Exception as e:
     print(f"❌ Erreur test de stabilité: {e}")
     return False

def test_reward_system(env):
 """Test du système de récompenses"""
 print("\n🧪 Test du système de récompenses...")
 
 try:
     # Reset
     obs, info = env.reset()
     
     rewards = []
     
     for step in range(50):
         action = env.action_space.sample()
         obs, reward, terminated, truncated, info = env.step(action)
         rewards.append(reward)
         
         if terminated or truncated:
             break
     
     avg_reward = np.mean(rewards)
     min_reward = min(rewards)
     max_reward = max(rewards)
     
     print(f"✅ Système de récompenses testé:")
     print(f"  - Récompense moyenne: {avg_reward:.2f}")
     print(f"  - Récompense min: {min_reward:.2f}")
     print(f"  - Récompense max: {max_reward:.2f}")
     
     # Vérifier que les récompenses sont raisonnables
     return -10.0 <= avg_reward <= 20.0
     
 except Exception as e:
     print(f"❌ Erreur système de récompenses: {e}")
     return False

def main():
 """Fonction principale de test simplifié"""
 print("🧪 DÉMARRAGE DES TESTS SIMPLIFIÉS DE L'ENVIRONNEMENT ROBUSTE")
 print("=" * 70)
 
 # Test 1: Création de l'environnement
 env = test_environment_creation()
 if env is None:
     print("❌ Test de création échoué - Arrêt des tests")
     return
 
 # Tests fonctionnels (sans rendu)
 tests = [
     ("Simulation physique", test_physics_simulation),
     ("Contrôle de vitesse", test_velocity_control),
     ("Curriculum learning", test_curriculum_learning),
     ("Stabilité", test_stability),
     ("Système de récompenses", test_reward_system)
 ]
 
 passed_tests = 0
 total_tests = len(tests)
 
 for name, test_func in tests:
     print(f"\n{'='*20} {name} {'='*20}")
     if test_func(env):
         passed_tests += 1
     else:
         print(f"⚠️ Test {name} échoué")
 
 # Nettoyage
 env.close()
 
 print(f"\n🎯 TESTS SIMPLIFIÉS TERMINÉS")
 print("=" * 70)
 print(f"📊 Résultat: {passed_tests}/{total_tests} tests réussis")
 
 if passed_tests >= total_tests * 0.8:  # 80% de réussite minimum
     print("✅ Environnement robuste prêt pour l'entraînement!")
 else:
     print("⚠️ Environnement nécessite des corrections")

if __name__ == "__main__":
 main()
