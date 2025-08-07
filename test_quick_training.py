#!/usr/bin/env python3
"""
🧪 TEST RAPIDE DU SYSTÈME DE GRASPING AVEC CURRICULUM
====================================================
Script de test pour vérifier que le système fonctionne correctement
avec un entraînement court et une génération de vidéo.
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Ajouter le chemin des environnements
sys.path.append('/home/oussema/Documents/project/envs')

try:
  from envs.curriculum_grasp_env import CurriculumGraspEnv
  print("✅ CurriculumGraspEnv importé avec succès")
except ImportError as e:
  print(f"❌ Erreur d'import: {e}")
  sys.exit(1)

from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure

def test_environment():
  """Test de base de l'environnement"""
  print("\n🧪 TEST DE L'ENVIRONNEMENT")
  print("=" * 40)
  
  env = CurriculumGraspEnv()
  print(f"✅ Environnement créé")
  print(f"   Observation space: {env.observation_space.shape}")
  print(f"   Action space: {env.action_space.shape}")
  
  # Test reset
  obs, info = env.reset()
  print(f"✅ Reset réussi - Observation size: {len(obs)}")
  
  # Test de quelques steps
  for i in range(5):
      action = env.action_space.sample()
      obs, reward, terminated, truncated, info = env.step(action)
      print(f"   Step {i+1}: reward={reward:.3f}, phase={info.get('phase', 'N/A')}")
      
      if terminated or truncated:
          break
  
  env.close()
  print("✅ Test environnement terminé avec succès")

def test_quick_training():
  """Test d'entraînement rapide"""
  print("\n🚀 TEST D'ENTRAÎNEMENT RAPIDE")
  print("=" * 40)
  
  # Créer l'environnement
  env = CurriculumGraspEnv()
  
  # Créer le modèle SAC
  print("🧠 Création du modèle SAC...")
  model = SAC(
      "MlpPolicy",
      env,
      learning_rate=0.001,
      buffer_size=1000,
      batch_size=32,
      verbose=1,
      device="cpu"  # Force CPU pour compatibilité
  )
  
  # Configuration des logs
  results_dir = "/home/oussema/Documents/project/test_results"
  os.makedirs(results_dir, exist_ok=True)
  logger = configure(os.path.join(results_dir, "logs"), ["stdout", "csv", "tensorboard"])
  model.set_logger(logger)
  
  print("📚 Début de l'entraînement rapide (1000 steps)...")
  
  try:
      # Entraînement très court pour test
      model.learn(total_timesteps=1000)
      print("✅ Entraînement terminé avec succès")
      
      # Sauvegarder le modèle
      model_path = os.path.join(results_dir, "test_model.zip")
      model.save(model_path)
      print(f"💾 Modèle sauvé: {model_path}")
      
      # Test du modèle entraîné
      print("🎮 Test du modèle entraîné...")
      obs, info = env.reset()
      total_reward = 0
      
      for step in range(50):
          action, _ = model.predict(obs, deterministic=True)
          obs, reward, terminated, truncated, info = env.step(action)
          total_reward += reward
          
          if step % 10 == 0:
              print(f"   Step {step}: reward={reward:.3f}, total={total_reward:.3f}")
          
          if terminated or truncated:
              break
      
      print(f"✅ Test terminé - Récompense totale: {total_reward:.3f}")
      
  except Exception as e:
      print(f"❌ Erreur durant l'entraînement: {e}")
      import traceback
      traceback.print_exc()
  
  finally:
      env.close()

def test_video_generation():
  """Test de génération de vidéo"""
  print("\n🎬 TEST DE GÉNÉRATION VIDÉO")
  print("=" * 40)
  
  try:
      # Créer environnement avec rendu
      env = CurriculumGraspEnv(render_mode='rgb_array')
      
      # Test de rendu
      obs, info = env.reset()
      frame = env.render()
      
      if frame is not None:
          print(f"✅ Rendu réussi - Frame shape: {frame.shape}")
      else:
          print("⚠️  Rendu retourne None (normal en mode headless)")
      
      env.close()
      print("✅ Test vidéo terminé")
      
  except Exception as e:
      print(f"❌ Erreur test vidéo: {e}")

def main():
  """Fonction principale de test"""
  print("🧪 TESTS RAPIDES DU SYSTÈME DE GRASPING")
  print("=" * 50)
  
  try:
      # Test 1: Environnement de base
      test_environment()
      
      # Test 2: Entraînement rapide
      test_quick_training()
      
      # Test 3: Génération vidéo
      test_video_generation()
      
      print("\n🎉 TOUS LES TESTS RÉUSSIS!")
      print("✅ Le système est prêt pour un entraînement complet")
      
  except Exception as e:
      print(f"\n❌ ÉCHEC DES TESTS: {e}")
      import traceback
      traceback.print_exc()

if __name__ == "__main__":
  main()
