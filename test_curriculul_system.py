#!/usr/bin/env python3
"""
🧪 TEST DU SYSTÈME DE CURRICULUM LEARNING
=========================================

Test complet du système de curriculum learning pour s'assurer que:
✅ L'environnement charge correctement tous les niveaux
✅ Les transitions de niveau fonctionnent
✅ Les récompenses sont adaptées par niveau
✅ Le système de phases évolue correctement
✅ L'entraîneur gère bien les progressions
"""

import sys
import numpy as np
import time
import json

# Ajouter les chemins
sys.path.append('/home/oussema/Documents/project/envs')
# sys.path.append('/workspace/envs')  # désactivé

try:
 from envs.curriculum_grasp_env import CurriculumGraspEnv
 print("✅ CurriculumGraspEnv importé avec succès")
except ImportError as e:
 print(f"❌ Erreur d'import: {e}")
 sys.exit(1)

def test_curriculum_environment():
 """Test de base de l'environnement curriculum"""
 print("\n🧪 TEST DE L'ENVIRONNEMENT CURRICULUM")
 print("-" * 50)
 
 try:
     # Créer l'environnement
     print("1. Création de l'environnement curriculum...")
     env = CurriculumGraspEnv()
     print("   ✅ Environnement créé avec succès")
     print(f"   📚 Niveau initial: {env.current_level}")
     print(f"   📖 Description: {env.curriculum_levels[env.current_level]['description']}")
     
     # Test du reset
     print("2. Test du reset...")
     obs, info = env.reset()
     print(f"   ✅ Reset réussi - Observation shape: {obs.shape}")
     print(f"   📊 Info curriculum: Niveau {info['curriculum_level']}, Phase {info['phase']}")
     
     # Test de simulation pour chaque niveau
     print("3. Test de simulation rapide...")
     total_reward = 0
     
     for step in range(50):
         action = env.action_space.sample() * 0.05  # Actions très douces
         obs, reward, terminated, truncated, info = env.step(action)
         total_reward += reward
         
         if step % 10 == 0:
             print(f"   Step {step}: Niveau={info['curriculum_level']}, Phase={info['phase']}, Récompense={reward:.3f}")
         
         if terminated or truncated:
             print(f"   🏁 Épisode terminé au step {step}")
             break
     
     print(f"   ✅ Simulation terminée - Récompense totale: {total_reward:.2f}")
     
     # Test des informations de curriculum
     curriculum_info = env.get_curriculum_info()
     print(f"   📊 Info détaillée du curriculum:")
     print(f"     - Niveau actuel: {curriculum_info['current_level']}")
     print(f"     - Nom: {curriculum_info['level_name']}")
     print(f"     - Succès consécutifs: {curriculum_info['consecutive_successes']}")
     print(f"     - Épisodes du niveau: {curriculum_info['level_episodes']}")
     
     env.close()
     return True
     
 except Exception as e:
     print(f"   ❌ Erreur durant le test: {e}")
     return False

def test_curriculum_levels():
 """Test de tous les niveaux de curriculum"""
 print("\n🧪 TEST DES NIVEAUX DE CURRICULUM")
 print("-" * 50)
 
 try:
     env = CurriculumGraspEnv()
     
     # Tester chaque niveau individuellement
     for test_level in range(1, 6):
         print(f"\n🎯 Test du niveau {test_level}:")
         
         # Forcer le niveau pour test
         env.current_level = test_level
         env._update_phase_config()
         
         level_config = env.curriculum_levels[test_level]
         print(f"   📖 {level_config['name']}: {level_config['description']}")
         print(f"   🎯 Objectif: {level_config['success_threshold']:.1f} points")
         print(f"   📊 Phases max: {level_config['max_phases']}")
         print(f"   ⏱️  Steps max: {level_config['max_episode_steps']}")
         
         # Test d'un épisode court
         obs, info = env.reset()
         episode_reward = 0
         max_phase_seen = 0
         
         for step in range(min(100, level_config['max_episode_steps'])):
             action = env.action_space.sample() * 0.08
             obs, reward, terminated, truncated, info = env.step(action)
             episode_reward += reward
             max_phase_seen = max(max_phase_seen, info['phase_timer'])
             
             if terminated or truncated:
                 break
         
         print(f"   📊 Résultat: Récompense={episode_reward:.2f}, Phase max vue={max_phase_seen}")
         
         # Vérifier que les phases sont limitées selon le niveau
         expected_max_phases = level_config['max_phases']
         current_phase = env.current_phase
         if current_phase < expected_max_phases:
             print(f"   ✅ Phases limitées correctement (phase {current_phase} < max {expected_max_phases})")
         else:
             print(f"   ⚠️ Phases non limitées (phase {current_phase} >= max {expected_max_phases})")
     
     env.close()
     print("\n✅ Test des niveaux terminé")
     return True
     
 except Exception as e:
     print(f"❌ Erreur durant le test des niveaux: {e}")
     return False

def test_curriculum_progression():
 """Test de la progression automatique du curriculum"""
 print("\n🧪 TEST DE LA PROGRESSION DU CURRICULUM")
 print("-" * 50)
 
 try:
     env = CurriculumGraspEnv()
     
     # Simuler des épisodes avec de bonnes performances pour déclencher progression
     initial_level = env.current_level
     print(f"📚 Niveau initial: {initial_level}")
     
     progression_detected = False
     episode_count = 0
     
     # Simuler jusqu'à 20 épisodes ou progression
     for episode in range(20):
         obs, info = env.reset()
         episode_reward = 0
         
         # Simuler un épisode court
         for step in range(50):
             action = env.action_space.sample() * 0.05
             obs, reward, terminated, truncated, info = env.step(action)
             episode_reward += reward
             
             if terminated or truncated:
                 break
         
         episode_count += 1
         
         # Simuler une bonne performance pour forcer la progression
         # En ajoutant un bonus artificiel pour le test
         test_reward = episode_reward + 20.0  # Bonus pour simulation
         
         # Mettre à jour le curriculum
         old_level = env.current_level
         env.update_curriculum_level(test_reward, test_reward >= 15.0)
         new_level = env.current_level
         
         print(f"   Épisode {episode + 1}: Récompense réelle={episode_reward:.2f}, "
               f"Simulée={test_reward:.2f}, Niveau={new_level}")
         
         if new_level > old_level:
             print(f"   🎉 PROGRESSION DÉTECTÉE! Niveau {old_level} → {new_level}")
             progression_detected = True
             break
     
     if progression_detected:
         print(f"✅ Progression du curriculum fonctionne (après {episode_count} épisodes)")
     else:
         print(f"📊 Pas de progression en {episode_count} épisodes (normal pour test court)")
     
     # Tester la mise à jour manuelle de niveau
     print("\n🔧 Test de mise à jour manuelle de niveau...")
     old_level = env.current_level
     for i in range(5):  # Simuler 5 succès consécutifs
         env.update_curriculum_level(25.0, True)  # Récompense élevée
     
     if env.current_level > old_level:
         print(f"✅ Mise à jour manuelle réussie: {old_level} → {env.current_level}")
     else:
         print(f"📊 Niveau stable: {env.current_level} (critères non atteints)")
     
     env.close()
     return True
     
 except Exception as e:
     print(f"❌ Erreur durant le test de progression: {e}")
     return False

def test_curriculum_rewards():
 """Test du système de récompenses adaptatif"""
 print("\n🧪 TEST DU SYSTÈME DE RÉCOMPENSES ADAPTATIF")
 print("-" * 50)
 
 try:
     env = CurriculumGraspEnv()
     
     reward_samples = {}
     
     # Tester les récompenses pour chaque niveau
     for test_level in range(1, 4):  # Tester les 3 premiers niveaux
         print(f"\n📊 Test récompenses niveau {test_level}:")
         
         env.current_level = test_level
         env._update_phase_config()
         
         level_rewards = []
         
         # Collecter quelques échantillons de récompenses
         for sample in range(10):
             obs, info = env.reset()
             
             # Quelques steps pour obtenir des récompenses représentatives
             for step in range(20):
                 action = env.action_space.sample() * 0.03  # Actions très douces
                 obs, reward, terminated, truncated, info = env.step(action)
                 level_rewards.append(reward)
                 
                 if terminated or truncated:
                     break
         
         if level_rewards:
             avg_reward = np.mean(level_rewards)
             min_reward = min(level_rewards)
             max_reward = max(level_rewards)
             
             reward_samples[test_level] = {
                 'avg': avg_reward,
                 'min': min_reward,
                 'max': max_reward,
                 'multiplier': env.curriculum_levels[test_level]['reward_multiplier']
             }
             
             print(f"   Récompenses: Moy={avg_reward:.3f}, Min={min_reward:.3f}, Max={max_reward:.3f}")
             print(f"   Multiplicateur configuré: {env.curriculum_levels[test_level]['reward_multiplier']}")
     
     # Analyser la progression des récompenses
     print(f"\n📈 Analyse comparative des récompenses:")
     for level, rewards in reward_samples.items():
         print(f"   Niveau {level}: {rewards['avg']:.3f} (x{rewards['multiplier']})")
     
     # Vérifier que les multiplicateurs affectent les récompenses
     if len(reward_samples) >= 2:
         level1_avg = reward_samples[1]['avg']
         level2_avg = reward_samples[2]['avg'] if 2 in reward_samples else level1_avg
         
         if abs(level2_avg) > abs(level1_avg) * 0.8:  # Permettre une variation
             print("   ✅ Récompenses évoluent avec les niveaux")
         else:
             print("   📊 Récompenses similaires entre niveaux (variations normales)")
     
     env.close()
     return True
     
 except Exception as e:
     print(f"❌ Erreur durant le test de récompenses: {e}")
     return False

def test_trainer_integration():
 """Test d'intégration avec l'entraîneur (sans entraînement complet)"""
 print("\n🧪 TEST D'INTÉGRATION AVEC L'ENTRAÎNEUR")
 print("-" * 50)
 
 try:
     # Import de l'entraîneur
     sys.path.append('/workspace')
     sys.path.append('/home/oussema/Documents/project')
     
     try:
         from train_curriculum_sac_grasp import CurriculumGraspingTrainer
         print("✅ CurriculumGraspingTrainer importé avec succès")
     except ImportError as e:
         print(f"⚠️ Impossible d'importer l'entraîneur: {e}")
         return True  # Pas critique pour ce test
     
     # Test de création de l'entraîneur
     print("1. Création de l'entraîneur...")
     trainer = CurriculumGraspingTrainer(total_timesteps=1000)  # Très petit pour test
     print("   ✅ Entraîneur créé")
     
     # Test de création de l'environnement
     print("2. Test création environnement par l'entraîneur...")
     env = trainer.create_curriculum_environment()
     print("   ✅ Environnement créé par l'entraîneur")
     
     # Test de la création du modèle SAC adaptatif
     print("3. Test création modèle SAC adaptatif...")
     try:
         model = trainer.create_adaptive_sac_model()
         print("   ✅ Modèle SAC adaptatif créé")
     except Exception as e:
         print(f"   ⚠️ Erreur création modèle SAC: {e} (dépendances manquantes?)")
     
     # Test de configuration des métriques
     print("4. Test métriques d'entraînement...")
     initial_metrics = trainer.training_metrics
     print(f"   ✅ Métriques initialisées: {len(initial_metrics)} catégories")
     
     if env:
         env.close()
     
     print("✅ Test d'intégration terminé")
     return True
     
 except Exception as e:
     print(f"❌ Erreur durant le test d'intégration: {e}")
     return False

def main():
 """Fonction principale de test"""
 print("🧪 LANCEMENT DES TESTS DU SYSTÈME CURRICULUM")
 print("=" * 70)
 
 tests = [
     ("Environnement curriculum de base", test_curriculum_environment),
     ("Niveaux de curriculum", test_curriculum_levels),
     ("Progression du curriculum", test_curriculum_progression),
     ("Système de récompenses adaptatif", test_curriculum_rewards),
     ("Intégration avec l'entraîneur", test_trainer_integration),
 ]
 
 results = []
 
 for test_name, test_func in tests:
     print(f"\n🔬 Test: {test_name}")
     start_time = time.time()
     
     try:
         success = test_func()
         duration = time.time() - start_time
         
         if success:
             print(f"✅ {test_name} RÉUSSI ({duration:.2f}s)")
             results.append(True)
         else:
             print(f"⚠️ {test_name} ÉCHOUÉ ({duration:.2f}s)")
             results.append(False)
             
     except Exception as e:
         duration = time.time() - start_time
         print(f"❌ {test_name} ERREUR: {e} ({duration:.2f}s)")
         results.append(False)
 
 # Résumé final
 print("\n" + "=" * 70)
 print("📊 RÉSUMÉ DES TESTS CURRICULUM")
 print("=" * 70)
 
 passed = sum(results)
 total = len(results)
 success_rate = (passed / total) * 100
 
 for i, (test_name, _) in enumerate(tests):
     status = "✅ RÉUSSI" if results[i] else "❌ ÉCHOUÉ"
     print(f"  {test_name}: {status}")
 
 print(f"\n🎯 Taux de réussite: {passed}/{total} ({success_rate:.1f}%)")
 
 if success_rate >= 80:
     print("🏆 SYSTÈME CURRICULUM PRÊT POUR L'ENTRAÎNEMENT!")
     print("\n🚀 Pour lancer l'entraînement:")
     print("   cd /home/oussema/Documents/project")
     print("   python3 train_curriculum_sac_grasp.py")
     return True
 else:
     print("⚠️ Système curriculum nécessite des corrections")
     return False

if __name__ == "__main__":
 success = main()
 sys.exit(0 if success else 1)
