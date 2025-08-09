#!/usr/bin/env python3
"""
🧪 TEST RAPIDE D'ENTRAÎNEMENT
===========================

Test ultra-rapide avec seulement 1000 steps pour vérifier 
que l'entraînement fonctionne sans erreurs.
"""

import sys
import os
import numpy as np
from pathlib import Path

def quick_test():
    print("🧪 TEST RAPIDE D'ENTRAÎNEMENT (1000 steps)")
    print("=" * 50)
    
    try:
        # Imports
        from stable_baselines3 import TD3
        from stable_baselines3.common.monitor import Monitor
        from envs.simple_robust_grasp_env import SimpleRobustGraspEnv
        
        print("✅ Imports réussis")
        
        # Créer l'environnement
        env = SimpleRobustGraspEnv(eval_mode=False)
        env = Monitor(env)
        print("✅ Environnement créé")
        
        # Créer le modèle avec configuration minimale
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=10000,  # Plus petit pour le test
            learning_starts=1000,
            batch_size=64,      # Plus petit
            tau=0.02,
            gamma=0.98,
            verbose=1,
            device='cpu'  # Forcer CPU pour plus de stabilité
        )
        print("✅ Modèle TD3 créé")
        
        # Test de quelques épisodes manuels d'abord
        print("\n🔍 Test de 3 épisodes manuels...")
        for episode in range(3):
            obs, info = env.reset()
            episode_reward = 0
            for step in range(50):  # Épisodes courts
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            print(f"  Épisode {episode+1}: Reward={episode_reward:.2f}, Steps={step+1}")
        
        print("✅ Épisodes manuels réussis")
        
        # Test d'entraînement ultra-court
        print("\n🏃 Test d'entraînement (1000 steps)...")
        
        start_time = time.time()
        model.learn(
            total_timesteps=1000,  # Très court pour test
            log_interval=1,
            progress_bar=True
        )
        training_time = time.time() - start_time
        
        print(f"✅ Entraînement terminé en {training_time:.1f} secondes")
        
        # Test du modèle entraîné
        print("\n🎯 Test du modèle entraîné...")
        obs, info = env.reset()
        for i in range(10):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i}: reward={reward:.3f}, distance={info.get('distance', 0):.3f}")
            if terminated or truncated:
                obs, info = env.reset()
        
        # Sauvegarder le modèle test
        test_dir = Path("test_results")
        test_dir.mkdir(exist_ok=True)
        model_path = test_dir / "test_model.zip"
        model.save(model_path)
        print(f"✅ Modèle test sauvegardé: {model_path}")
        
        # Fermeture
        env.close()
        print("✅ Test complet réussi!")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur pendant le test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import time
    
    print("🚀 TEST RAPIDE DU SYSTÈME D'ENTRAÎNEMENT")
    print("Ceci vérifie que tout fonctionne avant un entraînement complet")
    print()
    
    success = quick_test()
    
    if success:
        print("\n🎉 TEST RÉUSSI!")
        print("✅ Le système d'entraînement fonctionne correctement")
        print("💡 Vous pouvez maintenant lancer:")
        print("   - python3 simple_training_td3.py (entraînement complet)")
        print("   - python3 start_training.py (entraînement guidé)")
    else:
        print("\n❌ TEST ÉCHOUÉ!")
        print("Il y a un problème avec la configuration")
        sys.exit(1)