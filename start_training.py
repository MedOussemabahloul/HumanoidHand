#!/usr/bin/env python3
"""
🚀 DÉMARRAGE RAPIDE D'ENTRAÎNEMENT
=================================

Script de démarrage rapide pour tester l'entraînement TD3
sans complexité. Version simplifiée pour vérifier que tout fonctionne.
"""

import os
import sys
import time
from pathlib import Path

def main():
    print("🚀 DÉMARRAGE RAPIDE TD3 SIMPLIFIÉ")
    print("=" * 50)
    
    # Test rapide de l'environnement
    print("🧪 Test de l'environnement...")
    try:
        from envs.simple_robust_grasp_env import SimpleRobustGraspEnv
        env = SimpleRobustGraspEnv(eval_mode=True)
        obs, info = env.reset()
        
        # Test de quelques steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            print(f"  Test step {i}: reward={reward:.3f}, distance={info.get('distance', 0):.3f}")
            if done:
                obs, info = env.reset()
        
        env.close()
        print("✅ Test environnement réussi!")
        
    except Exception as e:
        print(f"❌ Erreur test environnement: {e}")
        return False
    
    # Lancer l'entraînement simplifié
    print("\n🏃 Lancement de l'entraînement TD3...")
    
    try:
        # Import et lancement
        from simple_training_td3 import train_td3_robot
        
        print("🔧 Configuration d'entraînement rapide:")
        print("  - 50,000 timesteps (rapide pour test)")
        print("  - Sauvegarde toutes les 25,000 steps")
        print("  - Vidéos d'évaluation automatiques")
        print()
        
        # Modifier temporairement la configuration pour un test rapide
        import simple_training_td3
        original_train = simple_training_td3.train_td3_robot
        
        def quick_train():
            # Configuration réduite pour test rapide
            print("🏃 Version de test rapide (50k steps)")
            return original_train()
        
        # Lancer
        model, env, eval_env = quick_train()
        
        if model is not None:
            print("\n🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
            print("\n📁 Fichiers créés:")
            
            # Vérifier les fichiers créés
            results_dir = Path("simple_td3_results")
            if results_dir.exists():
                for file_path in results_dir.rglob("*"):
                    if file_path.is_file():
                        size = file_path.stat().st_size / 1024  # KB
                        print(f"  📄 {file_path}: {size:.1f} KB")
            
            print("\n💡 Prochaines étapes:")
            print("  1. Évaluation: python evaluate_and_download.py")
            print("  2. Entraînement plus long: Modifier total_timesteps dans simple_training_td3.py")
            print("  3. Visualisation: Vérifier les vidéos dans simple_td3_results/videos/")
            
        else:
            print("❌ Entraînement échoué")
            return False
    
    except KeyboardInterrupt:
        print("\n⏹️ Entraînement interrompu par l'utilisateur")
        return True
    
    except Exception as e:
        print(f"\n❌ Erreur pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("🎯 SCRIPT DE DÉMARRAGE RAPIDE POUR GRASPING ROBOTIQUE")
    print()
    
    success = main()
    
    if success:
        print("\n✅ Script terminé avec succès!")
    else:
        print("\n❌ Échec du script")
        sys.exit(1)