#!/usr/bin/env python3
"""
🧪 TEST RAPIDE DU SYSTÈME D'ENTRAÎNEMENT ROBUSTE
=================================================

Script de test rapide pour vérifier que tous les composants fonctionnent
sans problèmes de rendu ou d'affichage.
"""
import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Ajouter le chemin des environnements
sys.path.append('/workspace/envs')

def test_environment_basic():
    """Test basique de l'environnement"""
    print("🧪 Test de création de l'environnement...")
    
    try:
        from envs.robust_curriculum_grasp_env import RobustCurriculumGraspEnv
        
        # Créer l'environnement sans rendu
        env = RobustCurriculumGraspEnv(
            model_path="/workspace/results/g1_combined.xml",
            render_mode=None,
            video_capture=False
        )
        
        print(f"✅ Environnement créé - Niveau: {env.current_level}")
        print(f"   - Espace d'action: {env.action_space.shape}")
        print(f"   - Espace d'observation: {env.observation_space.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur création environnement: {e}")
        return False

def test_observation_types():
    """Test des types d'observation"""
    print("\n🧪 Test des types d'observation...")
    
    try:
        from envs.robust_curriculum_grasp_env import RobustCurriculumGraspEnv
        
        env = RobustCurriculumGraspEnv(
            model_path="/workspace/results/g1_combined.xml",
            render_mode=None,
            video_capture=False
        )
        
        # Reset et obtenir observation
        obs, info = env.reset()
        
        # Vérifier le type
        if isinstance(obs, np.ndarray) and obs.dtype == np.float32:
            print(f"✅ Observation correcte - Shape: {obs.shape}, Type: {obs.dtype}")
        else:
            print(f"❌ Type d'observation incorrect: {type(obs)}, {obs.dtype if hasattr(obs, 'dtype') else 'N/A'}")
            return False
        
        # Vérifier qu'il n'y a pas de NaN/Inf
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print("❌ Observation contient NaN/Inf")
            return False
        
        print("✅ Types d'observation corrects")
        return True
        
    except Exception as e:
        print(f"❌ Erreur test observation: {e}")
        return False

def test_action_application():
    """Test de l'application des actions"""
    print("\n🧪 Test de l'application des actions...")
    
    try:
        from envs.robust_curriculum_grasp_env import RobustCurriculumGraspEnv
        
        env = RobustCurriculumGraspEnv(
            model_path="/workspace/results/g1_combined.xml",
            render_mode=None,
            video_capture=False
        )
        
        obs, info = env.reset()
        
        # Tester plusieurs actions
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if not isinstance(reward, (int, float)):
                print(f"❌ Type de récompense incorrect: {type(reward)}")
                return False
            
            if not isinstance(obs, np.ndarray):
                print(f"❌ Type d'observation incorrect après step: {type(obs)}")
                return False
        
        print("✅ Application des actions correcte")
        return True
        
    except Exception as e:
        print(f"❌ Erreur test actions: {e}")
        return False

def test_stability():
    """Test de la stabilité du système"""
    print("\n🧪 Test de la stabilité...")
    
    try:
        from envs.robust_curriculum_grasp_env import RobustCurriculumGraspEnv
        
        env = RobustCurriculumGraspEnv(
            model_path="/workspace/results/g1_combined.xml",
            render_mode=None,
            video_capture=False
        )
        
        # Test de plusieurs épisodes
        for episode in range(3):
            obs, info = env.reset()
            episode_reward = 0
            
            for step in range(50):  # 50 steps max par épisode
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                # Vérifier la stabilité des observations
                if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                    print(f"❌ Instabilité détectée à l'épisode {episode}, step {step}")
                    return False
                
                if terminated or truncated:
                    break
            
            print(f"   - Épisode {episode + 1}: Reward={episode_reward:.2f}")
        
        print("✅ Stabilité confirmée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur test stabilité: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🧪 DÉMARRAGE DES TESTS RAPIDES")
    print("=" * 50)
    
    tests = [
        ("Création environnement", test_environment_basic),
        ("Types d'observation", test_observation_types),
        ("Application actions", test_action_application),
        ("Stabilité", test_stability)
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Erreur lors du test {name}: {e}")
            results.append((name, False))
    
    # Résumé
    print(f"\n{'='*50}")
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHOUÉ"
        print(f"{name:25} : {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Résultat: {passed}/{total} tests réussis")
    
    if passed == total:
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print("Le système est prêt pour l'entraînement.")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) échoué(s)")
        print("Corrigez les problèmes avant de lancer l'entraînement.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
