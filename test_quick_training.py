#!/usr/bin/env python3
"""
🧪 TEST RAPIDE DE L'ENTRAÎNEMENT
=================================

Script de test rapide pour vérifier que l'entraînement fonctionne sans erreurs.
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

def test_environment_basic():
    """Test basique de l'environnement"""
    print("\n🧪 Test basique de l'environnement...")
    
    try:
        # Créer l'environnement
        env = RobustCurriculumGraspEnv(
            model_path="/workspace/results/g1_combined.xml",
            render_mode=None,
            video_capture=False
        )
        
        print("✅ Environnement créé")
        
        # Test de reset
        obs, info = env.reset()
        print(f"✅ Reset réussi - Observation shape: {obs.shape}")
        print(f"✅ Types d'observation: {obs.dtype}")
        
        # Test de quelques steps
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"  Step {step+1}: Reward={reward:.2f}, Obs shape={obs.shape}")
            
            if terminated or truncated:
                break
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Erreur test basique: {e}")
        return False

def test_observation_types():
    """Test des types d'observation"""
    print("\n🧪 Test des types d'observation...")
    
    try:
        env = RobustCurriculumGraspEnv(
            model_path="/workspace/results/g1_combined.xml",
            render_mode=None,
            video_capture=False
        )
        
        obs, info = env.reset()
        
        # Vérifier le type
        if obs.dtype == np.float32:
            print("✅ Type d'observation correct: float32")
        else:
            print(f"⚠️ Type d'observation incorrect: {obs.dtype}")
        
        # Vérifier qu'il n'y a pas de NaN/Inf
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print("❌ Observation contient NaN/Inf")
            return False
        else:
            print("✅ Observation sans NaN/Inf")
        
        # Vérifier la forme
        expected_shape = (59,)  # Basé sur le test précédent
        if obs.shape == expected_shape:
            print(f"✅ Forme d'observation correcte: {obs.shape}")
        else:
            print(f"⚠️ Forme d'observation inattendue: {obs.shape} (attendu: {expected_shape})")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Erreur test types: {e}")
        return False

def test_action_application():
    """Test de l'application des actions"""
    print("\n🧪 Test de l'application des actions...")
    
    try:
        env = RobustCurriculumGraspEnv(
            model_path="/workspace/results/g1_combined.xml",
            render_mode=None,
            video_capture=False
        )
        
        obs, info = env.reset()
        
        # Test avec différentes actions
        for i in range(5):
            # Action aléatoire
            action = env.action_space.sample()
            
            # Vérifier que l'action est valide
            if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                print(f"❌ Action {i} contient NaN/Inf")
                continue
            
            # Appliquer l'action
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"  Action {i+1}: Action shape={action.shape}, Reward={reward:.2f}")
            
            if terminated or truncated:
                break
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Erreur test actions: {e}")
        return False

def test_stability():
    """Test de stabilité"""
    print("\n🧪 Test de stabilité...")
    
    try:
        env = RobustCurriculumGraspEnv(
            model_path="/workspace/results/g1_combined.xml",
            render_mode=None,
            video_capture=False
        )
        
        obs, info = env.reset()
        
        stability_issues = 0
        
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Vérifier la stabilité
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                print(f"⚠️ Instabilité détectée à l'étape {step}")
                stability_issues += 1
            
            if terminated or truncated:
                break
        
        if stability_issues == 0:
            print("✅ Aucun problème de stabilité détecté")
        else:
            print(f"⚠️ {stability_issues} problèmes de stabilité détectés")
        
        env.close()
        return stability_issues == 0
        
    except Exception as e:
        print(f"❌ Erreur test stabilité: {e}")
        return False

def main():
    """Fonction principale de test rapide"""
    print("🧪 TEST RAPIDE DE L'ENTRAÎNEMENT")
    print("=" * 50)
    
    tests = [
        ("Test basique", test_environment_basic),
        ("Test types d'observation", test_observation_types),
        ("Test application des actions", test_action_application),
        ("Test stabilité", test_stability)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for name, test_func in tests:
        print(f"\n{'='*20} {name} {'='*20}")
        if test_func():
            passed_tests += 1
        else:
            print(f"⚠️ Test {name} échoué")
    
    print(f"\n🎯 RÉSULTAT DU TEST RAPIDE")
    print("=" * 50)
    print(f"📊 Résultat: {passed_tests}/{total_tests} tests réussis")
    
    if passed_tests == total_tests:
        print("✅ Système prêt pour l'entraînement complet!")
    else:
        print("⚠️ Système nécessite des corrections avant l'entraînement")

if __name__ == "__main__":
    main()
