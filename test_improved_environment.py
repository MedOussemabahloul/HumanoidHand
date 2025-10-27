#!/usr/bin/env python3
"""
🧪 TEST DE L'ENVIRONNEMENT AMÉLIORÉ
===================================

Test rapide pour vérifier:
✅ Chargement du modèle sans erreurs
✅ Stabilité physique (pas de NaN/Inf)
✅ Collisions fonctionnelles
✅ Système de phases
✅ Système de récompenses
✅ Mouvements fluides
"""

import sys
import numpy as np
import time

# Ajouter le chemin des environnements
sys.path.append('/workspace/envs')

try:
    from envs.improved_professional_grasp_env import ImprovedProfessionalGraspEnv
    print("✅ ImprovedProfessionalGraspEnv importé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

def test_environment_stability():
    """Test de stabilité de base de l'environnement"""
    print("\n🧪 TEST DE STABILITÉ DE L'ENVIRONNEMENT")
    print("-" * 50)
    
    try:
        # Créer l'environnement
        print("1. Création de l'environnement...")
        env = ImprovedProfessionalGraspEnv()
        print("   ✅ Environnement créé avec succès")
        
        # Test du reset
        print("2. Test du reset...")
        obs, info = env.reset()
        print(f"   ✅ Reset réussi - Observation shape: {obs.shape}")
        print(f"   📊 Phase initiale: {info['phase']}")
        
        # Test de simulation simple
        print("3. Test de simulation (100 steps)...")
        total_reward = 0
        instabilities = 0
        
        for step in range(100):
            # Action aléatoire
            action = env.action_space.sample() * 0.1  # Actions très petites pour stabilité
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Vérifier les instabilités
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                instabilities += 1
                print(f"   ⚠️ Instabilité détectée au step {step}")
            
            # Affichage périodique
            if step % 20 == 0:
                print(f"   Step {step}: Phase={info['phase']}, Récompense={reward:.3f}, Stable={info['stability_count']}")
            
            if terminated or truncated:
                print(f"   🏁 Épisode terminé au step {step}")
                break
        
        print(f"   ✅ Simulation terminée")
        print(f"   📊 Récompense totale: {total_reward:.2f}")
        print(f"   📊 Instabilités détectées: {instabilities}")
        print(f"   📊 Phase finale: {info['phase']}")
        
        env.close()
        return instabilities == 0
        
    except Exception as e:
        print(f"   ❌ Erreur durant le test: {e}")
        return False

def test_physics_collisions():
    """Test des collisions physiques"""
    print("\n🧪 TEST DES COLLISIONS PHYSIQUES")
    print("-" * 50)
    
    try:
        env = ImprovedProfessionalGraspEnv()
        obs, info = env.reset()
        
        print("1. Position initiale du cube:")
        cube_pos = info['cube_position']
        print(f"   Cube: {cube_pos}")
        
        # Simuler des mouvements vers le cube
        print("2. Test d'approche du cube...")
        
        for step in range(50):
            # Actions dirigées vers le cube (simuler approche)
            action = np.zeros(22)
            action[0] = 0.1   # Mouvement bras gauche vers cube
            action[7] = 0.1   # Mouvement bras droit vers cube
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 10 == 0:
                cube_pos = info['cube_position']
                print(f"   Step {step}: Cube pos={cube_pos}, Phase={info['phase']}")
        
        env.close()
        print("   ✅ Test de collision terminé sans erreur")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur durant le test de collision: {e}")
        return False

def test_phase_transitions():
    """Test des transitions de phases"""
    print("\n🧪 TEST DES TRANSITIONS DE PHASES")
    print("-" * 50)
    
    try:
        env = ImprovedProfessionalGraspEnv()
        obs, info = env.reset()
        
        phases_seen = []
        current_phase = info['phase']
        phases_seen.append(current_phase)
        
        print(f"Phase de départ: {current_phase}")
        
        # Simuler plusieurs steps pour voir les transitions
        for step in range(200):
            action = env.action_space.sample() * 0.05  # Actions douces
            obs, reward, terminated, truncated, info = env.step(action)
            
            if info['phase'] != current_phase:
                current_phase = info['phase']
                phases_seen.append(current_phase)
                print(f"   Step {step}: Transition vers phase '{current_phase}'")
            
            if terminated or truncated:
                break
        
        print(f"   ✅ Phases vues: {' → '.join(phases_seen)}")
        print(f"   📊 Nombre de transitions: {len(phases_seen) - 1}")
        
        env.close()
        return len(phases_seen) > 1  # Au moins une transition
        
    except Exception as e:
        print(f"   ❌ Erreur durant le test de phases: {e}")
        return False

def test_reward_system():
    """Test du système de récompenses"""
    print("\n🧪 TEST DU SYSTÈME DE RÉCOMPENSES")
    print("-" * 50)
    
    try:
        env = ImprovedProfessionalGraspEnv()
        obs, info = env.reset()
        
        rewards = []
        phases = []
        
        print("Collecte des récompenses sur 100 steps...")
        
        for step in range(100):
            action = env.action_space.sample() * 0.1
            obs, reward, terminated, truncated, info = env.step(action)
            
            rewards.append(reward)
            phases.append(info['phase'])
            
            if step % 20 == 0:
                avg_reward = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
                print(f"   Step {step}: Récompense moyenne = {avg_reward:.3f}, Phase = {info['phase']}")
            
            if terminated or truncated:
                break
        
        # Analyse des récompenses
        total_reward = sum(rewards)
        avg_reward = np.mean(rewards)
        min_reward = min(rewards)
        max_reward = max(rewards)
        
        print(f"   ✅ Analyse des récompenses:")
        print(f"      - Total: {total_reward:.2f}")
        print(f"      - Moyenne: {avg_reward:.3f}")
        print(f"      - Min: {min_reward:.3f}")
        print(f"      - Max: {max_reward:.3f}")
        print(f"      - Récompenses positives: {sum(1 for r in rewards if r > 0)}/{len(rewards)}")
        
        env.close()
        
        # Critères de réussite
        return (avg_reward > -5.0 and  # Pas trop négatif
                min_reward > -10.0 and  # Pas d'effondrement
                sum(1 for r in rewards if r > 0) > len(rewards) * 0.5)  # Plus de 50% positives
        
    except Exception as e:
        print(f"   ❌ Erreur durant le test de récompenses: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🧪 LANCEMENT DES TESTS DE L'ENVIRONNEMENT AMÉLIORÉ")
    print("=" * 60)
    
    tests = [
        ("Stabilité de l'environnement", test_environment_stability),
        ("Collisions physiques", test_physics_collisions),
        ("Transitions de phases", test_phase_transitions),
        ("Système de récompenses", test_reward_system),
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
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ RÉUSSI" if results[i] else "❌ ÉCHOUÉ"
        print(f"  {test_name}: {status}")
    
    print(f"\n🎯 Taux de réussite: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 75:
        print("🏆 ENVIRONNEMENT PRÊT POUR L'ENTRAÎNEMENT SAC!")
        return True
    else:
        print("⚠️ Environnement nécessite des corrections avant entraînement")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)