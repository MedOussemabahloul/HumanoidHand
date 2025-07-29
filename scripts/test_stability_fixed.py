#!/usr/bin/env python3
"""
Script de test de stabilité pour le modèle G1 fingers optimisé
Version corrigée pour résoudre le problème avec l'attribut frictionloss
"""

import mujoco
import numpy as np
import os
import sys
import time
from pathlib import Path

def test_model_loading(model_path):
    """
    Teste le chargement d'un modèle MuJoCo
    
    Args:
        model_path (str): Chemin vers le fichier XML du modèle
        
    Returns:
        tuple: (success, model, error_message)
    """
    try:
        print(f"🔍 Test de chargement: {model_path}")
        
        if not os.path.exists(model_path):
            return False, None, f"Fichier non trouvé: {model_path}"
        
        # Charger le modèle
        model = mujoco.MjModel.from_xml_path(model_path)
        
        print(f"✅ Modèle chargé avec succès!")
        print(f"   - Nombre de corps: {model.nbody}")
        print(f"   - Nombre de joints: {model.njnt}")
        print(f"   - Nombre de degrés de liberté: {model.nv}")
        print(f"   - Nombre d'actuateurs: {model.nu}")
        
        return True, model, None
        
    except Exception as e:
        error_msg = str(e)
        return False, None, error_msg

def test_model_simulation(model, duration=2.0, timestep=None):
    """
    Teste la simulation du modèle
    
    Args:
        model: Modèle MuJoCo
        duration (float): Durée de simulation en secondes
        timestep (float): Pas de temps (utilise celui du modèle si None)
        
    Returns:
        tuple: (success, stats, error_message)
    """
    try:
        print(f"\n🔄 Test de simulation (durée: {duration}s)")
        
        # Créer les données de simulation
        data = mujoco.MjData(model)
        
        # Statistiques
        stats = {
            'steps': 0,
            'avg_time_per_step': 0,
            'max_penetration': 0,
            'energy_initial': 0,
            'energy_final': 0,
            'joint_ranges': {},
            'stability_issues': 0
        }
        
        # Configuration initiale
        mujoco.mj_forward(model, data)
        stats['energy_initial'] = data.energy[0] + data.energy[1] if hasattr(data, 'energy') else 0
        
        # Simulation
        start_time = time.time()
        target_time = duration
        step_times = []
        
        while data.time < target_time:
            step_start = time.time()
            
            # Appliquer des contrôles légers pour tester les actuateurs
            if model.nu > 0:
                for i in range(model.nu):
                    # Contrôle sinusoïdal léger
                    data.ctrl[i] = 0.1 * np.sin(2 * np.pi * 0.5 * data.time + i)
            
            # Faire un pas de simulation
            mujoco.mj_step(model, data)
            
            step_end = time.time()
            step_times.append(step_end - step_start)
            stats['steps'] += 1
            
            # Vérifier la stabilité
            if np.any(np.isnan(data.qpos)) or np.any(np.isinf(data.qpos)):
                stats['stability_issues'] += 1
            
            if np.any(np.abs(data.qpos) > 100):  # Positions extrêmes
                stats['stability_issues'] += 1
            
            # Enregistrer les statistiques tous les 100 pas
            if stats['steps'] % 100 == 0:
                print(f"   Pas {stats['steps']}: t={data.time:.3f}s")
        
        # Calculer les statistiques finales
        total_time = time.time() - start_time
        stats['avg_time_per_step'] = np.mean(step_times) if step_times else 0
        stats['energy_final'] = data.energy[0] + data.energy[1] if hasattr(data, 'energy') else 0
        
        # Analyser les plages de mouvement des joints
        for i in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                # Obtenir la position actuelle du joint
                if i < len(data.qpos):
                    stats['joint_ranges'][joint_name] = data.qpos[i]
        
        print(f"✅ Simulation terminée!")
        print(f"   - {stats['steps']} pas de simulation")
        print(f"   - Temps moyen par pas: {stats['avg_time_per_step']*1000:.3f}ms")
        print(f"   - Problèmes de stabilité: {stats['stability_issues']}")
        
        return True, stats, None
        
    except Exception as e:
        error_msg = str(e)
        return False, None, error_msg

def main():
    """Fonction principale"""
    print("=" * 60)
    print("🚀 TESTS DE STABILITÉ G1 FINGERS OPTIMISÉ (VERSION CORRIGÉE)")
    print("=" * 60)
    
    # Chemins des modèles à tester
    models_to_test = [
        "results/g1_combined.xml",
        "assets/hands/g1_fingers.xml",
        "assets/hands/g1_body.xml"
    ]
    
    # Tester chaque modèle
    for model_path in models_to_test:
        print(f"\n{'='*40}")
        print(f"MODÈLE: {model_path}")
        print(f"{'='*40}")
        
        # Test de chargement
        success, model, error = test_model_loading(model_path)
        
        if not success:
            print(f"❌ Erreur de chargement: {error}")
            continue
        
        # Test de simulation
        success, stats, error = test_model_simulation(model, duration=1.0)
        
        if not success:
            print(f"❌ Erreur de simulation: {error}")
            continue
        
        # Rapport de stabilité
        print(f"\n📊 RAPPORT DE STABILITÉ:")
        print(f"   - Stabilité: {'✅ STABLE' if stats['stability_issues'] == 0 else '⚠️ INSTABLE'}")
        print(f"   - Performance: {1000/stats['avg_time_per_step']:.0f} FPS" if stats['avg_time_per_step'] > 0 else "   - Performance: N/A")
        
        if stats['joint_ranges']:
            print(f"   - Joints actifs: {len(stats['joint_ranges'])}")
    
    print(f"\n{'='*60}")
    print("🎉 Tests terminés!")
    print("💡 Si tous les modèles se chargent sans erreur, le problème frictionloss est résolu.")
    print("💡 Pour une utilisation optimale, utilisez le script build_combine.py pour créer le modèle combiné.")

if __name__ == "__main__":
    main()