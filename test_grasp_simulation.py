#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de la simulation de grasping pour G1 Robot
Vérifie que tous les composants fonctionnent correctement
"""

import numpy as np
import mujoco
import mujoco_viewer
import time
import os

def test_model_loading():
    """Teste le chargement du modèle"""
    print("=== Test de chargement du modèle ===")
    
    try:
        model = mujoco.MjModel.from_xml_path("results/g1_combined.xml")
        data = mujoco.MjData(model)
        print("✅ Modèle chargé avec succès")
        print(f"   Nombre de joints: {model.njnt}")
        print(f"   Nombre d'actuateurs: {model.nu}")
        print(f"   Nombre de capteurs: {model.nsensor}")
        return model, data
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None, None

def test_finger_joints(model, data):
    """Teste les joints des doigts"""
    print("\n=== Test des joints des doigts ===")
    
    finger_joint_names = [
        "left_thumb_joint_0", "left_thumb_joint_1",
        "left_index_joint_0", "left_index_joint_1",
        "left_middle_joint_0", "left_middle_joint_1",
        "left_ring_joint_0", "left_ring_joint_1",
        "right_thumb_joint_0", "right_thumb_joint_1",
        "right_index_joint_0", "right_index_joint_1",
        "right_middle_joint_0", "right_middle_joint_1",
        "right_ring_joint_0", "right_ring_joint_1"
    ]
    
    joint_ids = []
    for name in finger_joint_names:
        try:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            joint_ids.append(joint_id)
            print(f"✅ Joint trouvé: {name} (ID: {joint_id})")
        except:
            print(f"❌ Joint non trouvé: {name}")
    
    return joint_ids

def test_finger_actuators(model, data):
    """Teste les actuateurs des doigts"""
    print("\n=== Test des actuateurs des doigts ===")
    
    finger_actuator_names = [
        "act_left_thumb_joint_0", "act_left_thumb_joint_1",
        "act_left_index_joint_0", "act_left_index_joint_1",
        "act_left_middle_joint_0", "act_left_middle_joint_1",
        "act_left_ring_joint_0", "act_left_ring_joint_1",
        "act_right_thumb_joint_0", "act_right_thumb_joint_1",
        "act_right_index_joint_0", "act_right_index_joint_1",
        "act_right_middle_joint_0", "act_right_middle_joint_1",
        "act_right_ring_joint_0", "act_right_ring_joint_1"
    ]
    
    actuator_ids = []
    for name in finger_actuator_names:
        try:
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            actuator_ids.append(actuator_id)
            print(f"✅ Actuateur trouvé: {name} (ID: {actuator_id})")
        except:
            print(f"❌ Actuateur non trouvé: {name}")
    
    return actuator_ids

def test_cube_detection(model, data):
    """Teste la détection du cube"""
    print("\n=== Test de détection du cube ===")
    
    try:
        cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        print(f"✅ Cube trouvé (ID: {cube_id})")
        
        # Obtenir la position du cube
        cube_pos = data.xpos[cube_id]
        print(f"   Position du cube: {cube_pos}")
        
        return cube_id
    except:
        print("❌ Cube non trouvé")
        return None

def test_simple_simulation(model, data, joint_ids, actuator_ids, cube_id):
    """Teste une simulation simple"""
    print("\n=== Test de simulation simple ===")
    
    if not joint_ids or not actuator_ids or cube_id is None:
        print("❌ Impossible de tester la simulation - composants manquants")
        return False
    
    try:
        # Reset de la simulation
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        
        # Position initiale du cube
        initial_cube_pos = data.xpos[cube_id].copy()
        print(f"Position initiale du cube: {initial_cube_pos}")
        
        # Test de mouvement des doigts
        print("Test de mouvement des doigts...")
        
        # Ouvrir les doigts
        for actuator_id in actuator_ids:
            data.ctrl[actuator_id] = 0.0
        
        mujoco.mj_step(model, data)
        print("   Doigts ouverts")
        
        # Fermer les doigts partiellement
        for actuator_id in actuator_ids:
            data.ctrl[actuator_id] = 0.5
        
        mujoco.mj_step(model, data)
        print("   Doigts partiellement fermés")
        
        # Vérifier la position du cube
        final_cube_pos = data.xpos[cube_id]
        print(f"Position finale du cube: {final_cube_pos}")
        
        # Vérifier si le cube a bougé
        cube_movement = np.linalg.norm(final_cube_pos - initial_cube_pos)
        print(f"Mouvement du cube: {cube_movement:.6f}")
        
        if cube_movement > 0.001:
            print("✅ Le cube a bougé - simulation fonctionnelle")
            return True
        else:
            print("⚠️ Le cube n'a pas bougé - vérifier la configuration")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors de la simulation: {e}")
        return False

def test_viewer(model, data):
    """Teste le viewer MuJoCo"""
    print("\n=== Test du viewer ===")
    
    try:
        viewer = mujoco_viewer.MujocoViewer(model, data)
        print("✅ Viewer créé avec succès")
        
        # Afficher pendant quelques secondes
        print("Affichage de la simulation pendant 3 secondes...")
        start_time = time.time()
        
        while time.time() - start_time < 3.0:
            mujoco.mj_step(model, data)
            viewer.render()
            time.sleep(0.01)
        
        viewer.close()
        print("✅ Viewer fermé avec succès")
        return True
        
    except Exception as e:
        print(f"❌ Erreur avec le viewer: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🧪 Test de la simulation de grasping pour G1 Robot")
    print("=" * 50)
    
    # Test 1: Chargement du modèle
    model, data = test_model_loading()
    if model is None:
        print("❌ Impossible de continuer - modèle non chargé")
        return
    
    # Test 2: Joints des doigts
    joint_ids = test_finger_joints(model, data)
    
    # Test 3: Actuateurs des doigts
    actuator_ids = test_finger_actuators(model, data)
    
    # Test 4: Détection du cube
    cube_id = test_cube_detection(model, data)
    
    # Test 5: Simulation simple
    simulation_ok = test_simple_simulation(model, data, joint_ids, actuator_ids, cube_id)
    
    # Test 6: Viewer
    viewer_ok = test_viewer(model, data)
    
    # Résumé
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)
    
    print(f"Modèle chargé: {'✅' if model else '❌'}")
    print(f"Joints des doigts: {'✅' if joint_ids else '❌'} ({len(joint_ids)}/16)")
    print(f"Actuateurs des doigts: {'✅' if actuator_ids else '❌'} ({len(actuator_ids)}/16)")
    print(f"Cube détecté: {'✅' if cube_id is not None else '❌'}")
    print(f"Simulation simple: {'✅' if simulation_ok else '❌'}")
    print(f"Viewer: {'✅' if viewer_ok else '❌'}")
    
    if all([joint_ids, actuator_ids, cube_id is not None, simulation_ok, viewer_ok]):
        print("\n🎉 Tous les tests sont passés! La simulation peut être lancée.")
        print("Vous pouvez maintenant exécuter: python grasp_simulation_simple.py")
    else:
        print("\n⚠️ Certains tests ont échoué. Vérifiez la configuration.")
        print("Vérifiez que le fichier results/g1_combined.xml existe et est valide.")

if __name__ == "__main__":
    main()