#!/usr/bin/env python3
"""
Test rapide de la solution équilibrée
"""

import os
import warnings
import numpy as np
import mujoco

os.environ["MUJOCO_GL"] = "osmesa"
warnings.filterwarnings("ignore")

def test_solution():
    print("🧪 TEST RAPIDE DE LA SOLUTION ÉQUILIBRÉE")
    print("=" * 50)
    
    # 1. Test du modèle XML
    print("1️⃣ Test du modèle XML...")
    try:
        model = mujoco.MjModel.from_xml_path("/workspace/results/g1_combined_balanced.xml")
        data = mujoco.MjData(model)
        print(f"✅ Modèle OK - Timestep: {model.opt.timestep}")
        print(f"📊 DOFs: {model.nv}, Actuateurs: {model.nu}")
    except Exception as e:
        print(f"❌ Erreur modèle: {e}")
        return False
    
    # 2. Test de stabilité
    print("\n2️⃣ Test de stabilité (100 steps)...")
    warnings_count = 0
    
    for step in range(100):
        # Actions aléatoires modérées
        data.ctrl[:] = np.random.uniform(-0.3, 0.3, model.nu)
        mujoco.mj_step(model, data)
        
        # Vérifier stabilité
        if step % 25 == 0:
            stable = not (np.any(np.isnan(data.qpos)) or np.any(np.isinf(data.qpos)) or 
                         np.any(np.isnan(data.qvel)) or np.any(np.isinf(data.qvel)))
            print(f"  Step {step}: {'✅' if stable else '❌'} Stable")
    
    # 3. Test des IDs d'objets
    print("\n3️⃣ Test des IDs d'objets...")
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    right_hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    
    print(f"🎯 Cube ID: {cube_id}")
    print(f"🤚 Main droite ID: {right_hand_id}")
    
    if cube_id >= 0:
        cube_pos = data.xpos[cube_id]
        print(f"📍 Position cube: {cube_pos}")
    
    # 4. Test des actuateurs droits
    print("\n4️⃣ Test des actuateurs droits...")
    right_actuators = []
    for i in range(model.nu):
        actuator_name = model.actuator(i).name
        if 'right' in actuator_name:
            right_actuators.append(i)
    
    print(f"🎛️ Actuateurs droits ({len(right_actuators)}): {right_actuators}")
    
    # 5. Test de reward basique
    print("\n5️⃣ Test de reward...")
    if cube_id >= 0 and right_hand_id >= 0:
        cube_pos = data.xpos[cube_id]
        hand_pos = data.xpos[right_hand_id]
        distance = np.linalg.norm(cube_pos - hand_pos)
        
        # Reward simple
        distance_reward = -distance * 10.0
        height_reward = max(0, cube_pos[2] - 0.9) * 2.0
        contact_reward = data.ncon * 2.0
        
        total_reward = distance_reward + height_reward + contact_reward
        
        print(f"📏 Distance main-cube: {distance:.3f}")
        print(f"📈 Reward distance: {distance_reward:.2f}")
        print(f"📈 Reward hauteur: {height_reward:.2f}")
        print(f"📈 Reward contact: {contact_reward:.2f}")
        print(f"🎯 Reward total: {total_reward:.2f}")
        
        if total_reward > -50:
            print("✅ Rewards OK pour entraînement")
        else:
            print("⚠️ Rewards très négatifs - ajuster les paramètres")
    
    print("\n🎉 TEST TERMINÉ!")
    return True

if __name__ == "__main__":
    test_solution()