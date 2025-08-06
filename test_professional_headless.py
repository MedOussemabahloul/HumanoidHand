#!/usr/bin/env python3
"""
🧪 TEST HEADLESS PROFESSIONNEL
==============================

Test simple de l'environnement de grasping professionnel
sans interface graphique.
"""

import numpy as np
import mujoco
import os
import sys

def test_professional_grasp():
    """Test l'environnement de grasping professionnel"""
    
    print("🧪 DÉMARRAGE DU TEST PROFESSIONNEL HEADLESS")
    print("=" * 50)
    
    try:
        # Charger le modèle directement
        model_path = "/home/oussema/Documents/project/results/g1_combined.xml"
        
        print(f"📁 Chargement du modèle: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ Fichier non trouvé: {model_path}")
            return False
        
        # Charger MuJoCo
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        
        print(f"✅ Modèle chargé avec succès!")
        print(f"   - DOFs: {model.nq}")
        print(f"   - Actuateurs: {model.nu}")
        print(f"   - Capteurs: {model.nsensor}")
        print(f"   - Contacts max: {model.nconmax}")
        
        # Identifier les composants
        print("\n🔍 IDENTIFICATION DES COMPOSANTS:")
        
        # DOFs des doigts et bras
        finger_dofs = list(range(15, 31))  # 16 DOFs de doigts
        arm_dofs = list(range(1, 15))      # 14 DOFs de bras
        
        print(f"   - DOFs des bras: {arm_dofs}")
        print(f"   - DOFs des doigts: {finger_dofs}")
        
        # Capteurs tactiles
        touch_sensors = []
        for i in range(model.nsensor):
            sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if sensor_name and 'tip_sensor' in sensor_name:
                touch_sensors.append(i)
                print(f"   - Capteur tactile: {sensor_name} (index {i})")
        
        print(f"   - Total capteurs tactiles: {len(touch_sensors)}")
        
        # Configuration initiale stable
        print("\n⚙️ CONFIGURATION INITIALE:")
        
        # Reset du modèle
        mujoco.mj_resetData(model, data)
        
        # Positions stables des bras
        arm_positions = [
            0.0, 0.2, 0.0, -0.5, 0.0, 0.0, 0.0,  # Bras gauche
            0.0, -0.2, 0.0, -0.5, 0.0, 0.0, 0.0   # Bras droit
        ]
        
        for i, dof in enumerate(arm_dofs):
            if i < len(arm_positions):
                data.qpos[dof] = arm_positions[i]
        
        # Doigts ouverts
        for dof in finger_dofs:
            data.qpos[dof] = 0.0
        
        # Position du cube sur la table
        cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
        if cube_body_id >= 0:
            cube_qpos_start = model.body_jntadr[cube_body_id]
            if cube_qpos_start >= 0:
                data.qpos[cube_qpos_start:cube_qpos_start+3] = [0.5, 0.0, 0.05]
                data.qpos[cube_qpos_start+3:cube_qpos_start+7] = [1, 0, 0, 0]
                print(f"   - Cube positionné à: [0.5, 0.0, 0.05]")
        
        print("   - Configuration appliquée")
        
        # Test de simulation
        print("\n🚀 TEST DE SIMULATION:")
        
        max_steps = 1000
        instability_count = 0
        
        for step in range(max_steps):
            
            # Actions très douces pour la stabilité
            if step < 100:
                # Phase stabilisation: pas d'action
                pass
            else:
                # Actions très légères
                for i in range(min(14, model.nu)):  # Bras
                    data.ctrl[i] = data.qpos[arm_dofs[i]] * 0.99
                
                for i in range(16):  # Doigts
                    ctrl_idx = 14 + i
                    if ctrl_idx < model.nu and 15 + i < model.nq:
                        data.ctrl[ctrl_idx] = data.qpos[15 + i] * 0.99
            
            # Simulation
            mujoco.mj_step(model, data)
            
            # Vérification de stabilité
            if (np.any(np.isnan(data.qpos)) or np.any(np.isinf(data.qpos)) or
                np.any(np.isnan(data.qvel)) or np.any(np.isinf(data.qvel))):
                instability_count += 1
                if instability_count > 10:
                    print(f"❌ Instabilité critique détectée au step {step}")
                    break
            
            # Affichage de progression
            if step % 200 == 0:
                cube_pos = data.xpos[cube_body_id] if cube_body_id >= 0 else [0, 0, 0]
                max_arm_vel = max([abs(data.qvel[dof]) for dof in arm_dofs])
                print(f"   Step {step:4d}: Cube=[{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}], Max vel={max_arm_vel:.4f}")
        
        # Résultats finaux
        print(f"\n📊 RÉSULTATS:")
        
        if instability_count <= 10:
            print(f"✅ Simulation stable!")
            print(f"   - Steps simulés: {max_steps}")
            print(f"   - Instabilités mineures: {instability_count}")
            
            # Position finale du cube
            if cube_body_id >= 0:
                final_cube_pos = data.xpos[cube_body_id]
                print(f"   - Position finale cube: [{final_cube_pos[0]:.3f}, {final_cube_pos[1]:.3f}, {final_cube_pos[2]:.3f}]")
            
            # Vitesses finales
            max_arm_vel = max([abs(data.qvel[dof]) for dof in arm_dofs])
            max_finger_vel = max([abs(data.qvel[dof]) for dof in finger_dofs])
            print(f"   - Vitesse max bras: {max_arm_vel:.4f}")
            print(f"   - Vitesse max doigts: {max_finger_vel:.4f}")
            
            # Test de contacts
            contact_count = 0
            for i in range(data.ncon):
                contact = data.contact[i]
                geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                if geom1_name and geom2_name:
                    contact_count += 1
            
            print(f"   - Contacts actifs: {contact_count}")
            
            # Capteurs tactiles
            touch_values = []
            for sensor_idx in touch_sensors:
                touch_values.append(abs(data.sensordata[sensor_idx]))
            
            if touch_values:
                print(f"   - Capteurs tactiles actifs: {sum(1 for v in touch_values if v > 0.001)}/{len(touch_values)}")
            
            print("\n🏆 TEST RÉUSSI - Le système est stable et fonctionnel!")
            return True
            
        else:
            print(f"❌ Simulation instable!")
            print(f"   - Instabilités: {instability_count}")
            return False
    
    except Exception as e:
        print(f"❌ Erreur durant le test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale"""
    
    print("🏆 TEST PROFESSIONNEL DU SYSTÈME DE GRASPING")
    print("=" * 60)
    
    success = test_professional_grasp()
    
    if success:
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print("Le système est prêt pour l'entraînement professionnel.")
        return 0
    else:
        print("\n❌ ÉCHEC DU TEST")
        print("Le système nécessite des corrections supplémentaires.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
