#!/usr/bin/env python3
"""
🤖 G1 MANIPULATION TESTER
Test de manipulation pour le robot G1 avec mouvements de base
"""

import mujoco as mj
import mujoco.viewer as viewer
import numpy as np
import time

def main():
    print("🤖 G1 MANIPULATION TESTER")
    print("=" * 40)
    print(f"✅ MuJoCo version: {mj.__version__}")
    
    print("🚀 TEST DE MANIPULATION G1")
    print("=" * 50)
    print("🔍 Chargement du modèle...")
    
    try:
        # Charger le modèle
        model = mj.MjModel.from_xml_path("results/g1_combined.xml")
        data = mj.MjData(model)
        
        print("✅ Modèle chargé avec succès!")
        print(f"📊 Nombre de DOF: {model.nv}")
        print(f"🦾 Nombre d'actuateurs: {model.nu}")
        
        # Initialiser la simulation
        mj.mj_resetData(model, data)
        
        # Position initiale du robot (debout)
        if model.nq > 0:
            # Position stable pour le robot
            data.qpos[0:3] = [0, 0, 0.8]  # Position x, y, z du corps principal
            
        # Créer le viewer
        with viewer.launch_passive(model, data) as v:
            print("🎮 Contrôles:")
            print("  - ESC: Quitter")
            print("  - La simulation démarrera automatiquement")
            print("  - Le robot tentera de bouger ses bras vers le cube")
            
            # Variables pour le contrôle
            step = 0
            target_reached = False
            
            while v.is_running():
                step_start = time.time()
                
                # Mouvement simple des bras vers le cube
                if step < 1000:  # Premiers 1000 steps
                    # Mouvement des épaules vers l'avant
                    if "act_left_shoulder_pitch_joint" in [model.actuator(i).name for i in range(model.nu)]:
                        # Trouver les indices des actuateurs
                        for i in range(model.nu):
                            actuator_name = model.actuator(i).name
                            
                            # Mouvement des bras vers l'avant et vers le bas
                            if "left_shoulder_pitch" in actuator_name:
                                data.ctrl[i] = np.sin(step * 0.01) * 0.5
                            elif "right_shoulder_pitch" in actuator_name:
                                data.ctrl[i] = np.sin(step * 0.01) * 0.5
                            elif "left_elbow" in actuator_name:
                                data.ctrl[i] = np.sin(step * 0.02) * 0.3
                            elif "right_elbow" in actuator_name:
                                data.ctrl[i] = np.sin(step * 0.02) * 0.3
                            elif "wrist" in actuator_name:
                                data.ctrl[i] = np.sin(step * 0.015) * 0.2
                            elif "finger" in actuator_name or "thumb" in actuator_name:
                                # Mouvement des doigts (ouverture/fermeture)
                                data.ctrl[i] = np.sin(step * 0.05) * 0.1
                
                elif step < 2000:  # Deuxième phase
                    # Mouvement plus dirigé vers le cube
                    for i in range(model.nu):
                        actuator_name = model.actuator(i).name
                        
                        if "left_shoulder_pitch" in actuator_name:
                            data.ctrl[i] = 0.3  # Bras vers l'avant
                        elif "left_elbow" in actuator_name:
                            data.ctrl[i] = -0.5  # Plier le coude
                        elif "left_wrist" in actuator_name:
                            data.ctrl[i] = 0.2
                        elif "left" in actuator_name and ("finger" in actuator_name or "thumb" in actuator_name):
                            data.ctrl[i] = 0.3  # Fermer la main
                
                else:
                    # Phase de repos
                    for i in range(model.nu):
                        data.ctrl[i] = 0.0
                
                # Simuler un pas
                mj.mj_step(model, data)
                
                # Synchroniser avec le viewer
                v.sync()
                
                # Contrôler la fréquence
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                
                step += 1
                
                # Afficher des informations périodiquement
                if step % 500 == 0:
                    cube_pos = data.body("cube").xpos if "cube" in [model.body(i).name for i in range(model.nbody)] else [0, 0, 0]
                    print(f"Step {step}: Position cube: {cube_pos}")
        
        print("✅ Simulation terminée!")
        
    except Exception as e:
        print(f"❌ ERREUR: Erreur lors du chargement: {e}")
        print("💡 Vérifiez que le modèle a été créé avec: python scripts/create_combined_model.py")
        return False
    
    return True

if __name__ == "__main__":
    main()