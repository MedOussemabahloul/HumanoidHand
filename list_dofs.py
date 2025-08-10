#!/usr/bin/env python3
"""
Script simple pour lister tous les DOFs du modèle MuJoCo
"""

import mujoco
import os

def list_all_dofs():
    """Liste tous les DOFs du modèle avec leurs noms et indices"""
    
    # Chemin vers le modèle XML stable
    model_path = "/workspace/results/g1_combined_clean_stable.xml"
    
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        return
    
    try:
        # Charger le modèle
        print(f"📁 Chargement du modèle: {model_path}")
        model = mujoco.MjModel.from_xml_path(model_path)
        
        print(f"✅ Modèle chargé avec succès")
        print(f"📊 Nombre total de DOFs: {model.nv}")
        print(f"📊 Nombre total de joints: {model.njnt}")
        print(f"📊 Nombre total d'actuateurs: {model.nu}")
        print()
        
        print("=" * 80)
        print("LISTE COMPLÈTE DES DOFs:")
        print("=" * 80)
        
        # Lister tous les DOFs
        for i in range(model.nv):
            joint_id = model.dof_jntid[i]  # ID du joint pour ce DOF
            joint_name = model.joint(joint_id).name if joint_id < model.njnt else "N/A"
            dof_addr = model.jnt_dofadr[joint_id] if joint_id < model.njnt else -1
            
            print(f"DOF {i:2d}: Joint ID={joint_id:2d}, Joint Name='{joint_name}', DOF Addr={dof_addr}")
        
        print()
        print("=" * 80)
        print("LISTE DES JOINTS:")
        print("=" * 80)
        
        # Lister tous les joints
        for i in range(model.njnt):
            joint_name = model.joint(i).name
            joint_type = model.jnt_type[i]
            dof_addr = model.jnt_dofadr[i]
            
            type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
            type_str = type_names.get(joint_type, f"type_{joint_type}")
            
            print(f"Joint {i:2d}: '{joint_name}' ({type_str}), DOF addr={dof_addr}")
        
        print()
        print("=" * 80)
        print("LISTE DES ACTUATEURS:")
        print("=" * 80)
        
        # Lister tous les actuateurs
        for i in range(model.nu):
            actuator_name = model.actuator(i).name
            joint_id = model.actuator_trnid[i, 0]  # Premier joint contrôlé
            joint_name = model.joint(joint_id).name if joint_id >= 0 and joint_id < model.njnt else "N/A"
            
            print(f"Actuateur {i:2d}: '{actuator_name}' -> Joint {joint_id} ('{joint_name}')")
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    list_all_dofs()