#!/usr/bin/env python3
"""
Script de debug pour identifier les joints problématiques par nom
"""

import sys
import numpy as np
from pathlib import Path

# Ajouter les chemins locaux
sys.path.append('.')
sys.path.append('./envs')

try:
    import mujoco
    from mujoco import MjModel, MjData
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    print("❌ MuJoCo non disponible")

def debug_joints_mapping(xml_path="results/g1_combined.xml"):
    """Debug complet du mapping des joints"""
    if not HAS_MUJOCO:
        print("❌ MuJoCo requis pour le debug")
        return
    
    if not Path(xml_path).exists():
        print(f"❌ Modèle non trouvé: {xml_path}")
        return
    
    print("🔍 DEBUG DU MAPPING DES JOINTS")
    print("=" * 60)
    
    try:
        # Charger le modèle
        model = MjModel.from_xml_path(xml_path)
        data = MjData(model)
        
        print(f"📊 Informations générales:")
        print(f"   Total DOFs: {model.nv}")
        print(f"   Total joints: {model.njnt}")
        print(f"   Total actuateurs: {model.nu}")
        print()
        
        # Mapping DOF -> Joint
        print("🎯 MAPPING DOF -> JOINT:")
        print("-" * 40)
        finger_dofs = []
        arm_dofs = []
        
        for dof_id in range(model.nv):
            joint_id = model.dof_jntid[dof_id]
            
            if joint_id < model.njnt:
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                joint_type = model.jnt_type[joint_id]
                
                # Identifier le type de joint
                if joint_name:
                    if any(keyword in joint_name.lower() for keyword in ["finger", "thumb"]):
                        finger_dofs.append(dof_id)
                        category = "🖐️  FINGER"
                    elif any(keyword in joint_name.lower() for keyword in 
                           ["shoulder", "elbow", "wrist", "arm"]):
                        arm_dofs.append(dof_id)
                        category = "💪 ARM"
                    else:
                        category = "🤖 OTHER"
                else:
                    joint_name = "unknown"
                    category = "❓ UNKNOWN"
                
                print(f"DOF {dof_id:2d}: Joint {joint_id:2d} = {joint_name:25s} [{category}]")
            else:
                print(f"DOF {dof_id:2d}: FREE JOINT (floating base)")
        
        print()
        print("🖐️  JOINTS DE DOIGTS IDENTIFIÉS:")
        print(f"   DOFs fingers: {finger_dofs}")
        print(f"   DOFs problématiques signalés: [15, 16, 20]")
        
        # Vérifier les DOFs problématiques
        problematic_dofs = [15, 16, 20]
        print("\n⚠️  ANALYSE DES DOFs PROBLÉMATIQUES:")
        print("-" * 45)
        
        for dof_id in problematic_dofs:
            if dof_id < model.nv:
                joint_id = model.dof_jntid[dof_id]
                if joint_id < model.njnt:
                    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                    print(f"   DOF {dof_id}: Joint '{joint_name}' (ID {joint_id})")
                else:
                    print(f"   DOF {dof_id}: Free joint")
            else:
                print(f"   DOF {dof_id}: HORS LIMITES!")
        
        # Analyser les limites et amortissements
        print("\n🔧 PARAMÈTRES DES JOINTS FINGERS:")
        print("-" * 40)
        
        for dof_id in finger_dofs:
            joint_id = model.dof_jntid[dof_id]
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            
            # Limites
            if hasattr(model, 'jnt_range') and joint_id < len(model.jnt_range):
                limits = model.jnt_range[joint_id]
                print(f"   {joint_name}: Limites {limits}")
            
            # Amortissement
            if dof_id < len(model.dof_damping):
                damping = model.dof_damping[dof_id]
                print(f"   {joint_name}: Amortissement {damping}")
        
        return finger_dofs, arm_dofs, problematic_dofs
        
    except Exception as e:
        print(f"❌ Erreur lors du debug: {e}")
        import traceback
        traceback.print_exc()
        return [], [], []

def test_stable_simulation(xml_path="results/g1_combined.xml"):
    """Test de simulation avec joints bloqués"""
    if not HAS_MUJOCO:
        return
    
    print("\n🧪 TEST DE SIMULATION AVEC JOINTS BLOQUÉS")
    print("=" * 50)
    
    try:
        model = MjModel.from_xml_path(xml_path)
        data = MjData(model)
        
        # Identifier les DOFs de doigts
        finger_dofs = []
        for dof_id in range(model.nv):
            joint_id = model.dof_jntid[dof_id]
            if joint_id < model.njnt:
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                if joint_name and any(keyword in joint_name.lower() for keyword in ["finger", "thumb"]):
                    finger_dofs.append(dof_id)
        
        print(f"Finger DOFs à tester: {finger_dofs}")
        
        # Reset et stabilisation
        mujoco.mj_resetData(model, data)
        
        # Bloquer les joints de doigts
        for dof_id in finger_dofs:
            if dof_id < len(data.qpos):
                data.qpos[dof_id] = 0.0
            if dof_id < len(data.qvel):
                data.qvel[dof_id] = 0.0
        
        # Test de simulation
        print("Test avec joints bloqués...")
        for step in range(10):
            mujoco.mj_step(model, data)
            
            # Vérifier stabilité
            if np.any(np.isnan(data.qacc)) or np.any(np.isinf(data.qacc)):
                print(f"❌ Instabilité détectée à l'étape {step}")
                break
            elif step == 9:
                print("✅ Simulation stable avec joints bloqués")
        
    except Exception as e:
        print(f"❌ Erreur simulation: {e}")

def main():
    """Point d'entrée principal"""
    print("🐛 DEBUG DES JOINTS PROBLÉMATIQUES G1")
    print("=" * 60)
    
    # Debug du mapping
    finger_dofs, arm_dofs, problematic_dofs = debug_joints_mapping()
    
    # Test de simulation
    test_stable_simulation()
    
    print("\n💡 RECOMMANDATIONS:")
    print("-" * 20)
    print("1. Bloquer les joints de doigts au début")
    print("2. Utiliser un amortissement plus élevé")
    print("3. Limiter les vitesses des doigts")
    print("4. Démarrer avec seulement les bras")

if __name__ == "__main__":
    main()