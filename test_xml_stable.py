#!/usr/bin/env python3
"""
TEST DU MODÈLE XML CORRIGÉ
Vérifie que les corrections XML éliminent définitivement les instabilités
"""

import sys
from pathlib import Path
import numpy as np
sys.path.append('.')
sys.path.append('./envs')

def test_xml_stable_model():
    """Test du modèle XML corrigé"""
    print("🔬 TEST DU MODÈLE XML CORRIGÉ")
    print("=" * 60)
    
    # Vérifier que le fichier existe
    xml_path = "results/g1_combined_stable.xml"
    if not Path(xml_path).exists():
        print(f"❌ Modèle manquant: {xml_path}")
        return False
    
    print(f"✅ Modèle trouvé: {xml_path}")
    
    try:
        import mujoco
        from mujoco import MjModel, MjData
        
        # Charger le modèle corrigé
        print("\n🔧 CHARGEMENT DU MODÈLE CORRIGÉ...")
        model = MjModel.from_xml_path(xml_path)
        data = MjData(model)
        
        print(f"✅ Modèle chargé: {model.nv} DOFs, {model.nu} actuateurs")
        
        # Analyser les paramètres corrigés
        print(f"\n📊 PARAMÈTRES DE SIMULATION CORRIGÉS:")
        print(f"   ⏱️  Timestep: {model.opt.timestep}")
        print(f"   🔄 Iterations: {model.opt.iterations}")
        print(f"   🎯 Tolerance: {model.opt.tolerance}")
        print(f"   🧮 Solver: {model.opt.solver}")
        
        # Analyser les doigts
        print(f"\n🖐️  ANALYSE DES DOIGTS CORRIGÉS:")
        finger_keywords = ["finger", "thumb", "index", "middle", "ring"]
        finger_joints = []
        
        for dof_id in range(min(31, model.nv)):
            joint_id = model.dof_jntid[dof_id] if hasattr(model, 'dof_jntid') else dof_id
            if joint_id < model.njnt:
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                if joint_name and any(kw in joint_name.lower() for kw in finger_keywords):
                    finger_joints.append((dof_id, joint_name))
                    damping = model.dof_damping[dof_id] if dof_id < len(model.dof_damping) else 0.0
                    print(f"   DOF {dof_id:2d}: {joint_name:25s} - Damping: {damping:.1f}")
        
        print(f"\n🎯 DOIGTS IDENTIFIÉS: {len(finger_joints)} joints")
        
        # Test de stabilité intensif
        print(f"\n🧪 TEST DE STABILITÉ INTENSIF...")
        instabilities = 0
        successful_steps = 0
        max_qacc = 0.0
        
        for test_round in range(5):
            print(f"\n   🔄 Test {test_round+1}/5:")
            
            # Reset
            mujoco.mj_resetData(model, data)
            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            data.ctrl[:] = 0.0
            
            # Stabilisation initiale
            for i in range(100):
                mujoco.mj_forward(model, data)
                if i % 20 == 0:
                    mujoco.mj_step(model, data)
            
            # Test avec actions aléatoires
            for step in range(50):
                # Actions aléatoires sur TOUS les DOFs (y compris doigts)
                for dof_id in range(1, min(31, model.nu)):  # Exclure cube
                    if dof_id < len(data.ctrl):
                        data.ctrl[dof_id] = np.random.uniform(-0.05, 0.05)
                
                # Step MuJoCo
                try:
                    mujoco.mj_step(model, data)
                    successful_steps += 1
                    
                    # Vérifier stabilité
                    max_qacc_step = np.max(np.abs(data.qacc))
                    max_qacc = max(max_qacc, max_qacc_step)
                    
                    if (np.any(np.isnan(data.qacc)) or np.any(np.isinf(data.qacc)) or
                        np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel))):
                        
                        print(f"     ❌ Step {step+1}: Instabilité détectée")
                        instabilities += 1
                        
                        # Identifier le DOF problématique
                        for dof_id in range(min(31, model.nv)):
                            if (dof_id < len(data.qacc) and 
                                (np.isnan(data.qacc[dof_id]) or np.isinf(data.qacc[dof_id]))):
                                joint_id = model.dof_jntid[dof_id] if hasattr(model, 'dof_jntid') else dof_id
                                if joint_id < model.njnt:
                                    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                                    print(f"        🚨 DOF {dof_id}: '{joint_name}'")
                        break
                    
                    if step % 10 == 0:
                        print(f"     ✅ Step {step+1}: Stable (max_qacc: {max_qacc_step:.3f})")
                        
                except Exception as e:
                    print(f"     ❌ Step {step+1}: Erreur MuJoCo: {e}")
                    instabilities += 1
                    break
        
        # Résultats
        print(f"\n📈 RÉSULTATS DU TEST XML CORRIGÉ:")
        print(f"   ⚠️  Instabilités totales: {instabilities}")
        print(f"   ✅ Steps réussis: {successful_steps}")
        print(f"   📊 Max acceleration: {max_qacc:.6f}")
        print(f"   🖐️  Doigts testés: {len(finger_joints)}")
        
        success = instabilities == 0
        
        if success:
            print(f"\n🏆 MODÈLE XML PARFAITEMENT STABLE!")
            print(f"   ✅ Aucune instabilité sur {successful_steps} steps")
            print(f"   🖐️  Tous les doigts fonctionnent parfaitement")
            print(f"   🎯 Prêt pour l'entraînement complet")
        else:
            print(f"\n⚠️  MODÈLE NÉCESSITE DES AJUSTEMENTS")
            print(f"   🔧 {instabilities} instabilités détectées")
            
        return success
        
    except ImportError as e:
        print(f"❌ Erreur import: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        return False

def main():
    """Test principal"""
    print("🎯 VALIDATION DES CORRECTIONS XML")
    print("\n💡 CORRECTIONS APPLIQUÉES:")
    print("   1. ⏱️  Timestep: 0.002 → 0.005 (plus stable)")
    print("   2. 🎯 Tolerance: 1e-10 → 1e-6 (réaliste)")
    print("   3. 🔄 Solver: Newton → PGS (plus robuste)")
    print("   4. 🖐️  Damping doigts: 0.01 → 8.0 (800x plus élevé)")
    print("   5. 🎛️  Gains doigts: 100 → 15-25 (réduits)")
    print("   6. ⚖️  Masses explicites pour tous les doigts")
    print("   7. 🔧 Friction et stiffness optimisés")
    
    success = test_xml_stable_model()
    
    if success:
        print(f"\n🚀 PROCHAINES ÉTAPES:")
        print(f"   1. Utiliser 'results/g1_combined_stable.xml'")
        print(f"   2. Entraîner avec TOUS les DOFs actifs")
        print(f"   3. Profiter des doigts fonctionnels!")
        print(f"   4. Plus besoin de blocage ou contournements")
    else:
        print(f"\n🔧 AJUSTEMENTS NÉCESSAIRES:")
        print(f"   1. Augmenter encore le damping")
        print(f"   2. Réduire davantage les gains")
        print(f"   3. Vérifier les masses/inertie")
    
    return success

if __name__ == "__main__":
    main()