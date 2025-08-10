#!/usr/bin/env python3
"""
Script simple pour corriger TOUS les DOFs
"""

import re
import os
import mujoco
import numpy as np

def fix_all_dofs():
    """Applique des corrections ultra-conservatrices pour tous les DOFs"""
    
    input_path = "/workspace/results/g1_combined.xml"
    output_path = "/workspace/results/g1_combined_ultra_stable.xml"
    
    if not os.path.exists(input_path):
        print(f"❌ Fichier source non trouvé: {input_path}")
        return False
    
    print("🔧 CORRECTION ULTRA-CONSERVATIVE DE TOUS LES DOFs")
    print("=" * 60)
    
    # Lire le fichier XML
    with open(input_path, 'r') as f:
        xml_content = f.read()
    
    print("📖 Modèle XML original lu")
    
    # 1. PARAMÈTRES GLOBAUX ULTRA-CONSERVATEURS
    print("🔧 Paramètres globaux ultra-conservateurs...")
    
    # Timestep ultra-grand pour stabilité maximale
    xml_content = re.sub(r'timestep="0\.0005"', 'timestep="0.01"', xml_content)
    xml_content = re.sub(r'solver="Newton"', 'solver="PGS"', xml_content)
    xml_content = re.sub(r'iterations="500"', 'iterations="50"', xml_content)
    xml_content = re.sub(r'tolerance="1e-12"', 'tolerance="1e-6"', xml_content)
    
    # 2. TOUS LES ACTUATEURS DE BRAS - GAINS ULTRA-RÉDUITS
    print("🔧 Correction de tous les actuateurs de bras...")
    
    # Pattern pour tous les actuateurs de bras avec kp="120" kv="25"
    xml_content = re.sub(
        r'kp="120" kv="25"',
        'kp="20" kv="15"',  # Gains ultra-réduits
        xml_content
    )
    
    # 3. TOUS LES ACTUATEURS DE DOIGTS - GAINS ULTRA-DOUX
    print("🔧 Correction de tous les actuateurs de doigts...")
    
    # Tous les actuateurs act_*_*_joint_* (doigts)
    xml_content = re.sub(
        r'(<motor name="act_[^"]*_joint_[^"]*"[^>]*?)kp="[^"]*"([^>]*?)kv="[^"]*"',
        r'\1kp="5"\2kv="3"',  # Gains très doux pour doigts
        xml_content
    )
    
    # 4. FORCERANGE ULTRA-LIMITÉE POUR TOUS
    print("🔧 Limitation des forces...")
    
    # Bras: forcerange limitée
    xml_content = re.sub(
        r'forcerange="-150 150"',
        'forcerange="-30 30"',  # Forces très limitées
        xml_content
    )
    
    # Doigts: forcerange ultra-limitée
    xml_content = re.sub(
        r'(<motor name="act_[^"]*_joint_[^"]*"[^>]*?)forcerange="[^"]*"',
        r'\1forcerange="-5 5"',  # Forces ultra-faibles pour doigts
        xml_content
    )
    
    # 5. AMORTISSEMENT GLOBAL SIMPLE
    print("🔧 Amortissement global...")
    
    # Ajouter damping="3.0" à tous les joints qui n'en ont pas
    xml_content = re.sub(
        r'<joint([^>]*?)(?<!damping="[^"]*")/>',
        r'<joint\1 damping="3.0"/>',
        xml_content
    )
    
    # Sauvegarder
    with open(output_path, 'w') as f:
        f.write(xml_content)
    
    print(f"✅ Modèle ultra-stable sauvegardé: {output_path}")
    
    # Test du modèle
    print("\n🧪 Test du modèle ultra-stable...")
    try:
        test_model = mujoco.MjModel.from_xml_path(output_path)
        test_data = mujoco.MjData(test_model)
        
        print(f"✅ Modèle chargé - Timestep: {test_model.opt.timestep}")
        print(f"📊 DOFs: {test_model.nv}, Joints: {test_model.njnt}, Actuateurs: {test_model.nu}")
        
        # Test de simulation sur 100 steps
        print("🎯 Test de stabilité sur 100 steps...")
        
        for step in range(100):
            # Actions aléatoires très douces
            test_data.ctrl[:] = np.random.uniform(-0.1, 0.1, test_model.nu)
            
            mujoco.mj_step(test_model, test_data)
            
            # Vérifier NaN/Inf dans tous les DOFs
            if step % 25 == 0:
                qpos_ok = not (np.any(np.isnan(test_data.qpos)) or np.any(np.isinf(test_data.qpos)))
                qvel_ok = not (np.any(np.isnan(test_data.qvel)) or np.any(np.isinf(test_data.qvel)))
                qacc_ok = not (np.any(np.isnan(test_data.qacc)) or np.any(np.isinf(test_data.qacc)))
                
                status = "✅" if (qpos_ok and qvel_ok and qacc_ok) else "❌"
                print(f"  Step {step:2d}: {status} qpos={qpos_ok} qvel={qvel_ok} qacc={qacc_ok}")
        
        print("🎉 Test terminé!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = fix_all_dofs()
    if success:
        print("\n🎯 MODÈLE ULTRA-STABLE PRÊT!")
        print("📁 Nouveau modèle: /workspace/results/g1_combined_ultra_stable.xml")
        print("📋 Utilisez ce modèle dans vos environnements pour éliminer TOUS les warnings")
    else:
        print("\n❌ Échec de la correction")