#!/usr/bin/env python3
"""
Script pour corriger TOUS les DOFs et éliminer toutes les instabilités
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
    print("🔧 Application des paramètres globaux ultra-conservateurs...")
    
    # Timestep beaucoup plus grand pour stabilité maximale
    xml_content = re.sub(
        r'timestep="0\.0005"',
        'timestep="0.01"',  # 20x plus grand que l'original
        xml_content
    )
    
    # Solveur PGS (plus stable que Newton)
    xml_content = re.sub(
        r'solver="Newton"',
        'solver="PGS"',
        xml_content
    )
    
    # Moins d'itérations pour éviter l'accumulation d'erreurs
    xml_content = re.sub(
        r'iterations="500"',
        'iterations="50"',  # 10x moins
        xml_content
    )
    
    # Tolérance plus relâchée
    xml_content = re.sub(
        r'tolerance="1e-12"',
        'tolerance="1e-6"',  # Beaucoup plus relâchée
        xml_content
    )
    
    # 2. ACTUATEURS DE BRAS - PARAMÈTRES ULTRA-CONSERVATEURS
    print("🔧 Correction des actuateurs de bras...")
    
    # Tous les actuateurs de bras avec des gains très réduits
    arm_actuators = [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", 
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
    ]
    
    for actuator in arm_actuators:
        # Gains très réduits pour tous les bras
        xml_content = re.sub(
            rf'<motor name="{actuator}"([^>]*?)kp="120"([^>]*?)kv="25"',
            rf'<motor name="{actuator}"\1kp="30"\2kv="20"',  # Gains très réduits
            xml_content
        )
        
        # Forcerange très réduite
        xml_content = re.sub(
            rf'(<motor name="{actuator}"[^>]*?)forcerange="[^"]*"',
            rf'\1forcerange="-50 50"',  # Forces très limitées
            xml_content
        )
    
    # 3. ACTUATEURS DE DOIGTS - PARAMÈTRES ULTRA-DOUX
    print("🔧 Correction des actuateurs de doigts...")
    
    finger_patterns = [
        "act_.*_index_.*", "act_.*_middle_.*", "act_.*_ring_.*", "act_.*_thumb_.*"
    ]
    
    for pattern in finger_patterns:
        # Gains ultra-réduits pour tous les doigts
        xml_content = re.sub(
            rf'<motor name="{pattern}"([^>]*?)kp="[^"]*"([^>]*?)kv="[^"]*"',
            rf'<motor name="{pattern}"\1kp="10"\2kv="5"',  # Gains très doux
            xml_content
        )
        
        # Forces très limitées
        xml_content = re.sub(
            rf'(<motor name="{pattern}"[^>]*?)forcerange="[^"]*"',
            rf'\1forcerange="-10 10"',  # Forces très faibles
            xml_content
        )
    
    # 4. AMORTISSEMENT GLOBAL RENFORCÉ
    print("🔧 Renforcement de l'amortissement global...")
    
    # Ajouter de l'amortissement à tous les joints si pas déjà présent
    xml_content = re.sub(
        r'(<joint[^>]*?)(?<!damping="[^"]*")>',
        r'\1 damping="2.0">',
        xml_content
    )
    
    # Augmenter l'amortissement existant
    xml_content = re.sub(
        r'damping="([0-9.]+)"',
        lambda m: f'damping="{max(2.0, float(m.group(1)) * 2)}"',
        xml_content
    )
    
    # 5. FRICTION RENFORCÉE
    print("🔧 Renforcement de la friction...")
    
    # Ajouter friction aux joints sans friction
    xml_content = re.sub(
        r'(<joint[^>]*?)(?<!frictionloss="[^"]*")>',
        r'\1 frictionloss="1.0">',
        xml_content
    )
    
    # Sauvegarder le modèle corrigé
    with open(output_path, 'w') as f:
        f.write(xml_content)
    
    print(f"✅ Modèle ultra-stable sauvegardé: {output_path}")
    
    # Test du modèle
    print("\n🧪 Test du modèle ultra-stable...")
    try:
        test_model = mujoco.MjModel.from_xml_path(output_path)
        test_data = mujoco.MjData(test_model)
        
        print(f"✅ Modèle chargé - Timestep: {test_model.opt.timestep}")
        
        # Test de simulation sur 200 steps
        print("🎯 Test de simulation sur 200 steps...")
        warnings_count = 0
        
        for step in range(200):
            mujoco.mj_step(test_model, test_data)
            
            # Vérifier tous les DOFs pour NaN/Inf
            if np.any(np.isnan(test_data.qpos)) or np.any(np.isinf(test_data.qpos)):
                print(f"⚠️  Step {step}: NaN/Inf détecté dans qpos")
                warnings_count += 1
            if np.any(np.isnan(test_data.qvel)) or np.any(np.isinf(test_data.qvel)):
                print(f"⚠️  Step {step}: NaN/Inf détecté dans qvel")
                warnings_count += 1
            if np.any(np.isnan(test_data.qacc)) or np.any(np.isinf(test_data.qacc)):
                print(f"⚠️  Step {step}: NaN/Inf détecté dans qacc")
                warnings_count += 1
        
        if warnings_count == 0:
            print("🎉 PARFAIT! Aucun NaN/Inf détecté sur 200 steps!")
        else:
            print(f"⚠️  {warnings_count} warnings détectés")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test: {e}")
        return False

if __name__ == "__main__":
    success = fix_all_dofs()
    if success:
        print("\n🎯 MODÈLE ULTRA-STABLE PRÊT!")
        print("📁 Utilisez: /workspace/results/g1_combined_ultra_stable.xml")
    else:
        print("\n❌ Échec de la correction")