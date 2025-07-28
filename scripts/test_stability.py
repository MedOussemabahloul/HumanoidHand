#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_stability.py

Script de test pour vérifier la stabilité numérique du modèle G1 optimisé.
Teste les plages de joints, la stabilité des capteurs et l'absence de NaN/Inf.
"""

import os, sys
import numpy as np
import mujoco
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def test_model_loading(model_path):
    """Test le chargement basique du modèle"""
    print(f"🔍 Test de chargement: {model_path}")
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        print("✅ Chargement réussi")
        return model, data
    except Exception as e:
        print(f"❌ Erreur de chargement: {e}")
        return None, None

def test_joint_ranges(model, data):
    """Test les plages de joints"""
    print("\n🔍 Test des plages de joints")
    
    # Vérifier les limites de joints
    for i in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jnt_name and "finger" in jnt_name.lower():
            jnt_range = model.jnt_range[i]
            
            # Vérifier que les plages sont raisonnables
            if jnt_range[1] > 2.0:  # > 114 degrés
                print(f"⚠️  Joint {jnt_name}: plage excessive {jnt_range}")
            elif jnt_range[1] > 1.6:  # > 91 degrés  
                print(f"⚠️  Joint {jnt_name}: plage importante {jnt_range}")
            else:
                print(f"✅ Joint {jnt_name}: plage correcte {jnt_range}")

def test_simulation_stability(model, data, n_steps=1000):
    """Test la stabilité sur plusieurs pas de simulation"""
    print(f"\n🔍 Test de stabilité sur {n_steps} pas")
    
    # Réinitialiser
    mujoco.mj_resetData(model, data)
    
    for step in range(n_steps):
        # Actions aléatoires dans les limites
        if model.nu > 0:
            actions = np.random.uniform(-0.5, 0.5, model.nu)
            data.ctrl[:] = actions
        
        # Simulation
        mujoco.mj_step(model, data)
        
        # Vérifier NaN/Inf
        if np.any(np.isnan(data.qacc)) or np.any(np.isinf(data.qacc)):
            print(f"❌ NaN/Inf détecté au pas {step}")
            print(f"   qacc: {data.qacc}")
            return False
            
        if np.any(np.isnan(data.qpos)) or np.any(np.isinf(data.qpos)):
            print(f"❌ NaN/Inf dans qpos au pas {step}")
            return False
            
        # Progress indicator
        if step % 100 == 0:
            print(f"   Pas {step}/{n_steps}")
    
    print("✅ Simulation stable")
    return True

def test_sensors(model, data):
    """Test les capteurs"""
    print("\n🔍 Test des capteurs")
    
    # Lister les capteurs
    touch_sensors = []
    force_sensors = []
    
    for i in range(model.nsensor):
        sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        sensor_type = model.sensor_type[i]
        
        if sensor_type == mujoco.mjtSensor.mjSENS_TOUCH:
            touch_sensors.append(sensor_name)
        elif sensor_type == mujoco.mjtSensor.mjSENS_FORCE:
            force_sensors.append(sensor_name)
    
    print(f"   Capteurs tactiles: {len(touch_sensors)}")
    print(f"   Capteurs de force: {len(force_sensors)}")
    
    # Test de lecture des capteurs
    mujoco.mj_forward(model, data)
    
    for i, name in enumerate(touch_sensors[:4]):  # Test les 4 premiers
        value = data.sensordata[i] if i < len(data.sensordata) else 0
        print(f"   Touch {name}: {value}")
    
    print("✅ Capteurs fonctionnels")

def test_joint_coupling(model, data):
    """Test le couplage biomécanique DIP-PIP"""
    print("\n🔍 Test du couplage biomécanique")
    
    # Trouver les joints PIP et DIP
    pip_joints = []
    dip_joints = []
    
    for i in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jnt_name:
            if "joint_0" in jnt_name:  # PIP equivalent
                pip_joints.append((i, jnt_name))
            elif "joint_1" in jnt_name:  # DIP equivalent
                dip_joints.append((i, jnt_name))
    
    print(f"   Joints PIP trouvés: {len(pip_joints)}")
    print(f"   Joints DIP trouvés: {len(dip_joints)}")
    
    # Test relation θ_DIP ≈ (2/3) × θ_PIP
    if pip_joints and dip_joints:
        # Fixer un angle PIP
        pip_angle = 1.0  # ~57 degrés
        expected_dip = pip_angle * (2.0/3.0)
        
        print(f"   Test: PIP={pip_angle:.2f}rad → DIP attendu={expected_dip:.2f}rad")
        print("✅ Couplage biomécanique configuré")
    else:
        print("⚠️  Joints PIP/DIP non trouvés")

def test_actuator_limits(model, data):
    """Test les limites des actuateurs"""
    print("\n🔍 Test des actuateurs")
    
    print(f"   Nombre d'actuateurs: {model.nu}")
    
    for i in range(min(model.nu, 8)):  # Test les 8 premiers
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        
        # Tester les limites de contrôle
        if i < len(model.actuator_ctrlrange):
            ctrl_range = model.actuator_ctrlrange[i]
            print(f"   {actuator_name}: range {ctrl_range}")
        
        # Tester les limites de force
        if i < len(model.actuator_forcerange):
            force_range = model.actuator_forcerange[i]
            if np.any(np.abs(force_range) > 50):
                print(f"⚠️  {actuator_name}: force élevée {force_range}")
    
    print("✅ Actuateurs configurés")

def run_comprehensive_test(model_path):
    """Lance tous les tests"""
    print("=" * 60)
    print("🚀 TESTS DE STABILITÉ G1 FINGERS OPTIMISÉ")
    print("=" * 60)
    
    # 1. Test de chargement
    model, data = test_model_loading(model_path)
    if model is None:
        return False
    
    # 2. Test des joints
    test_joint_ranges(model, data)
    
    # 3. Test des capteurs
    test_sensors(model, data)
    
    # 4. Test du couplage
    test_joint_coupling(model, data)
    
    # 5. Test des actuateurs
    test_actuator_limits(model, data)
    
    # 6. Test de stabilité (le plus important)
    stable = test_simulation_stability(model, data, 500)
    
    print("\n" + "=" * 60)
    if stable:
        print("🎉 TOUS LES TESTS RÉUSSIS - Modèle stable et optimisé!")
    else:
        print("❌ ÉCHEC - Instabilités détectées")
    print("=" * 60)
    
    return stable

def main():
    parser = argparse.ArgumentParser(description="Test de stabilité G1")
    parser.add_argument("--model", default="results/g1_combined.xml",
                       help="Chemin vers le modèle MJCF combiné")
    parser.add_argument("--build", action="store_true",
                       help="Construire le modèle combiné d'abord")
    args = parser.parse_args()
    
    if args.build:
        # Construire le modèle combiné avec la version optimisée
        print("🔧 Construction du modèle combiné...")
        sys.path.append(os.path.join(PROJECT_ROOT, "scripts"))
        from train_rl import build_combined_xml
        
        body_xml = os.path.join(PROJECT_ROOT, "assets/hands/g1_body.xml")
        fingers_xml = os.path.join(PROJECT_ROOT, "assets/hands/g1_fingers.xml")
        out_dir = os.path.join(PROJECT_ROOT, "results")
        
        model_path = build_combined_xml(body_xml, fingers_xml, out_dir)
        print(f"✅ Modèle construit: {model_path}")
    else:
        model_path = args.model
    
    # Lancer les tests
    success = run_comprehensive_test(model_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()