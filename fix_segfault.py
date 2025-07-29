#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour corriger les problèmes courants de segmentation fault
"""

import os
import sys
import subprocess
import shutil

def set_environment_variables():
    """Configure les variables d'environnement pour éviter les segfaults"""
    print("🔧 Configuration des variables d'environnement...")
    
    env_vars = {
        'OMP_NUM_THREADS': '1',  # Limite les threads OpenMP
        'MKL_NUM_THREADS': '1',  # Limite les threads MKL
        'NUMEXPR_NUM_THREADS': '1',  # Limite les threads NumExpr
        'OPENBLAS_NUM_THREADS': '1',  # Limite les threads OpenBLAS
        'VECLIB_MAXIMUM_THREADS': '1',  # Limite les threads VecLib
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',  # Optimise allocation CUDA
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"   {var}={value}")
    
    return env_vars

def check_and_fix_memory():
    """Vérifie et optimise l'utilisation mémoire"""
    print("\n🔧 Vérification mémoire...")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        
        print(f"   Mémoire totale: {memory.total / 1024**3:.1f} GB")
        print(f"   Mémoire disponible: {memory.available / 1024**3:.1f} GB")
        
        if memory.available < 2 * 1024**3:
            print("   ⚠️  Peu de mémoire disponible")
            print("   💡 Recommandations:")
            print("      - Fermez les applications inutiles")
            print("      - Réduisez batch_size dans la config")
            print("      - Utilisez un replay buffer plus petit")
            return False
        else:
            print("   ✅ Mémoire suffisante")
            return True
    except ImportError:
        print("   ⚠️  psutil non installé")
        return True

def create_safe_config():
    """Crée une configuration sûre pour éviter les segfaults"""
    print("\n🔧 Création configuration sûre...")
    
    safe_config = """
task:
  cube_body_name: "cube"
  max_steps_per_episode: 200
  touch_sensors: []
  force_sensors: []
  include_orientation_reward: false
  force_reward_weight_normal: 0.0
  force_reward_weight_tangential: 0.0
  translation_penalty_weight: 0.0
  output_dir: "results"
  save_freq_steps: 50000

rl:
  gamma: 0.99
  alpha: 0.2
  learning_rate: 0.0003
  hidden_size: 256
  batch_size: 64
  replay_size: 10000
  start_steps: 1000
  update_after: 1000
  update_every: 50
  num_updates: 50
  total_steps: 100000
  tau: 0.005
  act_limit: 1.0
"""
    
    config_path = "config/train_config_safe.yaml"
    os.makedirs("config", exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(safe_config)
    
    print(f"   ✅ Configuration créée: {config_path}")
    return config_path

def create_simple_xml():
    """Crée un fichier XML MuJoCo simple pour test"""
    print("\n🔧 Création XML MuJoCo simple...")
    
    simple_xml = """<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="simple_test">
  <compiler angle="radian" coordinate="local" meshdir="assets" texturedir="assets"/>
  
  <default>
    <default class="main">
      <geom contype="1" conaffinity="1" rgba="0.8 0.6 0.4 1"/>
    </default>
  </default>
  
  <worldbody>
    <!-- Sol -->
    <geom name="ground" type="plane" pos="0 0 0" size="0 0 0.05" class="main"/>
    
    <!-- Cube à saisir -->
    <body name="cube" pos="0 0 0.1">
      <geom name="cube_geom" type="box" size="0.05 0.05 0.05" mass="0.1" class="main"/>
    </body>
    
    <!-- Robot simple (bras) -->
    <body name="base" pos="0 0 0.5">
      <geom name="base_geom" type="cylinder" size="0.1 0.2" class="main"/>
      
      <body name="arm1" pos="0 0 0.2">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom name="arm1_geom" type="cylinder" size="0.05 0.2" class="main"/>
        
        <body name="arm2" pos="0 0 0.2">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom name="arm2_geom" type="cylinder" size="0.05 0.2" class="main"/>
          
          <body name="gripper" pos="0 0 0.2">
            <geom name="gripper_geom" type="box" size="0.02 0.02 0.05" class="main"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="motor1" joint="joint1" gear="100"/>
    <motor name="motor2" joint="joint2" gear="100"/>
  </actuator>
  
  <sensor>
    <framepos name="cube_pos" objtype="body" objname="cube"/>
    <framepos name="gripper_pos" objtype="body" objname="gripper"/>
  </sensor>
</mujoco>
"""
    
    xml_path = "assets/simple_test.xml"
    os.makedirs("assets", exist_ok=True)
    
    with open(xml_path, 'w') as f:
        f.write(simple_xml)
    
    print(f"   ✅ XML créé: {xml_path}")
    return xml_path

def create_safe_launch_script():
    """Crée un script de lancement sûr"""
    print("\n🔧 Création script de lancement sûr...")
    
    launch_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de lancement sûr pour éviter les segfaults
"""

import os
import sys
import subprocess
import signal
import time

def setup_environment():
    """Configure l'environnement pour éviter les segfaults"""
    env_vars = {
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'VECLIB_MAXIMUM_THREADS': '1',
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value

def run_with_timeout(cmd, timeout=30):
    """Lance une commande avec timeout"""
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout.decode(), stderr.decode()
        
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        return -1, "", "Timeout"
    except Exception as e:
        return -1, "", str(e)

def main():
    print("🚀 LANCEMENT SÛR")
    print("=" * 50)
    
    setup_environment()
    
    # Test basique d'abord
    print("🔍 Test basique...")
    returncode, stdout, stderr = run_with_timeout([
        sys.executable, "debug_segfault.py"
    ], timeout=60)
    
    if returncode != 0:
        print(f"❌ Test basique échoué: {stderr}")
        return
    
    print("✅ Test basique réussi")
    
    # Lancement du training
    print("\\n🚀 Lancement training...")
    returncode, stdout, stderr = run_with_timeout([
        sys.executable, "scripts/train_rl.py",
        "-c", "config/train_config_safe.yaml",
        "--body_xml", "assets/simple_test.xml",
        "--fingers_xml", "assets/simple_test.xml",
        "-o", "results_safe"
    ], timeout=300)
    
    if returncode == 0:
        print("✅ Training terminé avec succès")
    else:
        print(f"❌ Training échoué: {stderr}")

if __name__ == "__main__":
    main()
'''
    
    with open("launch_safe.py", 'w') as f:
        f.write(launch_script)
    
    os.chmod("launch_safe.py", 0o755)
    print("   ✅ Script créé: launch_safe.py")
    return "launch_safe.py"

def main():
    print("🔧 CORRECTION SEGMENTATION FAULT")
    print("=" * 50)
    
    # 1. Variables d'environnement
    set_environment_variables()
    
    # 2. Vérification mémoire
    check_and_fix_memory()
    
    # 3. Configuration sûre
    config_path = create_safe_config()
    
    # 4. XML simple
    xml_path = create_simple_xml()
    
    # 5. Script de lancement
    launch_script = create_safe_launch_script()
    
    print("\n" + "=" * 50)
    print("✅ CORRECTION TERMINÉE")
    print("=" * 50)
    
    print("\\n📋 FICHIERS CRÉÉS:")
    print(f"   - {config_path}")
    print(f"   - {xml_path}")
    print(f"   - {launch_script}")
    
    print("\\n🚀 POUR LANCER:")
    print("   python debug_segfault.py")
    print("   python launch_safe.py")
    
    print("\\n💡 CONSEILS:")
    print("1. Si le segfault persiste, vérifiez les logs: dmesg | tail")
    print("2. Essayez avec moins de mémoire: réduisez batch_size et replay_size")
    print("3. Vérifiez que MuJoCo est correctement installé")
    print("4. Sur CPU, assurez-vous d'avoir les bonnes bibliothèques BLAS")

if __name__ == "__main__":
    main()