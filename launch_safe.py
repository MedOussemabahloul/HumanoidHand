#!/usr/bin/env python3
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
    print("\n🚀 Lancement training...")
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
