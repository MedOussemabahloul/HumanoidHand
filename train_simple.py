#!/usr/bin/env python3
"""
Script d'entraînement simple - Point d'entrée principal
"""

import sys
import os

# Ajouter le dossier courant au path
sys.path.append('/workspace')

from trainer import quick_train, production_train
from config import DEFAULT_CONFIG

def main():
    """Point d'entrée principal"""
    
    print("🤖 ENTRAÎNEMENT ROBOTIQUE PROFESSIONNEL")
    print("=" * 50)
    print()
    
    # Vérifier que le modèle XML existe
    if not os.path.exists(DEFAULT_CONFIG.system.model_path):
        print("❌ Modèle XML manquant!")
        print("🔧 Créez d'abord le modèle avec:")
        print('python3 -c "import re; xml=open(\'/workspace/results/g1_combined.xml\').read(); xml=re.sub(r\'timestep=\"0\.0005\"\', \'timestep=\"0.005\"\', xml); xml=re.sub(r\'solver=\"Newton\"\', \'solver=\"PGS\"\', xml); xml=re.sub(r\'iterations=\"500\"\', \'iterations=\"100\"\', xml); xml=re.sub(r\'tolerance=\"1e-12\"\', \'tolerance=\"1e-8\"\', xml); xml=re.sub(r\'kp=\"120\" kv=\"25\"\', \'kp=\"80\" kv=\"20\"\', xml); xml=re.sub(r\'kp=\"50\" kv=\"15\"\', \'kp=\"25\" kv=\"10\"\', xml); xml=re.sub(r\'forcerange=\"-150 150\"\', \'forcerange=\"-100 100\"\', xml); xml=re.sub(r\'forcerange=\"-50 50\"\', \'forcerange=\"-25 25\"\', xml); open(\'/workspace/results/g1_combined_balanced.xml\', \'w\').write(xml); print(\'✅ Modèle créé\')"')
        return
    
    print("📋 OPTIONS D'ENTRAÎNEMENT:")
    print("1. 🚀 Entraînement rapide (20k steps)")
    print("2. 🎯 Entraînement de production (100k steps)")
    print("3. 🧪 Test seulement")
    print()
    
    # Pour l'automatisation, choisir option 1 par défaut
    choice = "1"
    
    try:
        if choice == "1":
            print("🚀 Démarrage entraînement rapide...")
            model_path = quick_train(timesteps=20_000, algorithm="TD3")
            print(f"✅ Entraînement rapide terminé!")
            print(f"📁 Modèle: {model_path}")
            
        elif choice == "2":
            print("🎯 Démarrage entraînement de production...")
            model_path = production_train()
            print(f"✅ Entraînement de production terminé!")
            print(f"📁 Modèle: {model_path}")
            
        elif choice == "3":
            print("🧪 Test de l'environnement...")
            from envs.professional_grasp_env import make_professional_env
            
            env = make_professional_env()
            obs, _ = env.reset()
            
            total_reward = 0
            for i in range(50):
                action = env.action_space.sample()
                obs, reward, term, trunc, info = env.step(action)
                total_reward += reward
                
                if i % 10 == 0:
                    print(f"Step {i}: reward={reward:.2f}, distance={info.get('distance', 0):.3f}")
            
            print(f"✅ Test terminé - Reward total: {total_reward:.2f}")
            env.close()
        
        print("\n🎉 SUCCÈS!")
        print("📊 Consultez /workspace/results/ pour les modèles")
        print("📊 Consultez /workspace/logs/ pour les logs")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())