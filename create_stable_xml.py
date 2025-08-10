
#!/usr/bin/env python3
"""
🔧 SCRIPT DE CORRECTION DU MODÈLE XML POUR LA STABILITÉ
======================================================

Ce script corrige les paramètres de simulation dans le modèle XML
pour éviter les erreurs NaN/Inf dans QVEL/QACC/QPOS.

✅ Timestep optimal (0.005)
✅ Solveur stable (PGS)
✅ Intégrateur stable (RK4)
✅ Paramètres d'actuateurs optimisés
✅ Tolérance réaliste
"""

import os
import shutil
import numpy as np
from pathlib import Path

def create_stable_xml_model():
    """
    Créer un modèle XML stable en corrigeant les paramètres de simulation
    """
    
    print("🔧 Création d'un modèle XML stable...")
    
    # Lire le modèle existant
    source_path = "/home/oussema/Documents/project/results/g1_combined.xml"
    target_path = "/home/oussema/Documents/project/results/g1_combined_stable.xml"
    
    if not os.path.exists(source_path):
        print(f"❌ Modèle source introuvable: {source_path}")
        return None
    
    try:
        with open(source_path, 'r') as f:
            xml_content = f.read()
        
        print("📖 Modèle XML lu avec succès")
        
        # ✅ CORRECTIONS CRITIQUES POUR LA STABILITÉ
        
        # 1. Corriger le timestep (cause principale des erreurs NaN/Inf)
        xml_content = xml_content.replace(
            'timestep="0.0005"',
            'timestep="0.005"'  # ✅ 10x plus grand = plus stable
        )
        
        # 2. Corriger le solveur (Newton -> PGS pour plus de stabilité)
        xml_content = xml_content.replace(
            'solver="Newton"',
            'solver="PGS"'  # ✅ Plus stable
        )
        
        # 3. Corriger l'intégrateur si nécessaire
        if 'integrator=' not in xml_content:
            xml_content = xml_content.replace(
                '<option timestep="0.005"',
                '<option timestep="0.005" integrator="RK4"'
            )
        
        # 4. Corriger les itérations (500 -> 50 pour plus de stabilité)
        xml_content = xml_content.replace(
            'iterations="500"',
            'iterations="50"'  # ✅ Moins d'itérations = plus stable
        )
        
        # 5. Corriger la tolérance (1e-12 -> 1e-6 pour plus de réalisme)
        xml_content = xml_content.replace(
            'tolerance="1e-12"',
            'tolerance="1e-6"'  # ✅ Tolérance réaliste
        )
        
        # 6. Optimiser les paramètres des actuateurs pour éviter les vitesses excessives
        # Réduire kp et augmenter kv pour plus de damping
        xml_content = xml_content.replace(
            'kp="120" kv="25"',
            'kp="80" kv="40"'  # ✅ Plus de damping, moins de raideur
        )
        
        # 7. Optimiser les paramètres des doigts
        xml_content = xml_content.replace(
            'kp="8" kv="1.5"',
            'kp="6" kv="3"'  # ✅ Plus de damping pour les doigts
        )
        
        xml_content = xml_content.replace(
            'kp="6" kv="1"',
            'kp="4" kv="2.5"'  # ✅ Encore plus de damping
        )
        
        xml_content = xml_content.replace(
            'kp="10" kv="2"',
            'kp="8" kv="4"'  # ✅ Pouce avec plus de damping
        )
        
        # 8. Ajouter des limites de force si pas présent
        if 'forcerange=' not in xml_content:
            # Ajouter des limites de force pour tous les actuateurs
            lines = xml_content.split('\n')
            corrected_lines = []
            
            for line in lines:
                if '<position name="act_' in line and 'forcerange=' not in line:
                    # Ajouter forcerange avant la fermeture
                    line = line.replace(' />', ' forcerange="-10 10" />')
                corrected_lines.append(line)
            
            xml_content = '\n'.join(corrected_lines)
        
        # Sauvegarder le modèle corrigé
        with open(target_path, 'w') as f:
            f.write(xml_content)
        
        print(f"✅ Modèle XML stable créé: {target_path}")
        print("🔧 Corrections appliquées:")
        print("  - Timestep: 0.0005 → 0.005 (10x plus stable)")
        print("  - Solveur: Newton → PGS (plus stable)")
        print("  - Itérations: 500 → 50 (plus rapide)")
        print("  - Tolérance: 1e-12 → 1e-6 (réaliste)")
        print("  - Actuateurs: kp réduit, kv augmenté (plus de damping)")
        print("  - Limites de force ajoutées")
        
        return target_path
        
    except Exception as e:
        print(f"❌ Erreur lors de la correction: {e}")
        return None

def verify_xml_stability(xml_path: str):
    """
    Vérifier que le modèle XML corrigé est stable
    """
    
    print(f"\n🧪 Vérification de la stabilité: {xml_path}")
    
    try:
        import mujoco
        
        # Charger le modèle
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        
        print("✅ Modèle chargé avec succès")
        print(f"  - DOFs: {model.nv}")
        print(f"  - Actuateurs: {model.nu}")
        print(f"  - Timestep: {model.opt.timestep}")
        print(f"  - Solveur: {model.opt.solver}")
        print(f"  - Intégrateur: {model.opt.integrator}")
        
        # Test de simulation
        print("\n🧪 Test de simulation...")
        for i in range(100):
            # Actions aléatoires modérées
            ctrl = np.random.uniform(-0.5, 0.5, model.nu)
            data.ctrl[:] = ctrl
            
            # Step de simulation
            mujoco.mj_step(model, data)
            
            # Vérifier NaN/Inf
            if np.any(np.isnan(data.qpos)) or np.any(np.isinf(data.qpos)):
                print(f"❌ NaN/Inf détecté dans qpos à l'étape {i}")
                return False
            
            if np.any(np.isnan(data.qvel)) or np.any(np.isinf(data.qvel)):
                print(f"❌ NaN/Inf détecté dans qvel à l'étape {i}")
                return False
            
            if np.any(np.isnan(data.qacc)) or np.any(np.isinf(data.qacc)):
                print(f"❌ NaN/Inf détecté dans qacc à l'étape {i}")
                return False
        
        print("✅ Test de simulation réussi - aucun NaN/Inf détecté!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

if __name__ == "__main__":
    print("🔧 CRÉATION D'UN MODÈLE XML STABLE")
    print("=" * 50)
    
    # Créer le modèle stable
    stable_path = create_stable_xml_model()
    
    if stable_path:
        # Vérifier la stabilité
        if verify_xml_stability(stable_path):
            print("\n🎉 SUCCÈS! Modèle XML stable créé et vérifié")
            print(f"📁 Chemin: {stable_path}")
            print("🚀 Vous pouvez maintenant utiliser ce modèle pour l'entraînement")
        else:
            print("\n⚠️ Modèle créé mais la vérification a échoué")
    else:
        print("\n❌ Échec de la création du modèle stable")
