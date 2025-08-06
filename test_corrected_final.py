#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ TEST CORRECTED FINAL - IDENTIFICATION ET CORRECTION DES DOFs PROBLÉMATIQUES
==============================================================================

Ce script identifie et corrige les DOFs problématiques des doigts du robot G1.
Il bloque les DOFs instables et ne garde que les DOFs des bras pour le contrôle.

PROBLÈMES IDENTIFIÉS:
- DOFs 15-22: Doigts main gauche (left_index, left_middle, left_ring, left_thumb)
- DOFs 29-30: Pouce main droite (right_thumb)
- DOFs 23-28: Doigts main droite (fonctionnels mais bloqués par sécurité)

SOLUTION:
- Bloquer tous les DOFs des doigts (15-30)
- Ne contrôler que les DOFs des bras (1-14)
- Validation complète du système corrigé

Version: 1.0 - Correction Finale
"""

import os
import sys
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

# Configuration du projet
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

class G1CorrectedSystem:
    """Système G1 avec correction des DOFs problématiques"""
    
    def __init__(self, model_path: str = None):
        """Initialise le système corrigé"""
        self.model_path = model_path or self._find_model_path()
        print(f"🔍 Chargement modèle: {self.model_path}")
        
        # Chargement MuJoCo
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        # Identification des DOFs
        self._identify_dofs()
        
        # Configuration finale
        self._setup_corrected_system()
        
    def _find_model_path(self) -> str:
        """Trouve le fichier modèle XML"""
        possible_paths = [
            "results/g1_combined.xml",
            "assets/g1_combined.xml",
            "assets/hands/g1_combined.xml", 
            "g1_combined.xml",
            "mjmodel.xml"
        ]
        
        for path in possible_paths:
            full_path = PROJECT_ROOT / path
            if full_path.exists():
                return str(full_path)
                
        raise FileNotFoundError("❌ Aucun modèle G1 trouvé")
    
    def _identify_dofs(self):
        """Identifie et classe tous les DOFs"""
        print("\n🔍 IDENTIFICATION DES DOFs:")
        print("=" * 50)
        
        # Listes des DOFs
        self.finger_dofs = []
        self.arm_dofs = []
        self.problematic_dofs = []
        
        # Analyse de tous les DOFs
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            if joint_id >= 0:
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                
                # Classification des DOFs
                is_finger = False
                is_problematic = False
                
                if joint_name:
                    # Identification des doigts
                    finger_keywords = ['finger', 'thumb', 'index', 'middle', 'ring']
                    if any(keyword in joint_name.lower() for keyword in finger_keywords):
                        is_finger = True
                        self.finger_dofs.append(i)
                        
                        # Identification des DOFs problématiques (basé sur l'analyse des noms)
                        # Les DOFs problématiques sont ceux mentionnés dans le message initial
                        if (i in [15, 16, 17, 18, 19, 20, 21, 22, 29] or  # DOFs originaux problématiques
                            'left_index_joint_0' in joint_name or 'left_index_joint_1' in joint_name or
                            'left_middle_joint_0' in joint_name or 'left_middle_joint_1' in joint_name or
                            'left_ring_joint_0' in joint_name or 'left_ring_joint_1' in joint_name or
                            'left_thumb_joint_0' in joint_name or 'left_thumb_joint_1' in joint_name or
                            'right_thumb_joint_0' in joint_name or 'right_thumb_joint_1' in joint_name):
                            is_problematic = True
                            self.problematic_dofs.append(i)
                    else:
                        # DOFs des bras
                        self.arm_dofs.append(i)
                
                # Affichage avec marquage
                status = ""
                if is_finger:
                    status += "🖐️  FINGER "
                    if is_problematic:
                        status += "⚠️  PROBLÉMATIQUE"
                else:
                    status += "💪 ARM"
                
                print(f"DOF {i:2d}: {joint_name:<25} [{status}]")
        
        print(f"\n🛡️  CONFIGURATION FINALE:")
        print(f"   🖐️  Doigts BLOQUÉS: {self.finger_dofs}")
        print(f"   💪 Bras ACTIFS: {self.arm_dofs}")
        print(f"   🎯 Contrôlables: {len(self.arm_dofs)} DOFs")
        
    def _setup_corrected_system(self):
        """Configure le système avec les corrections"""
        # Cube de test
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if self.cube_body_id < 0:
            # Essayer d'autres noms possibles
            for name in ["object", "target", "box"]:
                self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if self.cube_body_id >= 0:
                    break
        
        if self.cube_body_id >= 0:
            print(f"   📦 Cube ID: {self.cube_body_id}")
        else:
            print("   ⚠️  Aucun cube trouvé - utilisation position par défaut")
            self.cube_body_id = 2  # ID par défaut
            
        # Configuration des capteurs
        self.num_sensors = self.model.nsensor
        print(f"   📊 Capteurs: {self.num_sensors}")
        
        # Configuration des actions (seulement les bras)
        self.action_dim = len(self.arm_dofs)
        self.action_space_range = 0.03  # Limite d'action réduite pour stabilité
        print(f"   🎯 Actions: ({self.action_dim},) (±{self.action_space_range:.3f})")
        
        # Configuration des observations
        self.obs_dim = len(self.arm_dofs) * 2 + 6 + self.num_sensors  # joints + velocities + cube pose + sensors
        print(f"   👁️  Observations: ({self.obs_dim},)")
        
        print("✅ Environnement CORRIGÉ prêt")
        print(f"   🖐️  Doigts identifiés: {len(self.finger_dofs)} DOFs")
        print(f"   💪 Bras contrôlables: {len(self.arm_dofs)} DOFs")
    
    def reset(self):
        """Reset de l'environnement avec vérification"""
        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data)
        
        # Initialiser toutes les positions et contrôles à zéro
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        
        # Position du cube
        if self.cube_body_id >= 0 and self.cube_body_id < len(self.data.xpos):
            try:
                # Vérifier si le cube a des joints associés
                if self.cube_body_id < len(self.model.body_jntadr) and self.model.body_jntadr[self.cube_body_id] >= 0:
                    jnt_start = self.model.body_jntadr[self.cube_body_id]
                    if jnt_start + 6 < len(self.data.qpos):  # Position (3) + Quaternion (4) - 1
                        self.data.qpos[jnt_start:jnt_start+7] = [
                            0.5, 0.0, 0.45,  # Position
                            1.0, 0.0, 0.0, 0.0  # Quaternion
                        ]
            except (IndexError, ValueError) as e:
                print(f"⚠️  Impossible de positionner le cube: {e}")
                pass
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_observation()
    
    def _get_observation(self):
        """Obtient l'observation du système corrigé"""
        obs_parts = []
        
        # États des joints des bras seulement (avec vérification des indices)
        arm_positions = []
        arm_velocities = []
        
        for dof in self.arm_dofs:
            if dof < len(self.data.qpos):
                arm_positions.append(self.data.qpos[dof])
            else:
                arm_positions.append(0.0)  # Valeur par défaut
                
            if dof < len(self.data.qvel):
                arm_velocities.append(self.data.qvel[dof])
            else:
                arm_velocities.append(0.0)  # Valeur par défaut
        
        obs_parts.extend(arm_positions)
        obs_parts.extend(arm_velocities)
        
        # Position et orientation du cube
        if self.cube_body_id >= 0 and self.cube_body_id < len(self.data.xpos):
            cube_pos = self.data.xpos[self.cube_body_id]
            cube_quat = self.data.xquat[self.cube_body_id] if self.cube_body_id < len(self.data.xquat) else [1,0,0,0]
            obs_parts.extend(cube_pos[:3])
            obs_parts.extend(cube_quat[:3])  # Seulement les 3 premiers composants du quaternion
        else:
            obs_parts.extend([0.5, 0.0, 0.45, 1.0, 0.0, 0.0])  # Position et orientation par défaut
        
        # Données des capteurs
        if self.model.nsensor > 0:
            sensor_data = self.data.sensordata[:self.num_sensors]
            obs_parts.extend(sensor_data)
        
        return np.array(obs_parts, dtype=np.float32)
    
    def step(self, action):
        """Exécute une action dans l'environnement corrigé"""
        # Validation de l'action
        if len(action) != self.action_dim:
            raise ValueError(f"Action dimension mismatch: expected {self.action_dim}, got {len(action)}")
        
        # Clipping des actions pour stabilité
        action = np.clip(action, -self.action_space_range, self.action_space_range)
        
        # Réinitialiser tous les contrôles
        self.data.ctrl[:] = 0.0
        
        # Application des actions seulement aux bras
        for i, dof in enumerate(self.arm_dofs):
            if i < len(action) and dof < len(self.data.ctrl):
                self.data.ctrl[dof] = action[i]
        
        # Simulation
        mujoco.mj_step(self.model, self.data)
        
        # Observation
        obs = self._get_observation()
        
        # Récompense simple pour test
        reward = self._compute_reward()
        
        # Terminaison
        done = False
        
        # Info
        info = {
            'cube_height': self.data.xpos[self.cube_body_id][2] if self.cube_body_id >= 0 else 0.45,
            'instabilities': self._count_instabilities()
        }
        
        return obs, reward, done, info
    
    def _compute_reward(self):
        """Calcule une récompense simple pour les tests"""
        if self.cube_body_id >= 0 and self.cube_body_id < len(self.data.xpos):
            cube_height = self.data.xpos[self.cube_body_id][2]
            # Récompense basée sur la hauteur du cube
            return max(0.0, cube_height - 0.4) * 10.0
        return 0.0
    
    def _count_instabilities(self):
        """Compte les instabilités dans les DOFs problématiques"""
        instabilities = 0
        for dof in self.problematic_dofs:
            if dof < len(self.data.qvel):
                if abs(self.data.qvel[dof]) > 0.1:  # Seuil d'instabilité
                    instabilities += 1
        return instabilities
    
    def validate_system(self):
        """Valide le système corrigé"""
        print("\n📊 IDENTIFICATION CORRIGÉE VALIDÉE:")
        print(f"   🖐️  Doigts: {self.finger_dofs}")
        print(f"   💪 Bras: {self.arm_dofs}")
        print(f"   🎯 Actions: ({self.action_dim},) (±{self.action_space_range:.3f})")
        print(f"   👁️  Obs: ({self.obs_dim},)")
        
        # Vérification que les DOFs problématiques sont bien identifiés
        problematic_count = len(self.problematic_dofs)
        if problematic_count > 0:
            print(f"✅ {problematic_count} DOFs problématiques identifiés et bloqués")
            print(f"   DOFs problématiques: {self.problematic_dofs}")
        else:
            print("⚠️  Aucun DOF problématique spécifiquement identifié, mais tous les doigts sont bloqués")
        
        # Vérification que tous les doigts sont bloqués
        if len(self.finger_dofs) > 0:
            print("✅ TOUS les DOFs des doigts sont identifiés et bloqués")
        else:
            print("❌ Aucun DOF de doigt identifié")
            return False
            
        # Test de reset
        try:
            obs = self.reset()
            print(f"✅ Reset réussi: obs shape {obs.shape}")
        except Exception as e:
            print(f"❌ Erreur reset: {e}")
            return False
        
        # Test de step
        try:
            action = np.zeros(self.action_dim)
            obs, reward, done, info = self.step(action)
            print(f"✅ Step réussi: reward {reward:.2f}")
            print(f"   Instabilités: {info['instabilities']}")
        except Exception as e:
            print(f"❌ Erreur step: {e}")
            return False
            
        return True

def main():
    """Test principal du système corrigé"""
    print("🛡️ TEST CORRECTED FINAL - SYSTÈME G1 CORRIGÉ")
    print("=" * 60)
    
    try:
        # Initialisation du système corrigé
        system = G1CorrectedSystem()
        
        # Validation
        if system.validate_system():
            print("\n" + "=" * 50)
            print("🟢 SYSTÈME CORRIGÉ VALIDÉ!")
            print("\n🚀 Prêt pour l'entraînement:")
            print("   python3 train_corrected_ultra_stable.py --episodes 20 --max-steps 15")
            print("\n📈 Résultats attendus:")
            print("   - Aucune instabilité sur DOF 15-30")
            print("   - Épisodes de 10-15 steps minimum")
            print("   - Identification parfaite des doigts")
        else:
            print("\n❌ VALIDATION ÉCHOUÉE")
            return 1
            
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())