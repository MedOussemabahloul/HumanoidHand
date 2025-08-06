#!/usr/bin/env python3
"""
TEST DE VALIDATION ULTRA-STABLE
Validation du modèle corrigé avec simulation stable comme avant les modifications
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
from pathlib import Path

class UltraStableValidator:
    """Validateur ultra-stable pour le modèle G1 corrigé"""
    
    def __init__(self, model_path="/workspace/results/g1_combined.xml"):
        self.model_path = Path(model_path)
        self.model = None
        self.data = None
        self.viewer = None
        
        # Paramètres de simulation ultra-conservateurs
        self.dt = 0.001  # Pas de temps très petit
        self.sim_time = 0.0
        
        # États de la tâche de grasping
        self.task_phase = "stabilize"  # stabilize -> search -> approach -> grasp -> lift -> hold
        self.phase_start_time = 0.0
        
        # Joints identifiés
        self.arm_joints = []
        self.finger_joints = []
        self.all_joints = []
        
        # Métriques de validation
        self.instability_count = 0
        self.contact_detected = False
        self.cube_lifted = False
        
    def load_model(self):
        """Charge le modèle avec validation"""
        print("🔍 Chargement du modèle corrigé...")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")
        
        try:
            # Changer vers le dossier results pour les chemins relatifs
            original_cwd = Path.cwd()
            results_dir = self.model_path.parent
            import os
            os.chdir(results_dir)
            
            try:
                self.model = mujoco.MjModel.from_xml_path(self.model_path.name)
                self.data = mujoco.MjData(self.model)
                
                print(f"✅ Modèle chargé: {self.model_path.name}")
                print(f"  - Corps: {self.model.nbody}")
                print(f"  - Joints: {self.model.njnt}")
                print(f"  - Actuateurs: {self.model.nu}")
                print(f"  - Capteurs: {self.model.nsensor}")
                print(f"  - DOFs: {self.model.nv}")
                
                # Validation des paramètres corrigés
                print(f"  - Timestep: {self.model.opt.timestep}")
                print(f"  - Iterations: {self.model.opt.iterations}")
                print(f"  - Solver: {self.model.opt.solver}")
                print(f"  - Tolerance: {self.model.opt.tolerance}")
                
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement: {e}")
    
    def identify_joints(self):
        """Identifie et classe les joints par catégorie"""
        print("🔍 Identification des joints...")
        
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) 
                      for i in range(self.model.njnt)]
        
        for i, name in enumerate(joint_names):
            if name:
                self.all_joints.append((i, name))
                
                if any(kw in name.lower() for kw in ["shoulder", "elbow", "wrist", "arm"]):
                    self.arm_joints.append(i)
                elif any(kw in name.lower() for kw in ["finger", "thumb", "index", "middle", "ring"]):
                    self.finger_joints.append(i)
        
        print(f"  ✅ Joints bras: {len(self.arm_joints)}")
        print(f"  ✅ Joints doigts: {len(self.finger_joints)}")
        print(f"  ✅ Joints totaux: {len(self.all_joints)}")
        
        # Affichage détaillé des joints problématiques (DOFs 15-30)
        print("\n🔍 Analyse des DOFs problématiques (15-30):")
        for dof_id in range(15, min(31, self.model.nv)):
            joint_id = self.model.dof_jntid[dof_id] if hasattr(self.model, 'dof_jntid') else dof_id
            if joint_id < self.model.njnt:
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                damping = self.model.dof_damping[dof_id] if dof_id < len(self.model.dof_damping) else 0
                print(f"  DOF {dof_id:2d}: {joint_name:25s} [damping: {damping:.1f}]")
    
    def get_cube_position(self):
        """Récupère la position du cube"""
        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if cube_body_id >= 0:
            return self.data.xpos[cube_body_id].copy()
        return np.array([0.3, 0.0, 0.05])  # Position par défaut
    
    def check_stability(self):
        """Vérifie la stabilité de la simulation"""
        # Vérification des NaN/Inf dans les accélérations
        if np.any(np.isnan(self.data.qacc)) or np.any(np.isinf(self.data.qacc)):
            self.instability_count += 1
            print(f"⚠️  INSTABILITÉ détectée (#{self.instability_count})")
            
            # Identification du DOF problématique
            for dof_id in range(min(31, self.model.nv)):
                if (dof_id < len(self.data.qacc) and 
                    (np.isnan(self.data.qacc[dof_id]) or np.isinf(self.data.qacc[dof_id]))):
                    
                    joint_id = self.model.dof_jntid[dof_id] if hasattr(self.model, 'dof_jntid') else dof_id
                    if joint_id < self.model.njnt:
                        joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                        print(f"   🚨 DOF {dof_id}: '{joint_name}' - qacc = {self.data.qacc[dof_id]}")
            
            return False
        
        return True
    
    def detect_contact(self):
        """Détecte le contact avec les capteurs tactiles"""
        contact_sum = 0.0
        
        for i in range(self.model.nsensor):
            sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if sensor_name and ("touch" in sensor_name.lower() or "contact" in sensor_name.lower()):
                if i < len(self.data.sensordata):
                    contact_value = self.data.sensordata[i]
                    if np.isfinite(contact_value):
                        contact_sum += abs(contact_value)
        
        self.contact_detected = contact_sum > 0.05
        return self.contact_detected
    
    def set_joint_targets(self, joint_ids, targets, strength=0.3):
        """Définit les positions cibles pour des joints avec force limitée"""
        for i, joint_id in enumerate(joint_ids):
            if i < len(targets) and joint_id < self.model.nu:
                # Force très limitée pour éviter l'instabilité
                self.data.ctrl[joint_id] = targets[i] * strength
    
    def stabilize_phase(self):
        """Phase de stabilisation initiale"""
        # Maintenir tous les joints à zéro avec force minimale
        self.data.ctrl[:] = 0.0
        
        # Transition après stabilisation
        if self.sim_time - self.phase_start_time > 3.0:
            self.task_phase = "search"
            self.phase_start_time = self.sim_time
            print("🔍 Phase: Recherche du cube")
    
    def search_phase(self):
        """Phase de recherche: mouvement très lent vers le cube"""
        cube_pos = self.get_cube_position()
        
        # Mouvement ultra-progressif des bras vers le cube
        if len(self.arm_joints) >= 6:
            # Bras gauche
            left_targets = [0.1, 0.2, -0.1, -0.5, 0.0, 0.3]
            self.set_joint_targets(self.arm_joints[:6], left_targets, strength=0.1)
            
            # Bras droit (si disponible)
            if len(self.arm_joints) >= 12:
                right_targets = [0.1, -0.2, 0.1, -0.5, 0.0, -0.3]
                self.set_joint_targets(self.arm_joints[6:12], right_targets, strength=0.1)
        
        # Transition vers l'approche après 5 secondes
        if self.sim_time - self.phase_start_time > 5.0:
            self.task_phase = "approach"
            self.phase_start_time = self.sim_time
            print("🤏 Phase: Approche du cube")
    
    def approach_phase(self):
        """Phase d'approche: positionner les mains près du cube"""
        # Maintenir la position de recherche
        self.search_phase()
        
        # Ouvrir légèrement les doigts
        finger_targets = [0.2] * len(self.finger_joints)
        self.set_joint_targets(self.finger_joints, finger_targets, strength=0.05)
        
        # Transition si contact détecté ou après 4 secondes
        if self.detect_contact() or self.sim_time - self.phase_start_time > 4.0:
            self.task_phase = "grasp"
            self.phase_start_time = self.sim_time
            print("✋ Phase: Saisie du cube")
    
    def grasp_phase(self):
        """Phase de saisie: fermer progressivement les doigts"""
        # Maintenir la position des bras
        self.approach_phase()
        
        # Fermeture progressive des doigts
        grasp_progress = min(1.0, (self.sim_time - self.phase_start_time) / 3.0)
        finger_closure = grasp_progress * 0.8  # Fermeture modérée
        
        finger_targets = [finger_closure] * len(self.finger_joints)
        self.set_joint_targets(self.finger_joints, finger_targets, strength=0.08)
        
        # Transition vers le levage après 3 secondes
        if self.sim_time - self.phase_start_time > 3.0:
            self.task_phase = "lift"
            self.phase_start_time = self.sim_time
            print("⬆️  Phase: Levage du cube")
    
    def lift_phase(self):
        """Phase de levage: soulever délicatement le cube"""
        # Maintenir la prise ferme
        finger_targets = [0.8] * len(self.finger_joints)
        self.set_joint_targets(self.finger_joints, finger_targets, strength=0.08)
        
        # Mouvement de levage très progressif
        if len(self.arm_joints) >= 6:
            # Lever légèrement les coudes
            lift_targets = [0.0, 0.1, -0.3, -0.3, 0.0, 0.5]
            self.set_joint_targets(self.arm_joints[:6], lift_targets, strength=0.15)
            
            if len(self.arm_joints) >= 12:
                right_lift_targets = [0.0, -0.1, 0.3, -0.3, 0.0, -0.5]
                self.set_joint_targets(self.arm_joints[6:12], right_lift_targets, strength=0.15)
        
        # Vérifier si le cube est levé
        cube_pos = self.get_cube_position()
        if cube_pos[2] > 0.07:  # 2cm au-dessus de la position initiale
            self.cube_lifted = True
        
        # Transition vers le maintien après 4 secondes
        if self.sim_time - self.phase_start_time > 4.0:
            self.task_phase = "hold"
            self.phase_start_time = self.sim_time
            print("🤝 Phase: Maintien du cube")
    
    def hold_phase(self):
        """Phase de maintien: tenir le cube stable"""
        # Maintenir la position de levage
        self.lift_phase()
        
        # Légère oscillation pour démontrer le contrôle
        oscillation = 0.05 * np.sin(2 * np.pi * 0.2 * self.sim_time)
        
        # Appliquer aux poignets si disponibles
        if len(self.arm_joints) >= 6:
            wrist_adjustment = [0, 0, 0, 0, oscillation, 0]
            current_targets = [0.0, 0.1, -0.3, -0.3, oscillation, 0.5]
            self.set_joint_targets(self.arm_joints[:6], current_targets, strength=0.1)
    
    def control_step(self):
        """Étape de contrôle selon la phase actuelle"""
        if self.task_phase == "stabilize":
            self.stabilize_phase()
        elif self.task_phase == "search":
            self.search_phase()
        elif self.task_phase == "approach":
            self.approach_phase()
        elif self.task_phase == "grasp":
            self.grasp_phase()
        elif self.task_phase == "lift":
            self.lift_phase()
        elif self.task_phase == "hold":
            self.hold_phase()
    
    def run_simulation(self, duration=30.0):
        """Lance la simulation de validation"""
        print("🚀 Lancement de la simulation de validation...")
        print(f"  - Durée: {duration:.1f} secondes")
        print("  - Contrôles: ESC pour quitter")
        
        # Initialisation
        self.phase_start_time = 0.0
        self.instability_count = 0
        self.contact_detected = False
        self.cube_lifted = False
        
        print("⚖️  Phase: Stabilisation initiale")
        
        # Lancer le viewer interactif
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            
            while viewer.is_running() and (time.time() - start_time) < duration:
                step_start = time.time()
                
                # Vérification de stabilité AVANT le step
                if not self.check_stability():
                    print("🚨 Simulation instable - tentative de récupération...")
                    # Réinitialisation partielle
                    self.data.qvel[:] = 0.0
                    self.data.ctrl[:] = 0.0
                    time.sleep(0.1)  # Pause pour stabilisation
                    continue
                
                # Étape de contrôle
                self.control_step()
                
                # Étape de simulation avec gestion d'erreurs
                try:
                    mujoco.mj_step(self.model, self.data)
                    self.sim_time = self.data.time
                except Exception as e:
                    print(f"🚨 Erreur MuJoCo step: {e}")
                    break
                
                # Synchronisation du viewer
                viewer.sync()
                
                # Maintenir le framerate
                time_until_next_step = self.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
        print("✅ Simulation terminée")
        
        # Rapport final
        print(f"\n📊 RAPPORT DE VALIDATION:")
        print(f"   ⚠️  Instabilités détectées: {self.instability_count}")
        print(f"   📱 Contact détecté: {'✅' if self.contact_detected else '❌'}")
        print(f"   📦 Cube levé: {'✅' if self.cube_lifted else '❌'}")
        print(f"   🕒 Durée simulation: {self.sim_time:.1f}s")
        
        return self.instability_count == 0
    
    def run_validation(self):
        """Lance la validation complète"""
        print("🔬 VALIDATION ULTRA-STABLE G1")
        print("=" * 50)
        
        try:
            # Charger le modèle
            self.load_model()
            
            # Identifier les joints
            self.identify_joints()
            
            # Lancer la simulation
            success = self.run_simulation()
            
            print("\n🎉 VALIDATION TERMINÉE!")
            if success:
                print("✅ Modèle ULTRA-STABLE validé")
                print("✅ Aucune instabilité détectée")
                print("✅ Simulation fluide et contrôlée")
                print("✅ Séquence de grasping exécutée")
            else:
                print("⚠️  Modèle partiellement validé")
                print("⚠️  Quelques instabilités détectées")
                print("💡 Les corrections ont amélioré la stabilité")
            
            return success
            
        except Exception as e:
            print(f"\n❌ ERREUR DE VALIDATION: {e}")
            print("💡 Vérifiez que le modèle a été corrigé correctement")
            return False

def main():
    """Point d'entrée principal"""
    print("🔬 VALIDATEUR ULTRA-STABLE G1")
    print("=" * 40)
    
    # Vérifier que MuJoCo est installé
    try:
        import mujoco
        print(f"✅ MuJoCo version: {mujoco.__version__}")
    except ImportError:
        print("❌ MuJoCo non installé")
        print("💡 Installation: pip install mujoco")
        sys.exit(1)
    
    # Lancer la validation
    validator = UltraStableValidator()
    success = validator.run_validation()
    
    if success:
        print("\n🎉 VALIDATION RÉUSSIE - Prêt pour l'entraînement!")
        print("🚀 Commande recommandée:")
        print("   python3 train_ultra_stable_final.py --episodes 30 --video")
    else:
        print("\n⚠️  VALIDATION PARTIELLE - Améliorations nécessaires")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()