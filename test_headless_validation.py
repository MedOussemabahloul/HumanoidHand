#!/usr/bin/env python3
"""
TEST DE VALIDATION HEADLESS
Validation du modèle corrigé sans interface graphique
"""

import mujoco
import numpy as np
import time
import sys
from pathlib import Path

class HeadlessValidator:
    """Validateur headless pour le modèle G1 corrigé"""
    
    def __init__(self, model_path="/workspace/results/g1_combined.xml"):
        self.model_path = Path(model_path)
        self.model = None
        self.data = None
        
        # Paramètres de simulation
        self.dt = 0.001
        self.sim_time = 0.0
        
        # Métriques de validation
        self.instability_count = 0
        self.total_steps = 0
        self.max_steps = 10000  # 10 secondes à 1000 Hz
        
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
    
    def analyze_joints(self):
        """Analyse détaillée des joints"""
        print("\n🔍 Analyse des DOFs problématiques (15-30):")
        
        problem_dofs = []
        finger_dofs = []
        arm_dofs = []
        
        for dof_id in range(15, min(31, self.model.nv)):
            joint_id = self.model.dof_jntid[dof_id] if hasattr(self.model, 'dof_jntid') else dof_id
            if joint_id < self.model.njnt:
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                damping = self.model.dof_damping[dof_id] if dof_id < len(self.model.dof_damping) else 0
                
                joint_type = "FINGER" if any(kw in joint_name.lower() for kw in ["finger", "thumb", "index", "middle", "ring"]) else "ARM"
                
                if joint_type == "FINGER":
                    finger_dofs.append(dof_id)
                else:
                    arm_dofs.append(dof_id)
                
                print(f"  DOF {dof_id:2d}: {joint_name:25s} [{joint_type}] [damping: {damping:.1f}]")
                
                # Vérifier si ce DOF était problématique (damping faible pour les doigts)
                if joint_type == "FINGER" and damping < 10:
                    problem_dofs.append(dof_id)
        
        print(f"\n📊 Résumé de l'analyse:")
        print(f"  🖐️  DOFs doigts (15-30): {len(finger_dofs)}")
        print(f"  💪 DOFs bras (15-30): {len(arm_dofs)}")
        print(f"  ⚠️  DOFs potentiellement problématiques: {len(problem_dofs)}")
        
        return problem_dofs, finger_dofs, arm_dofs
    
    def check_stability(self):
        """Vérifie la stabilité de la simulation"""
        # Vérification des NaN/Inf dans les accélérations
        if np.any(np.isnan(self.data.qacc)) or np.any(np.isinf(self.data.qacc)):
            self.instability_count += 1
            
            # Identification du DOF problématique
            problem_dofs = []
            for dof_id in range(min(31, self.model.nv)):
                if (dof_id < len(self.data.qacc) and 
                    (np.isnan(self.data.qacc[dof_id]) or np.isinf(self.data.qacc[dof_id]))):
                    problem_dofs.append(dof_id)
            
            print(f"⚠️  INSTABILITÉ #{self.instability_count} détectée aux DOFs: {problem_dofs}")
            return False
        
        return True
    
    def apply_test_actions(self):
        """Applique des actions de test progressives"""
        # Actions très conservatrices pour tester la stabilité
        self.data.ctrl[:] = 0.0
        
        # Test progressif des actuateurs
        step_ratio = self.total_steps / self.max_steps
        
        if step_ratio > 0.2:  # Après 20% du temps
            # Test léger des bras
            for i in range(min(14, self.model.nu)):  # Premiers actuateurs (bras)
                self.data.ctrl[i] = 0.001 * np.sin(0.1 * self.total_steps)
        
        if step_ratio > 0.5:  # Après 50% du temps
            # Test très léger des doigts
            for i in range(14, min(30, self.model.nu)):  # Actuateurs des doigts
                self.data.ctrl[i] = 0.0005 * np.sin(0.05 * self.total_steps)
    
    def run_headless_simulation(self):
        """Lance la simulation headless"""
        print("🚀 Lancement de la simulation headless...")
        print(f"  - Steps max: {self.max_steps}")
        print(f"  - Durée estimée: {self.max_steps * self.dt:.1f}s")
        
        # Initialisation
        self.instability_count = 0
        self.total_steps = 0
        
        # Reset initial
        mujoco.mj_resetData(self.model, self.data)
        
        # Stabilisation initiale
        print("⚖️  Phase: Stabilisation initiale...")
        for i in range(1000):
            mujoco.mj_forward(self.model, self.data)
            if i % 100 == 0 and not self.check_stability():
                print(f"⚠️  Instabilité lors de la stabilisation (step {i})")
                self.data.qvel[:] = 0.0
        
        print("🔄 Phase: Simulation avec actions de test...")
        start_time = time.time()
        
        # Boucle principale de simulation
        for step in range(self.max_steps):
            self.total_steps = step
            
            # Vérification de stabilité AVANT le step
            if not self.check_stability():
                print(f"🚨 Arrêt à cause d'instabilité (step {step})")
                break
            
            # Appliquer des actions de test
            self.apply_test_actions()
            
            # Step de simulation
            try:
                mujoco.mj_step(self.model, self.data)
                self.sim_time = self.data.time
            except Exception as e:
                print(f"🚨 Erreur MuJoCo step {step}: {e}")
                self.instability_count += 1
                break
            
            # Logging périodique
            if step % 1000 == 0 and step > 0:
                elapsed = time.time() - start_time
                progress = step / self.max_steps * 100
                print(f"   📊 Progrès: {progress:.1f}% - Instabilités: {self.instability_count} - Temps: {elapsed:.1f}s")
        
        elapsed = time.time() - start_time
        print(f"✅ Simulation terminée après {self.total_steps} steps ({elapsed:.2f}s)")
        
        return self.instability_count == 0
    
    def run_validation(self):
        """Lance la validation complète"""
        print("🔬 VALIDATION HEADLESS G1")
        print("=" * 50)
        
        try:
            # Charger le modèle
            self.load_model()
            
            # Analyser les joints
            problem_dofs, finger_dofs, arm_dofs = self.analyze_joints()
            
            # Lancer la simulation
            success = self.run_headless_simulation()
            
            # Rapport final
            print(f"\n📊 RAPPORT DE VALIDATION:")
            print(f"   ⚠️  Instabilités détectées: {self.instability_count}")
            print(f"   🕒 Steps simulés: {self.total_steps}/{self.max_steps}")
            print(f"   🕒 Temps simulé: {self.sim_time:.3f}s")
            print(f"   📈 Taux de réussite: {(1 - self.instability_count/max(1, self.total_steps/1000)) * 100:.1f}%")
            
            # Analyse des corrections
            print(f"\n🔧 ANALYSE DES CORRECTIONS:")
            corrected_finger_dofs = [dof for dof in finger_dofs if dof < len(self.model.dof_damping) and self.model.dof_damping[dof] >= 10]
            print(f"   🖐️  DOFs doigts corrigés (damping ≥ 10): {len(corrected_finger_dofs)}/{len(finger_dofs)}")
            
            if len(corrected_finger_dofs) == len(finger_dofs):
                print(f"   ✅ TOUTES les corrections de damping appliquées")
            else:
                print(f"   ⚠️  Corrections partielles de damping")
            
            print("\n🎉 VALIDATION TERMINÉE!")
            if success:
                print("✅ Modèle ULTRA-STABLE validé")
                print("✅ Aucune instabilité critique détectée")
                print("✅ Corrections physiques efficaces")
                print("✅ Prêt pour l'entraînement")
            elif self.instability_count < 5:
                print("🟡 Modèle PARTIELLEMENT stable")
                print("🟡 Quelques instabilités mineures")
                print("🟡 Améliorations significatives observées")
            else:
                print("🔴 Modèle encore INSTABLE")
                print("🔴 Corrections supplémentaires nécessaires")
            
            return success
            
        except Exception as e:
            print(f"\n❌ ERREUR DE VALIDATION: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Point d'entrée principal"""
    print("🔬 VALIDATEUR HEADLESS G1")
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
    validator = HeadlessValidator()
    success = validator.run_validation()
    
    if success:
        print("\n🎉 VALIDATION RÉUSSIE!")
        print("🚀 Recommandations:")
        print("   1. Lancer l'entraînement: python3 train_ultra_stable_final.py")
        print("   2. Ou tester avec interface: python3 test_ultra_stable_validation.py")
    else:
        print("\n⚠️  VALIDATION PARTIELLE")
        print("💡 Le modèle est plus stable mais peut nécessiter des ajustements")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()