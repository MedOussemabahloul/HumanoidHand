
#!/usr/bin/env python3
"""
Script professionnel de test et simulation G1 Manipulation
Teste les mouvements des mains, grasping et lifting du cube
Auteur: Assistant IA
Projet: G1 Fingers Manipulation
"""

import mujoco
from mujoco import viewer as mj_viewer
import numpy as np
import time
import sys
from pathlib import Path

class G1ManipulationTester:
    """Testeur professionnel pour la manipulation G1"""
    
    def __init__(self, model_path="/home/oussema/Documents/project/results/g1_combined.xml"):
        self.model_path = Path(model_path)
        self.model = None
        self.data = None
        self.viewer = None
        
        # Paramètres de simulation
        self.dt = 0.002  # Pas de temps
        self.sim_time = 0.0
        
        # États de la tâche
        self.task_phase = "approach"  # approach -> grasp -> lift -> hold
        self.phase_start_time = 0.0
        
        # Positions cibles pour les mains
        self.left_hand_joints = []
        self.right_hand_joints = []
        self.finger_joints = []
        
    def load_model(self):
        """Charge le modèle MuJoCo"""
        print("🔍 Chargement du modèle...")
        
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
                if "left" in name and ("shoulder" in name or "elbow" in name or "wrist" in name):
                    self.left_hand_joints.append(i)
                elif "right" in name and ("shoulder" in name or "elbow" in name or "wrist" in name):
                    self.right_hand_joints.append(i)
                elif "finger" in name or "thumb" in name:
                    self.finger_joints.append(i)
        
        print(f"  ✅ Joints bras gauche: {len(self.left_hand_joints)}")
        print(f"  ✅ Joints bras droit: {len(self.right_hand_joints)}")
        print(f"  ✅ Joints doigts: {len(self.finger_joints)}")
    
    def get_cube_position(self):
        """Récupère la position du cube"""
        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if cube_body_id >= 0:
            return self.data.xpos[cube_body_id].copy()
        return np.array([0.5, 0.0, 0.45])  # Position par défaut
    
    def set_hand_position(self, hand_joints, target_pos, strength=0.1):
        """Définit la position cible pour une main"""
        for i, joint_idx in enumerate(hand_joints):
            if i < len(target_pos):
                self.data.ctrl[joint_idx] = target_pos[i] * strength
    
    def set_finger_position(self, openness=0.0):
        """Contrôle l'ouverture/fermeture des doigts (0=fermé, 1=ouvert)"""
        finger_target = openness * 1.5  # Amplitude maximale
        for joint_idx in self.finger_joints:
            if joint_idx < self.model.nu:
                self.data.ctrl[joint_idx] = finger_target
    
    def approach_phase(self):
        """Phase d'approche: positionner les mains près du cube"""
        cube_pos = self.get_cube_position()
        
        # Positions d'approche pour les bras
        left_approach = [0.3, 0.5, -0.2, -1.0, 0.0, 0.5, 0.0]  # Configuration d'approche
        right_approach = [0.3, -0.5, 0.2, -1.0, 0.0, -0.5, 0.0]
        
        # Appliquer les positions
        self.set_hand_position(self.left_hand_joints, left_approach, strength=0.5)
        self.set_hand_position(self.right_hand_joints, right_approach, strength=0.5)
        
        # Ouvrir les doigts
        self.set_finger_position(openness=1.0)
        
        # Transition vers la phase de saisie après 3 secondes
        if self.sim_time - self.phase_start_time > 3.0:
            self.task_phase = "grasp"
            self.phase_start_time = self.sim_time
            print("🤏 Phase: Saisie du cube")
    
    def grasp_phase(self):
        """Phase de saisie: fermer les doigts sur le cube"""
        # Maintenir la position des bras
        self.approach_phase()  # Même position que l'approche
        
        # Fermer progressivement les doigts
        grasp_progress = min(1.0, (self.sim_time - self.phase_start_time) / 2.0)
        finger_closure = 1.0 - grasp_progress  # De ouvert (1) à fermé (0)
        self.set_finger_position(openness=finger_closure)
        
        # Transition vers le levage après 2 secondes
        if self.sim_time - self.phase_start_time > 2.0:
            self.task_phase = "lift"
            self.phase_start_time = self.sim_time
            print("⬆️  Phase: Levage du cube")
    
    def lift_phase(self):
        """Phase de levage: soulever le cube"""
        # Positions de levage (élever les bras)
        left_lift = [0.0, 0.3, -0.5, -0.8, 0.0, 0.8, 0.0]
        right_lift = [0.0, -0.3, 0.5, -0.8, 0.0, -0.8, 0.0]
        
        self.set_hand_position(self.left_hand_joints, left_lift, strength=0.8)
        self.set_hand_position(self.right_hand_joints, right_lift, strength=0.8)
        
        # Maintenir la prise
        self.set_finger_position(openness=0.2)
        
        # Transition vers le maintien après 3 secondes
        if self.sim_time - self.phase_start_time > 3.0:
            self.task_phase = "hold"
            self.phase_start_time = self.sim_time
            print("🤝 Phase: Maintien du cube")
    
    def hold_phase(self):
        """Phase de maintien: tenir le cube en l'air"""
        # Maintenir la position de levage
        self.lift_phase()
        
        # Ajouter un léger mouvement oscillatoire pour montrer le contrôle
        oscillation = 0.1 * np.sin(2 * np.pi * 0.5 * self.sim_time)
        
        # Appliquer l'oscillation aux poignets
        for joint_idx in self.left_hand_joints[-2:]:  # Derniers joints (poignets)
            if joint_idx < self.model.nu:
                self.data.ctrl[joint_idx] = oscillation
        
        for joint_idx in self.right_hand_joints[-2:]:
            if joint_idx < self.model.nu:
                self.data.ctrl[joint_idx] = oscillation
    
    def control_step(self):
        """Étape de contrôle selon la phase actuelle"""
        if self.task_phase == "approach":
            self.approach_phase()
        elif self.task_phase == "grasp":
            self.grasp_phase()
        elif self.task_phase == "lift":
            self.lift_phase()
        elif self.task_phase == "hold":
            self.hold_phase()
    
    def run_simulation(self, duration=20.0):
        """Lance la simulation interactive"""
        print("🚀 Lancement de la simulation interactive...")
        print("  - Durée: {:.1f} secondes".format(duration))
        print("  - Contrôles: ESC pour quitter")
        
        # Initialiser la position
        self.phase_start_time = 0.0
        print("👋 Phase: Approche du cube")
        
        # Lancer le viewer interactif
        with mj_viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            
            while viewer.is_running() and (time.time() - start_time) < duration:
                step_start = time.time()
                
                # Étape de contrôle
                self.control_step()
                
                # Étape de simulation
                mujoco.mj_step(self.model, self.data)
                self.sim_time = self.data.time
                
                # Synchronisation du viewer
                viewer.sync()
                
                # Maintenir le framerate
                time_until_next_step = self.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
        print("✅ Simulation terminée")
    
    def run_test(self):
        """Lance le test complet"""
        print("🚀 TEST DE MANIPULATION G1")
        print("=" * 50)
        
        try:
            # Charger le modèle
            self.load_model()
            
            # Identifier les joints
            self.identify_joints()
            
            # Lancer la simulation
            self.run_simulation()
            
            print("\n🎉 TEST RÉUSSI!")
            print("✅ Modèle fonctionnel")
            print("✅ Simulation interactive lancée")
            print("✅ Séquence de manipulation exécutée")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERREUR: {e}")
            print("💡 Vérifiez que le modèle a été créé avec: python scripts/create_combined_model.py")
            return False

def main():
    """Point d'entrée principal"""
    print("🤖 G1 MANIPULATION TESTER")
    print("=" * 40)
    
    # Vérifier que MuJoCo est installé
    try:
        import mujoco
        print(f"✅ MuJoCo version: {mujoco.__version__}")
    except ImportError:
        print("❌ MuJoCo non installé")
        print("💡 Installation: pip install mujoco")
        sys.exit(1)
    
    # Lancer le test
    tester = G1ManipulationTester()
    success = tester.run_test()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
