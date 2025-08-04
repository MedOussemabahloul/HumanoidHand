#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grasp Simulation for G1 Robot - Version Simplifiée
Simulation vidéo de la phase de grasping avec détection de contact
"""

import numpy as np
import mujoco
import mujoco_viewer
import time
import cv2
from pathlib import Path
import os

class GraspSimulationSimple:
    def __init__(self, model_path="results/g1_combined.xml"):
        # Charger le modèle MuJoCo
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Identifiants des joints des doigts
        self.finger_joint_names = [
            # Main gauche
            "left_thumb_joint_0", "left_thumb_joint_1",
            "left_index_joint_0", "left_index_joint_1",
            "left_middle_joint_0", "left_middle_joint_1",
            "left_ring_joint_0", "left_ring_joint_1",
            # Main droite
            "right_thumb_joint_0", "right_thumb_joint_1",
            "right_index_joint_0", "right_index_joint_1",
            "right_middle_joint_0", "right_middle_joint_1",
            "right_ring_joint_0", "right_ring_joint_1"
        ]
        
        self.finger_joint_ids = []
        for name in self.finger_joint_names:
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.finger_joint_ids.append(joint_id)
            except:
                print(f"Joint {name} non trouvé")
        
        # Identifiants des actuateurs des doigts
        self.finger_actuator_names = [f"act_{name}" for name in self.finger_joint_names]
        self.finger_actuator_ids = []
        for name in self.finger_actuator_names:
            try:
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                self.finger_actuator_ids.append(actuator_id)
            except:
                print(f"Actuateur {name} non trouvé")
        
        # ID du cube
        self.cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        
        # États de la simulation
        self.contact_detected = False
        self.grasp_phase = "approach"  # approach, contact, grasp, lift
        self.grasp_success = False
        
        # Paramètres de récompense
        self.reward_weights = {
            "contact": 10.0,
            "grasp_force": 5.0,
            "lift_height": 2.0,
            "stability": 1.0,
            "energy_penalty": -0.1,
            "grasp_success": 50.0
        }
        
        # Historique des récompenses
        self.reward_history = []
        
        # Configuration de la vidéo
        self.video_frames = []
        self.frame_count = 0
        
        # Position initiale du cube
        self.initial_cube_pos = None
        
        print(f"Simulation initialisée avec {len(self.finger_joint_ids)} joints de doigts")
        print(f"Actuateurs des doigts: {len(self.finger_actuator_ids)}")
    
    def detect_contact(self):
        """Détecte le contact via la distance entre les doigts et le cube"""
        # Obtenir la position du cube
        cube_pos = self.data.xpos[self.cube_id]
        
        # Obtenir les positions des doigts
        finger_positions = []
        for joint_id in self.finger_joint_ids:
            if joint_id < len(self.data.qpos):
                # Obtenir la position du joint dans l'espace mondial
                joint_pos = self.data.xpos[joint_id] if joint_id < len(self.data.xpos) else np.zeros(3)
                finger_positions.append(joint_pos)
        
        # Calculer la distance minimale entre les doigts et le cube
        min_distance = float('inf')
        for finger_pos in finger_positions:
            distance = np.linalg.norm(finger_pos - cube_pos)
            min_distance = min(min_distance, distance)
        
        # Seuil de détection de contact
        contact_threshold = 0.05
        contact_detected = min_distance < contact_threshold
        
        return contact_detected, min_distance
    
    def detect_grasp_force(self):
        """Détecte la force de préhension via la position des doigts"""
        # Calculer la force basée sur la fermeture des doigts
        total_force = 0.0
        for joint_id in self.finger_joint_ids:
            if joint_id < len(self.data.qpos):
                # Plus les doigts sont fermés, plus la force est élevée
                finger_angle = self.data.qpos[joint_id]
                total_force += max(0, finger_angle)  # Force positive seulement
        
        return total_force
    
    def get_finger_positions(self):
        """Obtient les positions actuelles des doigts"""
        positions = []
        for joint_id in self.finger_joint_ids:
            if joint_id < len(self.data.qpos):
                pos = self.data.qpos[joint_id]
                positions.append(pos)
            else:
                positions.append(0.0)
        return positions
    
    def set_finger_targets(self, targets):
        """Définit les positions cibles des doigts"""
        for i, actuator_id in enumerate(self.finger_actuator_ids):
            if i < len(targets) and actuator_id < len(self.data.ctrl):
                self.data.ctrl[actuator_id] = targets[i]
    
    def compute_reward(self, contact_detected, grasp_force, grasp_phase):
        """Calcule la récompense basée sur l'état actuel"""
        reward = 0.0
        
        # Récompense de contact
        if contact_detected:
            reward += self.reward_weights["contact"]
        
        # Récompense de force de préhension
        if grasp_phase in ["grasp", "lift"]:
            reward += self.reward_weights["grasp_force"] * min(grasp_force, 2.0)  # Limiter à 2.0
        
        # Récompense de hauteur de levage
        if grasp_phase == "lift":
            cube_height = self.data.xpos[self.cube_id][2]
            reward += self.reward_weights["lift_height"] * cube_height
        
        # Récompense de stabilité (pénalité pour mouvement du cube)
        if grasp_phase in ["grasp", "lift"]:
            cube_vel = np.linalg.norm(self.data.qvel[:3])  # Vitesse du cube
            reward += self.reward_weights["stability"] * (1.0 - min(cube_vel, 1.0))
        
        # Pénalité énergétique pour les mouvements des doigts
        finger_velocities = [abs(self.data.qvel[joint_id]) for joint_id in self.finger_joint_ids if joint_id < len(self.data.qvel)]
        energy_penalty = sum(finger_velocities) * self.reward_weights["energy_penalty"]
        reward += energy_penalty
        
        # Récompense de succès de préhension
        if self.grasp_success:
            reward += self.reward_weights["grasp_success"]
        
        return reward
    
    def grasp_control_loop(self):
        """Boucle de contrôle pour le grasping"""
        # Positions cibles pour les doigts (ouvert)
        open_positions = [0.0] * len(self.finger_joint_ids)
        
        # Positions cibles pour les doigts (fermé)
        closed_positions = [1.0] * len(self.finger_joint_ids)
        
        # Positions cibles pour les doigts (grasp)
        grasp_positions = [0.6] * len(self.finger_joint_ids)
        
        step = 0
        max_steps = 1500
        contact_stable_steps = 0
        required_stable_steps = 30
        
        # Enregistrer la position initiale du cube
        self.initial_cube_pos = self.data.xpos[self.cube_id].copy()
        
        while step < max_steps:
            # Détecter le contact
            contact_detected, distance = self.detect_contact()
            
            # Détecter la force de préhension
            grasp_force = self.detect_grasp_force()
            
            # Logique de contrôle basée sur la phase
            if self.grasp_phase == "approach":
                # Approche: doigts ouverts
                self.set_finger_targets(open_positions)
                
                if contact_detected:
                    self.grasp_phase = "contact"
                    print(f"Contact détecté! Distance: {distance:.3f}")
            
            elif self.grasp_phase == "contact":
                # Contact détecté, commencer à fermer les doigts
                self.set_finger_targets(grasp_positions)
                self.grasp_phase = "grasp"
                print("Fermeture des doigts pour la préhension...")
            
            elif self.grasp_phase == "grasp":
                # Phase de préhension: maintenir la force
                self.set_finger_targets(grasp_positions)
                
                # Vérifier si la préhension est stable
                if contact_detected and grasp_force > 0.3:
                    contact_stable_steps += 1
                    if contact_stable_steps >= required_stable_steps:
                        self.grasp_phase = "lift"
                        print(f"Préhension stable! Force: {grasp_force:.3f}")
                else:
                    contact_stable_steps = 0
            
            elif self.grasp_phase == "lift":
                # Phase de levage: maintenir la préhension
                self.set_finger_targets(grasp_positions)
                
                # Vérifier le succès du levage
                cube_height = self.data.xpos[self.cube_id][2]
                if cube_height > 0.06:  # Cube levé de plus de 6cm
                    self.grasp_success = True
                    print(f"Succès! Cube levé à {cube_height:.3f}m")
                    break
            
            # Calculer la récompense
            reward = self.compute_reward(contact_detected, grasp_force, self.grasp_phase)
            self.reward_history.append(reward)
            
            # Afficher les informations
            if step % 100 == 0:
                cube_height = self.data.xpos[self.cube_id][2]
                print(f"Step {step}: Phase={self.grasp_phase}, Contact={contact_detected}, "
                      f"Distance={distance:.3f}, Force={grasp_force:.3f}, Height={cube_height:.3f}, Reward={reward:.3f}")
            
            # Capturer la frame pour la vidéo
            self.capture_frame()
            
            # Avancer la simulation
            mujoco.mj_step(self.model, self.data)
            step += 1
            
            # Petite pause pour ralentir la simulation
            time.sleep(0.01)
        
        return self.grasp_success
    
    def capture_frame(self):
        """Capture une frame pour la vidéo"""
        try:
            # Créer un viewer temporaire pour capturer la frame
            viewer = mujoco_viewer.MujocoViewer(self.model, self.data, hide_menus=True)
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -25
            
            # Capturer la frame
            frame = viewer.read_pixels()
            if frame is not None:
                # Convertir de RGB à BGR pour OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.video_frames.append(frame_bgr)
            
            viewer.close()
        except Exception as e:
            print(f"Erreur lors de la capture de frame: {e}")
    
    def save_video(self, filename="grasp_simulation_simple.mp4"):
        """Sauvegarde la simulation en vidéo"""
        if not self.video_frames:
            print("Aucune frame à sauvegarder")
            return
        
        # Paramètres de la vidéo
        height, width = self.video_frames[0].shape[:2]
        fps = 30
        
        # Créer le writer vidéo
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        
        # Écrire les frames
        for frame in self.video_frames:
            out.write(frame)
        
        out.release()
        print(f"Vidéo sauvegardée: {filename}")
    
    def save_rewards(self, filename="grasp_rewards_simple.txt"):
        """Sauvegarde l'historique des récompenses"""
        with open(filename, 'w') as f:
            f.write("Step,Reward,Phase,Contact,GraspForce,CubeHeight\n")
            for i, reward in enumerate(self.reward_history):
                cube_height = self.data.xpos[self.cube_id][2] if i < len(self.reward_history) else 0.0
                f.write(f"{i},{reward:.6f},{self.grasp_phase},True,0.0,{cube_height:.3f}\n")
        print(f"Récompenses sauvegardées: {filename}")
    
    def run_simulation(self):
        """Exécute la simulation complète"""
        print("Démarrage de la simulation de grasping simplifiée...")
        
        # Reset de la simulation
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # Exécuter la boucle de contrôle
        success = self.grasp_control_loop()
        
        # Sauvegarder les résultats
        self.save_video()
        self.save_rewards()
        
        # Afficher les statistiques finales
        print(f"\n=== Résultats de la simulation ===")
        print(f"Succès: {success}")
        print(f"Nombre de frames: {len(self.video_frames)}")
        print(f"Récompense totale: {sum(self.reward_history):.3f}")
        print(f"Récompense moyenne: {np.mean(self.reward_history):.3f}")
        print(f"Récompense maximale: {max(self.reward_history):.3f}")
        
        return success

def main():
    """Fonction principale"""
    # Créer la simulation
    sim = GraspSimulationSimple()
    
    # Exécuter la simulation
    success = sim.run_simulation()
    
    if success:
        print("🎉 Simulation de grasping réussie!")
    else:
        print("❌ Simulation de grasping échouée")

if __name__ == "__main__":
    main()