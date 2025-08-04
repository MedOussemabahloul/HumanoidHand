#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grasp Simulation for G1 Robot - Version Améliorée
Simulation vidéo de la phase de grasping avec détection de contact via force sensors
"""

import numpy as np
import mujoco
import mujoco_viewer
import time
import cv2
from pathlib import Path
import os
import xml.etree.ElementTree as ET

class GraspSimulationImproved:
    def __init__(self, model_path="results/g1_combined.xml"):
        # Charger le modèle MuJoCo
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Ajouter des capteurs de force si nécessaire
        self.add_force_sensors_to_model()
        
        # Recharger le modèle avec les capteurs ajoutés
        self.model = mujoco.MjModel.from_xml_path("results/g1_combined_with_force_sensors.xml")
        self.data = mujoco.MjData(self.model)
        
        # Identifiants des capteurs de force
        self.force_sensor_names = [
            # Main gauche
            "left_thumb_force_sensor", "left_index_force_sensor", 
            "left_middle_force_sensor", "left_ring_force_sensor",
            # Main droite
            "right_thumb_force_sensor", "right_index_force_sensor", 
            "right_middle_force_sensor", "right_ring_force_sensor"
        ]
        
        # Obtenir les IDs des capteurs de force
        self.force_sensor_ids = []
        for name in self.force_sensor_names:
            try:
                sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                self.force_sensor_ids.append(sensor_id)
                print(f"Capteur de force trouvé: {name} (ID: {sensor_id})")
            except:
                print(f"Capteur {name} non trouvé")
        
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
        
        print(f"Simulation initialisée avec {len(self.force_sensor_ids)} capteurs de force")
        print(f"Joints des doigts: {len(self.finger_joint_ids)}")
        print(f"Actuateurs des doigts: {len(self.finger_actuator_ids)}")
    
    def add_force_sensors_to_model(self):
        """Ajoute des capteurs de force au modèle XML"""
        # Lire le fichier XML original
        tree = ET.parse("results/g1_combined.xml")
        root = tree.getroot()
        
        # Trouver la section sensor
        sensor_section = root.find('sensor')
        if sensor_section is None:
            sensor_section = ET.SubElement(root, 'sensor')
        
        # Ajouter des capteurs de force pour chaque doigt
        force_sensors = [
            ("left_thumb_force_sensor", "left_thumb_tip_site"),
            ("left_index_force_sensor", "left_index_tip_site"),
            ("left_middle_force_sensor", "left_middle_tip_site"),
            ("left_ring_force_sensor", "left_ring_tip_site"),
            ("right_thumb_force_sensor", "right_thumb_tip_site"),
            ("right_index_force_sensor", "right_index_tip_site"),
            ("right_middle_force_sensor", "right_middle_tip_site"),
            ("right_ring_force_sensor", "right_ring_tip_site")
        ]
        
        for sensor_name, site_name in force_sensors:
            # Vérifier si le capteur existe déjà
            existing_sensor = sensor_section.find(f"frameforce[@name='{sensor_name}']")
            if existing_sensor is None:
                # Ajouter le capteur de force
                force_sensor = ET.SubElement(sensor_section, 'frameforce')
                force_sensor.set('name', sensor_name)
                force_sensor.set('site', site_name)
        
        # Sauvegarder le modèle modifié
        tree.write("results/g1_combined_with_force_sensors.xml", encoding='utf-8', xml_declaration=True)
        print("Capteurs de force ajoutés au modèle")
    
    def detect_contact(self):
        """Détecte le contact via les capteurs de force"""
        contact_forces = []
        total_force = 0.0
        
        for sensor_id in self.force_sensor_ids:
            if sensor_id < len(self.data.sensordata):
                # Les capteurs frameforce donnent [fx, fy, fz, tx, ty, tz]
                force_data = self.data.sensordata[sensor_id:sensor_id+6]
                force_magnitude = np.linalg.norm(force_data[:3])  # Force linéaire seulement
                contact_forces.append(force_magnitude)
                total_force += force_magnitude
            else:
                contact_forces.append(0.0)
        
        # Seuil de détection de contact
        contact_threshold = 0.05
        
        return total_force > contact_threshold, contact_forces, total_force
    
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
    
    def compute_reward(self, contact_detected, contact_forces, total_force, grasp_phase):
        """Calcule la récompense basée sur l'état actuel"""
        reward = 0.0
        
        # Récompense de contact
        if contact_detected:
            reward += self.reward_weights["contact"]
        
        # Récompense de force de préhension
        if grasp_phase in ["grasp", "lift"]:
            reward += self.reward_weights["grasp_force"] * min(total_force, 2.0)  # Limiter à 2.0
        
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
        grasp_positions = [0.7] * len(self.finger_joint_ids)
        
        step = 0
        max_steps = 2000
        contact_stable_steps = 0
        required_stable_steps = 50
        
        while step < max_steps:
            # Détecter le contact
            contact_detected, contact_forces, total_force = self.detect_contact()
            
            # Logique de contrôle basée sur la phase
            if self.grasp_phase == "approach":
                # Approche: doigts ouverts
                self.set_finger_targets(open_positions)
                
                if contact_detected:
                    self.grasp_phase = "contact"
                    print(f"Contact détecté! Force totale: {total_force:.3f}")
            
            elif self.grasp_phase == "contact":
                # Contact détecté, commencer à fermer les doigts
                self.set_finger_targets(grasp_positions)
                self.grasp_phase = "grasp"
                print("Fermeture des doigts pour la préhension...")
            
            elif self.grasp_phase == "grasp":
                # Phase de préhension: maintenir la force
                self.set_finger_targets(grasp_positions)
                
                # Vérifier si la préhension est stable
                if contact_detected and total_force > 0.5:
                    contact_stable_steps += 1
                    if contact_stable_steps >= required_stable_steps:
                        self.grasp_phase = "lift"
                        print(f"Préhension stable! Force: {total_force:.3f}")
                else:
                    contact_stable_steps = 0
            
            elif self.grasp_phase == "lift":
                # Phase de levage: maintenir la préhension
                self.set_finger_targets(grasp_positions)
                
                # Vérifier le succès du levage
                cube_height = self.data.xpos[self.cube_id][2]
                if cube_height > 0.08:  # Cube levé de plus de 8cm
                    self.grasp_success = True
                    print(f"Succès! Cube levé à {cube_height:.3f}m")
                    break
            
            # Calculer la récompense
            reward = self.compute_reward(contact_detected, contact_forces, total_force, self.grasp_phase)
            self.reward_history.append(reward)
            
            # Afficher les informations
            if step % 100 == 0:
                cube_height = self.data.xpos[self.cube_id][2]
                print(f"Step {step}: Phase={self.grasp_phase}, Contact={contact_detected}, "
                      f"Force={total_force:.3f}, Height={cube_height:.3f}, Reward={reward:.3f}")
            
            # Capturer la frame pour la vidéo
            self.capture_frame()
            
            # Avancer la simulation
            mujoco.mj_step(self.model, self.data)
            step += 1
            
            # Petite pause pour ralentir la simulation
            time.sleep(0.005)
        
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
    
    def save_video(self, filename="grasp_simulation_improved.mp4"):
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
    
    def save_rewards(self, filename="grasp_rewards_improved.txt"):
        """Sauvegarde l'historique des récompenses"""
        with open(filename, 'w') as f:
            f.write("Step,Reward,Phase,Contact,TotalForce\n")
            for i, reward in enumerate(self.reward_history):
                f.write(f"{i},{reward:.6f},{self.grasp_phase},True,0.0\n")
        print(f"Récompenses sauvegardées: {filename}")
    
    def run_simulation(self):
        """Exécute la simulation complète"""
        print("Démarrage de la simulation de grasping améliorée...")
        
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
    sim = GraspSimulationImproved()
    
    # Exécuter la simulation
    success = sim.run_simulation()
    
    if success:
        print("🎉 Simulation de grasping réussie!")
    else:
        print("❌ Simulation de grasping échouée")

if __name__ == "__main__":
    main()