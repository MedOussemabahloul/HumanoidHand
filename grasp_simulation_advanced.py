#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Grasp Simulation for G1 Robot
Simulation vidéo avancée de la phase de grasping avec détection de contact via force sensors
"""

import numpy as np
import mujoco
import mujoco_viewer
import time
import cv2
from pathlib import Path
import os

class AdvancedGraspSimulation:
    def __init__(self, model_path="results/g1_combined.xml"):
        # Charger le modèle MuJoCo
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # IDs des éléments importants
        self.cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        
        # Analyser les sensors disponibles
        self._analyze_sensors()
        
        # Analyser les actuators disponibles
        self._analyze_actuators()
        
        # États de la simulation
        self.contact_detected = False
        self.grasp_completed = False
        self.grasp_stable = False
        self.step_count = 0
        self.max_steps = 3000
        
        # Paramètres de grasping
        self.contact_threshold = 0.05  # Seuil pour détecter le contact
        self.grasp_force = 0.8  # Force de fermeture des doigts
        self.open_position = 0.0  # Position ouverte des doigts
        self.closed_position = 1.2  # Position fermée des doigts
        
        # Historique pour la stabilité
        self.cube_position_history = []
        self.max_history_length = 50
        
        # Initialisation
        self.reset()
        
    def _analyze_sensors(self):
        """Analyser les sensors disponibles dans le modèle"""
        print("=== Analyse des sensors ===")
        
        # Lister tous les sensors
        sensor_names = []
        for i in range(self.model.nsensor):
            sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            sensor_names.append(sensor_name)
            print(f"Sensor {i}: {sensor_name}")
        
        # Identifier les sensors de force et de contact
        self.force_sensor_ids = []
        self.contact_sensor_ids = []
        self.joint_sensor_ids = []
        
        for i, name in enumerate(sensor_names):
            if name and "force" in name.lower():
                self.force_sensor_ids.append(i)
                print(f"Force sensor trouvé: {name}")
            elif name and "contact" in name.lower():
                self.contact_sensor_ids.append(i)
                print(f"Contact sensor trouvé: {name}")
            elif name and ("pos_" in name or "vel_" in name):
                self.joint_sensor_ids.append(i)
        
        # Si pas de force sensors, utiliser les sensors de position des doigts
        if not self.force_sensor_ids:
            print("Aucun force sensor trouvé, utilisation des sensors de position des doigts")
            finger_sensors = [i for i, name in enumerate(sensor_names) 
                            if name and any(finger in name for finger in 
                                          ["index", "middle", "ring", "thumb"])]
            self.force_sensor_ids = finger_sensors
        
        print(f"Force sensors: {len(self.force_sensor_ids)}")
        print(f"Contact sensors: {len(self.contact_sensor_ids)}")
        print(f"Joint sensors: {len(self.joint_sensor_ids)}")
    
    def _analyze_actuators(self):
        """Analyser les actuators disponibles dans le modèle"""
        print("=== Analyse des actuators ===")
        
        # Lister tous les actuators
        actuator_names = []
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            actuator_names.append(actuator_name)
            print(f"Actuator {i}: {actuator_name}")
        
        # Identifier les actuators des doigts
        self.finger_actuator_ids = []
        self.arm_actuator_ids = []
        
        for i, name in enumerate(actuator_names):
            if name and any(finger in name for finger in ["index", "middle", "ring", "thumb"]):
                self.finger_actuator_ids.append(i)
                print(f"Finger actuator trouvé: {name}")
            elif name and any(arm in name for arm in ["shoulder", "elbow", "wrist"]):
                self.arm_actuator_ids.append(i)
        
        print(f"Finger actuators: {len(self.finger_actuator_ids)}")
        print(f"Arm actuators: {len(self.arm_actuator_ids)}")
        
    def reset(self):
        """Reset de la simulation"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # Reset des états
        self.contact_detected = False
        self.grasp_completed = False
        self.grasp_stable = False
        self.step_count = 0
        self.cube_position_history = []
        
        # Position initiale des doigts (ouverts)
        self._set_finger_positions(self.open_position)
        
        # Position initiale des bras (position de préhension)
        self._set_arm_positions()
        
    def _set_finger_positions(self, position):
        """Définir la position des doigts"""
        for actuator_id in self.finger_actuator_ids:
            self.data.ctrl[actuator_id] = position
    
    def _set_arm_positions(self):
        """Définir la position des bras pour la préhension"""
        # Position de préhension - bras légèrement ouverts
        arm_positions = {
            'left_shoulder': [0.3, 0.2, 0.0],   # Épaule gauche
            'right_shoulder': [0.3, -0.2, 0.0], # Épaule droite
            'left_elbow': [0.0, 0.0, 0.0],      # Coude gauche
            'right_elbow': [0.0, 0.0, 0.0],     # Coude droit
            'left_wrist': [0.0, 0.0, 0.0],      # Poignet gauche
            'right_wrist': [0.0, 0.0, 0.0]      # Poignet droit
        }
        
        # Appliquer les positions aux actuators des bras
        for i, actuator_id in enumerate(self.arm_actuator_ids):
            if i < len(arm_positions):
                self.data.ctrl[actuator_id] = arm_positions[list(arm_positions.keys())[i]][0]
    
    def _detect_contact(self):
        """Détecter le contact via les sensors"""
        if not self.force_sensor_ids and not self.contact_sensor_ids:
            return False
            
        # Lire les valeurs des force sensors
        force_values = []
        for sensor_id in self.force_sensor_ids:
            if sensor_id < len(self.data.sensordata):
                force_values.append(abs(self.data.sensordata[sensor_id]))
        
        # Lire les valeurs des contact sensors
        contact_values = []
        for sensor_id in self.contact_sensor_ids:
            if sensor_id < len(self.data.sensordata):
                contact_values.append(self.data.sensordata[sensor_id])
        
        # Détecter le contact
        contact_detected = False
        
        # Vérifier les force sensors
        if force_values:
            max_force = max(force_values)
            if max_force > self.contact_threshold:
                contact_detected = True
        
        # Vérifier les contact sensors
        if contact_values:
            if any(val > 0 for val in contact_values):
                contact_detected = True
        
        return contact_detected
    
    def _check_grasp_stability(self):
        """Vérifier la stabilité du grasping"""
        if len(self.cube_position_history) < self.max_history_length:
            return False
        
        # Calculer la variance de la position du cube
        recent_positions = np.array(self.cube_position_history[-self.max_history_length:])
        variance = np.var(recent_positions, axis=0)
        
        # Le grasping est stable si la variance est faible
        position_stable = np.all(variance < 0.001)  # Seuil de stabilité
        
        # Vérifier aussi la vitesse du cube
        if len(self.cube_position_history) >= 2:
            current_pos = np.array(self.cube_position_history[-1])
            previous_pos = np.array(self.cube_position_history[-2])
            velocity = np.linalg.norm(current_pos - previous_pos)
            velocity_stable = velocity < 0.01  # Seuil de vitesse
        else:
            velocity_stable = False
        
        return position_stable and velocity_stable
    
    def _compute_reward(self):
        """Calculer la récompense basée sur le grasping"""
        reward = 0.0
        
        # Récompense pour le contact
        if self.contact_detected:
            reward += 2.0
        
        # Récompense pour le grasping réussi
        if self.grasp_completed:
            reward += 10.0
            
            # Récompense basée sur la stabilité du cube
            if self.grasp_stable:
                reward += 20.0
            
            # Récompense basée sur la hauteur du cube
            cube_pos = self.data.xpos[self.cube_id]
            cube_height = cube_pos[2]
            reward += cube_height * 5.0  # Plus le cube est haut, mieux c'est
        
        # Pénalité pour les mouvements excessifs
        if self.step_count > 200:  # Après 200 steps
            cube_pos = self.data.xpos[self.cube_id]
            initial_pos = np.array([0.3, 0, 0.05])  # Position initiale du cube
            distance = np.linalg.norm(cube_pos - initial_pos)
            if distance > 0.15:  # Si le cube s'éloigne trop
                reward -= distance * 15.0
        
        # Pénalité pour le temps
        if self.step_count > 1000:
            reward -= 0.1  # Pénalité temporelle
        
        return reward
    
    def step(self):
        """Exécuter un step de simulation"""
        self.step_count += 1
        
        # Détecter le contact
        contact = self._detect_contact()
        
        # Mettre à jour l'historique de position du cube
        cube_pos = self.data.xpos[self.cube_id].copy()
        self.cube_position_history.append(cube_pos)
        if len(self.cube_position_history) > self.max_history_length:
            self.cube_position_history.pop(0)
        
        # Logique de grasping
        if not self.contact_detected and contact:
            self.contact_detected = True
            print(f"Contact détecté à l'étape {self.step_count}")
        
        # Fermer les doigts si contact détecté
        if self.contact_detected and not self.grasp_completed:
            # Fermeture progressive des doigts
            current_position = self.data.ctrl[self.finger_actuator_ids[0]] if self.finger_actuator_ids else 0
            target_position = min(current_position + 0.1, self.closed_position)
            self._set_finger_positions(target_position)
            
            # Vérifier si le grasping est complet
            if target_position >= self.closed_position:
                self.grasp_completed = True
                print(f"Grasping complet à l'étape {self.step_count}")
        
        # Vérifier la stabilité du grasping
        if self.grasp_completed and not self.grasp_stable:
            if self._check_grasp_stability():
                self.grasp_stable = True
                print(f"Grasping stable à l'étape {self.step_count}")
        
        # Appliquer les actions
        mujoco.mj_step(self.model, self.data)
        
        # Calculer la récompense
        reward = self._compute_reward()
        
        # Vérifier si terminé
        done = (self.step_count >= self.max_steps) or (self.grasp_stable and self.step_count > 500)
        
        return reward, done
    
    def get_observation(self):
        """Obtenir l'observation actuelle"""
        obs = {
            'cube_position': self.data.xpos[self.cube_id].copy(),
            'cube_velocity': self.data.qvel[self.cube_id] if self.cube_id < len(self.data.qvel) else 0,
            'contact_detected': self.contact_detected,
            'grasp_completed': self.grasp_completed,
            'grasp_stable': self.grasp_stable,
            'step_count': self.step_count,
            'finger_positions': [self.data.ctrl[i] for i in self.finger_actuator_ids],
            'sensor_values': [self.data.sensordata[i] for i in self.force_sensor_ids if i < len(self.data.sensordata)]
        }
        return obs
    
    def run_simulation(self, save_video=True, video_path="advanced_grasp_simulation.mp4"):
        """Exécuter la simulation complète avec enregistrement vidéo"""
        print("Démarrage de la simulation avancée de grasping...")
        
        # Initialiser le viewer
        viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        
        # Paramètres vidéo
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = None
            frame_count = 0
        
        total_reward = 0.0
        
        try:
            while True:
                # Step de simulation
                reward, done = self.step()
                total_reward += reward
                
                # Mettre à jour le viewer
                viewer.render()
                
                # Capturer la frame pour la vidéo
                if save_video:
                    # Capturer l'image du viewer
                    frame = viewer.read_pixels()
                    if frame is not None:
                        if out is None:
                            height, width = frame.shape[:2]
                            out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
                        
                        # Convertir RGB vers BGR pour OpenCV
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        out.write(frame_bgr)
                        frame_count += 1
                
                # Afficher les informations
                if self.step_count % 100 == 0:
                    obs = self.get_observation()
                    print(f"Step {self.step_count}: Reward={reward:.2f}, Total={total_reward:.2f}")
                    print(f"  Cube pos: {obs['cube_position']}")
                    print(f"  Contact: {obs['contact_detected']}, Grasp: {obs['grasp_completed']}, Stable: {obs['grasp_stable']}")
                
                # Vérifier si terminé
                if done:
                    break
                
                # Petit délai pour ralentir la simulation
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("Simulation interrompue par l'utilisateur")
        
        finally:
            # Nettoyer
            viewer.close()
            if save_video and out is not None:
                out.release()
                print(f"Vidéo sauvegardée: {video_path}")
                print(f"Nombre de frames: {frame_count}")
        
        # Résultats finaux
        print(f"\n=== Résultats de la simulation avancée ===")
        print(f"Steps totaux: {self.step_count}")
        print(f"Récompense totale: {total_reward:.2f}")
        print(f"Contact détecté: {self.contact_detected}")
        print(f"Grasping réussi: {self.grasp_completed}")
        print(f"Grasping stable: {self.grasp_stable}")
        
        obs = self.get_observation()
        print(f"Position finale du cube: {obs['cube_position']}")
        
        return {
            'total_reward': total_reward,
            'steps': self.step_count,
            'contact_detected': self.contact_detected,
            'grasp_completed': self.grasp_completed,
            'grasp_stable': self.grasp_stable,
            'final_cube_position': obs['cube_position']
        }

def main():
    """Fonction principale"""
    print("=== Simulation Avancée de Grasping G1 ===")
    
    # Créer la simulation
    sim = AdvancedGraspSimulation()
    
    # Exécuter la simulation
    results = sim.run_simulation(save_video=True)
    
    print("\nSimulation terminée!")
    return results

if __name__ == "__main__":
    main()