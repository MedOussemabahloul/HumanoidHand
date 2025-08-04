#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grasp Simulation for G1 Robot
Simulation vidéo de la phase de grasping avec détection de contact via force sensors
"""

import numpy as np
import mujoco
import mujoco_viewer
import time
import cv2
from pathlib import Path
import os

class GraspSimulation:
    def __init__(self, model_path="results/g1_combined.xml"):
        # Charger le modèle MuJoCo
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # IDs des éléments importants
        self.cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        
        # Force sensors IDs (pour détecter le contact)
        self.force_sensor_names = [
            "left_thumb_force_sensor_0", "left_thumb_force_sensor_1", "left_thumb_force_sensor_2",
            "left_index_force_sensor_0", "left_index_force_sensor_1", "left_index_force_sensor_2",
            "left_middle_force_sensor_0", "left_middle_force_sensor_1", "left_middle_force_sensor_2",
            "left_ring_force_sensor_0", "left_ring_force_sensor_1", "left_ring_force_sensor_2",
            "right_thumb_force_sensor_0", "right_thumb_force_sensor_1", "right_thumb_force_sensor_2",
            "right_index_force_sensor_0", "right_index_force_sensor_1", "right_index_force_sensor_2",
            "right_middle_force_sensor_0", "right_middle_force_sensor_1", "right_middle_force_sensor_2",
            "right_ring_force_sensor_0", "right_ring_force_sensor_1", "right_ring_force_sensor_2"
        ]
        
        # Essayer de trouver les force sensors, sinon utiliser des sensors génériques
        self.force_sensor_ids = []
        for name in self.force_sensor_names:
            try:
                sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                self.force_sensor_ids.append(sensor_id)
            except:
                pass
        
        # Si pas de force sensors spécifiques, utiliser les sensors de position des doigts
        if not self.force_sensor_ids:
            print("Force sensors non trouvés, utilisation des sensors de position des doigts")
            finger_joint_names = [
                "left_index_joint_0", "left_index_joint_1",
                "left_middle_joint_0", "left_middle_joint_1", 
                "left_ring_joint_0", "left_ring_joint_1",
                "left_thumb_joint_0", "left_thumb_joint_1",
                "right_index_joint_0", "right_index_joint_1",
                "right_middle_joint_0", "right_middle_joint_1",
                "right_ring_joint_0", "right_ring_joint_1",
                "right_thumb_joint_0", "right_thumb_joint_1"
            ]
            for name in finger_joint_names:
                try:
                    sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"pos_{name}")
                    self.force_sensor_ids.append(sensor_id)
                except:
                    pass
        
        # Actuators pour les doigts
        self.finger_actuator_names = [
            "act_left_index_joint_0", "act_left_index_joint_1",
            "act_left_middle_joint_0", "act_left_middle_joint_1",
            "act_left_ring_joint_0", "act_left_ring_joint_1",
            "act_left_thumb_joint_0", "act_left_thumb_joint_1",
            "act_right_index_joint_0", "act_right_index_joint_1",
            "act_right_middle_joint_0", "act_right_middle_joint_1",
            "act_right_ring_joint_0", "act_right_ring_joint_1",
            "act_right_thumb_joint_0", "act_right_thumb_joint_1"
        ]
        
        self.finger_actuator_ids = []
        for name in self.finger_actuator_names:
            try:
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                self.finger_actuator_ids.append(actuator_id)
            except:
                pass
        
        # États de la simulation
        self.contact_detected = False
        self.grasp_completed = False
        self.step_count = 0
        self.max_steps = 2000
        
        # Paramètres de grasping
        self.contact_threshold = 0.1  # Seuil pour détecter le contact
        self.grasp_force = 0.5  # Force de fermeture des doigts
        self.open_position = 0.0  # Position ouverte des doigts
        self.closed_position = 1.0  # Position fermée des doigts
        
        # Initialisation
        self.reset()
        
    def reset(self):
        """Reset de la simulation"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # Reset des états
        self.contact_detected = False
        self.grasp_completed = False
        self.step_count = 0
        
        # Position initiale des doigts (ouverts)
        self._set_finger_positions(self.open_position)
        
    def _set_finger_positions(self, position):
        """Définir la position des doigts"""
        for actuator_id in self.finger_actuator_ids:
            self.data.ctrl[actuator_id] = position
    
    def _detect_contact(self):
        """Détecter le contact via les force sensors"""
        if not self.force_sensor_ids:
            return False
            
        # Lire les valeurs des sensors
        sensor_values = []
        for sensor_id in self.force_sensor_ids:
            if sensor_id < len(self.data.sensordata):
                sensor_values.append(abs(self.data.sensordata[sensor_id]))
        
        # Détecter le contact si une valeur dépasse le seuil
        if sensor_values:
            max_force = max(sensor_values)
            return max_force > self.contact_threshold
        
        return False
    
    def _compute_reward(self):
        """Calculer la récompense basée sur le grasping"""
        reward = 0.0
        
        # Récompense pour le contact
        if self.contact_detected:
            reward += 1.0
        
        # Récompense pour le grasping réussi
        if self.grasp_completed:
            reward += 5.0
            
            # Récompense basée sur la stabilité du cube
            cube_pos = self.data.xpos[self.cube_id]
            cube_height = cube_pos[2]
            reward += cube_height * 2.0  # Plus le cube est haut, mieux c'est
        
        # Pénalité pour les mouvements excessifs
        if self.step_count > 100:  # Après 100 steps
            cube_pos = self.data.xpos[self.cube_id]
            initial_pos = np.array([0.3, 0, 0.05])  # Position initiale du cube
            distance = np.linalg.norm(cube_pos - initial_pos)
            if distance > 0.1:  # Si le cube s'éloigne trop
                reward -= distance * 10.0
        
        return reward
    
    def step(self):
        """Exécuter un step de simulation"""
        self.step_count += 1
        
        # Détecter le contact
        contact = self._detect_contact()
        
        # Logique de grasping
        if not self.contact_detected and contact:
            self.contact_detected = True
            print(f"Contact détecté à l'étape {self.step_count}")
        
        # Fermer les doigts si contact détecté
        if self.contact_detected and not self.grasp_completed:
            self._set_finger_positions(self.closed_position)
            
            # Vérifier si le grasping est stable
            if self.step_count > 50:  # Attendre un peu pour la stabilité
                cube_vel = self.data.qvel[self.cube_id] if self.cube_id < len(self.data.qvel) else 0
                if abs(cube_vel) < 0.01:  # Cube stable
                    self.grasp_completed = True
                    print(f"Grasping réussi à l'étape {self.step_count}")
        
        # Appliquer les actions
        mujoco.mj_step(self.model, self.data)
        
        # Calculer la récompense
        reward = self._compute_reward()
        
        # Vérifier si terminé
        done = (self.step_count >= self.max_steps) or self.grasp_completed
        
        return reward, done
    
    def get_observation(self):
        """Obtenir l'observation actuelle"""
        obs = {
            'cube_position': self.data.xpos[self.cube_id].copy(),
            'cube_velocity': self.data.qvel[self.cube_id] if self.cube_id < len(self.data.qvel) else 0,
            'contact_detected': self.contact_detected,
            'grasp_completed': self.grasp_completed,
            'step_count': self.step_count,
            'finger_positions': [self.data.qpos[i] for i in range(len(self.finger_actuator_ids)) if i < len(self.data.qpos)]
        }
        return obs
    
    def run_simulation(self, save_video=True, video_path="grasp_simulation.mp4"):
        """Exécuter la simulation complète avec enregistrement vidéo"""
        print("Démarrage de la simulation de grasping...")
        
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
                    print(f"  Contact: {obs['contact_detected']}, Grasp: {obs['grasp_completed']}")
                
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
        print(f"\n=== Résultats de la simulation ===")
        print(f"Steps totaux: {self.step_count}")
        print(f"Récompense totale: {total_reward:.2f}")
        print(f"Contact détecté: {self.contact_detected}")
        print(f"Grasping réussi: {self.grasp_completed}")
        
        obs = self.get_observation()
        print(f"Position finale du cube: {obs['cube_position']}")
        
        return {
            'total_reward': total_reward,
            'steps': self.step_count,
            'contact_detected': self.contact_detected,
            'grasp_completed': self.grasp_completed,
            'final_cube_position': obs['cube_position']
        }

def main():
    """Fonction principale"""
    print("=== Simulation de Grasping G1 ===")
    
    # Créer la simulation
    sim = GraspSimulation()
    
    # Exécuter la simulation
    results = sim.run_simulation(save_video=True)
    
    print("\nSimulation terminée!")
    return results

if __name__ == "__main__":
    main()