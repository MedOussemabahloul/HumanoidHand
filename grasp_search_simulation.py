#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grasp Search Simulation for G1 Robot
Simulation de grasping avec recherche active du cube par mouvement des mains
Le robot cherche le cube avec ses mains, puis fait le grasping une fois trouvé
"""

import numpy as np
import mujoco
import time
import os

class GraspSearchSimulation:
    def __init__(self, model_path="results/g1_test_simple.xml"):
        # Charger le modèle MuJoCo
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # IDs des éléments importants
        self.cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        
        # Analyser les sensors et actuators disponibles
        self._analyze_sensors()
        self._analyze_actuators()
        
        # États de la simulation
        self.search_phase = True      # Phase de recherche du cube
        self.contact_detected = False # Contact détecté avec le cube
        self.grasp_completed = False  # Grasping terminé
        self.grasp_stable = False     # Grasping stable
        self.step_count = 0
        self.max_steps = 3000
        
        # Paramètres de recherche et grasping
        self.contact_threshold = 0.05  # Seuil pour détecter le contact
        self.search_radius = 0.2       # Rayon de recherche autour du cube
        self.search_speed = 0.1        # Vitesse de mouvement de recherche
        self.grasp_force = 1.5         # Force de fermeture des doigts
        self.open_position = 0.0       # Position ouverte des doigts
        self.closed_position = 1.5     # Position fermée des doigts
        
        # Paramètres de mouvement des bras
        self.search_patterns = [
            # Mouvements de recherche circulaires
            {"left_shoulder": [0.3, 0.2, 0.0], "right_shoulder": [0.3, -0.2, 0.0]},
            {"left_shoulder": [0.4, 0.1, 0.0], "right_shoulder": [0.4, -0.1, 0.0]},
            {"left_shoulder": [0.3, 0.0, 0.0], "right_shoulder": [0.3, 0.0, 0.0]},
            {"left_shoulder": [0.2, 0.1, 0.0], "right_shoulder": [0.2, -0.1, 0.0]},
            {"left_shoulder": [0.25, 0.15, 0.0], "right_shoulder": [0.25, -0.15, 0.0]},
        ]
        self.current_pattern = 0
        self.pattern_steps = 0
        self.pattern_duration = 100  # Steps par pattern de mouvement
        
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
        
        # Identifier les actuators des doigts et des bras
        self.finger_actuator_ids = []
        self.arm_actuator_ids = []
        
        # Actuators pour le mouvement des bras (recherche)
        self.arm_movement_actuators = {
            'left_shoulder_pitch': None,
            'left_elbow': None,
            'left_wrist': None,
            'right_shoulder_pitch': None,
            'right_elbow': None,
            'right_wrist': None,
        }
        
        for i, name in enumerate(actuator_names):
            if name and any(finger in name for finger in ["index", "middle", "ring", "thumb"]):
                self.finger_actuator_ids.append(i)
                print(f"Finger actuator trouvé: {name}")
            elif name and any(arm in name for arm in ["shoulder", "elbow", "wrist"]):
                self.arm_actuator_ids.append(i)
                print(f"Arm actuator trouvé: {name}")
                
                # Identifier les actuators spécifiques pour le mouvement
                if "left_shoulder_pitch" in name:
                    self.arm_movement_actuators['left_shoulder_pitch'] = i
                elif "left_elbow" in name:
                    self.arm_movement_actuators['left_elbow'] = i
                elif "left_wrist" in name:
                    self.arm_movement_actuators['left_wrist'] = i
                elif "right_shoulder_pitch" in name:
                    self.arm_movement_actuators['right_shoulder_pitch'] = i
                elif "right_elbow" in name:
                    self.arm_movement_actuators['right_elbow'] = i
                elif "right_wrist" in name:
                    self.arm_movement_actuators['right_wrist'] = i
        
        print(f"Finger actuators: {len(self.finger_actuator_ids)}")
        print(f"Arm actuators: {len(self.arm_actuator_ids)}")
        
        # Afficher les actuators de mouvement
        print("\n=== Actuators de mouvement des bras ===")
        for joint, actuator_id in self.arm_movement_actuators.items():
            if actuator_id is not None:
                print(f"{joint}: Actuator {actuator_id}")
            else:
                print(f"{joint}: Non trouvé")
        
    def reset(self):
        """Reset de la simulation"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # Reset des états
        self.search_phase = True
        self.contact_detected = False
        self.grasp_completed = False
        self.grasp_stable = False
        self.step_count = 0
        self.current_pattern = 0
        self.pattern_steps = 0
        self.cube_position_history = []
        
        # Position initiale des doigts (ouverts)
        self._set_finger_positions(self.open_position)
        
        # Position initiale des bras (position de recherche)
        self._set_initial_search_position()
        
    def _set_finger_positions(self, position):
        """Définir la position des doigts"""
        for actuator_id in self.finger_actuator_ids:
            self.data.ctrl[actuator_id] = position
    
    def _set_initial_search_position(self):
        """Définir la position initiale des bras pour la recherche"""
        # Position de départ pour la recherche
        initial_positions = {
            'left_shoulder_pitch': 0.3,   # Bras gauche légèrement levé
            'left_elbow': 0.0,            # Coude gauche droit
            'left_wrist': 0.0,            # Poignet gauche neutre
            'right_shoulder_pitch': 0.3,  # Bras droit légèrement levé
            'right_elbow': 0.0,           # Coude droit droit
            'right_wrist': 0.0,           # Poignet droit neutre
        }
        
        # Appliquer les positions aux actuators des bras
        for joint, position in initial_positions.items():
            actuator_id = self.arm_movement_actuators.get(joint)
            if actuator_id is not None:
                self.data.ctrl[actuator_id] = position
    
    def _move_arms_for_search(self):
        """Mouvoir les bras pour rechercher le cube"""
        if self.pattern_steps >= self.pattern_duration:
            # Changer de pattern de mouvement
            self.current_pattern = (self.current_pattern + 1) % len(self.search_patterns)
            self.pattern_steps = 0
        
        # Obtenir le pattern actuel
        pattern = self.search_patterns[self.current_pattern]
        
        # Appliquer les mouvements aux bras
        for joint, target_pos in pattern.items():
            if joint == "left_shoulder":
                actuator_id = self.arm_movement_actuators.get('left_shoulder_pitch')
                if actuator_id is not None:
                    current_pos = self.data.ctrl[actuator_id]
                    # Mouvement progressif vers la cible
                    new_pos = current_pos + self.search_speed * (target_pos[0] - current_pos)
                    self.data.ctrl[actuator_id] = new_pos
                    
            elif joint == "right_shoulder":
                actuator_id = self.arm_movement_actuators.get('right_shoulder_pitch')
                if actuator_id is not None:
                    current_pos = self.data.ctrl[actuator_id]
                    new_pos = current_pos + self.search_speed * (target_pos[0] - current_pos)
                    self.data.ctrl[actuator_id] = new_pos
        
        self.pattern_steps += 1
    
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
        """Calculer la récompense basée sur la recherche et le grasping"""
        reward = 0.0
        
        # Récompenses pour la phase de recherche
        if self.search_phase:
            # Récompense pour être proche du cube
            cube_pos = self.data.xpos[self.cube_id]
            left_hand_pos = self._get_hand_position('left')
            right_hand_pos = self._get_hand_position('right')
            
            if left_hand_pos is not None and right_hand_pos is not None:
                left_distance = np.linalg.norm(left_hand_pos - cube_pos)
                right_distance = np.linalg.norm(right_hand_pos - cube_pos)
                min_distance = min(left_distance, right_distance)
                
                # Récompense inversement proportionnelle à la distance
                if min_distance < self.search_radius:
                    reward += 5.0 * (1.0 - min_distance / self.search_radius)
                
                # Pénalité pour être trop loin
                if min_distance > 0.3:
                    reward -= 1.0
            
            # Récompense pour le mouvement (encourager l'exploration)
            reward += 0.1
        
        # Récompense pour le contact
        if self.contact_detected:
            reward += 10.0
            if self.search_phase:
                print(f"Contact détecté à l'étape {self.step_count} - Fin de la phase de recherche!")
                self.search_phase = False
        
        # Récompenses pour le grasping
        if not self.search_phase and self.grasp_completed:
            reward += 20.0
            
            # Récompense basée sur la stabilité du cube
            if self.grasp_stable:
                reward += 30.0
            
            # Récompense basée sur la hauteur du cube
            cube_pos = self.data.xpos[self.cube_id]
            cube_height = cube_pos[2]
            reward += cube_height * 10.0  # Plus le cube est haut, mieux c'est
        
        # Pénalités
        # Pénalité pour les mouvements excessifs du cube
        if self.step_count > 200:
            cube_pos = self.data.xpos[self.cube_id]
            initial_pos = np.array([0.3, 0, 0.05])  # Position initiale du cube
            distance = np.linalg.norm(cube_pos - initial_pos)
            if distance > 0.15:  # Si le cube s'éloigne trop
                reward -= distance * 20.0
        
        # Pénalité pour le temps (encourager l'efficacité)
        if self.step_count > 1000:
            reward -= 0.2  # Pénalité temporelle
        
        return reward
    
    def _get_hand_position(self, hand_side):
        """Obtenir la position de la main (gauche ou droite)"""
        try:
            if hand_side == 'left':
                # Utiliser la position du poignet gauche
                wrist_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand")
                if wrist_body_id >= 0:
                    return self.data.xpos[wrist_body_id].copy()
            elif hand_side == 'right':
                # Utiliser la position du poignet droit
                wrist_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand")
                if wrist_body_id >= 0:
                    return self.data.xpos[wrist_body_id].copy()
        except:
            pass
        return None
    
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
        
        # Logique de recherche et grasping
        if self.search_phase:
            # Phase de recherche : mouvoir les bras pour chercher le cube
            self._move_arms_for_search()
            
            # Détecter le contact
            if not self.contact_detected and contact:
                self.contact_detected = True
                print(f"Contact détecté à l'étape {self.step_count} - Début du grasping!")
        
        else:
            # Phase de grasping : fermer les doigts
            if not self.grasp_completed:
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
            'search_phase': self.search_phase,
            'cube_position': self.data.xpos[self.cube_id].copy(),
            'cube_velocity': self.data.qvel[self.cube_id] if self.cube_id < len(self.data.qvel) else 0,
            'contact_detected': self.contact_detected,
            'grasp_completed': self.grasp_completed,
            'grasp_stable': self.grasp_stable,
            'step_count': self.step_count,
            'current_pattern': self.current_pattern,
            'finger_positions': [self.data.ctrl[i] for i in self.finger_actuator_ids],
            'arm_positions': [self.data.ctrl[i] for i in self.arm_actuator_ids],
            'sensor_values': [self.data.sensordata[i] for i in self.force_sensor_ids if i < len(self.data.sensordata)]
        }
        return obs
    
    def run_simulation(self):
        """Exécuter la simulation complète"""
        print("Démarrage de la simulation de recherche et grasping...")
        print("Phase 1: Recherche du cube avec mouvement des bras")
        print("Phase 2: Grasping une fois le contact détecté")
        
        total_reward = 0.0
        
        try:
            while True:
                # Step de simulation
                reward, done = self.step()
                total_reward += reward
                
                # Afficher les informations
                if self.step_count % 100 == 0:
                    obs = self.get_observation()
                    phase = "RECHERCHE" if obs['search_phase'] else "GRASPING"
                    print(f"Step {self.step_count} ({phase}): Reward={reward:.2f}, Total={total_reward:.2f}")
                    print(f"  Cube pos: {obs['cube_position']}")
                    print(f"  Contact: {obs['contact_detected']}, Grasp: {obs['grasp_completed']}, Stable: {obs['grasp_stable']}")
                    print(f"  Pattern: {obs['current_pattern']}")
                
                # Vérifier si terminé
                if done:
                    break
                
                # Petit délai pour ralentir la simulation
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("Simulation interrompue par l'utilisateur")
        
        # Résultats finaux
        print(f"\n=== Résultats de la simulation ===")
        print(f"Steps totaux: {self.step_count}")
        print(f"Récompense totale: {total_reward:.2f}")
        print(f"Phase de recherche: {'Terminée' if not self.search_phase else 'En cours'}")
        print(f"Contact détecté: {self.contact_detected}")
        print(f"Grasping réussi: {self.grasp_completed}")
        print(f"Grasping stable: {self.grasp_stable}")
        
        obs = self.get_observation()
        print(f"Position finale du cube: {obs['cube_position']}")
        
        return {
            'total_reward': total_reward,
            'steps': self.step_count,
            'search_phase_completed': not self.search_phase,
            'contact_detected': self.contact_detected,
            'grasp_completed': self.grasp_completed,
            'grasp_stable': self.grasp_stable,
            'final_cube_position': obs['cube_position']
        }

def main():
    """Fonction principale"""
    print("=== Simulation de Recherche et Grasping G1 ===")
    
    # Créer la simulation
    sim = GraspSearchSimulation()
    
    # Exécuter la simulation
    results = sim.run_simulation()
    
    print("\nSimulation terminée!")
    return results

if __name__ == "__main__":
    main()