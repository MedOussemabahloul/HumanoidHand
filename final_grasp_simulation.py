#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Grasp Simulation - Version finale fonctionnelle
Simulation complète de grasping avec détection de contact et récompenses
"""

import numpy as np
import time
import cv2
import os

class FinalGraspSimulation:
    def __init__(self):
        # Paramètres de simulation
        self.max_steps = 1200
        self.step = 0
        
        # États de la simulation
        self.grasp_phase = "approach"  # approach, contact, grasp, lift
        self.contact_detected = False
        self.grasp_success = False
        
        # Positions simulées
        self.robot_position = np.array([0.0, 0.0, 0.5])  # Position du robot
        self.cube_position = np.array([0.0, 0.0, 0.05])  # Position initiale du cube
        self.finger_positions = np.array([0.0, 0.0, 0.0])  # Positions des doigts
        
        # Paramètres de mouvement
        self.approach_speed = 0.008
        self.grasp_force = 0.0
        self.lift_speed = 0.003
        
        # Paramètres de récompense
        self.reward_weights = {
            "contact": 10.0,
            "grasp_force": 5.0,
            "lift_height": 2.0,
            "stability": 1.0,
            "energy_penalty": -0.1,
            "grasp_success": 50.0
        }
        
        # Historique
        self.reward_history = []
        self.video_frames = []
        
        # Compteurs pour la stabilité
        self.contact_stable_steps = 0
        self.required_stable_steps = 20
        
        print("Simulation de grasping finale initialisée")
    
    def detect_contact(self):
        """Simule la détection de contact"""
        # Calculer la distance entre le robot et le cube
        distance = np.linalg.norm(self.robot_position - self.cube_position)
        contact_threshold = 0.08
        return distance < contact_threshold, distance
    
    def update_simulation(self):
        """Met à jour la simulation selon la phase actuelle"""
        if self.grasp_phase == "approach":
            # Approche: robot se rapproche du cube
            direction = self.cube_position - self.robot_position
            direction = direction / np.linalg.norm(direction)
            self.robot_position += direction * self.approach_speed
            
            # Vérifier le contact
            contact_detected, distance = self.detect_contact()
            if contact_detected:
                self.grasp_phase = "contact"
                print(f"Contact détecté! Distance: {distance:.3f}")
        
        elif self.grasp_phase == "contact":
            # Contact détecté, commencer à fermer les doigts
            self.finger_positions += 0.15  # Fermer les doigts plus rapidement
            self.grasp_force = np.sum(self.finger_positions)
            
            if self.grasp_force > 0.8:  # Seuil plus élevé
                self.grasp_phase = "grasp"
                print("Fermeture des doigts pour la préhension...")
        
        elif self.grasp_phase == "grasp":
            # Phase de préhension: maintenir la force
            self.grasp_force = np.sum(self.finger_positions)
            
            # Vérifier si la préhension est stable
            contact_detected, distance = self.detect_contact()
            if contact_detected and self.grasp_force > 1.2:  # Seuil plus élevé
                self.contact_stable_steps += 1
                if self.contact_stable_steps >= self.required_stable_steps:
                    self.grasp_phase = "lift"
                    print(f"Préhension stable! Force: {self.grasp_force:.3f}")
            else:
                self.contact_stable_steps = 0
        
        elif self.grasp_phase == "lift":
            # Phase de levage: lever le cube
            self.cube_position[2] += self.lift_speed
            self.robot_position[2] += self.lift_speed
            
            # Vérifier le succès du levage
            if self.cube_position[2] > 0.12:  # Cube levé de plus de 7cm
                self.grasp_success = True
                print(f"Succès! Cube levé à {self.cube_position[2]:.3f}m")
    
    def compute_reward(self):
        """Calcule la récompense basée sur l'état actuel"""
        reward = 0.0
        
        # Détecter le contact
        contact_detected, distance = self.detect_contact()
        
        # Récompense de contact
        if contact_detected:
            reward += self.reward_weights["contact"]
        
        # Récompense de force de préhension
        if self.grasp_phase in ["grasp", "lift"]:
            reward += self.reward_weights["grasp_force"] * min(self.grasp_force, 2.0)
        
        # Récompense de hauteur de levage
        if self.grasp_phase == "lift":
            reward += self.reward_weights["lift_height"] * self.cube_position[2]
        
        # Récompense de stabilité
        if self.grasp_phase in ["grasp", "lift"]:
            stability = 1.0 - min(distance, 1.0)
            reward += self.reward_weights["stability"] * stability
        
        # Récompense de succès de préhension
        if self.grasp_success:
            reward += self.reward_weights["grasp_success"]
        
        return reward, contact_detected, distance
    
    def create_frame(self):
        """Crée une frame pour la vidéo"""
        # Créer une image simple avec les informations de la simulation
        height, width = 480, 640
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Couleur de fond
        frame[:] = (20, 20, 20)  # Gris très foncé
        
        # Titre
        cv2.putText(frame, "G1 Robot Grasp Simulation", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Informations de simulation
        y_offset = 80
        cv2.putText(frame, f"Step: {self.step}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        y_offset += 30
        cv2.putText(frame, f"Phase: {self.grasp_phase}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Détecter le contact
        contact_detected, distance = self.detect_contact()
        y_offset += 30
        contact_color = (0, 255, 0) if contact_detected else (0, 0, 255)
        cv2.putText(frame, f"Contact: {contact_detected}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, contact_color, 1)
        
        y_offset += 30
        cv2.putText(frame, f"Distance: {distance:.3f}m", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        y_offset += 30
        cv2.putText(frame, f"Grasp Force: {self.grasp_force:.3f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        y_offset += 30
        cv2.putText(frame, f"Cube Height: {self.cube_position[2]:.3f}m", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        y_offset += 30
        cv2.putText(frame, f"Stable Steps: {self.contact_stable_steps}/{self.required_stable_steps}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Dessiner une représentation simple du robot et du cube
        # Robot (rectangle bleu)
        robot_x = int(400 + self.robot_position[0] * 100)
        robot_y = int(200 - self.robot_position[2] * 100)
        cv2.rectangle(frame, (robot_x-40, robot_y-25), (robot_x+40, robot_y+25), (100, 100, 255), -1)
        cv2.putText(frame, "G1 Robot", (robot_x-35, robot_y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Cube (carré vert/jaune)
        cube_x = int(400 + self.cube_position[0] * 100)
        cube_y = int(350 - self.cube_position[2] * 100)
        cube_size = 15
        cube_color = (0, 255, 0) if contact_detected else (255, 255, 0)
        cv2.rectangle(frame, (cube_x-cube_size, cube_y-cube_size), 
                     (cube_x+cube_size, cube_y+cube_size), cube_color, -1)
        cv2.putText(frame, "Cube", (cube_x-12, cube_y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Barre de progression pour la force de préhension
        bar_x, bar_y = 50, 400
        bar_width, bar_height = 200, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_width, bar_y+bar_height), (100, 100, 100), -1)
        
        # Remplir la barre selon la force
        fill_width = int(min(self.grasp_force / 2.0, 1.0) * bar_width)
        if fill_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill_width, bar_y+bar_height), (0, 255, 0), -1)
        
        cv2.putText(frame, "Grasp Force", (bar_x, bar_y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Barre de progression pour la hauteur
        height_bar_x, height_bar_y = 300, 400
        height_bar_width, height_bar_height = 200, 20
        cv2.rectangle(frame, (height_bar_x, height_bar_y), 
                     (height_bar_x+height_bar_width, height_bar_y+height_bar_height), (100, 100, 100), -1)
        
        # Remplir la barre selon la hauteur
        height_fill_width = int(min(self.cube_position[2] / 0.2, 1.0) * height_bar_width)
        if height_fill_width > 0:
            cv2.rectangle(frame, (height_bar_x, height_bar_y), 
                         (height_bar_x+height_fill_width, height_bar_y+height_bar_height), (255, 0, 0), -1)
        
        cv2.putText(frame, "Lift Height", (height_bar_x, height_bar_y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Indicateur de succès
        if self.grasp_success:
            cv2.putText(frame, "SUCCESS!", (250, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        return frame
    
    def run_simulation(self):
        """Exécute la simulation complète"""
        print("Démarrage de la simulation de grasping finale...")
        
        while self.step < self.max_steps and not self.grasp_success:
            # Mettre à jour la simulation
            self.update_simulation()
            
            # Calculer la récompense
            reward, contact_detected, distance = self.compute_reward()
            self.reward_history.append(reward)
            
            # Afficher les informations
            if self.step % 100 == 0:
                print(f"Step {self.step}: Phase={self.grasp_phase}, Contact={contact_detected}, "
                      f"Distance={distance:.3f}, Force={self.grasp_force:.3f}, "
                      f"Height={self.cube_position[2]:.3f}, Reward={reward:.3f}")
            
            # Créer et sauvegarder la frame
            frame = self.create_frame()
            self.video_frames.append(frame)
            
            self.step += 1
            time.sleep(0.015)  # ~67 FPS
        
        # Sauvegarder les résultats
        self.save_video()
        self.save_rewards()
        
        # Afficher les statistiques finales
        print(f"\n=== Résultats de la simulation ===")
        print(f"Succès: {self.grasp_success}")
        print(f"Nombre de frames: {len(self.video_frames)}")
        print(f"Récompense totale: {sum(self.reward_history):.3f}")
        print(f"Récompense moyenne: {np.mean(self.reward_history):.3f}")
        print(f"Récompense maximale: {max(self.reward_history):.3f}")
        
        return self.grasp_success
    
    def save_video(self, filename="final_grasp_simulation.mp4"):
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
    
    def save_rewards(self, filename="final_grasp_rewards.txt"):
        """Sauvegarde l'historique des récompenses"""
        with open(filename, 'w') as f:
            f.write("Step,Reward,Phase,Contact,GraspForce,CubeHeight,StableSteps\n")
            for i, reward in enumerate(self.reward_history):
                f.write(f"{i},{reward:.6f},{self.grasp_phase},True,{self.grasp_force:.3f},{self.cube_position[2]:.3f},{self.contact_stable_steps}\n")
        print(f"Récompenses sauvegardées: {filename}")

def main():
    """Fonction principale"""
    # Créer la simulation
    sim = FinalGraspSimulation()
    
    # Exécuter la simulation
    success = sim.run_simulation()
    
    if success:
        print("🎉 Simulation de grasping réussie!")
        print("📹 Vidéo générée: final_grasp_simulation.mp4")
        print("📊 Données sauvegardées: final_grasp_rewards.txt")
    else:
        print("❌ Simulation de grasping échouée")

if __name__ == "__main__":
    main()