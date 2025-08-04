#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Script for G1 Grasp Simulation
Script principal pour lancer les simulations de grasping du robot G1
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Simulation de grasping pour le robot G1")
    parser.add_argument("--model", type=str, default="results/g1_combined.xml",
                       help="Chemin vers le modèle MuJoCo")
    parser.add_argument("--enhanced", action="store_true",
                       help="Utiliser le modèle avec force sensors améliorés")
    parser.add_argument("--advanced", action="store_true",
                       help="Utiliser la simulation avancée")
    parser.add_argument("--no-video", action="store_true",
                       help="Ne pas sauvegarder de vidéo")
    parser.add_argument("--video-path", type=str, default="grasp_simulation.mp4",
                       help="Chemin de sortie pour la vidéo")
    parser.add_argument("--steps", type=int, default=2000,
                       help="Nombre maximum d'étapes")
    parser.add_argument("--contact-threshold", type=float, default=0.05,
                       help="Seuil de détection de contact")
    
    args = parser.parse_args()
    
    print("=== Simulation de Grasping G1 ===")
    print(f"Modèle: {args.model}")
    print(f"Simulation avancée: {args.advanced}")
    print(f"Sauvegarder vidéo: {not args.no_video}")
    print(f"Chemin vidéo: {args.video_path}")
    print(f"Steps max: {args.steps}")
    print(f"Seuil contact: {args.contact_threshold}")
    
    # Vérifier que le modèle existe
    if not os.path.exists(args.model):
        print(f"Erreur: Le modèle {args.model} n'existe pas!")
        return 1
    
    # Créer le modèle amélioré si demandé
    if args.enhanced:
        print("\nCréation du modèle amélioré avec force sensors...")
        try:
            from add_force_sensors import create_enhanced_grasp_model
            enhanced_model = create_enhanced_grasp_model()
            args.model = enhanced_model
            print(f"Modèle amélioré créé: {enhanced_model}")
        except Exception as e:
            print(f"Erreur lors de la création du modèle amélioré: {e}")
            return 1
    
    # Lancer la simulation appropriée
    try:
        if args.advanced:
            print("\nLancement de la simulation avancée...")
            from grasp_simulation_advanced import AdvancedGraspSimulation
            
            # Créer la simulation
            sim = AdvancedGraspSimulation(model_path=args.model)
            
            # Ajuster les paramètres
            sim.max_steps = args.steps
            sim.contact_threshold = args.contact_threshold
            
            # Lancer la simulation
            results = sim.run_simulation(
                save_video=not args.no_video,
                video_path=args.video_path
            )
        else:
            print("\nLancement de la simulation basique...")
            from grasp_simulation import GraspSimulation
            
            # Créer la simulation
            sim = GraspSimulation(model_path=args.model)
            
            # Ajuster les paramètres
            sim.max_steps = args.steps
            sim.contact_threshold = args.contact_threshold
            
            # Lancer la simulation
            results = sim.run_simulation(
                save_video=not args.no_video,
                video_path=args.video_path
            )
        
        # Afficher les résultats
        print("\n=== Résultats finaux ===")
        print(f"Récompense totale: {results['total_reward']:.2f}")
        print(f"Steps: {results['steps']}")
        print(f"Contact détecté: {results['contact_detected']}")
        print(f"Grasping réussi: {results['grasp_completed']}")
        if 'grasp_stable' in results:
            print(f"Grasping stable: {results['grasp_stable']}")
        print(f"Position finale du cube: {results['final_cube_position']}")
        
        return 0
        
    except ImportError as e:
        print(f"Erreur d'import: {e}")
        print("Assurez-vous que tous les modules sont installés:")
        print("pip install mujoco mujoco-viewer opencv-python numpy")
        return 1
    except Exception as e:
        print(f"Erreur lors de la simulation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())