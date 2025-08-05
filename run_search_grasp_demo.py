#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Search and Grasp Demo Runner
Script de démo pour la simulation de recherche et grasping du robot G1
"""

import os
import sys
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="Démo de recherche et grasping pour le robot G1")
    parser.add_argument("--model", type=str, default="results/g1_test_simple.xml",
                       help="Chemin vers le modèle MuJoCo")
    parser.add_argument("--steps", type=int, default=2000,
                       help="Nombre maximum d'étapes")
    parser.add_argument("--contact-threshold", type=float, default=0.05,
                       help="Seuil de détection de contact")
    parser.add_argument("--search-radius", type=float, default=0.2,
                       help="Rayon de recherche autour du cube")
    parser.add_argument("--search-speed", type=float, default=0.1,
                       help="Vitesse de mouvement de recherche")
    parser.add_argument("--grasp-force", type=float, default=1.5,
                       help="Force de fermeture des doigts")
    parser.add_argument("--demo", action="store_true",
                       help="Lancer une démo rapide")
    
    args = parser.parse_args()
    
    print("=== Démo de Recherche et Grasping G1 ===")
    print(f"Modèle: {args.model}")
    print(f"Steps max: {args.steps}")
    print(f"Seuil contact: {args.contact_threshold}")
    print(f"Rayon recherche: {args.search_radius}")
    print(f"Vitesse recherche: {args.search_speed}")
    print(f"Force grasping: {args.grasp_force}")
    
    # Vérifier que le modèle existe
    if not os.path.exists(args.model):
        print(f"Erreur: Le modèle {args.model} n'existe pas!")
        print("Création du modèle de test...")
        try:
            from fix_model_paths import create_simple_test_model
            args.model = create_simple_test_model()
        except Exception as e:
            print(f"Erreur lors de la création du modèle: {e}")
            return 1
    
    # Lancer la simulation
    try:
        from grasp_search_simulation import GraspSearchSimulation
        
        # Créer la simulation
        sim = GraspSearchSimulation(model_path=args.model)
        
        # Ajuster les paramètres
        sim.max_steps = args.steps
        sim.contact_threshold = args.contact_threshold
        sim.search_radius = args.search_radius
        sim.search_speed = args.search_speed
        sim.closed_position = args.grasp_force
        
        # Lancer la simulation
        print("\nDémarrage de la simulation...")
        results = sim.run_simulation()
        
        # Afficher les résultats
        print("\n=== Résultats finaux ===")
        print(f"Récompense totale: {results['total_reward']:.2f}")
        print(f"Steps: {results['steps']}")
        print(f"Phase de recherche terminée: {results['search_phase_completed']}")
        print(f"Contact détecté: {results['contact_detected']}")
        print(f"Grasping réussi: {results['grasp_completed']}")
        print(f"Grasping stable: {results['grasp_stable']}")
        print(f"Position finale du cube: {results['final_cube_position']}")
        
        # Évaluation du succès
        if results['grasp_stable']:
            print("\n✅ SUCCÈS COMPLET: Recherche et grasping réussis!")
        elif results['grasp_completed']:
            print("\n⚠️  GRASPING: Le grasping a été réalisé mais n'est pas stable")
        elif results['contact_detected']:
            print("\n⚠️  CONTACT: Le contact a été détecté mais le grasping n'est pas complet")
        elif results['search_phase_completed']:
            print("\n⚠️  RECHERCHE: La phase de recherche est terminée mais pas de contact")
        else:
            print("\n❌ ÉCHEC: La recherche n'a pas trouvé le cube")
        
        return 0
        
    except ImportError as e:
        print(f"Erreur d'import: {e}")
        print("Assurez-vous que tous les modules sont installés:")
        print("pip install mujoco numpy")
        return 1
    except Exception as e:
        print(f"Erreur lors de la simulation: {e}")
        return 1

def run_quick_demo():
    """Lancer une démo rapide avec différentes stratégies"""
    print("=== Démo Rapide de Recherche et Grasping ===")
    
    # Créer le modèle de test si nécessaire
    if not os.path.exists("results/g1_test_simple.xml"):
        print("Création du modèle de test...")
        try:
            from fix_model_paths import create_simple_test_model
            model_path = create_simple_test_model()
        except Exception as e:
            print(f"Erreur: {e}")
            return 1
    
    # Lancer plusieurs simulations avec différents paramètres
    demos = [
        {
            "steps": 1000, 
            "contact_threshold": 0.05, 
            "search_radius": 0.2, 
            "search_speed": 0.1, 
            "grasp_force": 1.5, 
            "name": "Démo 1: Paramètres par défaut"
        },
        {
            "steps": 1000, 
            "contact_threshold": 0.02, 
            "search_radius": 0.15, 
            "search_speed": 0.15, 
            "grasp_force": 1.5, 
            "name": "Démo 2: Recherche plus précise"
        },
        {
            "steps": 1000, 
            "contact_threshold": 0.1, 
            "search_radius": 0.25, 
            "search_speed": 0.05, 
            "grasp_force": 2.0, 
            "name": "Démo 3: Recherche large et force élevée"
        },
    ]
    
    results_summary = []
    
    for i, demo in enumerate(demos):
        print(f"\n--- {demo['name']} ---")
        
        try:
            from grasp_search_simulation import GraspSearchSimulation
            
            sim = GraspSearchSimulation(model_path="results/g1_test_simple.xml")
            sim.max_steps = demo["steps"]
            sim.contact_threshold = demo["contact_threshold"]
            sim.search_radius = demo["search_radius"]
            sim.search_speed = demo["search_speed"]
            sim.closed_position = demo["grasp_force"]
            
            results = sim.run_simulation()
            
            results_summary.append({
                "demo": demo["name"],
                "search_completed": results["search_phase_completed"],
                "contact": results["contact_detected"],
                "grasp": results["grasp_completed"],
                "stable": results["grasp_stable"],
                "reward": results["total_reward"],
                "steps": results["steps"]
            })
            
            print(f"Résultat: Recherche={results['search_phase_completed']}, Contact={results['contact_detected']}, Grasp={results['grasp_completed']}")
            
        except Exception as e:
            print(f"Erreur dans la démo {i+1}: {e}")
            results_summary.append({
                "demo": demo["name"],
                "search_completed": False,
                "contact": False,
                "grasp": False,
                "stable": False,
                "reward": 0,
                "steps": 0,
                "error": str(e)
            })
    
    # Résumé final
    print("\n" + "="*60)
    print("RÉSUMÉ DES DÉMOS DE RECHERCHE ET GRASPING")
    print("="*60)
    
    for result in results_summary:
        status = "❌"
        if result.get("error"):
            status = "💥"
        elif result["stable"]:
            status = "✅"
        elif result["grasp"]:
            status = "⚠️"
        elif result["contact"]:
            status = "🔍"
        elif result["search_completed"]:
            status = "🔎"
        
        print(f"{status} {result['demo']}")
        print(f"   Recherche: {result['search_completed']}, Contact: {result['contact']}")
        print(f"   Grasp: {result['grasp']}, Stable: {result['stable']}")
        print(f"   Récompense: {result['reward']:.2f}, Steps: {result['steps']}")
        if result.get("error"):
            print(f"   Erreur: {result['error']}")
        print()
    
    return 0

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        sys.exit(run_quick_demo())
    else:
        sys.exit(main())