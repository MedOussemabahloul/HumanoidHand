#!/usr/bin/env python3
"""
🚀 LANCEUR SIMPLE - Entraînement Grasping SAC
==============================================

Interface simple pour lancer l'entraînement du robot de grasping.
Choisissez votre mode et laissez le système faire le reste !

UTILISATION:
python3 lancer_entrainement.py
"""

import os
import sys

def main():
    """Launcher interactif pour l'entraînement"""
    
    print("🚀 LANCEUR D'ENTRAÎNEMENT GRASPING")
    print("=" * 40)
    print("🤖 Robot G1 - Apprentissage de Grasping")
    print("🎯 Choisissez votre mode d'entraînement")
    print("=" * 40)
    print()
    
    print("📋 MODES DISPONIBLES:")
    print("1. 🎭 Test Rapide      (5K steps - 2 minutes)")
    print("2. 🏃 Entraînement Rapide (50K steps - 10 minutes)")
    print("3. 💪 Entraînement Standard (100K steps - 20 minutes)")
    print("4. 🏆 Entraînement Expert (500K steps - 2 heures)")
    print("5. ⚙️  Configuration Personnalisée")
    print("6. 🧪 Tester un Modèle Existant")
    print("0. ❌ Quitter")
    print()
    
    while True:
        try:
            choix = input("🎯 Votre choix (0-6): ").strip()
            
            if choix == "0":
                print("👋 Au revoir !")
                return 0
            
            elif choix == "1":
                print("\n🎭 LANCEMENT DU TEST RAPIDE...")
                os.system("python3 train_final.py --quick")
                
            elif choix == "2":
                print("\n🏃 LANCEMENT DE L'ENTRAÎNEMENT RAPIDE...")
                os.system("python3 train_final.py --timesteps 50000")
                
            elif choix == "3":
                print("\n💪 LANCEMENT DE L'ENTRAÎNEMENT STANDARD...")
                os.system("python3 train_final.py --timesteps 100000")
                
            elif choix == "4":
                print("\n🏆 LANCEMENT DE L'ENTRAÎNEMENT EXPERT...")
                os.system("python3 train_final.py --timesteps 500000")
                
            elif choix == "5":
                print("\n⚙️ CONFIGURATION PERSONNALISÉE")
                timesteps = input("🎯 Nombre de timesteps (défaut 100000): ").strip()
                if not timesteps:
                    timesteps = "100000"
                
                results_dir = input("📁 Dossier résultats (défaut: /workspace/final_results): ").strip()
                if not results_dir:
                    results_dir = "/workspace/final_results"
                
                cmd = f"python3 train_final.py --timesteps {timesteps} --results-dir {results_dir}"
                print(f"\n🚀 Commande: {cmd}")
                os.system(cmd)
                
            elif choix == "6":
                print("\n🧪 TEST D'UN MODÈLE EXISTANT")
                model_path = input("🧠 Chemin du modèle (défaut: final_results/models/best_model.zip): ").strip()
                if not model_path:
                    model_path = "final_results/models/best_model.zip"
                
                episodes = input("📺 Nombre d'épisodes de test (défaut: 3): ").strip()
                if not episodes:
                    episodes = "3"
                
                if os.path.exists(model_path):
                    cmd = f"python3 test_trained_model.py --model {model_path} --episodes {episodes}"
                    print(f"\n🧪 Test en cours...")
                    os.system(cmd)
                else:
                    print(f"❌ Modèle non trouvé: {model_path}")
                    print("💡 Lancez d'abord un entraînement (option 1-4)")
                
            else:
                print("❌ Choix invalide. Utilisez 0-6.")
                continue
            
            # Proposer de continuer
            print("\n" + "=" * 40)
            continuer = input("🔄 Voulez-vous faire autre chose ? (o/N): ").strip().lower()
            if continuer not in ['o', 'oui', 'y', 'yes']:
                print("👋 Merci d'avoir utilisé le système de grasping !")
                return 0
            print()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Opération annulée par l'utilisateur")
            return 1
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)