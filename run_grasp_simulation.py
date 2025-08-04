#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de lancement pour les simulations de grasping G1 Robot
Permet de choisir entre différentes versions de simulation
"""

import os
import sys
import subprocess
import time

def print_banner():
    """Affiche la bannière du programme"""
    print("=" * 60)
    print("🤖 SIMULATION DE GRASPING G1 ROBOT")
    print("=" * 60)
    print("Ce script permet de lancer différentes versions de simulation")
    print("de grasping pour le robot G1 avec détection de contact.")
    print("=" * 60)

def check_dependencies():
    """Vérifie les dépendances nécessaires"""
    print("🔍 Vérification des dépendances...")
    
    required_packages = [
        "mujoco",
        "mujoco_viewer", 
        "numpy",
        "cv2",
        "xml"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "cv2":
                import cv2
            elif package == "xml":
                import xml.etree.ElementTree
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Packages manquants: {', '.join(missing_packages)}")
        print("Installez-les avec: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ Toutes les dépendances sont installées")
    return True

def check_model_file():
    """Vérifie que le fichier modèle existe"""
    model_path = "results/g1_combined.xml"
    
    if os.path.exists(model_path):
        print(f"✅ Modèle trouvé: {model_path}")
        return True
    else:
        print(f"❌ Modèle non trouvé: {model_path}")
        print("Assurez-vous que le fichier g1_combined.xml existe dans le dossier results/")
        return False

def show_menu():
    """Affiche le menu de sélection"""
    print("\n📋 MENU DE SÉLECTION")
    print("-" * 30)
    print("1. Test de configuration")
    print("2. Simulation simple (recommandée)")
    print("3. Simulation avec force sensors")
    print("4. Simulation améliorée")
    print("5. Afficher les fichiers de sortie")
    print("6. Quitter")
    print("-" * 30)

def run_test():
    """Lance le test de configuration"""
    print("\n🧪 Lancement du test de configuration...")
    
    try:
        result = subprocess.run([sys.executable, "test_grasp_simulation.py"], 
                              capture_output=True, text=True, timeout=60)
        
        print(result.stdout)
        if result.stderr:
            print("Erreurs:", result.stderr)
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Test interrompu (timeout)")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

def run_simple_simulation():
    """Lance la simulation simple"""
    print("\n🎯 Lancement de la simulation simple...")
    print("Cette version utilise la détection de contact basée sur la distance")
    print("et fonctionne avec le modèle existant sans modification.")
    
    try:
        result = subprocess.run([sys.executable, "grasp_simulation_simple.py"], 
                              capture_output=True, text=True, timeout=300)
        
        print(result.stdout)
        if result.stderr:
            print("Erreurs:", result.stderr)
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Simulation interrompue (timeout)")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de la simulation: {e}")
        return False

def run_force_sensor_simulation():
    """Lance la simulation avec force sensors"""
    print("\n⚡ Lancement de la simulation avec force sensors...")
    print("Cette version utilise les capteurs de force existants.")
    
    try:
        result = subprocess.run([sys.executable, "grasp_simulation.py"], 
                              capture_output=True, text=True, timeout=300)
        
        print(result.stdout)
        if result.stderr:
            print("Erreurs:", result.stderr)
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Simulation interrompue (timeout)")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de la simulation: {e}")
        return False

def run_improved_simulation():
    """Lance la simulation améliorée"""
    print("\n🚀 Lancement de la simulation améliorée...")
    print("Cette version ajoute des capteurs de force au modèle et utilise")
    print("une détection de contact plus sophistiquée.")
    
    try:
        result = subprocess.run([sys.executable, "grasp_simulation_improved.py"], 
                              capture_output=True, text=True, timeout=300)
        
        print(result.stdout)
        if result.stderr:
            print("Erreurs:", result.stderr)
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Simulation interrompue (timeout)")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de la simulation: {e}")
        return False

def show_output_files():
    """Affiche les fichiers de sortie disponibles"""
    print("\n📁 FICHIERS DE SORTIE DISPONIBLES")
    print("-" * 40)
    
    output_files = [
        "grasp_simulation_simple.mp4",
        "grasp_simulation.mp4", 
        "grasp_simulation_improved.mp4",
        "grasp_rewards_simple.txt",
        "grasp_rewards.txt",
        "grasp_rewards_improved.txt"
    ]
    
    found_files = []
    for filename in output_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✅ {filename} ({size} bytes)")
            found_files.append(filename)
        else:
            print(f"❌ {filename}")
    
    if not found_files:
        print("\nAucun fichier de sortie trouvé.")
        print("Lancez une simulation pour générer des fichiers.")
    else:
        print(f"\n{len(found_files)} fichier(s) trouvé(s)")

def main():
    """Fonction principale"""
    print_banner()
    
    # Vérifications préliminaires
    if not check_dependencies():
        print("\n❌ Impossible de continuer - dépendances manquantes")
        return
    
    if not check_model_file():
        print("\n❌ Impossible de continuer - modèle manquant")
        return
    
    print("\n✅ Configuration OK - Prêt à lancer les simulations")
    
    # Boucle principale
    while True:
        show_menu()
        
        try:
            choice = input("\nVotre choix (1-6): ").strip()
            
            if choice == "1":
                success = run_test()
                if success:
                    print("\n✅ Test réussi!")
                else:
                    print("\n❌ Test échoué")
                    
            elif choice == "2":
                print("\n⚠️ La simulation va prendre quelques minutes...")
                confirm = input("Continuer? (o/n): ").strip().lower()
                if confirm in ['o', 'oui', 'y', 'yes']:
                    success = run_simple_simulation()
                    if success:
                        print("\n✅ Simulation simple terminée!")
                    else:
                        print("\n❌ Simulation simple échouée")
                        
            elif choice == "3":
                print("\n⚠️ La simulation va prendre quelques minutes...")
                confirm = input("Continuer? (o/n): ").strip().lower()
                if confirm in ['o', 'oui', 'y', 'yes']:
                    success = run_force_sensor_simulation()
                    if success:
                        print("\n✅ Simulation avec force sensors terminée!")
                    else:
                        print("\n❌ Simulation avec force sensors échouée")
                        
            elif choice == "4":
                print("\n⚠️ La simulation va prendre quelques minutes...")
                confirm = input("Continuer? (o/n): ").strip().lower()
                if confirm in ['o', 'oui', 'y', 'yes']:
                    success = run_improved_simulation()
                    if success:
                        print("\n✅ Simulation améliorée terminée!")
                    else:
                        print("\n❌ Simulation améliorée échouée")
                        
            elif choice == "5":
                show_output_files()
                
            elif choice == "6":
                print("\n👋 Au revoir!")
                break
                
            else:
                print("\n❌ Choix invalide. Veuillez entrer un nombre entre 1 et 6.")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Interruption par l'utilisateur")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
        
        input("\nAppuyez sur Entrée pour continuer...")

if __name__ == "__main__":
    main()