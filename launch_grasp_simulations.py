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
        "numpy",
        "cv2",
        "time"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "cv2":
                import cv2
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

def show_menu():
    """Affiche le menu de sélection"""
    print("\n📋 MENU DE SÉLECTION")
    print("-" * 30)
    print("1. Démonstration simple")
    print("2. Simulation fonctionnelle")
    print("3. Simulation finale")
    print("4. Simulation réussie (recommandée)")
    print("5. Afficher les fichiers de sortie")
    print("6. Quitter")
    print("-" * 30)

def run_simple_demo():
    """Lance la démonstration simple"""
    print("\n🎯 Lancement de la démonstration simple...")
    print("Cette version montre le concept de base du grasping.")
    
    try:
        result = subprocess.run([sys.executable, "simple_grasp_demo.py"], 
                              capture_output=True, text=True, timeout=120)
        
        print(result.stdout)
        if result.stderr:
            print("Erreurs:", result.stderr)
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Démonstration interrompue (timeout)")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de la démonstration: {e}")
        return False

def run_working_simulation():
    """Lance la simulation fonctionnelle"""
    print("\n⚡ Lancement de la simulation fonctionnelle...")
    print("Cette version utilise des paramètres ajustés.")
    
    try:
        result = subprocess.run([sys.executable, "working_grasp_simulation.py"], 
                              capture_output=True, text=True, timeout=120)
        
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

def run_final_simulation():
    """Lance la simulation finale"""
    print("\n🚀 Lancement de la simulation finale...")
    print("Cette version utilise des paramètres optimisés.")
    
    try:
        result = subprocess.run([sys.executable, "final_grasp_simulation.py"], 
                              capture_output=True, text=True, timeout=120)
        
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

def run_successful_simulation():
    """Lance la simulation réussie"""
    print("\n🎉 Lancement de la simulation réussie...")
    print("Cette version est optimisée pour assurer le succès.")
    
    try:
        result = subprocess.run([sys.executable, "successful_grasp_simulation.py"], 
                              capture_output=True, text=True, timeout=120)
        
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
        "simple_grasp_demo.mp4",
        "working_grasp_simulation.mp4",
        "final_grasp_simulation.mp4",
        "successful_grasp_simulation.mp4",
        "simple_grasp_rewards.txt",
        "working_grasp_rewards.txt",
        "final_grasp_rewards.txt",
        "successful_grasp_rewards.txt"
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
    
    print("\n✅ Configuration OK - Prêt à lancer les simulations")
    
    # Boucle principale
    while True:
        show_menu()
        
        try:
            choice = input("\nVotre choix (1-6): ").strip()
            
            if choice == "1":
                print("\n⚠️ La démonstration va prendre quelques secondes...")
                confirm = input("Continuer? (o/n): ").strip().lower()
                if confirm in ['o', 'oui', 'y', 'yes']:
                    success = run_simple_demo()
                    if success:
                        print("\n✅ Démonstration simple terminée!")
                    else:
                        print("\n❌ Démonstration simple échouée")
                        
            elif choice == "2":
                print("\n⚠️ La simulation va prendre quelques secondes...")
                confirm = input("Continuer? (o/n): ").strip().lower()
                if confirm in ['o', 'oui', 'y', 'yes']:
                    success = run_working_simulation()
                    if success:
                        print("\n✅ Simulation fonctionnelle terminée!")
                    else:
                        print("\n❌ Simulation fonctionnelle échouée")
                        
            elif choice == "3":
                print("\n⚠️ La simulation va prendre quelques secondes...")
                confirm = input("Continuer? (o/n): ").strip().lower()
                if confirm in ['o', 'oui', 'y', 'yes']:
                    success = run_final_simulation()
                    if success:
                        print("\n✅ Simulation finale terminée!")
                    else:
                        print("\n❌ Simulation finale échouée")
                        
            elif choice == "4":
                print("\n⚠️ La simulation va prendre quelques secondes...")
                confirm = input("Continuer? (o/n): ").strip().lower()
                if confirm in ['o', 'oui', 'y', 'yes']:
                    success = run_successful_simulation()
                    if success:
                        print("\n✅ Simulation réussie terminée!")
                    else:
                        print("\n❌ Simulation réussie échouée")
                        
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