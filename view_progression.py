#!/usr/bin/env python3
"""
🔍 VISUALISEUR DE PROGRESSION CURRICULUM LEARNING G1
==================================================

Script simple pour explorer les visualisations de progression du robot G1.
Affiche un menu interactif pour naviguer dans les résultats.
"""

import os
import sys
import json
import webbrowser
from datetime import datetime

def print_header():
    """Affiche l'en-tête du visualiseur"""
    print("🔍 VISUALISEUR DE PROGRESSION G1 GRASPING")
    print("=" * 60)
    print("📊 Explorez l'évolution et la maîtrise du robot G1")
    print("🎓 Curriculum Learning - Visualisations Interactives")
    print("-" * 60)

def check_visualizations():
    """Vérifie que les visualisations existent"""
    vis_dir = "/workspace/curriculum_sac_results/visualizations"
    
    if not os.path.exists(vis_dir):
        print("❌ Dossier de visualisations non trouvé!")
        print(f"   Chemin recherché: {vis_dir}")
        print("\n🔧 Générez d'abord les visualisations:")
        print("   python3 create_visual_progression.py")
        return False
    
    files = os.listdir(vis_dir)
    if not files:
        print("❌ Aucune visualisation trouvée!")
        print("\n🔧 Générez d'abord les visualisations:")
        print("   python3 create_visual_progression.py")
        return False
    
    return True, vis_dir, files

def show_files_summary(vis_dir, files):
    """Affiche un résumé des fichiers disponibles"""
    print("\n📁 FICHIERS DISPONIBLES:")
    print("-" * 40)
    
    total_size = 0
    file_categories = {
        'Graphiques': [],
        'Rapports': [],
        'Données': []
    }
    
    for file in sorted(files):
        file_path = os.path.join(vis_dir, file)
        size_kb = os.path.getsize(file_path) / 1024
        total_size += size_kb
        
        if file.endswith('.png'):
            file_categories['Graphiques'].append((file, size_kb))
        elif file.endswith(('.html', '.txt')):
            file_categories['Rapports'].append((file, size_kb))
        elif file.endswith('.json'):
            file_categories['Données'].append((file, size_kb))
    
    for category, file_list in file_categories.items():
        if file_list:
            print(f"\n📊 {category}:")
            for file, size in file_list:
                print(f"  📄 {file:<35} ({size:.1f} KB)")
    
    print(f"\n📊 Total: {len(files)} fichiers ({total_size:.1f} KB)")

def load_training_data(vis_dir):
    """Charge les données d'entraînement"""
    data_file = os.path.join(vis_dir, "training_data.json")
    
    if not os.path.exists(data_file):
        return None
    
    try:
        with open(data_file, 'r') as f:
            return json.load(f)
    except:
        return None

def show_progression_summary(training_data):
    """Affiche un résumé de la progression"""
    if not training_data:
        print("⚠️ Données d'entraînement non disponibles")
        return
    
    print("\n🎓 RÉSUMÉ DE LA PROGRESSION:")
    print("-" * 40)
    
    levels = training_data.get('levels', {})
    progression_metrics = training_data.get('progression_metrics', [])
    
    if not levels:
        print("⚠️ Aucune donnée de niveau trouvée")
        return
    
    print("📈 Évolution par niveau:")
    for i, metric in enumerate(progression_metrics):
        level = metric['level']
        level_info = levels.get(str(level), {})
        name = level_info.get('name', f'Niveau {level}')
        avg_reward = metric['avg_reward']
        avg_stability = metric['avg_stability']
        
        improvement = ""
        if i > 0:
            prev_reward = progression_metrics[i-1]['avg_reward']
            if prev_reward > 0:
                pct_improvement = ((avg_reward - prev_reward) / prev_reward) * 100
                improvement = f" (+{pct_improvement:.0f}%)"
        
        print(f"  🎯 Niveau {level}: {name}")
        print(f"     💯 Récompense: {avg_reward:.1f}{improvement}")
        print(f"     ⚖️ Stabilité: {avg_stability:.1f}")
        print()

def show_interpretation_guide():
    """Affiche un guide d'interprétation"""
    print("\n📖 GUIDE D'INTERPRÉTATION:")
    print("-" * 40)
    print("🎯 curriculum_summary.png - Vue d'ensemble complète")
    print("   📊 Progression des récompenses, stabilité, efficacité")
    print("   💡 Regardez l'augmentation des barres bleues")
    print()
    print("📈 rewards_progression.png - Évolution détaillée")
    print("   📊 Graphiques multiples de progression")
    print("   💡 Tendance croissante = apprentissage réussi")
    print()
    print("📊 metrics_progression.png - Métriques techniques")
    print("   📊 Stabilité, durée, phases, contacts")
    print("   💡 Courbes ascendantes = amélioration")
    print()
    print("🎬 level_X_analysis.png - Analyses par niveau")
    print("   📊 Détails de chaque étape d'apprentissage")
    print("   💡 Compare les performances avant/après")

def open_file(file_path):
    """Ouvre un fichier selon son type"""
    if file_path.endswith('.html'):
        print(f"🌐 Ouverture dans le navigateur: {file_path}")
        try:
            webbrowser.open(f"file://{file_path}")
            return True
        except:
            print("⚠️ Impossible d'ouvrir automatiquement")
            print(f"📋 Copiez ce chemin: file://{file_path}")
            return False
    elif file_path.endswith('.png'):
        print(f"🖼️ Fichier image: {file_path}")
        print("📋 Utilisez un visualiseur d'images pour ouvrir ce fichier")
        return True
    elif file_path.endswith('.json'):
        print(f"📄 Fichier de données: {file_path}")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            print("📊 Aperçu des données:")
            if isinstance(data, dict):
                for key in list(data.keys())[:5]:
                    print(f"  🔑 {key}")
                if len(data) > 5:
                    print(f"  ... et {len(data)-5} autres clés")
            return True
        except:
            print("⚠️ Erreur lors de la lecture du fichier JSON")
            return False
    else:
        print(f"📄 Fichier texte: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print("📄 Contenu:")
            print("-" * 30)
            print(content[:500])
            if len(content) > 500:
                print("... (tronqué)")
            return True
        except:
            print("⚠️ Erreur lors de la lecture du fichier")
            return False

def interactive_menu(vis_dir, files, training_data):
    """Menu interactif pour explorer les visualisations"""
    while True:
        print("\n🔍 MENU DE NAVIGATION:")
        print("-" * 30)
        print("1. 📄 Ouvrir le rapport HTML principal")
        print("2. 📊 Voir la synthèse du curriculum")
        print("3. 📈 Explorer un graphique spécifique")
        print("4. 📋 Afficher les données d'entraînement")
        print("5. 📖 Guide d'interprétation")
        print("6. 📁 Lister tous les fichiers")
        print("0. ❌ Quitter")
        
        choice = input("\n🎯 Votre choix (0-6): ").strip()
        
        if choice == "0":
            print("👋 Au revoir!")
            break
        elif choice == "1":
            html_file = os.path.join(vis_dir, "progression_report.html")
            if os.path.exists(html_file):
                open_file(html_file)
            else:
                print("❌ Fichier HTML non trouvé")
        elif choice == "2":
            summary_file = os.path.join(vis_dir, "curriculum_summary.png")
            if os.path.exists(summary_file):
                open_file(summary_file)
            else:
                print("❌ Fichier synthèse non trouvé")
        elif choice == "3":
            print("\n📊 Graphiques disponibles:")
            png_files = [f for f in files if f.endswith('.png')]
            for i, f in enumerate(png_files, 1):
                print(f"  {i}. {f}")
            
            try:
                graph_choice = int(input("📈 Numéro du graphique: ")) - 1
                if 0 <= graph_choice < len(png_files):
                    file_path = os.path.join(vis_dir, png_files[graph_choice])
                    open_file(file_path)
                else:
                    print("❌ Numéro invalide")
            except ValueError:
                print("❌ Veuillez entrer un numéro")
        elif choice == "4":
            show_progression_summary(training_data)
        elif choice == "5":
            show_interpretation_guide()
        elif choice == "6":
            show_files_summary(vis_dir, files)
        else:
            print("❌ Choix invalide. Essayez à nouveau.")

def main():
    """Fonction principale"""
    print_header()
    
    # Vérifier les visualisations
    result = check_visualizations()
    if not result:
        return
    
    success, vis_dir, files = result
    
    # Charger les données
    training_data = load_training_data(vis_dir)
    
    # Afficher le résumé
    show_files_summary(vis_dir, files)
    show_progression_summary(training_data)
    
    # Menu interactif
    print("\n🎉 Visualisations prêtes à explorer!")
    interactive_menu(vis_dir, files, training_data)

if __name__ == "__main__":
    main()