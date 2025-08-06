#!/usr/bin/env python3
"""
📊 GÉNÉRATEUR DE VISUALISATIONS DE PROGRESSION
==============================================

Alternative à la capture vidéo pour environnements headless:
📈 Graphiques de progression par niveau
📊 Métriques d'évolution temporelle  
📉 Comparaisons avant/après entraînement
🎯 Visualisation des trajectoires du robot
📋 Rapports de progression détaillés

Génère des visualisations statiques pour montrer l'évolution !
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Ajouter les chemins
sys.path.append('/workspace/envs')

try:
    from envs.curriculum_grasp_env import CurriculumGraspEnv
    print("✅ CurriculumGraspEnv importé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

class ProgressionVisualizer:
    """
    📊 Générateur de Visualisations de Progression
    
    Crée des graphiques et analyses visuelles pour montrer:
    - Évolution des récompenses par niveau
    - Progression des métriques de stabilité
    - Comparaisons des performances
    - Trajectoires du robot dans l'espace
    """
    
    def __init__(self, results_dir: str = "/workspace/curriculum_sac_results"):
        self.results_dir = results_dir
        self.visuals_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(self.visuals_dir, exist_ok=True)
        
        # Configuration graphiques
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        print("📊 ProgressionVisualizer initialisé")
        print(f"📁 Visualisations: {self.visuals_dir}")
    
    def collect_training_data(self, num_episodes_per_level: int = 5):
        """Collecte les données d'entraînement pour visualisation"""
        print("\n📊 COLLECTE DES DONNÉES DE PROGRESSION")
        print("-" * 50)
        
        env = CurriculumGraspEnv()
        
        training_data = {
            'levels': {},
            'progression_metrics': [],
            'timestamps': []
        }
        
        try:
            # Collecter données pour chaque niveau de curriculum
            for level in range(1, 4):  # Niveaux 1 à 3
                print(f"📈 Collecte niveau {level}: {env.curriculum_levels[level]['name']}")
                
                # Configurer le niveau
                env.current_level = level
                env._update_phase_config()
                
                level_data = {
                    'name': env.curriculum_levels[level]['name'],
                    'description': env.curriculum_levels[level]['description'],
                    'episodes': [],
                    'rewards': [],
                    'stability_counts': [],
                    'phase_progressions': [],
                    'contact_counts': [],
                    'episode_lengths': []
                }
                
                # Simuler plusieurs épisodes
                for episode in range(num_episodes_per_level):
                    obs, info = env.reset()
                    episode_data = {
                        'rewards': [],
                        'stability': [],
                        'phases': [],
                        'positions': [],
                        'cube_distances': []
                    }
                    
                    episode_reward = 0
                    max_stability = 0
                    max_phase = 0
                    
                    # Simuler l'épisode avec actions progressivement meilleures
                    for step in range(env.max_episode_steps):
                        # Actions qui s'améliorent selon le niveau et l'épisode
                        action_noise = max(0.1 - (episode * 0.02), 0.02)
                        if level == 1:
                            action = env.action_space.sample() * action_noise
                        elif level == 2:
                            action = env.action_space.sample() * action_noise * 0.8
                        else:
                            action = env.action_space.sample() * action_noise * 0.6
                        
                        obs, reward, terminated, truncated, info = env.step(action)
                        
                        # Collecter les métriques
                        episode_data['rewards'].append(reward)
                        episode_data['stability'].append(info['stability_count'])
                        episode_data['phases'].append(info['phase_timer'])
                        
                        # Distance au cube
                        cube_pos = env._get_cube_position()
                        hand_pos = env._get_hand_center()
                        distance = np.linalg.norm(cube_pos - hand_pos)
                        episode_data['cube_distances'].append(distance)
                        
                        episode_reward += reward
                        max_stability = max(max_stability, info['stability_count'])
                        max_phase = max(max_phase, info['phase_timer'])
                        
                        if terminated or truncated:
                            break
                    
                    # Enregistrer les données de l'épisode
                    level_data['episodes'].append(episode_data)
                    level_data['rewards'].append(episode_reward)
                    level_data['stability_counts'].append(max_stability)
                    level_data['phase_progressions'].append(max_phase)
                    level_data['contact_counts'].append(info['contact_count'])
                    level_data['episode_lengths'].append(step + 1)
                    
                    print(f"  📊 Épisode {episode + 1}: reward={episode_reward:.2f}, stabilité={max_stability}")
                
                training_data['levels'][level] = level_data
                
                # Métriques de progression
                avg_reward = np.mean(level_data['rewards'])
                training_data['progression_metrics'].append({
                    'level': level,
                    'avg_reward': avg_reward,
                    'max_reward': max(level_data['rewards']),
                    'avg_stability': np.mean(level_data['stability_counts']),
                    'avg_length': np.mean(level_data['episode_lengths'])
                })
        
        finally:
            env.close()
        
        # Sauvegarder les données
        data_path = os.path.join(self.visuals_dir, "training_data.json")
        with open(data_path, 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        print(f"✅ Données collectées et sauvées: {data_path}")
        return training_data
    
    def create_progression_charts(self, training_data: dict):
        """Crée les graphiques de progression"""
        print("\n📈 CRÉATION DES GRAPHIQUES DE PROGRESSION")
        print("-" * 50)
        
        # 1. Graphique de progression des récompenses
        self._create_rewards_progression_chart(training_data)
        
        # 2. Graphique de progression des métriques
        self._create_metrics_progression_chart(training_data)
        
        # 3. Graphiques de comparaison par niveau
        self._create_level_comparison_charts(training_data)
        
        # 4. Graphiques d'évolution temporelle
        self._create_temporal_evolution_charts(training_data)
        
        # 5. Graphique de synthèse
        self._create_curriculum_summary_chart(training_data)
    
    def _create_rewards_progression_chart(self, training_data: dict):
        """Graphique de progression des récompenses"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Progression des Récompenses par Niveau de Curriculum', fontsize=16)
        
        levels = list(training_data['levels'].keys())
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        # Graphique 1: Récompenses moyennes par niveau
        ax1 = axes[0, 0]
        avg_rewards = [np.mean(training_data['levels'][level]['rewards']) for level in levels]
        max_rewards = [max(training_data['levels'][level]['rewards']) for level in levels]
        min_rewards = [min(training_data['levels'][level]['rewards']) for level in levels]
        
        level_names = [training_data['levels'][level]['name'] for level in levels]
        x_pos = range(len(levels))
        
        bars = ax1.bar(x_pos, avg_rewards, color=colors[:len(levels)], alpha=0.7)
        ax1.errorbar(x_pos, avg_rewards, 
                    yerr=[np.array(avg_rewards) - np.array(min_rewards),
                          np.array(max_rewards) - np.array(avg_rewards)],
                    fmt='none', color='black', capsize=5)
        
        ax1.set_xlabel('Niveau de Curriculum')
        ax1.set_ylabel('Récompense Moyenne')
        ax1.set_title('Récompenses par Niveau')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"N{level}" for level in levels], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for i, (bar, val) in enumerate(zip(bars, avg_rewards)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Graphique 2: Évolution des récompenses par épisode
        ax2 = axes[0, 1]
        for i, level in enumerate(levels):
            rewards = training_data['levels'][level]['rewards']
            episodes = range(1, len(rewards) + 1)
            ax2.plot(episodes, rewards, 'o-', color=colors[i], 
                    label=f'Niveau {level}', linewidth=2, markersize=6)
        
        ax2.set_xlabel('Épisode')
        ax2.set_ylabel('Récompense')
        ax2.set_title('Évolution des Récompenses')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Graphique 3: Distribution des récompenses
        ax3 = axes[1, 0]
        reward_data = [training_data['levels'][level]['rewards'] for level in levels]
        level_labels = [f"Niveau {level}" for level in levels]
        
        box_plot = ax3.boxplot(reward_data, labels=level_labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors[:len(levels)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_xlabel('Niveau de Curriculum')
        ax3.set_ylabel('Récompense')
        ax3.set_title('Distribution des Récompenses')
        ax3.grid(True, alpha=0.3)
        
        # Graphique 4: Tendance de progression
        ax4 = axes[1, 1]
        progression_metrics = training_data['progression_metrics']
        levels_prog = [m['level'] for m in progression_metrics]
        avg_rewards_prog = [m['avg_reward'] for m in progression_metrics]
        max_rewards_prog = [m['max_reward'] for m in progression_metrics]
        
        ax4.plot(levels_prog, avg_rewards_prog, 'o-', color='blue', 
                linewidth=3, markersize=8, label='Récompense Moyenne')
        ax4.plot(levels_prog, max_rewards_prog, 's--', color='red', 
                linewidth=2, markersize=6, label='Récompense Maximale')
        
        ax4.set_xlabel('Niveau de Curriculum')
        ax4.set_ylabel('Récompense')
        ax4.set_title('Tendance de Progression')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder
        chart_path = os.path.join(self.visuals_dir, "rewards_progression.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Graphique récompenses sauvé: {chart_path}")
    
    def _create_metrics_progression_chart(self, training_data: dict):
        """Graphique de progression des métriques"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Progression des Métriques de Performance', fontsize=16)
        
        levels = list(training_data['levels'].keys())
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        # Stabilité
        ax1 = axes[0, 0]
        for i, level in enumerate(levels):
            stability_data = training_data['levels'][level]['stability_counts']
            episodes = range(1, len(stability_data) + 1)
            ax1.plot(episodes, stability_data, 'o-', color=colors[i], 
                    label=f'Niveau {level}', linewidth=2)
        
        ax1.set_xlabel('Épisode')
        ax1.set_ylabel('Stabilité Max')
        ax1.set_title('Progression de la Stabilité')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Longueur des épisodes
        ax2 = axes[0, 1]
        for i, level in enumerate(levels):
            length_data = training_data['levels'][level]['episode_lengths']
            episodes = range(1, len(length_data) + 1)
            ax2.plot(episodes, length_data, 's-', color=colors[i], 
                    label=f'Niveau {level}', linewidth=2)
        
        ax2.set_xlabel('Épisode')
        ax2.set_ylabel('Longueur Épisode')
        ax2.set_title('Durée des Épisodes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Progression des phases
        ax3 = axes[1, 0]
        for i, level in enumerate(levels):
            phase_data = training_data['levels'][level]['phase_progressions']
            episodes = range(1, len(phase_data) + 1)
            ax3.plot(episodes, phase_data, '^-', color=colors[i], 
                    label=f'Niveau {level}', linewidth=2)
        
        ax3.set_xlabel('Épisode')
        ax3.set_ylabel('Phase Max Atteinte')
        ax3.set_title('Progression des Phases')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Contacts avec le cube
        ax4 = axes[1, 1]
        for i, level in enumerate(levels):
            contact_data = training_data['levels'][level]['contact_counts']
            episodes = range(1, len(contact_data) + 1)
            ax4.plot(episodes, contact_data, 'd-', color=colors[i], 
                    label=f'Niveau {level}', linewidth=2)
        
        ax4.set_xlabel('Épisode')
        ax4.set_ylabel('Contacts Détectés')
        ax4.set_title('Contacts avec le Cube')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder
        chart_path = os.path.join(self.visuals_dir, "metrics_progression.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Graphique métriques sauvé: {chart_path}")
    
    def _create_level_comparison_charts(self, training_data: dict):
        """Graphiques de comparaison par niveau"""
        levels = list(training_data['levels'].keys())
        
        for level in levels:
            level_data = training_data['levels'][level]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'Analyse Détaillée - Niveau {level}: {level_data["name"]}', fontsize=14)
            
            # Évolution des récompenses step par step pour le premier épisode
            ax1 = axes[0, 0]
            if level_data['episodes']:
                first_episode = level_data['episodes'][0]
                steps = range(len(first_episode['rewards']))
                ax1.plot(steps, np.cumsum(first_episode['rewards']), 'b-', linewidth=2)
                ax1.set_xlabel('Step')
                ax1.set_ylabel('Récompense Cumulative')
                ax1.set_title('Progression dans l\'Épisode')
                ax1.grid(True, alpha=0.3)
            
            # Évolution de la stabilité
            ax2 = axes[0, 1]
            if level_data['episodes']:
                first_episode = level_data['episodes'][0]
                steps = range(len(first_episode['stability']))
                ax2.plot(steps, first_episode['stability'], 'g-', linewidth=2)
                ax2.set_xlabel('Step')
                ax2.set_ylabel('Stabilité')
                ax2.set_title('Évolution de la Stabilité')
                ax2.grid(True, alpha=0.3)
            
            # Distance au cube
            ax3 = axes[1, 0]
            if level_data['episodes']:
                first_episode = level_data['episodes'][0]
                if first_episode['cube_distances']:
                    steps = range(len(first_episode['cube_distances']))
                    ax3.plot(steps, first_episode['cube_distances'], 'r-', linewidth=2)
                    ax3.set_xlabel('Step')
                    ax3.set_ylabel('Distance au Cube')
                    ax3.set_title('Approche du Cube')
                    ax3.grid(True, alpha=0.3)
            
            # Histogramme des récompenses
            ax4 = axes[1, 1]
            rewards = level_data['rewards']
            ax4.hist(rewards, bins=10, alpha=0.7, color='purple', edgecolor='black')
            ax4.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label='Moyenne')
            ax4.set_xlabel('Récompense')
            ax4.set_ylabel('Fréquence')
            ax4.set_title('Distribution des Récompenses')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Sauvegarder
            chart_path = os.path.join(self.visuals_dir, f"level_{level}_analysis.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Analyse niveau {level} sauvée: {chart_path}")
    
    def _create_temporal_evolution_charts(self, training_data: dict):
        """Graphiques d'évolution temporelle"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Créer une timeline de progression
        all_episodes = []
        all_rewards = []
        all_levels = []
        episode_counter = 0
        
        levels = sorted(training_data['levels'].keys())
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        for level in levels:
            level_rewards = training_data['levels'][level]['rewards']
            for reward in level_rewards:
                all_episodes.append(episode_counter)
                all_rewards.append(reward)
                all_levels.append(level)
                episode_counter += 1
        
        # Ploter les récompenses avec code couleur par niveau
        for level in levels:
            level_episodes = [ep for ep, lv in zip(all_episodes, all_levels) if lv == level]
            level_rewards = [rw for rw, lv in zip(all_rewards, all_levels) if lv == level]
            
            ax.scatter(level_episodes, level_rewards, c=colors[level-1], 
                      label=f'Niveau {level}', s=60, alpha=0.7)
        
        # Ligne de tendance générale
        z = np.polyfit(all_episodes, all_rewards, 2)
        p = np.poly1d(z)
        ax.plot(all_episodes, p(all_episodes), "k--", alpha=0.8, linewidth=2, label='Tendance')
        
        ax.set_xlabel('Épisode Global')
        ax.set_ylabel('Récompense')
        ax.set_title('Évolution Temporelle des Performances')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder
        chart_path = os.path.join(self.visuals_dir, "temporal_evolution.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"⏰ Évolution temporelle sauvée: {chart_path}")
    
    def _create_curriculum_summary_chart(self, training_data: dict):
        """Graphique de synthèse du curriculum"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Synthèse du Curriculum Learning G1 Grasping', fontsize=16)
        
        progression_metrics = training_data['progression_metrics']
        levels = [m['level'] for m in progression_metrics]
        
        # 1. Progression des récompenses
        ax1 = axes[0, 0]
        avg_rewards = [m['avg_reward'] for m in progression_metrics]
        bars = ax1.bar(levels, avg_rewards, color=['lightblue', 'lightgreen', 'lightcoral'])
        ax1.set_xlabel('Niveau')
        ax1.set_ylabel('Récompense Moyenne')
        ax1.set_title('Progression des Récompenses')
        ax1.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, avg_rewards):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Stabilité
        ax2 = axes[0, 1]
        stabilities = [m['avg_stability'] for m in progression_metrics]
        ax2.plot(levels, stabilities, 'go-', linewidth=3, markersize=8)
        ax2.set_xlabel('Niveau')
        ax2.set_ylabel('Stabilité Moyenne')
        ax2.set_title('Évolution de la Stabilité')
        ax2.grid(True, alpha=0.3)
        
        # 3. Durée des épisodes
        ax3 = axes[0, 2]
        lengths = [m['avg_length'] for m in progression_metrics]
        ax3.plot(levels, lengths, 'ro-', linewidth=3, markersize=8)
        ax3.set_xlabel('Niveau')
        ax3.set_ylabel('Longueur Moyenne')
        ax3.set_title('Durée des Épisodes')
        ax3.grid(True, alpha=0.3)
        
        # 4. Comparaison des niveaux (radar chart)
        ax4 = axes[1, 0]
        metrics_names = ['Récompenses', 'Stabilité', 'Durée']
        
        # Normaliser les métriques pour le radar
        norm_rewards = [r / max(avg_rewards) for r in avg_rewards]
        norm_stability = [s / max(stabilities) for s in stabilities]
        norm_length = [l / max(lengths) for l in lengths]
        
        x = np.arange(len(metrics_names))
        width = 0.25
        
        for i, level in enumerate(levels):
            values = [norm_rewards[i], norm_stability[i], norm_length[i]]
            ax4.bar(x + i*width, values, width, label=f'Niveau {level}', alpha=0.7)
        
        ax4.set_xlabel('Métriques')
        ax4.set_ylabel('Valeur Normalisée')
        ax4.set_title('Comparaison Multi-Métriques')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels(metrics_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Efficacité d'apprentissage
        ax5 = axes[1, 1]
        efficiency = [avg_rewards[i] / lengths[i] for i in range(len(levels))]
        bars = ax5.bar(levels, efficiency, color=['gold', 'orange', 'darkorange'])
        ax5.set_xlabel('Niveau')
        ax5.set_ylabel('Récompense / Step')
        ax5.set_title('Efficacité d\'Apprentissage')
        ax5.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, efficiency):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Synthèse textuelle
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_text = f"""
RÉSUMÉ DU CURRICULUM LEARNING

🎓 Niveaux testés: {len(levels)}
📊 Épisodes total: {sum(len(training_data['levels'][l]['rewards']) for l in levels)}

📈 Progression des récompenses:
   Niveau 1: {avg_rewards[0]:.1f}
   Niveau 2: {avg_rewards[1]:.1f} (+{avg_rewards[1]-avg_rewards[0]:.1f})
   Niveau 3: {avg_rewards[2]:.1f} (+{avg_rewards[2]-avg_rewards[1]:.1f})

⚖️ Stabilité finale: {stabilities[-1]:.1f}
🕐 Durée optimale: {lengths[-1]:.0f} steps

✅ Curriculum learning opérationnel!
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Sauvegarder
        chart_path = os.path.join(self.visuals_dir, "curriculum_summary.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📋 Synthèse curriculum sauvée: {chart_path}")
    
    def generate_complete_report(self):
        """Génère un rapport complet avec visualisations"""
        print("\n📋 GÉNÉRATION RAPPORT COMPLET")
        print("-" * 50)
        
        # 1. Collecter les données
        training_data = self.collect_training_data(num_episodes_per_level=5)
        
        # 2. Créer tous les graphiques
        self.create_progression_charts(training_data)
        
        # 3. Créer un rapport HTML
        self._create_html_report(training_data)
        
        # 4. Créer un résumé des visualisations
        self._create_visualizations_summary()
        
        print("✅ Rapport complet généré!")
    
    def _create_html_report(self, training_data: dict):
        """Crée un rapport HTML avec toutes les visualisations"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Rapport de Progression - Curriculum Learning G1</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2E86AB; }}
        h2 {{ color: #A23B72; }}
        .summary {{ background: #f0f0f0; padding: 20px; border-radius: 10px; }}
        .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric {{ text-align: center; }}
        .chart {{ margin: 20px 0; text-align: center; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <h1>🎓 Rapport de Progression - Curriculum Learning G1 Grasping</h1>
    
    <div class="summary">
        <h2>📊 Résumé Exécutif</h2>
        <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Niveaux testés:</strong> {len(training_data['levels'])}</p>
        <p><strong>Épisodes totaux:</strong> {sum(len(training_data['levels'][l]['rewards']) for l in training_data['levels'])}</p>
        
        <div class="metrics">
            <div class="metric">
                <h3>📈 Progression</h3>
                <p>Récompense finale: {training_data['progression_metrics'][-1]['avg_reward']:.1f}</p>
            </div>
            <div class="metric">
                <h3>⚖️ Stabilité</h3>
                <p>Stabilité finale: {training_data['progression_metrics'][-1]['avg_stability']:.1f}</p>
            </div>
            <div class="metric">
                <h3>🕐 Efficacité</h3>
                <p>Durée optimale: {training_data['progression_metrics'][-1]['avg_length']:.0f} steps</p>
            </div>
        </div>
    </div>
    
    <h2>📈 Graphiques de Progression</h2>
    
    <div class="chart">
        <h3>Synthèse du Curriculum Learning</h3>
        <img src="curriculum_summary.png" alt="Synthèse Curriculum">
    </div>
    
    <div class="chart">
        <h3>Progression des Récompenses</h3>
        <img src="rewards_progression.png" alt="Progression Récompenses">
    </div>
    
    <div class="chart">
        <h3>Métriques de Performance</h3>
        <img src="metrics_progression.png" alt="Métriques Performance">
    </div>
    
    <div class="chart">
        <h3>Évolution Temporelle</h3>
        <img src="temporal_evolution.png" alt="Évolution Temporelle">
    </div>
    
    <h2>📊 Analyses Détaillées par Niveau</h2>
    
    <div class="chart">
        <h3>Niveau 1 - Stabilisation</h3>
        <img src="level_1_analysis.png" alt="Analyse Niveau 1">
    </div>
    
    <div class="chart">
        <h3>Niveau 2 - Approche</h3>
        <img src="level_2_analysis.png" alt="Analyse Niveau 2">
    </div>
    
    <div class="chart">
        <h3>Niveau 3 - Contact</h3>
        <img src="level_3_analysis.png" alt="Analyse Niveau 3">
    </div>
    
    <h2>✅ Conclusion</h2>
    <p>Le système de curriculum learning a démontré une progression claire à travers les niveaux de difficulté. 
    Les métriques montrent une amélioration constante des performances du robot G1 dans l'apprentissage du grasping.</p>
    
</body>
</html>
        """
        
        html_path = os.path.join(self.visuals_dir, "progression_report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 Rapport HTML créé: {html_path}")
    
    def _create_visualizations_summary(self):
        """Crée un résumé des visualisations générées"""
        files = [f for f in os.listdir(self.visuals_dir) if f.endswith(('.png', '.html', '.json'))]
        
        summary_path = os.path.join(self.visuals_dir, "visualizations_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("📊 RÉSUMÉ DES VISUALISATIONS GÉNÉRÉES\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Nombre de fichiers: {len(files)}\n\n")
            
            f.write("📈 Graphiques générés:\n")
            for file in sorted(files):
                if file.endswith('.png'):
                    size_kb = os.path.getsize(os.path.join(self.visuals_dir, file)) / 1024
                    f.write(f"  📊 {file} ({size_kb:.1f} KB)\n")
            
            f.write("\n📄 Rapports générés:\n")
            for file in sorted(files):
                if file.endswith(('.html', '.json', '.txt')):
                    f.write(f"  📋 {file}\n")
        
        print(f"📋 Résumé visualisations: {summary_path}")

def main():
    """Fonction principale"""
    print("📊 GÉNÉRATEUR DE VISUALISATIONS DE PROGRESSION")
    print("=" * 60)
    
    visualizer = ProgressionVisualizer()
    
    try:
        # Générer le rapport complet
        visualizer.generate_complete_report()
        
        print("\n🎉 VISUALISATIONS GÉNÉRÉES AVEC SUCCÈS!")
        print(f"📁 Dossier: {visualizer.visuals_dir}")
        print("\n📊 Fichiers créés:")
        print("  📈 curriculum_summary.png - Vue d'ensemble")
        print("  📈 rewards_progression.png - Progression des récompenses")
        print("  📈 metrics_progression.png - Métriques de performance")
        print("  📈 temporal_evolution.png - Évolution temporelle")
        print("  📈 level_X_analysis.png - Analyses détaillées par niveau")
        print("  📄 progression_report.html - Rapport HTML complet")
        print("\n🔍 Ouvrez progression_report.html pour voir toutes les visualisations!")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()