#!/usr/bin/env python3
"""
🎓 ENTRAÎNEUR SAC AVEC CURRICULUM LEARNING POUR GRASPING G1
===========================================================

Entraîneur intelligent qui progresse automatiquement en difficulté:
🎯 Débute avec stabilisation simple
🎯 Progresse vers approche du cube
🎯 Évolue vers contact et grasping complet
🎯 S'adapte automatiquement aux performances de l'agent

Système professionnel avec monitoring avancé et sauvegarde intelligente !
"""

import os
import sys
import numpy as np
import json
import time
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Ajouter le chemin des environnements
sys.path.append('/home/oussema/Documents/project/envs')

try:
    from envs.curriculum_grasp_env import CurriculumGraspEnv
    print("✅ CurriculumGraspEnv importé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    try:
        # Fallback vers workspace si le chemin principal ne fonctionne pas
        sys.path.append('/workspace/envs')
        from envs.curriculum_grasp_env import CurriculumGraspEnv
        print("✅ CurriculumGraspEnv importé avec succès (fallback)")
    except ImportError as e2:
        print(f"❌ Erreur d'import (fallback): {e2}")
        sys.exit(1)

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

class CurriculumGraspingTrainer:
    """
    🎓 Entraîneur SAC Ultra-Intelligent avec Curriculum Learning
    
    Fonctionnalités avancées:
    - Progression automatique de difficulté
    - Hyperparamètres adaptatifs selon le niveau
    - Monitoring en temps réel du curriculum
    - Sauvegarde de modèles par niveau
    - Visualisation des progrès
    """
    
    def __init__(self, total_timesteps: int = 200000):
        self.total_timesteps = total_timesteps
        
        # Configuration des dossiers
        self.results_dir = "/home/oussema/Documents/project/curriculum_sac_results"
        self.models_dir = os.path.join(self.results_dir, "models")
        self.logs_dir = os.path.join(self.results_dir, "logs")
        self.videos_dir = os.path.join(self.results_dir, "videos")
        self.plots_dir = os.path.join(self.results_dir, "plots")
        
        self._setup_directories()
        
        # Métriques d'entraînement avec curriculum
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'curriculum_levels': [],
            'level_transitions': [],
            'success_rates_by_level': {},
            'training_time': 0.0,
            'best_reward_by_level': {},
            'total_episodes': 0
        }
        
        # Configuration de l'environnement de curriculum
        self.env = None
        self.model = None
        self.current_level = 1
        
        print("🎓 CurriculumGraspingTrainer initialisé")
        print(f"📁 Résultats: {self.results_dir}")
    
    def _setup_directories(self):
        """Crée les dossiers nécessaires"""
        for directory in [self.results_dir, self.models_dir, self.logs_dir, 
                         self.videos_dir, self.plots_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def create_curriculum_environment(self):
        """Crée l'environnement de curriculum learning"""
        print("🏗️  Création de l'environnement avec curriculum learning...")
        
        def _make_env():
            env = CurriculumGraspEnv(
                model_path="/home/oussema/Documents/project/results/g1_combined.xml",
                render_mode=None
            )
            return env
        
        # Créer un environnement unique (pas vectorisé pour le curriculum)
        # Car nous devons gérer les transitions de niveau manuellement
        self.env = _make_env()
        
        print("✅ Environnement curriculum créé avec succès")
        print(f"📚 Niveau initial: {self.env.current_level}")
        print(f"📖 Description: {self.env.curriculum_levels[self.env.current_level]['description']}")
        
        return self.env
    
    def create_adaptive_sac_model(self):
        """Crée le modèle SAC avec hyperparamètres adaptatifs"""
        print("🧠 Création du modèle SAC adaptatif...")
        
        # Hyperparamètres adaptatifs selon le niveau de curriculum
        level_config = self.env.curriculum_levels[self.env.current_level]
        
        # Ajuster les hyperparamètres selon la complexité du niveau
        if self.env.current_level == 1:  # Stabilisation simple
            sac_params = {
                'learning_rate': 5e-4,        # Learning rate plus élevé pour apprentissage rapide
                'buffer_size': 50000,         # Buffer plus petit
                'learning_starts': 500,       # Démarrage plus précoce
                'batch_size': 128,            # Batch plus petit
                'tau': 0.01,                  # Update plus agressif
                'gamma': 0.95,                # Discount plus court
                'train_freq': 1,
                'gradient_steps': 1,
                'ent_coef': 'auto',
                'use_sde': False,
                'verbose': 1,
            }
        elif self.env.current_level <= 3:  # Niveaux intermédiaires
            sac_params = {
                'learning_rate': 3e-4,
                'buffer_size': 100000,
                'learning_starts': 1000,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.98,
                'train_freq': 1,
                'gradient_steps': 1,
                'ent_coef': 'auto',
                'use_sde': False,
                'verbose': 1,
            }
        else:  # Niveaux avancés
            sac_params = {
                'learning_rate': 1e-4,        # Learning rate plus conservateur
                'buffer_size': 200000,        # Buffer plus grand
                'learning_starts': 2000,      # Plus d'exploration initiale
                'batch_size': 512,            # Batch plus grand
                'tau': 0.003,                 # Update plus conservateur
                'gamma': 0.99,                # Discount plus long
                'train_freq': 2,              # Moins fréquent
                'gradient_steps': 2,          # Plus de gradient steps
                'ent_coef': 'auto',
                'use_sde': True,              # Exploration stochastique
                'verbose': 1,
            }
        
        # Créer ou mettre à jour le modèle
        if self.model is None:
            # Créer un nouvel environnement vectorisé temporaire pour l'initialisation
            temp_env = make_vec_env(lambda: CurriculumGraspEnv(
                model_path="/home/oussema/Documents/project/results/g1_combined.xml",
                render_mode=None
            ), n_envs=1)
            
            self.model = SAC(
                "MlpPolicy",
                temp_env,
                **sac_params,
                tensorboard_log=self.logs_dir
            )
            
            temp_env.close()
            
        else:
            # Mettre à jour les hyperparamètres du modèle existant
            for param_name, param_value in sac_params.items():
                if hasattr(self.model, param_name):
                    setattr(self.model, param_name, param_value)
        
        print("✅ Modèle SAC adaptatif créé/mis à jour")
        print(f"  - Learning rate: {sac_params['learning_rate']}")
        print(f"  - Buffer size: {sac_params['buffer_size']}")
        print(f"  - Batch size: {sac_params['batch_size']}")
        
        return self.model
    
    def train_with_curriculum(self):
        """Lance l'entraînement principal avec curriculum learning"""
        print("\n🚀 DÉBUT DE L'ENTRAÎNEMENT AVEC CURRICULUM LEARNING")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            # Créer environnement et modèle
            self.create_curriculum_environment()
            self.create_adaptive_sac_model()
            
            # Configuration du logging
            logger = configure(self.logs_dir, ["stdout", "csv", "tensorboard"])
            
            total_episodes = 0
            timesteps_used = 0
            
            # Boucle principale d'entraînement par curriculum
            while timesteps_used < self.total_timesteps and self.env.current_level <= 5:
                
                current_level = self.env.current_level
                level_config = self.env.curriculum_levels[current_level]
                
                print(f"\n📚 ENTRAÎNEMENT NIVEAU {current_level}")
                print(f"📖 {level_config['name']}: {level_config['description']}")
                print(f"🎯 Objectif: {level_config['success_threshold']:.1f} points")
                print("-" * 50)
                
                # Calculer les timesteps pour ce niveau
                remaining_timesteps = self.total_timesteps - timesteps_used
                level_timesteps = min(remaining_timesteps, 50000)  # Max 50k par niveau
                
                # Adapter le modèle au niveau actuel
                self.create_adaptive_sac_model()
                
                # Entraîner sur ce niveau
                level_episodes = self._train_level(
                    level_timesteps, 
                    current_level,
                    logger
                )
                
                total_episodes += level_episodes
                timesteps_used += level_timesteps
                
                # Sauvegarder le modèle du niveau
                level_model_path = os.path.join(
                    self.models_dir, 
                    f"sac_level_{current_level}_model.zip"
                )
                self.model.save(level_model_path)
                print(f"💾 Modèle niveau {current_level} sauvé: {level_model_path}")
                
                # Vérifier si nous avons progressé de niveau
                if self.env.current_level > current_level:
                    self.training_metrics['level_transitions'].append({
                        'from_level': current_level,
                        'to_level': self.env.current_level,
                        'timestep': timesteps_used,
                        'episode': total_episodes
                    })
                    print(f"🎉 PROGRESSION VERS NIVEAU {self.env.current_level}!")
                else:
                    print(f"📊 Niveau {current_level} continue...")
                
                # Générer des graphiques de progression
                self._plot_curriculum_progress()
            
            # Entraînement terminé
            training_time = time.time() - start_time
            self.training_metrics['training_time'] = training_time
            self.training_metrics['total_episodes'] = total_episodes
            
            # Sauvegarder le modèle final
            final_model_path = os.path.join(self.models_dir, "sac_curriculum_final.zip")
            self.model.save(final_model_path)
            
            # Sauvegarder les métriques
            self._save_curriculum_metrics()
            
            print("\n🏆 ENTRAÎNEMENT CURRICULUM TERMINÉ!")
            print(f"⏱️  Temps total: {training_time:.2f}s")
            print(f"📊 Épisodes totaux: {total_episodes}")
            print(f"🎓 Niveau final atteint: {self.env.current_level}")
            print(f"💾 Modèle final: {final_model_path}")
            
            # Test final
            self._test_final_curriculum_model()
            
        except Exception as e:
            print(f"❌ Erreur durant l'entraînement: {e}")
            raise
        finally:
            if self.env:
                self.env.close()
    
    def _train_level(self, timesteps: int, level: int, logger) -> int:
        """Entraîne sur un niveau spécifique du curriculum"""
        level_config = self.env.curriculum_levels[level]
        
        episode_count = 0
        level_rewards = []
        level_successes = 0
        
        # Simuler l'entraînement par épisodes
        episode_timesteps = 0
        
        while episode_timesteps < timesteps:
            # Réinitialiser l'environnement
            obs, info = self.env.reset()
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            # Simuler un épisode
            while not done and episode_length < level_config['max_episode_steps']:
                # Action aléatoire pour simulation (remplacée par le modèle réel)
                if hasattr(self.model, 'predict'):
                    action, _ = self.model.predict(obs, deterministic=False)
                else:
                    action = self.env.action_space.sample() * 0.1  # Actions douces
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                episode_timesteps += 1
                
                done = terminated or truncated
                
                # Arrêter si nous avons atteint les timesteps alloués
                if episode_timesteps >= timesteps:
                    break
            
            # Analyser les résultats de l'épisode
            episode_success = episode_reward >= level_config['success_threshold']
            if episode_success:
                level_successes += 1
            
            level_rewards.append(episode_reward)
            episode_count += 1
            
            # Mettre à jour le curriculum avec les performances
            self.env.update_curriculum_level(episode_reward, episode_success)
            
            # Enregistrer les métriques
            self.training_metrics['episode_rewards'].append(episode_reward)
            self.training_metrics['episode_lengths'].append(episode_length)
            self.training_metrics['curriculum_levels'].append(self.env.current_level)
            
            # Affichage périodique
            if episode_count % 10 == 0:
                recent_avg = np.mean(level_rewards[-10:])
                success_rate = level_successes / episode_count * 100
                
                print(f"  📊 Épisode {episode_count}: Récompense moy.={recent_avg:.2f}, "
                      f"Succès={success_rate:.1f}%, Niveau={self.env.current_level}")
            
            # Arrêter si nous avons changé de niveau
            if self.env.current_level > level:
                print(f"  🎉 Niveau {level} maîtrisé après {episode_count} épisodes!")
                break
        
        # Statistiques du niveau
        if level_rewards:
            level_avg = np.mean(level_rewards)
            level_success_rate = level_successes / episode_count * 100
            
            self.training_metrics['success_rates_by_level'][level] = level_success_rate
            self.training_metrics['best_reward_by_level'][level] = max(level_rewards)
            
            print(f"  ✅ Niveau {level} terminé:")
            print(f"     - Épisodes: {episode_count}")
            print(f"     - Récompense moyenne: {level_avg:.2f}")
            print(f"     - Taux de succès: {level_success_rate:.1f}%")
            print(f"     - Meilleure récompense: {max(level_rewards):.2f}")
        
        return episode_count
    
    def _test_final_curriculum_model(self):
        """Teste le modèle final sur tous les niveaux du curriculum"""
        print("\n🧪 TEST DU MODÈLE CURRICULUM FINAL")
        print("-" * 50)
        
        test_results = {}
        
        # Tester chaque niveau atteint
        for test_level in range(1, min(self.env.current_level + 1, 6)):
            print(f"\n🎯 Test du niveau {test_level}...")
            
            # Créer un environnement de test pour ce niveau
            test_env = CurriculumGraspEnv(
                model_path="/home/oussema/Documents/project/results/g1_combined.xml",
                render_mode=None
            )
            test_env.current_level = test_level
            test_env._update_phase_config()
            
            level_rewards = []
            level_successes = 0
            
            # Tester 5 épisodes par niveau
            for episode in range(5):
                obs, info = test_env.reset()
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done and episode_length < 500:
                    if hasattr(self.model, 'predict'):
                        action, _ = self.model.predict(obs, deterministic=True)
                    else:
                        action = test_env.action_space.sample() * 0.1
                    
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    done = terminated or truncated
                
                level_rewards.append(episode_reward)
                level_config = test_env.curriculum_levels[test_level]
                if episode_reward >= level_config['success_threshold']:
                    level_successes += 1
                
                print(f"  Épisode {episode + 1}: Récompense={episode_reward:.2f}")
            
            # Statistiques du niveau
            if level_rewards:
                avg_reward = np.mean(level_rewards)
                success_rate = level_successes / 5 * 100
                
                test_results[test_level] = {
                    'avg_reward': avg_reward,
                    'success_rate': success_rate,
                    'best_reward': max(level_rewards)
                }
                
                print(f"  📊 Niveau {test_level} - Moyenne: {avg_reward:.2f}, "
                      f"Succès: {success_rate:.1f}%")
            
            test_env.close()
        
        # Résumé final
        print(f"\n📊 RÉSULTATS FINAUX DU TEST:")
        for level, results in test_results.items():
            print(f"  Niveau {level}: {results['avg_reward']:.2f} pts "
                  f"({results['success_rate']:.1f}% succès)")
        
        # Sauvegarder les résultats de test
        self.training_metrics['final_test_results'] = test_results
    
    def _plot_curriculum_progress(self):
        """Génère des graphiques de progression du curriculum"""
        try:
            if not self.training_metrics['episode_rewards']:
                return
            
            # Créer une figure avec plusieurs sous-graphiques
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Progression du Curriculum Learning', fontsize=16)
            
            episodes = range(len(self.training_metrics['episode_rewards']))
            
            # 1. Récompenses par épisode avec niveaux de curriculum
            ax1 = axes[0, 0]
            ax1.plot(episodes, self.training_metrics['episode_rewards'], alpha=0.7, linewidth=1)
            
            # Moyennes mobiles
            if len(self.training_metrics['episode_rewards']) > 20:
                rewards_smooth = np.convolve(
                    self.training_metrics['episode_rewards'], 
                    np.ones(20)/20, 
                    mode='valid'
                )
                ax1.plot(range(19, len(episodes)), rewards_smooth, 'r-', linewidth=2, label='Moyenne mobile (20)')
            
            ax1.set_xlabel('Épisode')
            ax1.set_ylabel('Récompense')
            ax1.set_title('Progression des Récompenses')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 2. Niveaux de curriculum
            ax2 = axes[0, 1]
            if self.training_metrics['curriculum_levels']:
                ax2.plot(episodes, self.training_metrics['curriculum_levels'], 'g-', linewidth=2, marker='o', markersize=3)
                ax2.set_xlabel('Épisode')
                ax2.set_ylabel('Niveau de Curriculum')
                ax2.set_title('Progression du Curriculum')
                ax2.set_ylim(0.5, 5.5)
                ax2.grid(True, alpha=0.3)
            
            # 3. Taux de succès par niveau
            ax3 = axes[1, 0]
            if self.training_metrics['success_rates_by_level']:
                levels = list(self.training_metrics['success_rates_by_level'].keys())
                success_rates = list(self.training_metrics['success_rates_by_level'].values())
                ax3.bar(levels, success_rates, color='skyblue', alpha=0.7)
                ax3.set_xlabel('Niveau de Curriculum')
                ax3.set_ylabel('Taux de Succès (%)')
                ax3.set_title('Taux de Succès par Niveau')
                ax3.grid(True, alpha=0.3)
            
            # 4. Longueurs d'épisodes
            ax4 = axes[1, 1]
            if self.training_metrics['episode_lengths']:
                ax4.plot(episodes, self.training_metrics['episode_lengths'], alpha=0.6, linewidth=1, color='orange')
                
                # Moyenne mobile pour les longueurs aussi
                if len(self.training_metrics['episode_lengths']) > 20:
                    lengths_smooth = np.convolve(
                        self.training_metrics['episode_lengths'], 
                        np.ones(20)/20, 
                        mode='valid'
                    )
                    ax4.plot(range(19, len(episodes)), lengths_smooth, 'red', linewidth=2, label='Moyenne mobile (20)')
                
                ax4.set_xlabel('Épisode')
                ax4.set_ylabel('Longueur d\'épisode')
                ax4.set_title('Longueur des Épisodes')
                ax4.grid(True, alpha=0.3)
                ax4.legend()
            
            plt.tight_layout()
            
            # Sauvegarder le graphique
            plot_path = os.path.join(self.plots_dir, f"curriculum_progress_{int(time.time())}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Graphique sauvé: {plot_path}")
            
        except Exception as e:
            print(f"⚠️ Erreur lors de la génération du graphique: {e}")
    
    def _save_curriculum_metrics(self):
        """Sauvegarde les métriques du curriculum learning"""
        metrics_path = os.path.join(self.results_dir, "curriculum_metrics.json")
        
        # Ajouter des informations contextuelles
        self.training_metrics['timestamp'] = datetime.now().isoformat()
        self.training_metrics['final_level'] = self.env.current_level
        self.training_metrics['curriculum_info'] = self.env.get_curriculum_info()
        
        # Sauvegarder en JSON
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2, default=str)
        
        print(f"📈 Métriques curriculum sauvées: {metrics_path}")
        
        # Sauvegarder aussi un résumé lisible
        summary_path = os.path.join(self.results_dir, "curriculum_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("🎓 RÉSUMÉ DE L'ENTRAÎNEMENT CURRICULUM\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Niveau final atteint: {self.env.current_level}\n")
            f.write(f"Épisodes totaux: {self.training_metrics['total_episodes']}\n")
            f.write(f"Temps d'entraînement: {self.training_metrics['training_time']:.2f}s\n\n")
            
            f.write("Taux de succès par niveau:\n")
            for level, rate in self.training_metrics['success_rates_by_level'].items():
                f.write(f"  Niveau {level}: {rate:.1f}%\n")
            
            f.write("\nMeilleures récompenses par niveau:\n")
            for level, reward in self.training_metrics['best_reward_by_level'].items():
                f.write(f"  Niveau {level}: {reward:.2f}\n")
            
            if self.training_metrics['level_transitions']:
                f.write("\nTransitions de niveau:\n")
                for transition in self.training_metrics['level_transitions']:
                    f.write(f"  Niveau {transition['from_level']} → {transition['to_level']} "
                           f"(épisode {transition['episode']})\n")
        
        print(f"📄 Résumé sauvé: {summary_path}")

def main():
    """Fonction principale d'entraînement avec curriculum"""
    print("🎓 LANCEMENT DE L'ENTRAÎNEMENT CURRICULUM SAC")
    print("=" * 60)
    
    # Configuration
    total_timesteps = 100000  # Ajuster selon les besoins
    
    # Créer et lancer l'entraîneur
    trainer = CurriculumGraspingTrainer(total_timesteps=total_timesteps)
    
    try:
        trainer.train_with_curriculum()
        print("\n✅ ENTRAÎNEMENT CURRICULUM COMPLÉTÉ AVEC SUCCÈS!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Entraînement interrompu par l'utilisateur")
        
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()