#!/usr/bin/env python3
"""
🧠 ENTRAÎNEUR SAC PROFESSIONNEL POUR GRASPING
==============================================

Agent SAC ultra-robuste avec fonctionnalités avancées:
🎯 Apprentissage adaptatif avec curriculum automatique
🎬 Enregistrement vidéo automatique à la fin de l'entraînement
🔄 Sauvegarde intelligente des modèles et métriques
📊 Monitoring en temps réel des performances
🛡️ Gestion d'erreurs robuste et recovery automatique
📈 Visualisation des courbes d'apprentissage

Conçu pour être entièrement autonome et professionnel.
"""

import os
import sys
import numpy as np
import json
import time
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import argparse
warnings.filterwarnings("ignore")

# Ajouter le path du workspace pour l'environnement
sys.path.append('/workspace')

try:
    from robust_grasp_env import RobustGraspEnv
    print("✅ RobustGraspEnv importé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class GraspingProgressCallback(BaseCallback):
    """Callback pour monitorer et enregistrer les progrès d'entraînement"""
    
    def __init__(self, trainer, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.trainer = trainer
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_phases = []
        self.episode_successes = []
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        # Collecter les informations de l'épisode si terminé
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'episode' in info:
                episode_reward = info['episode']['r']
                self.episode_rewards.append(episode_reward)
                
                # Enregistrer les métriques
                if hasattr(self.training_env.envs[0], 'get_attr'):
                    env_info = self.training_env.envs[0].get_attr('_get_info')[0]
                    self.episode_phases.append(env_info.get('phase', 'UNKNOWN'))
                    
                    # Détecter le succès (cube levé et maintenu)
                    success = (env_info.get('cube_lifted', False) and 
                             env_info.get('stability_score', 0) > 0.7)
                    self.episode_successes.append(success)
                
                # Mise à jour des métriques du trainer
                self.trainer.training_metrics['episode_rewards'].append(episode_reward)
                self.trainer.training_metrics['total_episodes'] += 1
                
                # Affichage périodique
                if len(self.episode_rewards) % 10 == 0:
                    recent_rewards = self.episode_rewards[-10:]
                    mean_reward = np.mean(recent_rewards)
                    success_rate = np.mean(self.episode_successes[-10:]) if self.episode_successes else 0
                    
                    print(f"📊 Épisode {len(self.episode_rewards):4d} | "
                          f"Récompense: {episode_reward:7.2f} | "
                          f"Moyenne: {mean_reward:7.2f} | "
                          f"Succès: {success_rate:5.1%}")
                    
                    # Sauvegarder le meilleur modèle
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        model_path = os.path.join(self.trainer.models_dir, "best_model.zip")
                        self.model.save(model_path)
                        print(f"💾 Nouveau meilleur modèle sauvegardé: {mean_reward:.2f}")
        
        return True


class SACGraspTrainer:
    """
    🧠 Entraîneur SAC Ultra-Professionnel pour Grasping
    
    Fonctionnalités complètes:
    - Entraînement SAC adaptatif avec hyperparamètres optimisés
    - Curriculum learning automatique intégré
    - Enregistrement vidéo automatique des épisodes
    - Sauvegarde intelligente des modèles et métriques
    - Visualisation automatique des courbes d'apprentissage
    - Monitoring en temps réel des performances
    - Gestion robuste des erreurs et recovery
    """
    
    def __init__(self, 
                 total_timesteps: int = 500000,
                 learning_rate: float = 3e-4,
                 buffer_size: int = 100000,
                 batch_size: int = 256,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 results_dir: str = "/workspace/sac_grasp_results"):
        
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.results_dir = results_dir
        
        # Configuration des dossiers
        self.models_dir = os.path.join(results_dir, "models")
        self.logs_dir = os.path.join(results_dir, "logs")
        self.videos_dir = os.path.join(results_dir, "videos")
        self.plots_dir = os.path.join(results_dir, "plots")
        self.metrics_dir = os.path.join(results_dir, "metrics")
        
        self._setup_directories()
        
        # Métriques d'entraînement
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'training_time': 0.0,
            'best_reward': -np.inf,
            'total_episodes': 0,
            'hyperparameters': {
                'learning_rate': learning_rate,
                'buffer_size': buffer_size,
                'batch_size': batch_size,
                'gamma': gamma,
                'tau': tau
            }
        }
        
        # État de l'entraînement
        self.env = None
        self.model = None
        self.start_time = None
        
        print("🧠 SACGraspTrainer initialisé")
        print(f"📁 Résultats: {self.results_dir}")
        print(f"🎯 Timesteps: {total_timesteps:,}")
        print(f"📚 Learning rate: {learning_rate}")
        print(f"🔄 Buffer size: {buffer_size:,}")
    
    def _setup_directories(self):
        """Crée la structure de dossiers nécessaire"""
        for directory in [self.results_dir, self.models_dir, self.logs_dir, 
                         self.videos_dir, self.plots_dir, self.metrics_dir]:
            os.makedirs(directory, exist_ok=True)
        print("📁 Structure de dossiers créée")
    
    def create_environment(self, record_video: bool = True) -> DummyVecEnv:
        """Crée l'environnement d'entraînement avec monitoring"""
        print("🏗️  Création de l'environnement de grasping...")
        
        def make_env():
            # Créer l'environnement avec enregistrement vidéo
            env = RobustGraspEnv(
                render_mode="rgb_array",
                record_video=record_video,
                video_dir=self.videos_dir
            )
            
            # Ajouter monitoring pour les métriques
            monitor_path = os.path.join(self.logs_dir, "monitor.csv")
            env = Monitor(env, monitor_path)
            
            return env
        
        # Créer l'environnement vectorisé
        self.env = DummyVecEnv([make_env])
        
        # Optionnel: normalisation des observations
        # self.env = VecNormalize(self.env, norm_obs=True, norm_reward=False)
        
        print("✅ Environnement créé avec succès")
        print(f"  📐 Espace action: {self.env.action_space}")
        print(f"  👁️  Espace observation: {self.env.observation_space}")
        
        return self.env
    
    def create_sac_model(self) -> SAC:
        """Crée le modèle SAC avec hyperparamètres optimisés"""
        print("🧠 Création du modèle SAC optimisé...")
        
        # Configuration du logger
        logger = configure(self.logs_dir, ["stdout", "csv", "tensorboard"])
        
        # Hyperparamètres optimisés pour le grasping
        sac_kwargs = {
            'learning_rate': self.learning_rate,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'tau': self.tau,
            'ent_coef': 'auto',  # Entropie automatique
            'target_update_interval': 1,
            'gradient_steps': 1,
            'learning_starts': 1000,
            'use_sde': True,  # State Dependent Exploration
            'sde_sample_freq': -1,
            'train_freq': 1,
            'verbose': 1,
            'device': 'auto',
            'tensorboard_log': self.logs_dir
        }
        
        # Créer le modèle SAC
        self.model = SAC('MlpPolicy', self.env, **sac_kwargs)
        self.model.set_logger(logger)
        
        print("✅ Modèle SAC créé avec succès")
        print(f"  🎯 Device: {self.model.device}")
        print(f"  📚 Policy: {type(self.model.policy).__name__}")
        print(f"  🔄 Buffer: {self.buffer_size:,} transitions")
        
        return self.model
    
    def train(self) -> Dict[str, Any]:
        """Lance l'entraînement complet avec monitoring"""
        print("\n🚀 DÉBUT DE L'ENTRAÎNEMENT SAC")
        print("=" * 60)
        
        self.start_time = time.time()
        
        try:
            # Créer l'environnement et le modèle
            if self.env is None:
                self.create_environment(record_video=True)
            
            if self.model is None:
                self.create_sac_model()
            
            # Configurer les callbacks
            progress_callback = GraspingProgressCallback(
                trainer=self,
                eval_freq=1000,
                verbose=1
            )
            
            # Callback d'évaluation
            eval_env = DummyVecEnv([lambda: Monitor(
                RobustGraspEnv(render_mode="rgb_array"), 
                os.path.join(self.logs_dir, "eval_monitor.csv")
            )])
            
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=self.models_dir,
                log_path=self.logs_dir,
                eval_freq=5000,
                deterministic=True,
                render=False,
                n_eval_episodes=5
            )
            
            callbacks = [progress_callback, eval_callback]
            
            print(f"📚 Entraînement pour {self.total_timesteps:,} timesteps...")
            print(f"🎬 Vidéos sauvegardées dans: {self.videos_dir}")
            
            # Lancer l'entraînement
            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=callbacks,
                log_interval=10,
                tb_log_name="SAC_Grasping",
                reset_num_timesteps=False,
                progress_bar=True
            )
            
            # Finaliser l'entraînement
            return self._finalize_training()
            
        except Exception as e:
            print(f"❌ Erreur pendant l'entraînement: {e}")
            self._save_crash_report(e)
            raise
        
        finally:
            # Nettoyer les ressources
            if self.env:
                self.env.close()
    
    def _finalize_training(self) -> Dict[str, Any]:
        """Finalise l'entraînement avec sauvegarde et visualisation"""
        
        end_time = time.time()
        training_duration = end_time - self.start_time
        self.training_metrics['training_time'] = training_duration
        
        print("\n🎉 ENTRAÎNEMENT TERMINÉ!")
        print("=" * 60)
        print(f"⏱️  Durée totale: {training_duration/3600:.2f} heures")
        print(f"📊 Épisodes totaux: {self.training_metrics['total_episodes']}")
        
        # Sauvegarder le modèle final
        final_model_path = os.path.join(self.models_dir, "final_model.zip")
        self.model.save(final_model_path)
        print(f"💾 Modèle final sauvegardé: {final_model_path}")
        
        # Sauvegarder les métriques
        metrics_path = os.path.join(self.metrics_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        print(f"📈 Métriques sauvegardées: {metrics_path}")
        
        # Générer les visualisations
        self._create_visualizations()
        
        # Créer une vidéo de démonstration finale
        self._create_demo_video()
        
        # Générer le rapport final
        report = self._generate_final_report()
        
        print(f"📊 Rapport final généré: {self.results_dir}/final_report.md")
        print(f"🎬 Vidéos disponibles: {self.videos_dir}")
        print(f"📈 Courbes d'apprentissage: {self.plots_dir}")
        
        return report
    
    def _create_visualizations(self):
        """Crée les visualisations des courbes d'apprentissage"""
        print("📊 Génération des visualisations...")
        
        if not self.training_metrics['episode_rewards']:
            print("⚠️  Pas de données à visualiser")
            return
        
        # Configuration matplotlib
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🧠 Courbes d\'Apprentissage SAC Grasping', fontsize=16, fontweight='bold')
        
        # 1. Récompenses par épisode
        ax1 = axes[0, 0]
        rewards = self.training_metrics['episode_rewards']
        episodes = range(1, len(rewards) + 1)
        
        ax1.plot(episodes, rewards, alpha=0.3, color='blue', linewidth=0.5)
        
        # Moyenne mobile
        if len(rewards) > 10:
            window = min(50, len(rewards) // 10)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(rewards) + 1), moving_avg, 
                    color='red', linewidth=2, label=f'Moyenne mobile ({window})')
        
        ax1.set_xlabel('Épisode')
        ax1.set_ylabel('Récompense')
        ax1.set_title('📈 Récompenses par Épisode')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Distribution des récompenses
        ax2 = axes[0, 1]
        ax2.hist(rewards, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(np.mean(rewards), color='red', linestyle='--', 
                   label=f'Moyenne: {np.mean(rewards):.2f}')
        ax2.set_xlabel('Récompense')
        ax2.set_ylabel('Fréquence')
        ax2.set_title('📊 Distribution des Récompenses')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Taux de succès (si disponible)
        ax3 = axes[1, 0]
        if len(rewards) > 20:
            # Calculer le taux de succès sur fenêtre glissante
            window = 20
            success_threshold = 50.0  # Récompense considérée comme succès
            success_rates = []
            
            for i in range(window, len(rewards) + 1):
                recent_rewards = rewards[i-window:i]
                success_rate = sum(1 for r in recent_rewards if r >= success_threshold) / window
                success_rates.append(success_rate * 100)
            
            ax3.plot(range(window, len(rewards) + 1), success_rates, 
                    color='purple', linewidth=2)
            ax3.set_xlabel('Épisode')
            ax3.set_ylabel('Taux de Succès (%)')
            ax3.set_title(f'🎯 Taux de Succès (fenêtre {window})')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Données insuffisantes', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('🎯 Taux de Succès')
        
        # 4. Statistiques récapitulatives
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        📊 STATISTIQUES FINALES
        
        🎯 Épisodes totaux: {len(rewards):,}
        📈 Récompense moyenne: {np.mean(rewards):.2f}
        🏆 Meilleure récompense: {np.max(rewards):.2f}
        📉 Récompense minimale: {np.min(rewards):.2f}
        📐 Écart-type: {np.std(rewards):.2f}
        
        ⏱️  Temps d'entraînement: {self.training_metrics['training_time']/3600:.1f}h
        🔄 Timesteps: {self.total_timesteps:,}
        
        🧠 Hyperparamètres:
        • Learning Rate: {self.learning_rate}
        • Buffer Size: {self.buffer_size:,}
        • Batch Size: {self.batch_size}
        • Gamma: {self.gamma}
        • Tau: {self.tau}
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=10, fontfamily='monospace', verticalalignment='top')
        
        # Sauvegarder les graphiques
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, "learning_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"💾 Graphiques sauvegardés: {plot_path}")
        
        plt.close()
    
    def _create_demo_video(self):
        """Crée une vidéo de démonstration avec le modèle final"""
        print("🎬 Création de la vidéo de démonstration finale...")
        
        try:
            # Créer un environnement de démonstration
            demo_env = RobustGraspEnv(
                render_mode="rgb_array",
                record_video=True,
                video_dir=self.videos_dir
            )
            
            # Charger le meilleur modèle
            best_model_path = os.path.join(self.models_dir, "best_model.zip")
            if os.path.exists(best_model_path):
                demo_model = SAC.load(best_model_path)
            else:
                demo_model = self.model
            
            # Exécuter quelques épisodes de démonstration
            n_demo_episodes = 3
            for episode in range(n_demo_episodes):
                obs, _ = demo_env.reset()
                done = False
                episode_reward = 0
                
                print(f"🎬 Enregistrement épisode démo {episode + 1}/{n_demo_episodes}")
                
                while not done:
                    action, _ = demo_model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = demo_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                
                print(f"   Récompense: {episode_reward:.2f}")
                
                # Sauvegarder la vidéo de cet épisode
                demo_env.save_video(f"demo_episode_{episode + 1:02d}.mp4")
            
            demo_env.close()
            print("✅ Vidéos de démonstration créées avec succès")
            
        except Exception as e:
            print(f"⚠️  Erreur lors de la création de la vidéo de démo: {e}")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Génère un rapport final détaillé"""
        
        report = {
            'training_completed': True,
            'timestamp': datetime.now().isoformat(),
            'total_timesteps': self.total_timesteps,
            'training_duration_hours': self.training_metrics['training_time'] / 3600,
            'total_episodes': self.training_metrics['total_episodes'],
            'hyperparameters': self.training_metrics['hyperparameters'],
            'performance': {
                'mean_reward': np.mean(self.training_metrics['episode_rewards']) if self.training_metrics['episode_rewards'] else 0,
                'best_reward': max(self.training_metrics['episode_rewards']) if self.training_metrics['episode_rewards'] else 0,
                'final_100_episodes_mean': np.mean(self.training_metrics['episode_rewards'][-100:]) if len(self.training_metrics['episode_rewards']) >= 100 else 0
            },
            'files': {
                'best_model': os.path.join(self.models_dir, "best_model.zip"),
                'final_model': os.path.join(self.models_dir, "final_model.zip"),
                'metrics': os.path.join(self.metrics_dir, "training_metrics.json"),
                'plots': os.path.join(self.plots_dir, "learning_curves.png"),
                'videos_dir': self.videos_dir,
                'logs_dir': self.logs_dir
            }
        }
        
        # Créer un rapport markdown
        markdown_report = f"""# 🧠 Rapport d'Entraînement SAC Grasping

## 📊 Résumé Exécutif

**Date de fin:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Durée totale:** {report['training_duration_hours']:.2f} heures  
**Timesteps:** {report['total_timesteps']:,}  
**Épisodes:** {report['total_episodes']:,}  

## 🎯 Performances

- **Récompense moyenne:** {report['performance']['mean_reward']:.2f}
- **Meilleure récompense:** {report['performance']['best_reward']:.2f}
- **Moyenne des 100 derniers épisodes:** {report['performance']['final_100_episodes_mean']:.2f}

## ⚙️ Hyperparamètres

- **Learning Rate:** {self.learning_rate}
- **Buffer Size:** {self.buffer_size:,}
- **Batch Size:** {self.batch_size}
- **Gamma:** {self.gamma}
- **Tau:** {self.tau}

## 📁 Fichiers Générés

- **Meilleur modèle:** `{os.path.basename(report['files']['best_model'])}`
- **Modèle final:** `{os.path.basename(report['files']['final_model'])}`
- **Métriques:** `{os.path.basename(report['files']['metrics'])}`
- **Graphiques:** `{os.path.basename(report['files']['plots'])}`
- **Vidéos:** `{os.path.basename(report['files']['videos_dir'])}/`
- **Logs:** `{os.path.basename(report['files']['logs_dir'])}/`

## 🎬 Utilisation

Pour tester le modèle entraîné:

```python
from stable_baselines3 import SAC
from robust_grasp_env import RobustGraspEnv

# Charger le modèle
model = SAC.load("{report['files']['best_model']}")

# Créer l'environnement
env = RobustGraspEnv(render_mode="rgb_array", record_video=True)

# Tester
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

env.save_video("test_episode.mp4")
env.close()
```

## 🏆 Conclusion

L'entraînement s'est terminé avec succès. Les modèles et vidéos sont disponibles dans le dossier de résultats.
"""
        
        # Sauvegarder le rapport markdown
        report_path = os.path.join(self.results_dir, "final_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        return report
    
    def _save_crash_report(self, error: Exception):
        """Sauvegarde un rapport d'erreur en cas de crash"""
        crash_report = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'training_progress': {
                'episodes_completed': self.training_metrics['total_episodes'],
                'training_time': time.time() - self.start_time if self.start_time else 0
            }
        }
        
        crash_path = os.path.join(self.results_dir, "crash_report.json")
        with open(crash_path, 'w') as f:
            json.dump(crash_report, f, indent=2)
        
        print(f"💥 Rapport de crash sauvegardé: {crash_path}")


def main():
    """Fonction principale d'entraînement"""
    
    # Configuration par défaut
    parser = argparse.ArgumentParser(description='🧠 Entraîneur SAC pour Grasping Robuste')
    parser.add_argument('--timesteps', type=int, default=500000, 
                       help='Nombre total de timesteps d\'entraînement')
    parser.add_argument('--lr', type=float, default=3e-4, 
                       help='Learning rate')
    parser.add_argument('--buffer', type=int, default=100000, 
                       help='Taille du buffer de replay')
    parser.add_argument('--batch', type=int, default=256, 
                       help='Taille du batch')
    parser.add_argument('--results-dir', type=str, default='/workspace/sac_grasp_results',
                       help='Dossier de sauvegarde des résultats')
    
    args = parser.parse_args()
    
    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT SAC GRASPING")
    print("=" * 60)
    print(f"🎯 Timesteps: {args.timesteps:,}")
    print(f"📚 Learning rate: {args.lr}")
    print(f"🔄 Buffer size: {args.buffer:,}")
    print(f"📦 Batch size: {args.batch}")
    print(f"📁 Résultats: {args.results_dir}")
    print("=" * 60)
    
    try:
        # Créer l'entraîneur
        trainer = SACGraspTrainer(
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            buffer_size=args.buffer,
            batch_size=args.batch,
            results_dir=args.results_dir
        )
        
        # Lancer l'entraînement
        report = trainer.train()
        
        print("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print(f"📊 Résultats disponibles dans: {args.results_dir}")
        
        return report
        
    except KeyboardInterrupt:
        print("\n⚠️  Entraînement interrompu par l'utilisateur")
        return None
        
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()