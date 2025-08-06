#!/usr/bin/env python3
"""
ENTRAÎNEMENT ULTRA-STABLE FINAL
Corrige définitivement les instabilités DOF 15, 16, 20
"""

import os
import sys
import argparse
import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import time
from pathlib import Path
import json

# Imports locaux
sys.path.append('.')
sys.path.append('./envs')
sys.path.append('./agents')

if HAS_TORCH:
    from envs.ultra_stable_grasp_env import UltraStableGraspEnv
    from agents.improved_sac_agent import ImprovedSACAgent

class UltraStableTrainer:
    """Entraîneur ultra-stable FINAL"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        if not HAS_TORCH:
            print("❌ PyTorch requis")
            return
        
        # Environnement ultra-stable
        print("🛡️  Initialisation environnement ULTRA-STABLE...")
        self.env = UltraStableGraspEnv(
            xml_path=config['model_path'],
            max_episode_steps=config['max_episode_steps'],
            block_fingers=config['block_fingers']
        )
        
        # Agent SAC ultra-conservateur
        print("🧠 Initialisation agent SAC...")
        self.agent = ImprovedSACAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.shape[0],
            lr=config['learning_rate'],
            hidden_sizes=config['hidden_sizes'],
            buffer_size=config['buffer_size'],
            gamma=config['gamma'],
            tau=config['tau']
        )
        
        # Métriques
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.instability_counts = []
        self.training_metrics = []
        
        # Monitoring ultra-stable
        self.total_instabilities = 0
        self.consecutive_crashes = 0
        
        print("✅ Entraîneur ULTRA-STABLE prêt")
    
    def train(self):
        """Entraînement ultra-stable"""
        if not HAS_TORCH:
            print("❌ PyTorch manquant")
            return
            
        print("\n🛡️  DÉBUT ENTRAÎNEMENT ULTRA-STABLE")
        print("=" * 60)
        print(f"🖐️  Doigts bloqués: {self.config['block_fingers']}")
        print(f"⏱️  Steps max: {self.config['max_episode_steps']}")
        print(f"🎯 Actions: ±{self.env.action_space.high[0]:.2f}")
        
        start_time = time.time()
        total_episodes = self.config['total_episodes']
        
        for episode in range(total_episodes):
            try:
                # Reset ultra-sécurisé
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                episode_success = False
                episode_instabilities = 0
                
                done = False
                crashed = False
                
                while not done and episode_length < self.config['max_episode_steps']:
                    # Actions ultra-conservatives
                    if episode < 10:  # Phase d'acclimatation
                        action = np.zeros(self.env.action_space.shape[0])
                    elif episode < 50:
                        action = self.agent.select_action(obs, evaluate=True)
                        action = action * 0.01  # Actions minuscules
                    else:
                        action = self.agent.select_action(obs)
                        action = action * 0.1  # Actions réduites
                    
                    # Step avec monitoring
                    try:
                        next_obs, reward, terminated, truncated, info = self.env.step(action)
                        done = terminated or truncated
                        
                        # Vérifier instabilité
                        if "error" in info:
                            episode_instabilities += 1
                            self.total_instabilities += 1
                            crashed = True
                            print(f"⚠️  Crash épisode {episode}: {info['error']}")
                            reward = -100.0
                            done = True
                            break
                        
                        # Stocker transition si stable
                        if not crashed and episode >= 10:
                            self.agent.store_transition(obs, action, reward, next_obs, done)
                        
                        episode_reward += reward
                        episode_length += 1
                        episode_success = terminated and not crashed
                        
                        obs = next_obs
                        
                    except Exception as e:
                        print(f"⚠️  Exception épisode {episode}: {e}")
                        crashed = True
                        episode_instabilities += 1
                        self.total_instabilities += 1
                        done = True
                        break
                
                # Gestion crashes
                if crashed:
                    self.consecutive_crashes += 1
                    if self.consecutive_crashes >= 3:
                        print("🛑 Trop de crashes - arrêt temporaire")
                        time.sleep(1)
                        self.consecutive_crashes = 0
                else:
                    self.consecutive_crashes = 0
                
                # Arrêt si trop d'instabilités
                if self.total_instabilities >= 15:
                    print("🛑 Trop d'instabilités totales - arrêt")
                    break
                
                # Entraînement de l'agent (très conservateur)
                if (len(self.agent.replay_buffer) > self.config['batch_size'] and 
                    not crashed and 
                    episode >= 30 and
                    episode % self.config['training_frequency'] == 0):
                    
                    training_info = self.agent.update(self.config['batch_size'])
                    if training_info:
                        self.training_metrics.append(training_info)
                
                # Enregistrer métriques
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_successes.append(episode_success)
                self.instability_counts.append(episode_instabilities)
                
                # Logging détaillé
                if (episode + 1) % self.config['log_interval'] == 0:
                    self._log_progress(episode + 1, total_episodes, start_time)
                
                # Sauvegarde fréquente
                if (episode + 1) % self.config['save_interval'] == 0:
                    self._save_checkpoint(episode + 1)
                
            except Exception as e:
                print(f"❌ Erreur critique épisode {episode}: {e}")
                self.consecutive_crashes += 1
                continue
        
        # Fin entraînement
        total_time = time.time() - start_time
        print(f"\n✅ ENTRAÎNEMENT ULTRA-STABLE TERMINÉ")
        print(f"   Durée: {total_time/3600:.1f}h")
        print(f"   Épisodes: {len(self.episode_rewards)}")
        print(f"   Instabilités totales: {self.total_instabilities}")
        
        if self.episode_rewards:
            print(f"   Récompense moyenne: {np.mean(self.episode_rewards[-20:]):.2f}")
            print(f"   Longueur moyenne: {np.mean(self.episode_lengths[-20:]):.1f}")
            print(f"   Taux succès: {np.mean(self.episode_successes[-20:]) * 100:.1f}%")
            
            stable_episodes = sum(1 for x in self.instability_counts[-20:] if x == 0)
            print(f"   Épisodes stables: {stable_episodes}/20")
        
        self._save_final_results()
    
    def _log_progress(self, episode, total_episodes, start_time):
        """Log détaillé des progrès"""
        recent_episodes = min(self.config['log_interval'], len(self.episode_rewards))
        
        if recent_episodes > 0:
            recent_rewards = self.episode_rewards[-recent_episodes:]
            recent_lengths = self.episode_lengths[-recent_episodes:]
            recent_successes = self.episode_successes[-recent_episodes:]
            recent_instabilities = self.instability_counts[-recent_episodes:]
            
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            success_rate = np.mean(recent_successes) * 100
            avg_instabilities = np.mean(recent_instabilities)
            stable_episodes = sum(1 for x in recent_instabilities if x == 0)
            
            elapsed_time = time.time() - start_time
            
            print(f"\n🛡️  ULTRA-STABLE PROGRESS - Épisode {episode}/{total_episodes}")
            print("-" * 50)
            print(f"   📊 Récompense: {avg_reward:.2f} ± {np.std(recent_rewards):.2f}")
            print(f"   📏 Longueur: {avg_length:.1f} steps")
            print(f"   ✅ Succès: {success_rate:.1f}%")
            print(f"   🛡️  Stables: {stable_episodes}/{recent_episodes}")
            print(f"   ⚠️  Instabilités moy: {avg_instabilities:.1f}")
            print(f"   💥 Crashes consécutifs: {self.consecutive_crashes}")
            print(f"   📊 Instabilités totales: {self.total_instabilities}")
            print(f"   💾 Buffer: {len(self.agent.replay_buffer)}")
            print(f"   ⏱️  Temps: {elapsed_time/60:.1f}min")
            
            # État stabilité
            if avg_instabilities == 0:
                print("   🟢 ÉTAT: STABLE")
            elif avg_instabilities < 1:
                print("   🟡 ÉTAT: QUASI-STABLE")
            else:
                print("   🔴 ÉTAT: INSTABLE")
    
    def _save_checkpoint(self, episode):
        """Sauvegarde checkpoint"""
        try:
            checkpoint_path = self.output_dir / "models" / f"ultra_stable_ep_{episode}.pth"
            self.agent.save(checkpoint_path)
            
            metrics = {
                "episode": episode,
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
                "episode_successes": self.episode_successes,
                "instability_counts": self.instability_counts,
                "total_instabilities": self.total_instabilities,
                "consecutive_crashes": self.consecutive_crashes,
                "config": self.config
            }
            
            metrics_path = self.output_dir / "logs" / f"metrics_ep_{episode}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Erreur sauvegarde: {e}")
    
    def _save_final_results(self):
        """Sauvegarde finale"""
        try:
            final_model_path = self.output_dir / "models" / "ultra_stable_final.pth"
            self.agent.save(final_model_path)
            
            final_metrics = {
                "config": self.config,
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
                "episode_successes": self.episode_successes,
                "instability_counts": self.instability_counts,
                "training_metrics": self.training_metrics,
                "total_instabilities": self.total_instabilities,
                "final_stats": {
                    "total_episodes": len(self.episode_rewards),
                    "avg_reward": float(np.mean(self.episode_rewards[-20:])) if self.episode_rewards else 0,
                    "success_rate": float(np.mean(self.episode_successes[-20:])) if self.episode_successes else 0,
                    "avg_length": float(np.mean(self.episode_lengths[-20:])) if self.episode_lengths else 0,
                    "stability_rate": float(sum(1 for x in self.instability_counts[-20:] if x == 0) / min(20, len(self.instability_counts))) if self.instability_counts else 0,
                    "total_instabilities": self.total_instabilities
                }
            }
            
            final_metrics_path = self.output_dir / "logs" / "ultra_stable_final.json"
            with open(final_metrics_path, 'w') as f:
                json.dump(final_metrics, f, indent=2)
            
            print(f"✅ Résultats sauvegardés: {self.output_dir}")
            
        except Exception as e:
            print(f"⚠️  Erreur sauvegarde finale: {e}")

def load_ultra_stable_config():
    """Configuration ultra-stable finale"""
    return {
        'model_path': 'results/g1_combined.xml',
        'max_episode_steps': 40,       # Très court
        'block_fingers': True,         # DOIGTS BLOQUÉS
        'total_episodes': 100,         # Modéré
        'learning_rate': 5e-5,         # Très bas
        'batch_size': 32,              # Petit
        'buffer_size': 5000,           # Petit
        'training_frequency': 25,      # Rare
        'hidden_sizes': [64, 64],      # Petit
        'gamma': 0.9,                  # Court terme
        'tau': 0.005,                  # Lent
        'log_interval': 5,             # Fréquent
        'save_interval': 25,           # Fréquent
        'output_dir': 'ultra_stable_results'
    }

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description='Entraînement ULTRA-STABLE final G1')
    parser.add_argument('--episodes', type=int, default=100, help='Nombre épisodes')
    parser.add_argument('--max-steps', type=int, default=40, help='Steps max par épisode')
    parser.add_argument('--output', type=str, default='ultra_stable_results', help='Dossier sortie')
    
    args = parser.parse_args()
    
    config = load_ultra_stable_config()
    config['total_episodes'] = args.episodes
    config['max_episode_steps'] = args.max_steps
    config['output_dir'] = args.output
    
    print("🛡️  ENTRAÎNEMENT ULTRA-STABLE FINAL G1")
    print("=" * 50)
    print(f"Épisodes: {config['total_episodes']}")
    print(f"Steps max: {config['max_episode_steps']}")
    print(f"Doigts bloqués: {config['block_fingers']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Actions: ±{0.1}")
    print(f"Sortie: {config['output_dir']}")
    
    # Vérifier modèle
    if not Path(config['model_path']).exists():
        print(f"❌ Modèle manquant: {config['model_path']}")
        print("💡 Placez g1_combined.xml dans results/")
        return
    
    try:
        trainer = UltraStableTrainer(config)
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n⏹️  Arrêt manuel")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🏁 Fin entraînement ultra-stable")

if __name__ == "__main__":
    main()
