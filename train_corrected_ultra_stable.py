
#!/usr/bin/env python3
"""
ENTRAÎNEUR CORRIGÉ - Identification exacte des doigts
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

sys.path.append('.')
sys.path.append('./envs')
sys.path.append('./agents')

if HAS_TORCH:
  from envs.corrected_ultra_stable_env import CorrectedUltraStableGraspEnv
  from agents.improved_sac_agent import ImprovedSACAgent

class CorrectedTrainer:
    """Entraîneur avec identification CORRIGÉE"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        if not HAS_TORCH:
            print("❌ PyTorch requis")
            return
        
        print("🔧 Initialisation environnement CORRIGÉ...")
        self.env = CorrectedUltraStableGraspEnv(
            xml_path=config['model_path'],
            max_episode_steps=config['max_episode_steps'],
            block_fingers=config['block_fingers']
        )
        
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
        
        # Métriques de stabilité
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.instability_counts = []
        self.training_metrics = []
        self.total_instabilities = 0
        self.consecutive_stable_episodes = 0
        self.best_stability_streak = 0
        
        print("✅ Entraîneur CORRIGÉ prêt")
    
    def train(self):
        """Entraînement avec identification corrigée"""
        if not HAS_TORCH:
            return
            
        print("\n🔧 DÉBUT ENTRAÎNEMENT IDENTIFICATION CORRIGÉE")
        print("=" * 70)
        print(f"🖐️  Doigts identifiés: {len(self.env.finger_dofs)} DOFs")
        print(f"🔒 Doigts bloqués: {self.env.finger_dofs}")
        print(f"💪 Bras actifs: {self.env.arm_dofs}")
        print(f"🎯 Actions: ±{self.env.action_space.high[0]:.3f}")
        
        start_time = time.time()
        total_episodes = self.config['total_episodes']
        
        for episode in range(total_episodes):
            try:
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                episode_success = False
                episode_instabilities = 0
                
                done = False
                crashed = False
                
                while not done and episode_length < self.config['max_episode_steps']:
                    # Actions ultra-progressives
                    if episode < 3:
                        action = np.zeros(self.env.action_space.shape[0])
                    elif episode < 15:
                        action = self.agent.select_action(obs, evaluate=True) * 0.0001
                    elif episode < 50:
                        action = self.agent.select_action(obs, evaluate=True) * 0.001
                    else:
                        action = self.agent.select_action(obs) * 0.01
                    
                    try:
                        next_obs, reward, terminated, truncated, info = self.env.step(action)
                        done = terminated or truncated
                        
                        if "error" in info:
                            episode_instabilities += 1
                            self.total_instabilities += 1
                            crashed = True
                            
                            print(f"⚠️  CRASH épisode {episode} step {episode_length}: {info['error']}")
                            print(f"   Doigts bloqués: {info.get('blocked_fingers', 'N/A')}")
                            reward = -100.0
                            done = True
                            break
                        
                        if not crashed and episode >= 3:
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
                
                # Tracking stabilité
                if not crashed and episode_instabilities == 0:
                    self.consecutive_stable_episodes += 1
                    self.best_stability_streak = max(
                        self.best_stability_streak, self.consecutive_stable_episodes
                    )
                else:
                    self.consecutive_stable_episodes = 0
                
                # Arrêt si trop d'instabilités
                if self.total_instabilities >= 15:
                    print("🛑 Trop d'instabilités - arrêt")
                    break
                
                # Entraînement très conservateur
                if (len(self.agent.replay_buffer) > self.config['batch_size'] and 
                    not crashed and episode >= 75 and
                    episode % self.config['training_frequency'] == 0):
                    
                    training_info = self.agent.update(self.config['batch_size'])
                    if training_info:
                        self.training_metrics.append(training_info)
                
                # Métriques
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_successes.append(episode_success)
                self.instability_counts.append(episode_instabilities)
                
                # Logging
                if (episode + 1) % self.config['log_interval'] == 0:
                    self._log_corrected_progress(episode + 1, total_episodes, start_time)
                
                # Sauvegarde
                if (episode + 1) % self.config['save_interval'] == 0:
                    self._save_checkpoint(episode + 1)
                
            except Exception as e:
                print(f"❌ Erreur épisode {episode}: {e}")
                continue
        
        # Fin
        total_time = time.time() - start_time
        print(f"\n✅ ENTRAÎNEMENT CORRIGÉ TERMINÉ")
        print(f"   Durée: {total_time/60:.1f}min")
        print(f"   Épisodes: {len(self.episode_rewards)}")
        print(f"   Instabilités totales: {self.total_instabilities}")
        print(f"   Meilleure série stable: {self.best_stability_streak}")
        
        if self.episode_rewards:
            print(f"   Récompense moyenne: {np.mean(self.episode_rewards[-15:]):.2f}")
            print(f"   Longueur moyenne: {np.mean(self.episode_lengths[-15:]):.1f}")
            stable_recent = sum(1 for x in self.instability_counts[-15:] if x == 0)
            print(f"   Épisodes stables récents: {stable_recent}/15")
        
        self._save_final_results()
    
    def _log_corrected_progress(self, episode, total_episodes, start_time):
        """Log avec détails d'identification"""
        recent = min(self.config['log_interval'], len(self.episode_rewards))
        
        if recent > 0:
            rewards = self.episode_rewards[-recent:]
            lengths = self.episode_lengths[-recent:]
            successes = self.episode_successes[-recent:]
            instabilities = self.instability_counts[-recent:]
            
            avg_reward = np.mean(rewards)
            avg_length = np.mean(lengths)
            success_rate = np.mean(successes) * 100
            avg_instabilities = np.mean(instabilities)
            stable_episodes = sum(1 for x in instabilities if x == 0)
            
            elapsed = time.time() - start_time
            
            print(f"\n🔧 PROGRÈS CORRIGÉ - Épisode {episode}/{total_episodes}")
            print("-" * 60)
            print(f"   📊 Récompense: {avg_reward:.2f} ± {np.std(rewards):.2f}")
            print(f"   📏 Longueur: {avg_length:.1f} steps")
            print(f"   ✅ Succès: {success_rate:.1f}%")
            print(f"   🛡️  Stables: {stable_episodes}/{recent}")
            print(f"   🔥 Série actuelle: {self.consecutive_stable_episodes}")
            print(f"   🏆 Record: {self.best_stability_streak}")
            print(f"   ⚠️  Instab. moy: {avg_instabilities:.1f}")
            print(f"   📊 Instab. totales: {self.total_instabilities}")
            print(f"   💾 Buffer: {len(self.agent.replay_buffer)}")
            print(f"   ⏱️  Temps: {elapsed/60:.1f}min")
            print(f"   🖐️  Doigts bloqués: {len(self.env.finger_dofs)}")
            print(f"   💪 Bras actifs: {len(self.env.arm_dofs)}")
            
            if avg_instabilities == 0 and stable_episodes >= recent - 1:
                print("   🟢 ÉTAT: ULTRA-STABLE")
            elif avg_instabilities == 0:
                print("   🟢 ÉTAT: STABLE")
            elif avg_instabilities < 0.3:
                print("   🟡 ÉTAT: QUASI-STABLE")
            else:
                print("   🔴 ÉTAT: INSTABLE")
    
    def _save_checkpoint(self, episode):
        """Sauvegarde avec métriques de stabilité"""
        try:
            path = self.output_dir / "models" / f"corrected_ep_{episode}.pth"
            self.agent.save(path)
            
            metrics = {
                "episode": episode,
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
                "episode_successes": self.episode_successes,
                "instability_counts": self.instability_counts,
                "total_instabilities": self.total_instabilities,
                "consecutive_stable_episodes": self.consecutive_stable_episodes,
                "best_stability_streak": self.best_stability_streak,
                "finger_dofs_blocked": self.env.finger_dofs,
                "arm_dofs_active": self.env.arm_dofs,
                "config": self.config
            }
            
            metrics_path = self.output_dir / "logs" / f"corrected_ep_{episode}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Erreur sauvegarde: {e}")
    
    def _save_final_results(self):
        """Sauvegarde finale complète"""
        try:
            final_model = self.output_dir / "models" / "corrected_final.pth"
            self.agent.save(final_model)
            
            final_metrics = {
                "config": self.config,
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
                "episode_successes": self.episode_successes,
                "instability_counts": self.instability_counts,
                "training_metrics": self.training_metrics,
                "total_instabilities": self.total_instabilities,
                "best_stability_streak": self.best_stability_streak,
                "finger_identification": {
                    "correctly_identified_fingers": self.env.finger_dofs,
                    "total_finger_dofs": len(self.env.finger_dofs),
                    "arm_dofs": self.env.arm_dofs,
                    "total_arm_dofs": len(self.env.arm_dofs)
                },
                "final_stats": {
                    "total_episodes": len(self.episode_rewards),
                    "avg_reward": float(np.mean(self.episode_rewards[-15:])) if self.episode_rewards else 0,
                    "success_rate": float(np.mean(self.episode_successes[-15:])) if self.episode_successes else 0,
                    "avg_length": float(np.mean(self.episode_lengths[-15:])) if self.episode_lengths else 0,
                    "stability_rate": float(sum(1 for x in self.instability_counts[-15:] if x == 0) / min(15, len(self.instability_counts))) if self.instability_counts else 0,
                    "total_instabilities": self.total_instabilities,
                    "best_stability_streak": self.best_stability_streak
                }
            }
            
            final_path = self.output_dir / "logs" / "corrected_final.json"
            with open(final_path, 'w') as f:
                json.dump(final_metrics, f, indent=2)
            
            print(f"✅ Résultats CORRIGÉS: {self.output_dir}")
            
        except Exception as e:
            print(f"⚠️  Erreur finale: {e}")

    def load_corrected_config():
        """Configuration ultra-conservative pour identification corrigée"""
        return {
            'model_path': 'results/g1_combined.xml',
            'max_episode_steps': 25,       # Très court
            'block_fingers': True,         # Doigts bloqués
            'total_episodes': 100,
            'learning_rate': 5e-6,         # Très bas
            'batch_size': 8,               # Très petit
            'buffer_size': 1000,           # Petit
            'training_frequency': 75,      # Très rare
            'hidden_sizes': [16, 16],      # Très petit
            'gamma': 0.9,
            'tau': 0.0005,                 # Très lent
            'log_interval': 3,             # Très fréquent
            'save_interval': 15,
            'output_dir': 'corrected_results'
        }

def main():
  """Point d'entrée corrigé"""
  parser = argparse.ArgumentParser(description='Entraînement CORRIGÉ identification exacte')
  parser.add_argument('--episodes', type=int, default=100, help='Épisodes')
  parser.add_argument('--max-steps', type=int, default=25, help='Steps max')
  parser.add_argument('--output', type=str, default='corrected_results', help='Sortie')
  
  args = parser.parse_args()
  
  config = CorrectedTrainer.load_corrected_config()
  config['total_episodes'] = args.episodes
  config['max_episode_steps'] = args.max_steps
  config['output_dir'] = args.output
  
  print("🔧 ENTRAÎNEMENT CORRIGÉ G1")
  print("=" * 50)
  print(f"Épisodes: {config['total_episodes']}")
  print(f"Steps max: {config['max_episode_steps']}")
  print(f"Actions: ±{0.03}")
  print(f"Sortie: {config['output_dir']}")
  
  if not Path(config['model_path']).exists():
      print(f"❌ Modèle manquant: {config['model_path']}")
      return
  
  try:
      trainer = CorrectedTrainer(config)
      trainer.train()
      
  except KeyboardInterrupt:
      print("\n⏹️  Arrêt")
  except Exception as e:
      print(f"\n❌ Erreur: {e}")
      import traceback
      traceback.print_exc()
  finally:
      print("\n🏁 Fin")

if __name__ == "__main__":
  main()

