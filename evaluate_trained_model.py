#!/usr/bin/env python3
"""
🎥 ÉVALUATION ET CORRECTIONS CRITIQUES
===================================

1. Script d'évaluation avec vidéos
2. Corrections pour débloquer l'apprentissage figé
3. Nouvelles stratégies d'exploration
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
from PIL import Image
import json
from stable_baselines3 import TD3
from envs.optimized_grasp_env import OptimizedGraspEnv

# ============================================================================
# 1. SCRIPT D'ÉVALUATION COMPLET
# ============================================================================

def evaluate_trained_model(model_path: str, num_episodes: int = 10, 
                          save_video: bool = True, results_dir: str = "evaluation_results"):
    """
    Évalue un modèle entraîné et génère des vidéos
    """
    
    print(f"🎯 ÉVALUATION DU MODÈLE: {model_path}")
    print("=" * 50)
    
    # Créer dossiers de résultats
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    (results_path / "videos").mkdir(exist_ok=True)
    (results_path / "plots").mkdir(exist_ok=True)
    
    # Charger le modèle
    try:
        model = TD3.load(model_path)
        print(f"✅ Modèle chargé: {model_path}")
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        return
    
    # Créer environnement d'évaluation
    try:
        env = OptimizedGraspEnv(render_mode="rgb_array")
        print("✅ Environnement créé")
    except Exception as e:
        print(f"❌ Erreur environnement: {e}")
        return
    
    # Métriques d'évaluation
    episode_rewards = []
    episode_contacts = []
    episode_distances = []
    episode_frames = []  # Pour les vidéos
    
    print(f"\n📊 Évaluation sur {num_episodes} épisodes...")
    
    for episode in range(num_episodes):
        print(f"\n🎮 Épisode {episode + 1}/{num_episodes}")
        
        obs, _ = env.reset()
        episode_reward = 0
        max_contacts = 0
        final_distance = float('inf')
        frames = []
        
        step_rewards = []  # Pour analyser la progression
        step_distances = []
        step_contacts = []
        
        for step in range(500):
            # Prédiction du modèle
            action, _ = model.predict(obs, deterministic=True)
            
            # Step dans l'environnement
            obs, reward, terminated, _, info = env.step(action)
            
            # Collecter métriques
            episode_reward += reward
            current_contacts = info.get('contact_count', 0)
            current_distance = info.get('distance', float('inf'))
            
            max_contacts = max(max_contacts, current_contacts)
            final_distance = current_distance
            
            # Historique pour analyse
            step_rewards.append(reward)
            step_distances.append(current_distance)
            step_contacts.append(current_contacts)
            
            # Capturer frames pour vidéo
            if save_video:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            
            # Debug périodique
            if step % 100 == 0:
                print(f"   Step {step}: dist={current_distance:.4f}, "
                      f"contacts={current_contacts}, reward={reward:.2f}")
            
            if terminated:
                print(f"   ⏹️ Terminé au step {step}")
                break
        
        # Sauvegarder métriques d'épisode
        episode_rewards.append(episode_reward)
        episode_contacts.append(max_contacts)
        episode_distances.append(final_distance)
        
        print(f"   📈 Résultats: Reward={episode_reward:.1f}, "
              f"Max contacts={max_contacts}, Distance finale={final_distance:.4f}")
        
        # Sauvegarder vidéo d'épisode
        if save_video and frames:
            video_path = results_path / "videos" / f"episode_{episode+1}.mp4"
            save_episode_video(frames, video_path)
            print(f"   🎥 Vidéo sauvegardée: {video_path.name}")
        
        # Analyser pourquoi pas de contacts
        if max_contacts == 0:
            min_distance = min(step_distances)
            print(f"   ⚠️ AUCUN CONTACT - Distance minimale atteinte: {min_distance:.4f}")
            if min_distance < 0.1:
                print("      → Robot très proche mais collision detection échoue")
            elif min_distance > 0.15:
                print("      → Robot n'arrive pas à s'approcher")
    
    # Statistiques finales
    print(f"\n📈 STATISTIQUES FINALES:")
    print(f"   Reward moyen: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"   Contacts moyens: {np.mean(episode_contacts):.1f}")
    print(f"   Distance finale: {np.mean(episode_distances):.4f} ± {np.std(episode_distances):.4f}")
    
    success_rate = sum(1 for c in episode_contacts if c >= 2) / len(episode_contacts) * 100
    print(f"   Taux de succès: {success_rate:.1f}% (≥2 contacts)")
    
    # Diagnostic du problème
    diagnose_training_issues(episode_rewards, episode_contacts, episode_distances)
    
    # Sauvegarder résultats
    save_evaluation_results(results_path, episode_rewards, episode_contacts, episode_distances)
    
    env.close()
    return episode_rewards, episode_contacts, episode_distances

def save_episode_video(frames, video_path):
    """Sauvegarde une vidéo d'épisode"""
    try:
        # Convertir en images PIL si nécessaire
        if frames and hasattr(frames[0], 'dtype'):
            pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
        else:
            pil_frames = frames
        
        # Sauvegarder à 30 FPS
        imageio.mimsave(str(video_path), pil_frames, fps=30)
        
    except Exception as e:
        print(f"❌ Erreur sauvegarde vidéo: {e}")

def diagnose_training_issues(rewards, contacts, distances):
    """Diagnostique les problèmes d'entraînement"""
    
    print(f"\n🔍 DIAGNOSTIC DES PROBLÈMES:")
    
    # Analyse des rewards
    reward_std = np.std(rewards)
    if reward_std < 0.01:
        print("   ❌ PROBLÈME: Rewards identiques → Comportement déterministe figé")
        print("      → Solution: Augmenter l'exploration (noise, epsilon)")
    
    # Analyse des contacts
    if max(contacts) == 0:
        print("   ❌ PROBLÈME: Aucun contact → Robot n'apprend pas à saisir")
        print("      → Solutions:")
        print("        1. Récompense plus agressive pour proximité")
        print("        2. Vérifier collision detection")
        print("        3. Augmenter la taille des doigts/cube")
    
    # Analyse des distances
    distance_std = np.std(distances)
    min_dist = min(distances)
    
    if distance_std < 0.005:
        print("   ❌ PROBLÈME: Distance constante → Robot figé dans position locale")
        print("      → Solution: Restart training avec exploration forcée")
    
    if min_dist > 0.15:
        print("   ❌ PROBLÈME: Robot trop loin → N'apprend pas à s'approcher")
        print("      → Solution: Curriculum learning ou position initiale plus proche")

def save_evaluation_results(results_path, rewards, contacts, distances):
    """Sauvegarde les résultats d'évaluation"""
    
    # Données JSON
    results_data = {
        "episode_rewards": rewards,
        "episode_contacts": contacts,
        "episode_distances": distances,
        "statistics": {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_contacts": float(np.mean(contacts)),
            "mean_distance": float(np.mean(distances)),
            "success_rate": float(sum(1 for c in contacts if c >= 2) / len(contacts) * 100)
        }
    }
    
    with open(results_path / "evaluation_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Graphiques
    create_evaluation_plots(results_path, rewards, contacts, distances)
    
    print(f"💾 Résultats sauvegardés dans {results_path}")

def create_evaluation_plots(results_path, rewards, contacts, distances):
    """Crée des graphiques d'évaluation"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Rewards par épisode
    axes[0, 0].bar(range(1, len(rewards)+1), rewards)
    axes[0, 0].set_title('Rewards par épisode')
    axes[0, 0].set_xlabel('Épisode')
    axes[0, 0].set_ylabel('Reward')
    
    # Contacts par épisode
    axes[0, 1].bar(range(1, len(contacts)+1), contacts, color='orange')
    axes[0, 1].set_title('Contacts max par épisode')
    axes[0, 1].set_xlabel('Épisode')
    axes[0, 1].set_ylabel('Contacts')
    
    # Distances par épisode
    axes[1, 0].bar(range(1, len(distances)+1), distances, color='red')
    axes[1, 0].set_title('Distance finale par épisode')
    axes[1, 0].set_xlabel('Épisode')
    axes[1, 0].set_ylabel('Distance (m)')
    
    # Histogramme des distances
    axes[1, 1].hist(distances, bins=10, color='green', alpha=0.7)
    axes[1, 1].set_title('Distribution des distances finales')
    axes[1, 1].set_xlabel('Distance (m)')
    axes[1, 1].set_ylabel('Fréquence')
    
    plt.tight_layout()
    plt.savefig(results_path / "plots" / "evaluation_summary.png", dpi=300)
    plt.close()

# ============================================================================
# 2. CORRECTIONS CRITIQUES POUR DÉBLOQUER L'APPRENTISSAGE
# ============================================================================

def create_fixed_environment():
    """Crée un environnement avec corrections critiques"""
    
    class UnstuckGraspEnv(OptimizedGraspEnv):
        """Environnement avec corrections anti-blocage"""
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            
            # Variables anti-blocage
            self.stuck_counter = 0
            self.last_positions = []
            self.force_exploration = False
            self.exploration_steps = 0
            
        def _compute_reward(self, positions):
            """Récompense anti-blocage avec exploration forcée"""
            
            dist = positions['palm_to_cube_dist']
            cube_vel = positions['cube_velocity']
            contact_count = positions['contact_count']
            
            reward = 0.0
            
            # 1. RÉCOMPENSE DISTANCE TRÈS AGRESSIVE
            if dist < 0.05:
                reward += 100.0  # ÉNORME bonus proximité
            elif dist < 0.1:
                reward += 50.0
            elif dist < 0.15:
                reward += 20.0
            elif dist < 0.2:
                reward += 5.0
            else:
                reward -= 10.0  # Pénalité être loin
            
            # 2. BONUS CONTACT GIGANTESQUE
            if contact_count >= 1:
                reward += 200.0  # Premier contact = jackpot
            if contact_count >= 2:
                reward += 500.0  # Deux contacts = super jackpot
            if contact_count >= 3:
                reward += 1000.0  # Grasp = méga jackpot
            
            # 3. DÉTECTION DE BLOCAGE
            self.last_positions.append(dist)
            if len(self.last_positions) > 100:
                self.last_positions.pop(0)
                
                # Si distance varie peu sur 100 steps = BLOQUÉ
                if len(self.last_positions) == 100:
                    distance_variation = np.std(self.last_positions)
                    if distance_variation < 0.01:  # Très peu de variation
                        self.stuck_counter += 1
                        if self.stuck_counter > 5:  # Bloqué depuis 500 steps
                            reward -= 50.0  # GROSSE pénalité blocage
                            self.force_exploration = True
                            self.exploration_steps = 200
                            print(f"🚨 ROBOT BLOQUÉ! Force exploration activée")
                    else:
                        self.stuck_counter = 0
            
            # 4. BONUS EXPLORATION FORCÉE
            if self.force_exploration:
                self.exploration_steps -= 1
                reward += 10.0  # Bonus pour bouger quand bloqué
                if self.exploration_steps <= 0:
                    self.force_exploration = False
            
            # 5. BONUS VARIABILITÉ (anti-déterminisme)
            if len(self.last_positions) >= 10:
                recent_variation = np.std(self.last_positions[-10:])
                reward += recent_variation * 20.0  # Bonus bouger
            
            # 6. PÉNALITÉ TEMPS RÉDUITE
            reward -= 0.1  # Très faible
            
            return float(reward)
        
        def step(self, action):
            """Step avec anti-blocage"""
            
            # EXPLORATION FORCÉE si robot bloqué
            if self.force_exploration:
                # Ajouter du bruit aléatoire à l'action
                noise = np.random.normal(0, 0.5, action.shape)
                action = np.clip(action + noise, -1.0, 1.0)
                
                # Occasionnellement action complètement aléatoire
                if np.random.random() < 0.3:
                    action = np.random.uniform(-1.0, 1.0, action.shape)
            
            return super().step(action)
        
        def reset(self, seed=None, options=None):
            """Reset avec position aléatoire pour éviter blocage"""
            
            obs, info = super().reset(seed=seed, options=options)
            
            # Reset variables anti-blocage
            self.stuck_counter = 0
            self.last_positions = []
            self.force_exploration = False
            self.exploration_steps = 0
            
            # POSITION INITIALE ALÉATOIRE pour éviter convergence locale
            if hasattr(self.data, 'qpos') and len(self.data.qpos) > 4:
                # Randomiser position initiale du bras
                for i in range(min(4, len(self.data.qpos))):
                    self.data.qpos[i] += np.random.normal(0, 0.2)
            
            return obs, info
    
    return UnstuckGraspEnv

# ============================================================================
# 3. NOUVEAU TRAINING AVEC EXPLORATION FORCÉE
# ============================================================================

def retrain_with_forced_exploration(model_path=None):
    """Nouveau training avec exploration forcée"""
    
    print("🚀 NOUVEAU TRAINING AVEC EXPLORATION FORCÉE")
    print("=" * 50)
    
    # Environnement corrigé
    UnstuckEnv = create_fixed_environment()
    env = UnstuckEnv(render_mode="rgb_array")
    
    # Hyperparamètres TRÈS exploratoires
    config = {
        'learning_rate': 1e-3,      # Plus élevé
        'batch_size': 64,           # Plus petit = plus instable = plus exploration
        'buffer_size': 50_000,      # Plus petit
        'gamma': 0.9,              # Moins patient
        'tau': 0.1,                # Mise à jour rapide
        'noise_std': 1.0,          # BEAUCOUP de bruit!
    }
    
    # Bruit d'action très élevé
    from stable_baselines3.common.noise import NormalActionNoise
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=config['noise_std'] * np.ones(n_actions)
    )
    
    # Modèle avec exploration maximale
    if model_path and Path(model_path).exists():
        print(f"📚 Chargement et fine-tuning du modèle: {model_path}")
        model = TD3.load(model_path, env=env)
        model.action_noise = action_noise  # Nouveau bruit
    else:
        print("🆕 Nouveau modèle avec exploration maximale")
        model = TD3(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            buffer_size=config['buffer_size'],
            gamma=config['gamma'],
            tau=config['tau'],
            verbose=1
        )
    
    print("⚡ Configuration exploratory training:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Training court mais intensif
    try:
        print("\n🎯 Training exploratoire (20,000 steps)...")
        model.learn(
            total_timesteps=20_000,
            progress_bar=True
        )
        
        # Sauvegarder modèle débloqué
        unstuck_path = "optimized_results/models/unstuck_model"
        model.save(unstuck_path)
        print(f"💾 Modèle débloqué sauvegardé: {unstuck_path}")
        
        # Test rapide
        print("\n🧪 Test du modèle débloqué...")
        test_rewards, test_contacts, test_distances = [], [], []
        
        for episode in range(3):
            obs, _ = env.reset()
            episode_reward = 0
            max_contacts = 0
            
            for step in range(200):
                action, _ = model.predict(obs, deterministic=False)  # Non-déterministe!
                obs, reward, terminated, _, info = env.step(action)
                
                episode_reward += reward
                max_contacts = max(max_contacts, info.get('contact_count', 0))
                
                if terminated:
                    break
            
            test_rewards.append(episode_reward)
            test_contacts.append(max_contacts)
            test_distances.append(info.get('distance', 0))
            
            print(f"   Test {episode+1}: reward={episode_reward:.1f}, "
                  f"contacts={max_contacts}, dist={info.get('distance', 0):.4f}")
        
        # Vérifier si débloqué
        reward_variation = np.std(test_rewards)
        if reward_variation > 1.0:
            print("✅ Robot débloqué! Variation des rewards détectée")
        else:
            print("❌ Robot encore bloqué...")
        
        return unstuck_path
        
    except Exception as e:
        print(f"❌ Erreur training: {e}")
        return None
    finally:
        env.close()

# ============================================================================
# 4. SCRIPT PRINCIPAL D'ÉVALUATION
# ============================================================================

if __name__ == "__main__":
    
    print("🎯 SCRIPT D'ÉVALUATION ET CORRECTION")
    print("=" * 50)
    
    # Chemins des modèles
    final_model = "optimized_results/models/final_model"
    best_model = "optimized_results/models/best_model"
    
    # 1. Évaluer le modèle existant
    if Path(final_model + ".zip").exists():
        print("📊 Évaluation du modèle final...")
        evaluate_trained_model(final_model, num_episodes=5, save_video=True)
    
    # 2. Tentative de déblocage
    print("\n🔧 Tentative de déblocage du modèle...")
    unstuck_model = retrain_with_forced_exploration(final_model)
    
    # 3. Évaluer le modèle débloqué
    if unstuck_model:
        print("\n📊 Évaluation du modèle débloqué...")
        evaluate_trained_model(unstuck_model, num_episodes=5, 
                              save_video=True, results_dir="unstuck_evaluation")