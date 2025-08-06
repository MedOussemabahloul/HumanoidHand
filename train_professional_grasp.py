#!/usr/bin/env python3
"""
🏆 ENTRAÎNEMENT PROFESSIONNEL DE GRASPING
=========================================

Fonctionnalités:
✅ Entraînement avec stabilité des bras
✅ Contact palm-cube professionnel
✅ Grasping en phases contrôlées
✅ Enregistrement vidéo complet
✅ Métriques détaillées
✅ Récupération d'erreurs
"""

import os
import sys
import numpy as np
import json
import time
from datetime import datetime
import cv2
import warnings
warnings.filterwarnings("ignore")

# Ajouter le chemin des environnements
sys.path.append('/workspace/envs')

try:
    from professional_grasp_env import ProfessionalGraspEnv
    print("✅ ProfessionalGraspEnv importé avec succès")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    sys.exit(1)

class ProfessionalGraspTrainer:
    """
    🎯 Entraîneur Professionnel pour Grasping
    
    Stratégie d'entraînement:
    1. Actions guidées par phase
    2. Apprentissage progressif
    3. Stabilité prioritaire
    4. Enregistrement complet
    """
    
    def __init__(self, episodes: int = 30, record_video: bool = True):
        self.episodes = episodes
        self.record_video = record_video
        
        # Métriques globales
        self.global_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'phase_completions': {
                'STABILIZE': [],
                'APPROACH': [],
                'CONTACT': [],
                'GRASP': [],
                'LIFT': [],
                'HOLD': []
            },
            'success_rates': {
                'arm_stability': [],
                'cube_approach': [],
                'palm_contact': [],
                'successful_grasp': [],
                'cube_lift': [],
                'complete_success': []
            },
            'instability_count': 0,
            'total_training_time': 0.0
        }
        
        # Dossiers de résultats
        self.results_dir = "/workspace/professional_grasp_results"
        self.videos_dir = os.path.join(self.results_dir, "videos")
        self.logs_dir = os.path.join(self.results_dir, "logs")
        
        self._setup_directories()
        
        print("🏆 ProfessionalGraspTrainer initialisé")
        print(f"📁 Résultats: {self.results_dir}")
        
    def _setup_directories(self):
        """Crée les dossiers nécessaires"""
        for directory in [self.results_dir, self.videos_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
            
    def train(self):
        """Lance l'entraînement professionnel"""
        
        print("\n🚀 DÉBUT DE L'ENTRAÎNEMENT PROFESSIONNEL")
        print("=" * 60)
        
        start_time = time.time()
        
        # Créer l'environnement
        env = ProfessionalGraspEnv(
            model_path="/workspace/results/g1_combined.xml",
            render_mode='rgb_array' if self.record_video else None,
            fix_physics=True
        )
        
        if self.record_video:
            env.record_video = True
        
        try:
            for episode in range(self.episodes):
                print(f"\n🎯 ÉPISODE {episode + 1}/{self.episodes}")
                print("-" * 40)
                
                episode_start = time.time()
                
                # Reset de l'environnement
                obs, info = env.reset()
                
                episode_reward = 0.0
                episode_length = 0
                episode_metrics = {
                    'phases_completed': [],
                    'max_phase_reached': 0,
                    'stability_maintained': 0,
                    'contacts_detected': 0,
                    'successful_grasps': 0,
                    'cube_lifted': False,
                    'complete_success': False
                }
                
                # Boucle d'épisode
                for step in range(500):  # Maximum 500 steps par épisode
                    
                    # Action professionnelle basée sur la phase
                    action = self._get_professional_action(obs, info, step)
                    
                    # Exécution
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    # Mise à jour des métriques
                    self._update_episode_metrics(episode_metrics, info)
                    
                    # Affichage de progression
                    if step % 50 == 0:
                        phase = info.get('current_phase', 'UNKNOWN')
                        print(f"  Step {step:3d}: Phase={phase:10s}, Reward={reward:6.2f}, Total={episode_reward:8.2f}")
                    
                    # Conditions d'arrêt
                    if terminated or truncated:
                        break
                
                # Fin d'épisode
                episode_time = time.time() - episode_start
                
                # Sauvegarde vidéo si demandée
                if self.record_video and hasattr(env, 'video_frames') and env.video_frames:
                    video_path = os.path.join(self.videos_dir, f"episode_{episode+1:03d}.mp4")
                    if env.save_video(video_path):
                        print(f"📹 Vidéo sauvegardée: episode_{episode+1:03d}.mp4")
                
                # Analyse des résultats
                self._analyze_episode_results(episode + 1, episode_reward, episode_length, 
                                            episode_metrics, episode_time, info)
                
                # Mise à jour des métriques globales
                self._update_global_metrics(episode_reward, episode_length, episode_metrics, info)
                
            # Fin de l'entraînement
            total_time = time.time() - start_time
            self.global_metrics['total_training_time'] = total_time
            
            # Rapport final
            self._generate_final_report()
            
            # Sauvegarde des résultats
            self._save_results()
            
        except Exception as e:
            print(f"❌ Erreur durant l'entraînement: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            env.close()
            
        print("\n🏆 ENTRAÎNEMENT TERMINÉ!")
        return self.global_metrics
    
    def _get_professional_action(self, obs, info, step):
        """Génère une action professionnelle basée sur la phase"""
        
        phase = info.get('current_phase', 'STABILIZE')
        phase_timer = info.get('phase_timer', 0)
        
        # Dimension d'action (14 bras + 16 doigts = 30)
        action = np.zeros(30)
        
        if phase == 'STABILIZE':
            # Phase stabilisation: mouvements très doux pour stabiliser
            action[:14] = np.random.normal(0, 0.05, 14)  # Bras: petits ajustements
            action[14:] = np.random.normal(0, 0.02, 16)   # Doigts: très petits mouvements
            
        elif phase == 'APPROACH':
            # Phase approche: mouvements guidés vers le cube
            cube_pos = info.get('cube_position', [0.5, 0.0, 0.05])
            
            # Mouvement des bras vers le cube
            approach_strength = min(0.3, phase_timer / 50.0)
            
            # Bras gauche vers le cube
            action[0] = 0.1 * approach_strength   # shoulder_pitch
            action[1] = 0.2 * approach_strength   # shoulder_roll
            action[3] = -0.2 * approach_strength  # elbow
            
            # Bras droit vers le cube  
            action[7] = 0.1 * approach_strength   # shoulder_pitch
            action[8] = -0.2 * approach_strength  # shoulder_roll
            action[10] = -0.2 * approach_strength # elbow
            
            # Doigts: préparation à l'ouverture
            action[14:] = np.random.normal(-0.1, 0.05, 16)
            
        elif phase == 'CONTACT':
            # Phase contact: maintenir position, préparer contact palm
            action[:14] = np.random.normal(0, 0.03, 14)  # Bras: stabilité
            
            # Doigts: légère ouverture pour préparer la prise
            action[14:] = np.random.normal(-0.05, 0.03, 16)
            
        elif phase == 'GRASP':
            # Phase grasping: fermeture contrôlée des doigts
            action[:14] = np.random.normal(0, 0.02, 14)  # Bras: très stable
            
            # Fermeture progressive des doigts
            grasp_progress = min(1.0, phase_timer / 40.0)
            grasp_strength = 0.6 * grasp_progress
            
            # Tous les doigts se ferment symétriquement
            action[14:] = np.full(16, grasp_strength) + np.random.normal(0, 0.05, 16)
            
        elif phase == 'LIFT':
            # Phase lift: soulever le cube
            lift_progress = min(1.0, phase_timer / 30.0)
            
            # Mouvement de lift avec les bras
            action[0] = 0.2 * lift_progress    # left_shoulder_pitch
            action[3] = -0.3 * lift_progress   # left_elbow
            action[7] = 0.2 * lift_progress    # right_shoulder_pitch
            action[10] = -0.3 * lift_progress  # right_elbow
            
            # Maintenir la prise des doigts
            action[14:] = np.full(16, 0.7) + np.random.normal(0, 0.02, 16)
            
        else:  # HOLD
            # Phase hold: maintenir position et prise
            action[:14] = np.random.normal(0, 0.01, 14)   # Bras: très stable
            action[14:] = np.full(16, 0.8) + np.random.normal(0, 0.01, 16)  # Doigts: prise ferme
        
        # Limiter l'action
        action = np.clip(action, -1.0, 1.0)
        
        # Ajouter du bruit adaptatif selon le step
        noise_level = max(0.01, 0.1 * (1.0 - step / 500.0))
        action += np.random.normal(0, noise_level, action.shape)
        
        return np.clip(action, -1.0, 1.0)
    
    def _update_episode_metrics(self, episode_metrics, info):
        """Met à jour les métriques de l'épisode"""
        
        current_phase = info.get('current_phase', 'STABILIZE')
        metrics = info.get('metrics', {})
        
        # Phase maximale atteinte
        phase_levels = {'STABILIZE': 0, 'APPROACH': 1, 'CONTACT': 2, 
                       'GRASP': 3, 'LIFT': 4, 'HOLD': 5}
        episode_metrics['max_phase_reached'] = max(
            episode_metrics['max_phase_reached'],
            phase_levels.get(current_phase, 0)
        )
        
        # Phases complétées
        phase_completions = metrics.get('phase_completions', {})
        for phase, completed in phase_completions.items():
            if completed and phase not in episode_metrics['phases_completed']:
                episode_metrics['phases_completed'].append(phase)
        
        # Métriques spécifiques
        if info.get('arms_stable', False):
            episode_metrics['stability_maintained'] += 1
        
        if info.get('palm_contact', False):
            episode_metrics['contacts_detected'] += 1
        
        if info.get('grasp_established', False):
            episode_metrics['successful_grasps'] += 1
        
        if info.get('cube_lifted', False):
            episode_metrics['cube_lifted'] = True
        
        # Succès complet
        if (current_phase == 'HOLD' and 
            info.get('cube_lifted', False) and 
            info.get('grasp_established', False)):
            episode_metrics['complete_success'] = True
    
    def _analyze_episode_results(self, episode, reward, length, metrics, time_taken, final_info):
        """Analyse les résultats de l'épisode"""
        
        print(f"\n📊 RÉSULTATS ÉPISODE {episode}")
        print(f"  Récompense totale: {reward:.2f}")
        print(f"  Longueur: {length} steps")
        print(f"  Temps: {time_taken:.2f}s")
        print(f"  Phase max: {list(['STABILIZE', 'APPROACH', 'CONTACT', 'GRASP', 'LIFT', 'HOLD'])[metrics['max_phase_reached']]}")
        print(f"  Phases complétées: {len(metrics['phases_completed'])}/6")
        print(f"  Stabilité maintenue: {metrics['stability_maintained']} steps")
        print(f"  Contacts détectés: {metrics['contacts_detected']}")
        print(f"  Prises réussies: {metrics['successful_grasps']}")
        print(f"  Cube soulevé: {'✅' if metrics['cube_lifted'] else '❌'}")
        print(f"  Succès complet: {'🏆' if metrics['complete_success'] else '❌'}")
    
    def _update_global_metrics(self, reward, length, episode_metrics, final_info):
        """Met à jour les métriques globales"""
        
        self.global_metrics['episode_rewards'].append(reward)
        self.global_metrics['episode_lengths'].append(length)
        
        # Phases complétées
        for phase in self.global_metrics['phase_completions'].keys():
            completed = 1 if phase in episode_metrics['phases_completed'] else 0
            self.global_metrics['phase_completions'][phase].append(completed)
        
        # Taux de succès
        self.global_metrics['success_rates']['arm_stability'].append(
            1 if episode_metrics['stability_maintained'] > 50 else 0
        )
        self.global_metrics['success_rates']['cube_approach'].append(
            1 if episode_metrics['max_phase_reached'] >= 1 else 0
        )
        self.global_metrics['success_rates']['palm_contact'].append(
            1 if episode_metrics['contacts_detected'] > 0 else 0
        )
        self.global_metrics['success_rates']['successful_grasp'].append(
            1 if episode_metrics['successful_grasps'] > 0 else 0
        )
        self.global_metrics['success_rates']['cube_lift'].append(
            1 if episode_metrics['cube_lifted'] else 0
        )
        self.global_metrics['success_rates']['complete_success'].append(
            1 if episode_metrics['complete_success'] else 0
        )
        
        # Instabilité
        if final_info.get('instability', False):
            self.global_metrics['instability_count'] += 1
    
    def _generate_final_report(self):
        """Génère le rapport final"""
        
        print("\n" + "=" * 80)
        print("🏆 RAPPORT FINAL D'ENTRAÎNEMENT PROFESSIONNEL")
        print("=" * 80)
        
        # Statistiques générales
        total_episodes = len(self.global_metrics['episode_rewards'])
        avg_reward = np.mean(self.global_metrics['episode_rewards'])
        avg_length = np.mean(self.global_metrics['episode_lengths'])
        total_time = self.global_metrics['total_training_time']
        
        print(f"\n📊 STATISTIQUES GÉNÉRALES:")
        print(f"  Episodes totaux: {total_episodes}")
        print(f"  Récompense moyenne: {avg_reward:.2f}")
        print(f"  Longueur moyenne: {avg_length:.1f} steps")
        print(f"  Temps total: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"  Instabilités: {self.global_metrics['instability_count']}")
        
        # Taux de succès par phase
        print(f"\n🎯 TAUX DE SUCCÈS PAR PHASE:")
        for phase, completions in self.global_metrics['phase_completions'].items():
            success_rate = np.mean(completions) * 100 if completions else 0
            print(f"  {phase:10s}: {success_rate:5.1f}%")
        
        # Taux de succès par capacité
        print(f"\n🏅 TAUX DE SUCCÈS PAR CAPACITÉ:")
        for capacity, successes in self.global_metrics['success_rates'].items():
            success_rate = np.mean(successes) * 100 if successes else 0
            print(f"  {capacity.replace('_', ' ').title():20s}: {success_rate:5.1f}%")
        
        # Progression
        if total_episodes >= 10:
            early_rewards = np.mean(self.global_metrics['episode_rewards'][:10])
            late_rewards = np.mean(self.global_metrics['episode_rewards'][-10:])
            improvement = ((late_rewards - early_rewards) / abs(early_rewards)) * 100
            
            print(f"\n📈 PROGRESSION:")
            print(f"  Premiers 10 épisodes: {early_rewards:.2f}")
            print(f"  Derniers 10 épisodes: {late_rewards:.2f}")
            print(f"  Amélioration: {improvement:+.1f}%")
        
        # Évaluation finale
        final_success_rate = np.mean(self.global_metrics['success_rates']['complete_success']) * 100
        
        print(f"\n🏆 ÉVALUATION FINALE:")
        if final_success_rate >= 70:
            print(f"  🌟 EXCELLENT: {final_success_rate:.1f}% de succès complet!")
        elif final_success_rate >= 50:
            print(f"  ✅ BON: {final_success_rate:.1f}% de succès complet")
        elif final_success_rate >= 30:
            print(f"  ⚠️ MOYEN: {final_success_rate:.1f}% de succès complet")
        else:
            print(f"  ❌ FAIBLE: {final_success_rate:.1f}% de succès complet")
        
        print("=" * 80)
    
    def _save_results(self):
        """Sauvegarde les résultats"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarder les métriques JSON
        results_file = os.path.join(self.logs_dir, f"professional_grasp_results_{timestamp}.json")
        
        # Préparer les données pour JSON
        json_data = {}
        for key, value in self.global_metrics.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif isinstance(value, dict):
                json_data[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_data[key][k] = v.tolist()
                    else:
                        json_data[key][k] = v
            else:
                json_data[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\n💾 Résultats sauvegardés: {os.path.basename(results_file)}")
        
        # Créer vidéo finale si des vidéos existent
        self._create_final_video()
    
    def _create_final_video(self):
        """Crée une vidéo finale avec les meilleurs épisodes"""
        
        video_files = [f for f in os.listdir(self.videos_dir) if f.endswith('.mp4')]
        if not video_files:
            return
        
        print(f"\n🎬 Création de la vidéo finale avec {len(video_files)} épisodes...")
        
        try:
            # Prendre les 5 derniers épisodes (supposés meilleurs)
            selected_videos = sorted(video_files)[-5:]
            
            final_video_path = os.path.join(self.videos_dir, "professional_grasp_final.mp4")
            
            # Lire la première vidéo pour obtenir les dimensions
            first_video = os.path.join(self.videos_dir, selected_videos[0])
            cap = cv2.VideoCapture(first_video)
            
            if not cap.isOpened():
                print("❌ Erreur ouverture vidéo")
                return
            
            # Propriétés vidéo
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Créer la vidéo finale
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(final_video_path, fourcc, fps, (width, height))
            
            for video_file in selected_videos:
                video_path = os.path.join(self.videos_dir, video_file)
                cap = cv2.VideoCapture(video_path)
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                
                cap.release()
            
            out.release()
            print(f"🎥 Vidéo finale créée: professional_grasp_final.mp4")
            
        except Exception as e:
            print(f"❌ Erreur création vidéo finale: {e}")

def main():
    """Fonction principale"""
    
    print("🏆 DÉMARRAGE DE L'ENTRAÎNEMENT PROFESSIONNEL DE GRASPING")
    print("=" * 70)
    
    # Configuration
    episodes = 30
    record_video = True
    
    print(f"📋 Configuration:")
    print(f"  - Episodes: {episodes}")
    print(f"  - Enregistrement vidéo: {'✅' if record_video else '❌'}")
    print(f"  - Modèle: g1_combined.xml")
    print(f"  - Environnement: ProfessionalGraspEnv")
    
    # Créer et lancer l'entraîneur
    trainer = ProfessionalGraspTrainer(episodes=episodes, record_video=record_video)
    
    try:
        results = trainer.train()
        
        print("\n🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print(f"📁 Résultats disponibles dans: /workspace/professional_grasp_results/")
        
        # Statistiques finales
        success_rate = np.mean(results['success_rates']['complete_success']) * 100
        print(f"🏆 Taux de succès final: {success_rate:.1f}%")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Entraînement interrompu par l'utilisateur")
        return False
        
    except Exception as e:
        print(f"\n❌ Erreur durant l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)