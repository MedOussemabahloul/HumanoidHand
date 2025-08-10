#!/usr/bin/env python3
"""
🎥 ÉVALUATION ET GÉNÉRATION VIDÉO - ULTRA-ROBUSTE
==================================================

Script pour évaluer le modèle entraîné et générer des vidéos de démonstration.
Compatible avec tous les formats et environnements.
"""

import os
import sys
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import json
import argparse

# Imports ML
try:
    from stable_baselines3 import TD3
    print("✅ Stable-Baselines3 importé")
except ImportError as e:
    print(f"❌ Erreur import TD3: {e}")
    sys.exit(1)

# Imports vidéo
try:
    import imageio
    from PIL import Image
    import matplotlib.pyplot as plt
    print("✅ Outils vidéo et graphiques importés")
except ImportError as e:
    print(f"⚠️ Outils vidéo limités: {e}")
    imageio = None
    Image = None

# Import de notre environnement
try:
    from envs.ultra_robust_grasp_env import UltraRobustGraspEnv
    print("✅ Environnement ultra-robuste importé")
except ImportError:
    print("❌ ERREUR: Impossible d'importer l'environnement ultra-robuste")
    print("Assurez-vous que le fichier ultra_grasp_env.py est dans le même dossier")
    sys.exit(1)


class UltraEvaluator:
    """
    Évaluateur ultra-robuste pour modèles entraînés
    """
    
    def __init__(self, 
                 model_path: str,
                 results_dir: str = "evaluation_results",
                 render_mode: str = "rgb_array"):
        
        self.model_path = Path(model_path)
        self.results_dir = Path(results_dir)
        self.render_mode = render_mode
        
        # Créer le dossier de résultats
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "videos").mkdir(exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)
        
        # Charger le modèle
        self.model = self._load_model()
        
        # Créer l'environnement
        self.env = self._create_environment()
        
        print(f"🎯 UltraEvaluator initialisé")
        print(f"   Modèle: {self.model_path}")
        print(f"   Résultats: {self.results_dir}")
    
    def _load_model(self):
        """Charge le modèle entraîné"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")
            
            print(f"📥 Chargement du modèle: {self.model_path}")
            model = TD3.load(str(self.model_path))
            print(f"✅ Modèle chargé avec succès")
            
            return model
            
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            raise
    
    def _create_environment(self):
        """Crée l'environnement d'évaluation"""
        try:
            env = UltraRobustGraspEnv(
                render_mode=self.render_mode,
                enable_assistance=False  # Pas d'assistance pour l'évaluation
            )
            print("✅ Environnement d'évaluation créé")
            return env
            
        except Exception as e:
            print(f"❌ Erreur création environnement: {e}")
            raise
    
    def evaluate_model(self, 
                      n_episodes: int = 10,
                      max_steps: int = 600,
                      verbose: bool = True) -> Dict:
        """
        Évalue le modèle sur plusieurs épisodes
        """
        print(f"\n📊 ÉVALUATION DU MODÈLE")
        print(f"   Épisodes: {n_episodes}")
        print(f"   Steps max par épisode: {max_steps}")
        print("-" * 50)
        
        results = {
            'episodes': [],
            'summary': {}
        }
        
        for episode in range(n_episodes):
            if verbose:
                print(f"Épisode {episode + 1}/{n_episodes}...", end=" ")
            
            episode_result = self._run_single_episode(max_steps, episode)
            results['episodes'].append(episode_result)
            
            if verbose:
                print(f"Reward: {episode_result['total_reward']:.2f}, "
                      f"Distance finale: {episode_result['final_distance']:.3f}, "
                      f"Contacts max: {episode_result['max_contacts']}")
        
        # Calculer les statistiques résumées
        results['summary'] = self._calculate_summary_stats(results['episodes'])
        
        # Afficher le résumé
        self._print_evaluation_summary(results['summary'])
        
        # Sauvegarder les résultats
        self._save_evaluation_results(results)
        
        return results
    
    def _run_single_episode(self, max_steps: int, episode_num: int) -> Dict:
        """Exécute un seul épisode d'évaluation"""
        obs, info = self.env.reset()
        
        episode_data = {
            'episode_number': episode_num,
            'total_reward': 0.0,
            'steps': 0,
            'final_distance': float('inf'),
            'min_distance': float('inf'),
            'max_contacts': 0,
            'contact_history': [],
            'distance_history': [],
            'reward_history': [],
            'successful_grasp': False,
            'trajectory': []
        }
        
        for step in range(max_steps):
            # Prédiction
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Step environnement
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Enregistrer les données
            episode_data['total_reward'] += reward
            episode_data['steps'] += 1
            
            distance = info.get('distance', float('inf'))
            contacts = info.get('contact_count', 0)
            
            episode_data['distance_history'].append(distance)
            episode_data['reward_history'].append(reward)
            episode_data['contact_history'].append(contacts)
            
            # Métriques
            if distance < episode_data['min_distance']:
                episode_data['min_distance'] = distance
            
            if contacts > episode_data['max_contacts']:
                episode_data['max_contacts'] = contacts
            
            # Enregistrer position pour trajectoire
            cube_pos = info.get('cube_position', [0, 0, 0])
            palm_pos = info.get('palm_position', [0, 0, 0])
            episode_data['trajectory'].append({
                'step': step,
                'cube_pos': cube_pos,
                'palm_pos': palm_pos,
                'distance': distance,
                'contacts': contacts
            })
            
            if terminated or truncated:
                break
        
        # Métriques finales
        episode_data['final_distance'] = episode_data['distance_history'][-1] if episode_data['distance_history'] else float('inf')
        
        # Grasp réussi si distance finale < 0.06 et au moins 2 contacts
        if episode_data['final_distance'] < 0.06 and episode_data['max_contacts'] >= 2:
            episode_data['successful_grasp'] = True
        
        return episode_data
    
    def _calculate_summary_stats(self, episodes: List[Dict]) -> Dict:
        """Calcule les statistiques résumées"""
        if not episodes:
            return {}
        
        rewards = [ep['total_reward'] for ep in episodes]
        distances = [ep['final_distance'] for ep in episodes]
        min_distances = [ep['min_distance'] for ep in episodes]
        lengths = [ep['steps'] for ep in episodes]
        max_contacts = [ep['max_contacts'] for ep in episodes]
        successes = [ep['successful_grasp'] for ep in episodes]
        
        return {
            'n_episodes': len(episodes),
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'mean_final_distance': float(np.mean(distances)),
            'std_final_distance': float(np.std(distances)),
            'best_distance_achieved': float(np.min(min_distances)),
            'mean_episode_length': float(np.mean(lengths)),
            'mean_max_contacts': float(np.mean(max_contacts)),
            'success_rate': float(np.mean(successes)) * 100,
            'successful_episodes': int(np.sum(successes))
        }
    
    def _print_evaluation_summary(self, summary: Dict):
        """Affiche le résumé de l'évaluation"""
        print("\n📈 RÉSULTATS D'ÉVALUATION")
        print("=" * 50)
        print(f"📊 Épisodes évalués: {summary['n_episodes']}")
        print(f"🏆 Reward moyen: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
        print(f"   Meilleur reward: {summary['max_reward']:.2f}")
        print(f"   Pire reward: {summary['min_reward']:.2f}")
        print(f"📏 Distance finale moyenne: {summary['mean_final_distance']:.3f} ± {summary['std_final_distance']:.3f}")
        print(f"   Meilleure distance atteinte: {summary['best_distance_achieved']:.3f}")
        print(f"⏱️  Longueur moyenne des épisodes: {summary['mean_episode_length']:.1f} steps")
        print(f"🤝 Contacts max moyens: {summary['mean_max_contacts']:.1f}")
        print(f"🎯 Taux de succès: {summary['success_rate']:.1f}% ({summary['successful_episodes']}/{summary['n_episodes']})")
        print("=" * 50)
    
    def _save_evaluation_results(self, results: Dict):
        """Sauvegarde les résultats d'évaluation"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Sauvegarder les résultats complets
            results_file = self.results_dir / f"evaluation_{timestamp}.json"
            
            # Convertir les numpy types en types Python pour JSON
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                return obj
            
            # Préparer les données pour JSON
            json_data = {
                'timestamp': datetime.now().isoformat(),
                'model_path': str(self.model_path),
                'summary': results['summary'],
                'episodes': []
            }
            
            # Ajouter les épisodes avec conversion
            for episode in results['episodes']:
                episode_json = {}
                for key, value in episode.items():
                    if key in ['distance_history', 'reward_history', 'contact_history']:
                        episode_json[key] = [convert_for_json(v) for v in value]
                    elif key == 'trajectory':
                        episode_json[key] = []
                        for traj_point in value:
                            traj_json = {}
                            for k, v in traj_point.items():
                                traj_json[k] = convert_for_json(v)
                            episode_json[key].append(traj_json)
                    else:
                        episode_json[key] = convert_for_json(value)
                json_data['episodes'].append(episode_json)
            
            with open(results_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"💾 Résultats sauvegardés: {results_file}")
            
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde résultats: {e}")
    
    def create_evaluation_video(self, 
                              video_length: int = 500,
                              fps: int = 30,
                              video_name: Optional[str] = None) -> Optional[str]:
        """
        Crée une vidéo de démonstration du modèle
        """
        if not imageio or not Image:
            print("⚠️ Outils vidéo non disponibles, impossible de créer la vidéo")
            return None
        
        if video_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = f"demonstration_{timestamp}.mp4"
        
        video_path = self.results_dir / "videos" / video_name
        
        print(f"\n🎬 CRÉATION VIDÉO DE DÉMONSTRATION")
        print(f"   Longueur: {video_length} frames")
        print(f"   FPS: {fps}")
        print(f"   Fichier: {video_path}")
        print("-" * 50)
        
        try:
            frames = []
            obs, _ = self.env.reset()
            
            print("📹 Capture en cours...", end="")
            
            for step in range(video_length):
                # Prédiction
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Step environnement
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Capturer frame
                frame = self.env.render()
                if frame is not None:
                    if isinstance(frame, np.ndarray):
                        frames.append(Image.fromarray(frame.astype(np.uint8)))
                    else:
                        frames.append(frame)
                
                # Affichage progression
                if step % 50 == 0:
                    print(".", end="", flush=True)
                
                # Reset si épisode terminé
                if terminated or truncated:
                    obs, _ = self.env.reset()
            
            print(f" {len(frames)} frames capturées")
            
            # Sauvegarder la vidéo
            if len(frames) > 10:
                print("💾 Sauvegarde vidéo...")
                imageio.mimsave(str(video_path), frames, fps=fps)
                print(f"✅ Vidéo sauvegardée: {video_path}")
                
                return str(video_path)
            else:
                print("⚠️ Pas assez de frames capturées")
                return None
            
        except Exception as e:
            print(f"❌ Erreur création vidéo: {e}")
            return None
    
    def create_performance_plots(self, results: Dict):
        """Crée des graphiques de performance"""
        try:
            import matplotlib.pyplot as plt
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Préparer les données
            episodes = results['episodes']
            if not episodes:
                print("⚠️ Pas de données pour les graphiques")
                return
            
            # Créer la figure avec subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Analyse de Performance du Modèle', fontsize=16)
            
            # 1. Rewards par épisode
            rewards = [ep['total_reward'] for ep in episodes]
            axes[0, 0].bar(range(len(rewards)), rewards, alpha=0.7, color='blue')
            axes[0, 0].set_title('Rewards par Épisode')
            axes[0, 0].set_xlabel('Épisode')
            axes[0, 0].set_ylabel('Reward Total')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Distances finales
            final_distances = [ep['final_distance'] for ep in episodes]
            axes[0, 1].bar(range(len(final_distances)), final_distances, alpha=0.7, color='red')
            axes[0, 1].set_title('Distances Finales')
            axes[0, 1].set_xlabel('Épisode')
            axes[0, 1].set_ylabel('Distance (m)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Contacts maximum par épisode
            max_contacts = [ep['max_contacts'] for ep in episodes]
            axes[0, 2].bar(range(len(max_contacts)), max_contacts, alpha=0.7, color='green')
            axes[0, 2].set_title('Contacts Maximum par Épisode')
            axes[0, 2].set_xlabel('Épisode')
            axes[0, 2].set_ylabel('Nombre de Contacts')
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. Évolution de la distance (épisode le plus long)
            longest_episode = max(episodes, key=lambda x: len(x['distance_history']))
            axes[1, 0].plot(longest_episode['distance_history'], 'b-', linewidth=2)
            axes[1, 0].set_title('Évolution Distance (Meilleur Épisode)')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Distance (m)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. Évolution des contacts (même épisode)
            axes[1, 1].plot(longest_episode['contact_history'], 'g-', linewidth=2)
            axes[1, 1].set_title('Évolution Contacts (Meilleur Épisode)')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Nombre de Contacts')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Statistiques résumées
            summary = results['summary']
            stats_text = f"""Statistiques de Performance:

• Épisodes: {summary['n_episodes']}
• Reward moyen: {summary['mean_reward']:.2f}
• Écart-type: {summary['std_reward']:.2f}
• Meilleur reward: {summary['max_reward']:.2f}
• Distance finale moy.: {summary['mean_final_distance']:.3f}m
• Meilleure distance: {summary['best_distance_achieved']:.3f}m
• Longueur moyenne: {summary['mean_episode_length']:.1f} steps
• Taux de succès: {summary['success_rate']:.1f}%
• Épisodes réussis: {summary['successful_episodes']}
"""
            
            axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[1, 2].set_title('Résumé des Performances')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # Sauvegarder
            plot_path = self.results_dir / "plots" / f"performance_analysis_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Graphiques sauvegardés: {plot_path}")
            
        except ImportError:
            print("⚠️ Matplotlib non disponible pour les graphiques")
        except Exception as e:
            print(f"❌ Erreur création graphiques: {e}")
    
    def run_complete_evaluation(self, 
                              n_episodes: int = 10,
                              create_video: bool = True,
                              create_plots: bool = True,
                              video_length: int = 500) -> Dict:
        """
        Exécute une évaluation complète avec vidéo et graphiques
        """
        print(f"🚀 ÉVALUATION COMPLÈTE DU MODÈLE")
        print(f"{'='*60}")
        
        # 1. Évaluation quantitative
        results = self.evaluate_model(n_episodes=n_episodes)
        
        # 2. Création vidéo
        if create_video:
            video_path = self.create_evaluation_video(video_length=video_length)
            results['video_path'] = video_path
        
        # 3. Création graphiques
        if create_plots:
            self.create_performance_plots(results)
        
        print(f"\n🎉 ÉVALUATION COMPLÈTE TERMINÉE")
        print(f"📁 Tous les résultats sont dans: {self.results_dir}")
        
        return results
    
    def close(self):
        """Fermeture propre"""
        if self.env:
            self.env.close()
        print("🔒 Évaluateur fermé proprement")


def main():
    """Fonction principale avec interface CLI"""
    parser = argparse.ArgumentParser(description="Évaluateur ultra-robuste pour modèles TD3")
    parser.add_argument("model_path", help="Chemin vers le modèle (.zip)")
    parser.add_argument("--episodes", type=int, default=10, help="Nombre d'épisodes d'évaluation")
    parser.add_argument("--video-length", type=int, default=500, help="Longueur vidéo en frames")
    parser.add_argument("--results-dir", default="evaluation_results", help="Dossier de résultats")
    parser.add_argument("--no-video", action="store_true", help="Désactiver création vidéo")
    parser.add_argument("--no-plots", action="store_true", help="Désactiver création graphiques")
    parser.add_argument("--fps", type=int, default=30, help="FPS pour la vidéo")
    
    args = parser.parse_args()
    
    # Vérifier que le modèle existe
    if not Path(args.model_path).exists():
        print(f"❌ ERREUR: Modèle non trouvé: {args.model_path}")
        print("Modèles disponibles:")
        
        # Chercher des modèles dans les dossiers communs
        search_dirs = ["ultra_robust_results", ".", "models"]
        for search_dir in search_dirs:
            search_path = Path(search_dir)
            if search_path.exists():
                models = list(search_path.glob("**/*.zip"))
                if models:
                    print(f"  Dans {search_dir}:")
                    for model in models[:5]:  # Limiter à 5
                        print(f"    {model}")
        sys.exit(1)
    
    print(f"🎯 ÉVALUATION ULTRA-ROBUSTE")
    print(f"{'='*60}")
    print(f"📥 Modèle: {args.model_path}")
    print(f"📊 Épisodes: {args.episodes}")
    print(f"📁 Résultats: {args.results_dir}")
    print(f"🎥 Vidéo: {'Non' if args.no_video else f'Oui ({args.video_length} frames)'}")
    print(f"📈 Graphiques: {'Non' if args.no_plots else 'Oui'}")
    print(f"{'='*60}")
    
    try:
        # Créer l'évaluateur
        evaluator = UltraEvaluator(
            model_path=args.model_path,
            results_dir=args.results_dir,
            render_mode="rgb_array"
        )
        
        # Exécuter l'évaluation complète
        results = evaluator.run_complete_evaluation(
            n_episodes=args.episodes,
            create_video=not args.no_video,
            create_plots=not args.no_plots,
            video_length=args.video_length
        )
        
        # Résumé final
        summary = results['summary']
        print(f"\n🏆 RÉSULTATS FINAUX:")
        print(f"   Taux de succès: {summary['success_rate']:.1f}%")
        print(f"   Reward moyen: {summary['mean_reward']:.2f}")
        print(f"   Meilleure distance: {summary['best_distance_achieved']:.3f}m")
        
        if 'video_path' in results and results['video_path']:
            print(f"🎥 Vidéo: {results['video_path']}")
        
        evaluator.close()
        
        print(f"\n🎉 ÉVALUATION TERMINÉE AVEC SUCCÈS!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Évaluation interrompue")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def quick_test(model_path: str):
    """Test rapide d'un modèle"""
    print(f"🧪 TEST RAPIDE: {model_path}")
    
    try:
        evaluator = UltraEvaluator(model_path, results_dir="quick_test")
        
        # Test sur 3 épisodes courts
        print("Exécution de 3 épisodes de test...")
        results = evaluator.evaluate_model(n_episodes=3, max_steps=100, verbose=True)
        
        summary = results['summary']
        print(f"\n✅ TEST TERMINÉ:")
        print(f"   Reward moyen: {summary['mean_reward']:.2f}")
        print(f"   Distance moyenne: {summary['mean_final_distance']:.3f}")
        print(f"   Succès: {summary['successful_episodes']}/3")
        
        evaluator.close()
        return True
        
    except Exception as e:
        print(f"❌ ERREUR TEST: {e}")
        return False


if __name__ == "__main__":
    # Si aucun argument, chercher des modèles et proposer un test rapide
    if len(sys.argv) == 1:
        print("🔍 RECHERCHE DE MODÈLES DISPONIBLES...")
        
        search_dirs = ["ultra_robust_results", ".", "models", "ultra_robust_results/models"]
        found_models = []
        
        for search_dir in search_dirs:
            search_path = Path(search_dir)
            if search_path.exists():
                models = list(search_path.glob("**/*.zip"))
                found_models.extend(models)
        
        if found_models:
            print(f"📂 Modèles trouvés:")
            for i, model in enumerate(found_models[:10], 1):
                print(f"   {i}. {model}")
            
            print(f"\n🚀 Utilisation:")
            print(f"   python evaluate_and_video.py {found_models[0]} --episodes 10")
            print(f"   python evaluate_and_video.py {found_models[0]} --no-video --episodes 5")
            
            # Test rapide du premier modèle
            if input("\n🧪 Lancer un test rapide du premier modèle? (y/N): ").lower().strip() == 'y':
                quick_test(str(found_models[0]))
        else:
            print("❌ Aucun modèle trouvé")
            print("Entraînez d'abord un modèle avec ultra_robust_training.py")
    else:
        main()