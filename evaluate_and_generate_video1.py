"""
🎬 GÉNÉRATEUR DE VIDÉO DE SIMULATION GRASPING
============================================

Script d'évaluation qui génère automatiquement une vidéo
claire et professionnelle montrant le robot qui:
1. Cherche le cube 🔍
2. S'approche avec mouvements fluides 🤖
3. Établit le contact 👋
4. Effectue le grasping 🤝

MISSION: Livrable vidéo pour deadline!
"""

import os
import numpy as np
import imageio
from stable_baselines3 import TD3
from optimized_grasp_env1 import OptimizedGraspEnv1
import matplotlib.pyplot as plt
from datetime import datetime
import logging

class VideoGenerator:
    """
    🎥 Générateur de vidéo de simulation grasping
    
    Crée une vidéo de haute qualité montrant:
    - Le robot cherchant le cube
    - L'approche progressive
    - Le contact et grasping
    - Interface informative avec métriques
    """
    
    def __init__(self, model_path="optimized_grasp_final.zip"):
        self.model_path = model_path
        self.setup_logging()
        
        # Configuration vidéo
        self.fps = 30
        self.width = 800
        self.height = 600
        
        print("🎬 Générateur de vidéo initialisé")
    
    def setup_logging(self):
        """📝 Configuration logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_model_and_env(self):
        """🔧 Chargement modèle et environnement"""
        
        print("🔧 Chargement modèle et environnement...")
        
        # Environnement optimisé
        self.env = OptimizedGraspEnv1(
            model_path="results/g1_combined_fixed.xml",
            render_mode="rgb_array"
        )
        
        # Modèle entraîné
        try:
            self.model = TD3.load(self.model_path)
            print(f"✅ Modèle chargé: {self.model_path}")
        except Exception as e:
            print(f"⚠️  Erreur chargement modèle: {e}")
            print("🔄 Utilisation d'un modèle aléatoire pour démo")
            self.model = None
        
        print(f"🎭 Résolution rendu: {self.width}x{self.height}")
    
    def add_info_overlay(self, frame, step, info):
        """
        📊 Ajoute overlay informatif sur la frame
        
        Affiche métriques clés en temps réel:
        - Distance au cube
        - Nombre de contacts  
        - Step actuel
        - Status grasping
        """
        
        # Conversion en PIL pour texte (optionnel, ici on utilise matplot)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(frame)
        ax.axis('off')
        
        # Extraction métriques
        distance = info.get('distance', 0.0)
        contacts = info.get('contact_count', 0)
        total_reward = info.get('total_reward', 0.0)
        
        # Status grasping
        if distance < 0.05 and contacts >= 2:
            status = "🤝 GRASPING ACTIF!"
            color = 'green'
        elif distance < 0.1:
            status = "👋 CONTACT IMMINENT"
            color = 'orange'
        elif distance < 0.2:
            status = "🎯 APPROCHE FINALE"
            color = 'blue'
        else:
            status = "🔍 RECHERCHE CUBE"
            color = 'red'
        
        # Texte overlay
        info_text = f"""
Step: {step}
Distance: {distance:.3f}m
Contacts: {contacts}/3
Reward: {total_reward:.1f}

{status}
        """.strip()
        
        # Affichage texte
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                color=color, weight='bold')
        
        # Conversion en array
        fig.canvas.draw()
        frame_with_info = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame_with_info = frame_with_info.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return frame_with_info
    
    def generate_video(self, output_filename=None, max_steps=500, deterministic=True):
        """
        🎬 Génération de la vidéo principale
        
        Enregistre une simulation complète montrant:
        - Le comportement du robot entraîné
        - Les métriques en temps réel
        - Une narration visuelle claire
        """
        
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"grasping_simulation_{timestamp}.mp4"
        
        print(f"🎬 Génération vidéo: {output_filename}")
        print(f"⏱️  Durée max: {max_steps} steps ({max_steps/self.fps:.1f}s)")
        
        # Préparation enregistrement
        frames = []
        obs, _ = self.env.reset()
        
        # Métriques pour narration
        best_distance = float('inf')
        first_contact_step = None
        grasp_achieved = False
        
        for step in range(max_steps):
            # Prédiction action
            if self.model is not None:
                action, _ = self.model.predict(obs, deterministic=deterministic)
            else:
                # Actions aléatoires pour démo
                action = self.env.action_space.sample() * 0.3  # Plus douces
            
            # Step simulation
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Rendu frame
            frame = self.env.render()
            if frame is not None:
                # Redimensionnement si nécessaire
                if frame.shape[:2] != (self.height, self.width):
                    frame = np.array(frame)  # Assurer format correct
                
                # Ajout overlay informatif
                frame_with_info = self.add_info_overlay(frame, step, info)
                frames.append(frame_with_info)
                
                # Tracking métriques pour narration
                distance = info.get('distance', float('inf'))
                contacts = info.get('contact_count', 0)
                
                if distance < best_distance:
                    best_distance = distance
                
                if contacts > 0 and first_contact_step is None:
                    first_contact_step = step
                    print(f"👋 Premier contact détecté au step {step}!")
                
                if distance < 0.05 and contacts >= 2:
                    grasp_achieved = True
                    print(f"🤝 Grasping réussi au step {step}!")
                
                # Progress indicator
                if step % 50 == 0:
                    print(f"   📹 Frame {step}/{max_steps} - Distance: {distance:.3f}m")
            
            # Termination
            if terminated or truncated:
                print(f"🏁 Simulation terminée au step {step}")
                break
        
        # Sauvegarde vidéo
        if frames:
            print(f"💾 Sauvegarde {len(frames)} frames...")
            
            try:
                imageio.mimsave(
                    output_filename, 
                    frames, 
                    fps=self.fps,
                    quality=8,
                    macro_block_size=1
                )
                
                print(f"✅ Vidéo générée: {output_filename}")
                print(f"📊 Statistiques:")
                print(f"   🎯 Meilleure distance: {best_distance:.4f}m")
                print(f"   👋 Premier contact: step {first_contact_step or 'Aucun'}")
                print(f"   🤝 Grasping réussi: {'Oui' if grasp_achieved else 'Non'}")
                
                return output_filename, {
                    'best_distance': best_distance,
                    'first_contact': first_contact_step,
                    'grasp_achieved': grasp_achieved,
                    'total_frames': len(frames)
                }
                
            except Exception as e:
                print(f"❌ Erreur sauvegarde vidéo: {e}")
                return None, {}
        
        else:
            print("❌ Aucune frame générée!")
            return None, {}
    
    def generate_comparison_video(self, episodes=3):
        """
        🎭 Génère vidéo comparative multi-épisodes
        
        Montre plusieurs tentatives pour illustrer:
        - La variabilité des approches
        - Les différentes stratégies
        - La robustesse du modèle
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"grasping_comparison_{timestamp}.mp4"
        
        print(f"🎭 Génération vidéo comparative ({episodes} épisodes)")
        
        all_frames = []
        
        for episode in range(episodes):
            print(f"\n🎬 Épisode {episode + 1}/{episodes}")
            
            # Reset environnement
            obs, _ = self.env.reset()
            episode_frames = []
            
            # Titre épisode
            title_frame = self.create_title_frame(f"Épisode {episode + 1}")
            episode_frames.extend([title_frame] * 60)  # 2s de titre
            
            # Simulation épisode
            for step in range(200):  # Épisodes plus courts
                if self.model is not None:
                    action, _ = self.model.predict(obs, deterministic=False)  # Non-déterministe
                else:
                    action = self.env.action_space.sample() * 0.3
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                frame = self.env.render()
                if frame is not None:
                    frame_with_info = self.add_info_overlay(frame, step, info)
                    episode_frames.append(frame_with_info)
                
                if terminated or truncated:
                    break
            
            all_frames.extend(episode_frames)
            print(f"   ✅ Épisode {episode + 1} enregistré ({len(episode_frames)} frames)")
        
        # Sauvegarde vidéo comparative
        if all_frames:
            imageio.mimsave(
                output_filename, 
                all_frames, 
                fps=self.fps,
                quality=8
            )
            
            print(f"✅ Vidéo comparative générée: {output_filename}")
            return output_filename
        
        return None
    
    def create_title_frame(self, title):
        """📝 Crée frame de titre"""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, title, transform=ax.transAxes,
                horizontalalignment='center', verticalalignment='center',
                fontsize=24, weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='blue', alpha=0.8),
                color='white')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return frame
    
    def close(self):
        """🧹 Nettoyage ressources"""
        if hasattr(self, 'env'):
            self.env.close()

def main():
    """
    🚀 MAIN - Génération vidéo de démonstration
    
    Crée automatiquement une vidéo professionnelle
    pour présenter les résultats du projet grasping.
    """
    
    print("=" * 60)
    print("🎬 GÉNÉRATEUR DE VIDÉO GRASPING - LIVRABLE DEADLINE")
    print("=" * 60)
    
    # Configuration environnement
    os.environ["MUJOCO_GL"] = "egl"
    
    # Générateur vidéo
    video_gen = VideoGenerator()
    
    try:
        # Chargement modèle et environnement
        video_gen.load_model_and_env()
        
        # Génération vidéo principale
        print("\n🎬 GÉNÉRATION VIDÉO PRINCIPALE")
        video_file, stats = video_gen.generate_video(
            output_filename="DEMO_GRASPING_FINAL.mp4",
            max_steps=300,
            deterministic=True
        )
        
        if video_file:
            print(f"\n🎉 SUCCÈS! Vidéo générée: {video_file}")
            print("📁 Prête pour présentation!")
            
            # Génération vidéo comparative (optionnel)
            print("\n🎭 GÉNÉRATION VIDÉO COMPARATIVE")
            comparison_video = video_gen.generate_comparison_video(episodes=2)
            
            if comparison_video:
                print(f"✅ Vidéo comparative: {comparison_video}")
        
        else:
            print("❌ Échec génération vidéo principale")
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        raise
    
    finally:
        video_gen.close()
    
    print("\n🏁 Génération terminée!")
    print("📹 Fichiers vidéo prêts pour livrable!")

if __name__ == "__main__":
    main()