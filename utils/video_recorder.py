#!/usr/bin/env python3
"""
Enregistreur vidéo pour les simulations G1
Auteur: Assistant IA
"""

import numpy as np
import imageio
import os
from pathlib import Path
import time
from datetime import datetime

class VideoRecorder:
    """Enregistreur vidéo pour les épisodes de simulation"""
    
    def __init__(self, 
                 output_dir="videos",
                 fps=30,
                 quality=8,
                 format="mp4"):
        """
        Args:
            output_dir: Dossier de sortie pour les vidéos
            fps: Images par seconde
            quality: Qualité de compression (1-10, 10 = meilleure)
            format: Format vidéo (mp4, avi, etc.)
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.quality = quality
        self.format = format
        
        # Créer le dossier de sortie
        self.output_dir.mkdir(exist_ok=True)
        
        # État de l'enregistrement
        self.is_recording = False
        self.frames = []
        self.episode_info = {}
        
        print(f"✅ Enregistreur vidéo initialisé")
        print(f"   Dossier: {self.output_dir}")
        print(f"   FPS: {self.fps}")
        print(f"   Qualité: {self.quality}")
    
    def start_recording(self, episode_info=None):
        """Démarre l'enregistrement d'un nouvel épisode"""
        if self.is_recording:
            self.stop_recording()
        
        self.is_recording = True
        self.frames = []
        self.episode_info = episode_info or {}
        
        print(f"🎬 Enregistrement démarré")
    
    def add_frame(self, frame):
        """Ajoute une frame à l'enregistrement"""
        if not self.is_recording:
            return
        
        # Convertir en numpy array si nécessaire
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        
        # S'assurer que la frame est en RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Convertir de float [0,1] à uint8 [0,255] si nécessaire
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 255).astype(np.uint8)
            
            self.frames.append(frame)
    
    def stop_recording(self, filename=None):
        """Arrête l'enregistrement et sauvegarde la vidéo"""
        if not self.is_recording or len(self.frames) == 0:
            return None
        
        self.is_recording = False
        
        # Générer le nom de fichier
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode = self.episode_info.get("episode", "unknown")
            reward = self.episode_info.get("total_reward", 0)
            filename = f"episode_{episode}_{timestamp}_reward_{reward:.1f}.{self.format}"
        
        filepath = self.output_dir / filename
        
        try:
            # Sauvegarder la vidéo
            imageio.mimsave(
                str(filepath),
                self.frames,
                fps=self.fps,
                quality=self.quality
            )
            
            print(f"✅ Vidéo sauvegardée: {filepath}")
            print(f"   Frames: {len(self.frames)}")
            print(f"   Durée: {len(self.frames) / self.fps:.1f}s")
            
            # Ajouter les métadonnées dans un fichier texte
            self._save_metadata(filepath)
            
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            return None
        finally:
            self.frames = []
    
    def _save_metadata(self, video_path):
        """Sauvegarde les métadonnées de l'épisode"""
        metadata_path = video_path.with_suffix('.txt')
        
        try:
            with open(metadata_path, 'w') as f:
                f.write(f"Métadonnées vidéo - {datetime.now()}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Fichier vidéo: {video_path.name}\n")
                f.write(f"Nombre de frames: {len(self.frames)}\n")
                f.write(f"FPS: {self.fps}\n")
                f.write(f"Durée: {len(self.frames) / self.fps:.1f}s\n\n")
                
                if self.episode_info:
                    f.write("Informations de l'épisode:\n")
                    for key, value in self.episode_info.items():
                        f.write(f"  {key}: {value}\n")
                
        except Exception as e:
            print(f"⚠️  Impossible de sauvegarder les métadonnées: {e}")
    
    def record_episode(self, env, agent, max_steps=500, render_mode="rgb_array"):
        """Enregistre un épisode complet"""
        print(f"🎬 Enregistrement d'un épisode...")
        
        # Démarrer l'enregistrement
        obs, info = env.reset()
        self.start_recording()
        
        total_reward = 0
        step = 0
        done = False
        
        while not done and step < max_steps:
            # Rendu de l'environnement
            frame = env.render(mode=render_mode)
            if frame is not None:
                self.add_frame(frame)
            
            # Action de l'agent
            if agent is not None:
                action = agent.select_action(obs, evaluate=True)
            else:
                action = env.action_space.sample()
            
            # Étape de simulation
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1
        
        # Informations de l'épisode
        episode_info = {
            "total_reward": total_reward,
            "steps": step,
            "success": terminated and total_reward > 0,
            "phase": info.get("phase", "unknown"),
            "contact": info.get("contact", False),
            "cube_height": info.get("cube_height", 0)
        }
        
        self.episode_info = episode_info
        
        # Arrêter l'enregistrement
        video_path = self.stop_recording()
        
        return video_path, episode_info
    
    def create_training_video(self, env, agent, num_episodes=5):
        """Crée une vidéo de plusieurs épisodes d'entraînement"""
        print(f"🎬 Création d'une vidéo d'entraînement ({num_episodes} épisodes)")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_compilation_{timestamp}.{self.format}"
        
        self.start_recording({"type": "training_compilation", "episodes": num_episodes})
        
        total_rewards = []
        
        for episode in range(num_episodes):
            print(f"   Épisode {episode + 1}/{num_episodes}")
            
            obs, _ = env.reset()
            episode_reward = 0
            step = 0
            done = False
            
            while not done and step < 200:  # Episodes plus courts pour la compilation
                frame = env.render(mode="rgb_array")
                if frame is not None:
                    self.add_frame(frame)
                
                action = agent.select_action(obs, evaluate=True) if agent else env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step += 1
            
            total_rewards.append(episode_reward)
            
            # Ajouter quelques frames de pause entre les épisodes
            if episode < num_episodes - 1:
                last_frame = self.frames[-1] if self.frames else None
                if last_frame is not None:
                    for _ in range(self.fps // 2):  # 0.5 secondes de pause
                        self.add_frame(last_frame)
        
        # Informations de compilation
        compilation_info = {
            "type": "training_compilation",
            "episodes": num_episodes,
            "avg_reward": np.mean(total_rewards),
            "total_frames": len(self.frames)
        }
        
        self.episode_info = compilation_info
        video_path = self.stop_recording(filename)
        
        return video_path, compilation_info
    
    def cleanup_old_videos(self, max_videos=20):
        """Nettoie les anciennes vidéos pour économiser l'espace"""
        video_files = list(self.output_dir.glob(f"*.{self.format}"))
        
        if len(video_files) <= max_videos:
            return
        
        # Trier par date de modification
        video_files.sort(key=lambda x: x.stat().st_mtime)
        
        # Supprimer les plus anciens
        to_delete = video_files[:-max_videos]
        
        for video_file in to_delete:
            try:
                video_file.unlink()
                # Supprimer aussi le fichier de métadonnées s'il existe
                metadata_file = video_file.with_suffix('.txt')
                if metadata_file.exists():
                    metadata_file.unlink()
                
                print(f"🗑️  Vidéo supprimée: {video_file.name}")
            except Exception as e:
                print(f"⚠️  Impossible de supprimer {video_file.name}: {e}")
    
    def get_video_list(self):
        """Retourne la liste des vidéos enregistrées"""
        video_files = list(self.output_dir.glob(f"*.{self.format}"))
        video_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        videos_info = []
        for video_file in video_files:
            stat = video_file.stat()
            size_mb = stat.st_size / (1024 * 1024)
            
            videos_info.append({
                "filename": video_file.name,
                "path": str(video_file),
                "size_mb": size_mb,
                "created": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return videos_info