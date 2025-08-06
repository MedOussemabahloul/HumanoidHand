#!/usr/bin/env python3
"""
Enregistreur vidéo simplifié pour les simulations G1
"""

import numpy as np
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("⚠️  imageio non disponible - enregistrement vidéo désactivé")

import os
from pathlib import Path
import time
from datetime import datetime

class VideoRecorder:
    """Enregistreur vidéo pour les épisodes de simulation"""
    
    def __init__(self, 
                 output_dir="training_results/videos",
                 fps=30,
                 quality=8,
                 format="mp4"):
        
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.quality = quality
        self.format = format
        self.has_imageio = HAS_IMAGEIO
        
        # Créer le dossier de sortie
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # État de l'enregistrement
        self.is_recording = False
        self.frames = []
        self.episode_info = {}
        
        if self.has_imageio:
            print(f"✅ Enregistreur vidéo initialisé")
            print(f"   Dossier: {self.output_dir}")
        else:
            print(f"⚠️  Enregistreur vidéo en mode simulation (pas d'imageio)")
    
    def start_recording(self, episode_info=None):
        """Démarre l'enregistrement d'un nouvel épisode"""
        if not self.has_imageio:
            return
            
        if self.is_recording:
            self.stop_recording()
        
        self.is_recording = True
        self.frames = []
        self.episode_info = episode_info or {}
    
    def add_frame(self, frame):
        """Ajoute une frame à l'enregistrement"""
        if not self.is_recording or not self.has_imageio:
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
        if not self.is_recording or len(self.frames) == 0 or not self.has_imageio:
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
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            return None
        finally:
            self.frames = []
    
    def record_episode(self, env, agent, max_steps=500, render_mode="rgb_array"):
        """Enregistre un épisode complet"""
        if not self.has_imageio:
            print("⚠️  Enregistrement vidéo désactivé (imageio manquant)")
            return None, {}
            
        print(f"🎬 Enregistrement d'un épisode...")
        
        # Démarrer l'enregistrement
        obs, info = env.reset()
        self.start_recording()
        
        total_reward = 0
        step = 0
        done = False
        
        while not done and step < max_steps:
            # Rendu de l'environnement
            try:
                frame = env.render(mode=render_mode)
                if frame is not None:
                    self.add_frame(frame)
            except:
                pass  # Ignorer les erreurs de rendu
            
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
