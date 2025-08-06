#!/usr/bin/env python3
"""
Enregistreur vidéo alternatif sans FFmpeg
"""

import numpy as np
import os
from pathlib import Path
from datetime import datetime
import pickle

class AlternativeVideoRecorder:
    """Enregistreur sans dépendance FFmpeg"""
    
    def __init__(self, output_dir="training_results/videos", fps=30):
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.is_recording = False
        self.frames = []
        self.episode_info = {}
        
        print(f"✅ Enregistreur alternatif initialisé (sans FFmpeg)")
    
    def start_recording(self, episode_info=None):
        """Démarre l'enregistrement"""
        self.is_recording = True
        self.frames = []
        self.episode_info = episode_info or {}
    
    def add_frame(self, frame):
        """Ajoute une frame"""
        if self.is_recording and frame is not None:
            self.frames.append(frame)
    
    def stop_recording(self, filename=None):
        """Arrête et sauvegarde en format pickle"""
        if not self.is_recording or len(self.frames) == 0:
            return None
        
        self.is_recording = False
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode = self.episode_info.get("episode", "unknown")
            reward = self.episode_info.get("total_reward", 0)
            filename = f"episode_{episode}_{timestamp}_reward_{reward:.1f}.pkl"
        
        filepath = self.output_dir / filename
        
        try:
            # Sauvegarder en pickle au lieu de MP4
            data = {
                'frames': self.frames,
                'episode_info': self.episode_info,
                'fps': self.fps
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"✅ Données vidéo sauvegardées: {filepath}")
            
            # Créer un fichier de métadonnées lisible
            meta_path = filepath.with_suffix('.txt')
            with open(meta_path, 'w') as f:
                f.write(f"Métadonnées épisode\n")
                f.write(f"Fichier: {filepath.name}\n")
                f.write(f"Frames: {len(self.frames)}\n")
                f.write(f"FPS: {self.fps}\n")
                for key, value in self.episode_info.items():
                    f.write(f"{key}: {value}\n")
            
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
            return None
        finally:
            self.frames = []
    
    def record_episode(self, env, agent, max_steps=500, render_mode="rgb_array"):
        """Enregistre un épisode"""
        print("⚠️  Enregistrement en mode alternatif (sans vidéo MP4)")
        
        obs, info = env.reset()
        self.start_recording()
        
        total_reward = 0
        step = 0
        done = False
        
        while not done and step < max_steps:
            try:
                frame = env.render(mode=render_mode)
                self.add_frame(frame)
            except:
                pass
            
            if agent:
                action = agent.select_action(obs, evaluate=True)
            else:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward = reward
            done = terminated or truncated
            step = 1
        
        episode_info = {
            "total_reward": total_reward,
            "steps": step,
            "success": terminated and total_reward > 0
        }
        
        self.episode_info = episode_info
        video_path = self.stop_recording()
        
        return video_path, episode_info
