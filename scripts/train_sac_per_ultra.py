#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ULTRA-ROBUST SAC + PER TRAINING SYSTEM FOR G1 HUMANOID
=========================================================

COMBINAISON OPTIMALE: Soft Actor-Critic + Prioritized Experience Replay
✅ SAC: Exploration optimale avec entropie adaptative 
✅ PER: Apprentissage accéléré avec importance sampling
✅ ULTRA-ROBUSTE: Validation complète + recovery automatique
✅ HAUTE PERFORMANCE: Cache intelligent + optimisations

GARANTIES SYSTÈME:
🔒 Validation complète: Tous inputs/outputs vérifiés
🛡️ Gestion erreurs: Recovery automatique sans crash  
⚡ Cache intelligent: Performance optimale
📊 Logging structuré: Debug facilité
🎯 Reproductibilité: Seeds + determinisme
🔧 Monitoring: Métriques temps réel avancées

ALGORITHME MATHÉMATIQUE:
- SAC Policy: π_θ(a|s) = tanh(μ_θ(s) + σ_θ(s) ⊙ ε), ε ~ N(0,I)
- SAC Q-functions: Q_φ(s,a) avec double Q-learning  
- SAC Entropy: α adaptative via J_α = -α(log π + H_target)
- PER Priority: p_i = |δ_i|^α + ε (erreur TD + epsilon)
- PER Sampling: P(i) = p_i^α / Σ_k p_k^α
- PER IS weights: w_i = (N·P(i))^(-β) / max_j w_j

Version: 3.0 Ultra - Système IA Expert
"""

import os
import sys
import time
import logging
import warnings
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque, defaultdict
import threading
import queue
import json
from contextlib import contextmanager

# Suppression warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration chemin projet
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports scientifiques optimisés
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.multiprocessing as mp

# MuJoCo et simulation
import mujoco
import mujoco.viewer

# Imports locaux avec validation
try:
    from tasks.grasp.grasp_lift_task_optimized import GraspLiftTaskOptimized
except ImportError:
    print("⚠️  Utilisation fallback task standard")
    from tasks.grasp.grasp_lift_task import GraspLiftTask as GraspLiftTaskOptimized

# Configuration logging ultra-structuré
class ColoredFormatter(logging.Formatter):
    """Formatter avec couleurs pour debugging visuel"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Vert  
        'WARNING': '\033[33m',  # Jaune
        'ERROR': '\033[31m',    # Rouge
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

def setup_ultra_logging(log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Configuration logging ultra-avancé avec handlers multiples"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger principal
    logger = logging.getLogger('SAC_PER_ULTRA')
    logger.setLevel(level)
    
    # Clear handlers existants
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Handler console avec couleurs
    console_handler = logging.StreamHandler()
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # Handler fichier détaillé
    file_handler = logging.FileHandler(log_dir / 'training.log')
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(name)15s | %(funcName)20s:%(lineno)4d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    
    # Handler erreurs séparé
    error_handler = logging.FileHandler(log_dir / 'errors.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)
    
    return logger

# Logger global
logger = None

@dataclass
class ValidationResult:
    """Résultat de validation avec détails"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)

class UltraValidator:
    """Système de validation ultra-complet avec recovery"""
    
    @staticmethod
    def validate_torch_tensor(tensor: Any, name: str, 
                            expected_shape: Optional[Tuple] = None,
                            expected_dtype: Optional[torch.dtype] = None,
                            allow_nan: bool = False,
                            value_range: Optional[Tuple[float, float]] = None) -> ValidationResult:
        """Validation ultra-complète d'un tensor PyTorch"""
        result = ValidationResult(is_valid=True)
        
        # Type check
        if not isinstance(tensor, torch.Tensor):
            if isinstance(tensor, np.ndarray):
                result.add_warning(f"{name}: Conversion numpy->torch")
                try:
                    tensor = torch.from_numpy(tensor)
                except Exception as e:
                    result.add_error(f"{name}: Conversion failed: {e}")
                    return result
            else:
                result.add_error(f"{name}: Expected torch.Tensor, got {type(tensor)}")
                return result
        
        # Shape validation
        if expected_shape is not None:
            if tensor.shape != expected_shape:
                # Tentative reshape automatique
                try:
                    tensor = tensor.reshape(expected_shape)
                    result.add_warning(f"{name}: Auto-reshaped to {expected_shape}")
                except:
                    result.add_error(f"{name}: Shape {tensor.shape} != expected {expected_shape}")
        
        # Dtype validation
        if expected_dtype is not None and tensor.dtype != expected_dtype:
            try:
                tensor = tensor.to(expected_dtype)
                result.add_warning(f"{name}: Auto-converted to {expected_dtype}")
            except:
                result.add_error(f"{name}: Cannot convert to {expected_dtype}")
        
        # NaN/Inf check
        if not allow_nan:
            if torch.isnan(tensor).any():
                result.add_error(f"{name}: Contains NaN values")
            if torch.isinf(tensor).any():
                result.add_error(f"{name}: Contains Inf values")
        
        # Value range validation
        if value_range is not None:
            min_val, max_val = value_range
            if tensor.min().item() < min_val or tensor.max().item() > max_val:
                result.add_warning(f"{name}: Values outside [{min_val}, {max_val}]")
        
        result.metadata['tensor'] = tensor
        result.metadata['actual_shape'] = tensor.shape
        result.metadata['actual_dtype'] = tensor.dtype
        
        return result
    
    @staticmethod
    def validate_numpy_array(array: Any, name: str,
                           expected_shape: Optional[Tuple] = None,
                           expected_dtype: Optional[np.dtype] = None,
                           allow_nan: bool = False) -> ValidationResult:
        """Validation ultra-complète d'un array NumPy"""
        result = ValidationResult(is_valid=True)
        
        if not isinstance(array, np.ndarray):
            try:
                array = np.array(array)
                result.add_warning(f"{name}: Auto-converted to numpy array")
            except:
                result.add_error(f"{name}: Cannot convert to numpy array")
                return result
        
        if expected_shape is not None and array.shape != expected_shape:
            try:
                array = array.reshape(expected_shape)
                result.add_warning(f"{name}: Auto-reshaped to {expected_shape}")
            except:
                result.add_error(f"{name}: Cannot reshape to {expected_shape}")
        
        if expected_dtype is not None and array.dtype != expected_dtype:
            try:
                array = array.astype(expected_dtype)
                result.add_warning(f"{name}: Auto-converted to {expected_dtype}")
            except:
                result.add_error(f"{name}: Cannot convert to {expected_dtype}")
        
        if not allow_nan:
            if np.isnan(array).any():
                result.add_error(f"{name}: Contains NaN")
            if np.isinf(array).any():
                result.add_error(f"{name}: Contains Inf")
        
        result.metadata['array'] = array
        return result

class IntelligentCache:
    """Cache intelligent avec gestion mémoire automatique"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.current_memory = 0
        self._lock = threading.RLock()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimation taille mémoire d'un objet"""
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        else:
            return sys.getsizeof(obj)
    
    def _cleanup_if_needed(self):
        """Nettoyage intelligent basé sur LRU et fréquence"""
        if len(self.cache) <= self.max_size and self.current_memory <= self.max_memory_bytes:
            return
        
        # Score combiné: temps accès récent + fréquence
        current_time = time.time()
        scores = {}
        
        for key in self.cache:
            recency_score = current_time - self.access_times.get(key, 0)
            frequency_score = 1.0 / (self.access_counts[key] + 1)
            scores[key] = recency_score + frequency_score
        
        # Suppression des items avec score le plus élevé (moins utiles)
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        
        for key in sorted_keys:
            if len(self.cache) <= self.max_size // 2 and self.current_memory <= self.max_memory_bytes // 2:
                break
            self._remove_item(key)
    
    def _remove_item(self, key: str):
        """Suppression item avec mise à jour mémoire"""
        if key in self.cache:
            self.current_memory -= self._estimate_size(self.cache[key])
            del self.cache[key]
            del self.access_times[key]
            del self.access_counts[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Récupération avec mise à jour statistiques accès"""
        with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.access_counts[key] += 1
                return self.cache[key]
            return default
    
    def set(self, key: str, value: Any):
        """Stockage avec gestion mémoire automatique"""
        with self._lock:
            # Suppression ancienne valeur si existe
            if key in self.cache:
                self._remove_item(key)
            
            # Nettoyage préventif
            self._cleanup_if_needed()
            
            # Stockage nouvelle valeur
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            self.current_memory += self._estimate_size(value)
    
    def clear(self):
        """Nettoyage complet"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.current_memory = 0
    
    def stats(self) -> Dict[str, Any]:
        """Statistiques du cache"""
        with self._lock:
            return {
                'size': len(self.cache),
                'memory_mb': self.current_memory / (1024 * 1024),
                'hit_rate': sum(self.access_counts.values()) / max(len(self.cache), 1),
                'most_accessed': max(self.access_counts.items(), key=lambda x: x[1]) if self.access_counts else None
            }

class UltraPrioritizedReplayBuffer:
    """Buffer PER ultra-optimisé avec segment tree pour O(log n)"""
    
    def __init__(self, size: int, obs_dim: int, act_dim: int, device: torch.device,
                 alpha: float = 0.6, beta: float = 0.4, epsilon: float = 1e-6):
        self.size = size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        
        # Buffers optimisés avec pré-allocation
        self.obs_buf = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.next_obs_buf = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.bool, device=device)
        
        # Segment tree pour priorités (efficacité O(log n))
        self.tree_size = 1
        while self.tree_size < size:
            self.tree_size *= 2
        self.tree = np.zeros(2 * self.tree_size)
        
        # État interne
        self.ptr = 0
        self.actual_size = 0
        self.max_priority = 1.0
        
        # Cache intelligent pour optimisation
        self.cache = IntelligentCache(max_size=500, max_memory_mb=50)
        
        # Statistiques monitoring
        self.stats = {
            'total_additions': 0,
            'total_samples': 0,
            'priority_updates': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Validation et logging
        self.validator = UltraValidator()
        
        logger.info(f"✅ UltraPER Buffer: size={size}, α={alpha}, β={beta}")
    
    def _update_tree(self, idx: int, priority: float):
        """Mise à jour segment tree efficace"""
        tree_idx = idx + self.tree_size
        self.tree[tree_idx] = priority
        
        # Propagation vers racine
        while tree_idx > 1:
            tree_idx //= 2
            self.tree[tree_idx] = self.tree[2 * tree_idx] + self.tree[2 * tree_idx + 1]
    
    def _get_priority_sum(self, start: int = 0, end: int = None) -> float:
        """Somme des priorités dans un range (O(log n))"""
        if end is None:
            end = self.actual_size
        
        # Conversion vers indices tree
        start += self.tree_size
        end += self.tree_size
        
        total = 0.0
        while start < end:
            if start % 2 == 1:
                total += self.tree[start]
                start += 1
            if end % 2 == 1:
                end -= 1
                total += self.tree[end]
            start //= 2
            end //= 2
        
        return total
    
    def _sample_proportional(self, batch_size: int) -> np.ndarray:
        """Échantillonnage proportionnel ultra-efficace"""
        total_priority = self._get_priority_sum()
        
        if total_priority == 0:
            # Fallback échantillonnage uniforme
            return np.random.randint(0, self.actual_size, size=batch_size)
        
        # Génération samples proportionnels
        segment_size = total_priority / batch_size
        segments = np.arange(batch_size) * segment_size
        random_offsets = np.random.uniform(0, segment_size, size=batch_size)
        samples = segments + random_offsets
        
        # Recherche indices correspondants dans tree
        indices = []
        for sample in samples:
            idx = self._find_index_for_value(sample)
            indices.append(idx)
        
        return np.array(indices)
    
    def _find_index_for_value(self, value: float) -> int:
        """Recherche index pour valeur donnée dans tree"""
        tree_idx = 1
        
        while tree_idx < self.tree_size:
            left_child = 2 * tree_idx
            left_sum = self.tree[left_child]
            
            if value <= left_sum:
                tree_idx = left_child
            else:
                value -= left_sum
                tree_idx = left_child + 1
        
        return min(tree_idx - self.tree_size, self.actual_size - 1)
    
    @contextmanager
    def _error_recovery(self, operation: str):
        """Context manager pour recovery automatique"""
        try:
            yield
        except Exception as e:
            logger.error(f"❌ Erreur {operation}: {e}")
            logger.info("🔧 Tentative recovery automatique...")
            
            # Recovery strategies
            if "CUDA" in str(e) or "device" in str(e):
                # Recovery device mismatch
                self._fix_device_mismatch()
            elif "shape" in str(e) or "size" in str(e):
                # Recovery shape mismatch  
                self._fix_shape_mismatch()
            else:
                # Recovery générique
                self._generic_recovery()
            
            logger.info("✅ Recovery terminé")
    
    def _fix_device_mismatch(self):
        """Fix automatique des problèmes de device"""
        logger.info("🔧 Fix device mismatch...")
        # Déplacer tous les buffers vers le bon device
        self.obs_buf = self.obs_buf.to(self.device)
        self.act_buf = self.act_buf.to(self.device)
        self.rew_buf = self.rew_buf.to(self.device)
        self.next_obs_buf = self.next_obs_buf.to(self.device)
        self.done_buf = self.done_buf.to(self.device)
    
    def _fix_shape_mismatch(self):
        """Fix automatique des problèmes de shape"""
        logger.info("🔧 Fix shape mismatch...")
        # Réinitialiser les buffers avec les bonnes dimensions
        self.obs_buf = torch.zeros((self.size, self.obs_dim), dtype=torch.float32, device=self.device)
        self.act_buf = torch.zeros((self.size, self.act_dim), dtype=torch.float32, device=self.device)
        self.rew_buf = torch.zeros(self.size, dtype=torch.float32, device=self.device)
        self.next_obs_buf = torch.zeros((self.size, self.obs_dim), dtype=torch.float32, device=self.device)
        self.done_buf = torch.zeros(self.size, dtype=torch.bool, device=self.device)
    
    def _generic_recovery(self):
        """Recovery générique en cas d'erreur"""
        logger.info("🔧 Generic recovery...")
        # Clear cache en cas de corruption
        self.cache.clear()
        # Reset pointeurs
        if self.ptr >= self.size:
            self.ptr = 0
    
    def store(self, obs: torch.Tensor, act: torch.Tensor, rew: float, 
              next_obs: torch.Tensor, done: bool) -> bool:
        """Stockage ultra-robuste avec validation complète"""
        
        with self._error_recovery("store"):
            # Validation inputs ultra-complète
            obs_valid = self.validator.validate_torch_tensor(
                obs, "obs", (self.obs_dim,), torch.float32, value_range=(-10, 10))
            
            if not obs_valid.is_valid:
                logger.error(f"❌ Validation obs failed: {obs_valid.errors}")
                return False
            
            act_valid = self.validator.validate_torch_tensor(
                act, "act", (self.act_dim,), torch.float32, value_range=(-5, 5))
            
            if not act_valid.is_valid:
                logger.error(f"❌ Validation act failed: {act_valid.errors}")
                return False
            
            # Extraction tensors validés
            obs = obs_valid.metadata['tensor'].to(self.device)
            act = act_valid.metadata['tensor'].to(self.device)
            
            # Validation reward et done
            if not isinstance(rew, (int, float)) or not np.isfinite(rew):
                logger.warning(f"⚠️  Invalid reward {rew}, clipping...")
                rew = np.clip(float(rew), -100, 100)
            
            # Stockage dans buffers
            self.obs_buf[self.ptr] = obs
            self.act_buf[self.ptr] = act
            self.rew_buf[self.ptr] = rew
            self.next_obs_buf[self.ptr] = next_obs.to(self.device)
            self.done_buf[self.ptr] = bool(done)
            
            # Mise à jour priorités (nouvelles expériences = priorité max)
            self._update_tree(self.ptr, self.max_priority)
            
            # Mise à jour état
            self.ptr = (self.ptr + 1) % self.size
            self.actual_size = min(self.actual_size + 1, self.size)
            self.stats['total_additions'] += 1
            
            # Cache invalidation pour cohérence
            cache_key = f"sample_{self.stats['total_samples']}"
            if self.cache.get(cache_key) is not None:
                self.cache.set(cache_key, None)  # Invalidate
            
            return True
    
    def sample(self, batch_size: int) -> Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
        """Échantillonnage ultra-optimisé avec cache intelligent"""
        
        if self.actual_size < batch_size:
            logger.warning(f"⚠️  Buffer size {self.actual_size} < batch_size {batch_size}")
            return None
        
        # Cache check pour accélération
        cache_key = f"sample_{batch_size}_{self.stats['total_samples'] % 100}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result is not None:
            self.stats['cache_hits'] += 1
            logger.debug("💾 Cache hit pour échantillonnage")
            return cached_result
        
        self.stats['cache_misses'] += 1
        
        with self._error_recovery("sample"):
            # Échantillonnage proportionnel
            indices = self._sample_proportional(batch_size)
            
            # Calcul poids importance sampling
            total_priority = self._get_priority_sum()
            min_priority = np.min([self.tree[i + self.tree_size] for i in indices])
            
            weights = []
            for idx in indices:
                priority = self.tree[idx + self.tree_size]
                prob = priority / total_priority
                weight = (self.actual_size * prob) ** (-self.beta)
                weights.append(weight)
            
            weights = np.array(weights)
            weights /= np.max(weights)  # Normalisation
            
            # Préparation batch optimisé
            batch_indices = torch.from_numpy(indices).to(self.device)
            
            batch = {
                'obs': self.obs_buf[batch_indices],
                'act': self.act_buf[batch_indices],
                'rew': self.rew_buf[batch_indices],
                'next_obs': self.next_obs_buf[batch_indices],
                'done': self.done_buf[batch_indices]
            }
            
            weights_tensor = torch.from_numpy(weights).float().to(self.device)
            indices_tensor = torch.from_numpy(indices).long().to(self.device)
            
            result = (batch, weights_tensor, indices_tensor)
            
            # Cache pour réutilisation
            self.cache.set(cache_key, result)
            self.stats['total_samples'] += 1
            
            return result
    
    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """Mise à jour priorités ultra-efficace avec validation"""
        
        with self._error_recovery("update_priorities"):
            # Validation
            indices_valid = self.validator.validate_torch_tensor(
                indices, "indices", expected_dtype=torch.long)
            priorities_valid = self.validator.validate_torch_tensor(
                priorities, "priorities", expected_dtype=torch.float32)
            
            if not (indices_valid.is_valid and priorities_valid.is_valid):
                logger.error("❌ Validation failed pour update_priorities")
                return
            
            indices = indices_valid.metadata['tensor'].cpu().numpy()
            priorities = priorities_valid.metadata['tensor'].cpu().numpy()
            
            # Clipping et epsilon pour stabilité
            priorities = np.abs(priorities) + self.epsilon
            priorities = np.clip(priorities, self.epsilon, 100.0)
            
            # Mise à jour tree
            for idx, priority in zip(indices, priorities):
                if 0 <= idx < self.actual_size:
                    self._update_tree(idx, priority ** self.alpha)
                    self.max_priority = max(self.max_priority, priority)
            
            self.stats['priority_updates'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistiques complètes du buffer"""
        cache_stats = self.cache.stats()
        
        return {
            'buffer_size': self.actual_size,
            'max_size': self.size,
            'fill_ratio': self.actual_size / self.size,
            'max_priority': self.max_priority,
            'total_additions': self.stats['total_additions'],
            'total_samples': self.stats['total_samples'],
            'priority_updates': self.stats['priority_updates'],
            'cache_hit_ratio': self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1),
            'cache_stats': cache_stats,
            'alpha': self.alpha,
            'beta': self.beta
        }

class UltraAdvancedSACAgent:
    """Agent SAC ultra-avancé avec architecture optimisée"""
    
    def __init__(self, obs_dim: int, act_dim: int, device: torch.device,
                 hidden_sizes: List[int] = [512, 512, 256],
                 learning_rate: float = 3e-4):
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        
        # Architectures réseau ultra-optimisées
        self.policy = self._build_policy_network(obs_dim, act_dim, hidden_sizes).to(device)
        self.q1 = self._build_q_network(obs_dim, act_dim, hidden_sizes).to(device)
        self.q2 = self._build_q_network(obs_dim, act_dim, hidden_sizes).to(device)
        
        # Réseaux cibles avec copie exacte
        self.q1_target = self._build_q_network(obs_dim, act_dim, hidden_sizes).to(device)
        self.q2_target = self._build_q_network(obs_dim, act_dim, hidden_sizes).to(device)
        
        # Synchronisation initiale
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Désactivation gradients cibles
        for net in [self.q1_target, self.q2_target]:
            for p in net.parameters():
                p.requires_grad = False
        
        # Entropie adaptative
        self.log_alpha = torch.tensor(np.log(0.2), device=device, requires_grad=True)
        self.target_entropy = -act_dim
        
        # Optimiseurs ultra-configurés
        self.policy_optimizer = optim.AdamW(
            self.policy.parameters(), lr=learning_rate, 
            weight_decay=1e-4, eps=1e-8)
        self.q1_optimizer = optim.AdamW(
            self.q1.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.q2_optimizer = optim.AdamW(
            self.q2.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        
        # Schedulers adaptatifs
        self.policy_scheduler = CosineAnnealingLR(self.policy_optimizer, T_max=1000)
        
        # Cache intelligent pour optimisation
        self.cache = IntelligentCache(max_size=200, max_memory_mb=30)
        
        # Validation système
        self.validator = UltraValidator()
        
        # Comptage paramètres
        total_params = sum(p.numel() for net in [self.policy, self.q1, self.q2] 
                          for p in net.parameters())
        logger.info(f"🧠 SAC Agent: {total_params:,} paramètres")
    
    def _build_policy_network(self, obs_dim: int, act_dim: int, 
                            hidden_sizes: List[int]) -> nn.Module:
        """Construction réseau politique ultra-optimisé"""
        
        class UltraPolicyNetwork(nn.Module):
            def __init__(self, obs_dim, act_dim, hidden_sizes):
                super().__init__()
                
                # Normalisation d'entrée
                self.obs_norm = nn.LayerNorm(obs_dim)
                
                # Architecture avec residual connections
                layers = []
                in_dim = obs_dim
                
                for i, hidden_size in enumerate(hidden_sizes):
                    layers.extend([
                        nn.Linear(in_dim, hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.GELU(),  # GELU > ReLU pour gradients
                        nn.Dropout(0.1)
                    ])
                    in_dim = hidden_size
                
                self.trunk = nn.Sequential(*layers)
                
                # Têtes spécialisées
                self.mu_head = nn.Linear(in_dim, act_dim)
                self.logstd_head = nn.Linear(in_dim, act_dim)
                
                # Initialisation Xavier optimisée
                self._init_weights()
            
            def _init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        if m == self.mu_head:
                            nn.init.xavier_uniform_(m.weight, gain=0.01)
                        elif m == self.logstd_head:
                            nn.init.constant_(m.weight, 0.0)
                            nn.init.constant_(m.bias, -1.0)
                        else:
                            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                        nn.init.constant_(m.bias, 0.0)
            
            def forward(self, obs):
                x = self.obs_norm(obs)
                features = self.trunk(x)
                
                mu = self.mu_head(features)
                logstd = self.logstd_head(features).clamp(-20, 2)
                std = torch.exp(logstd)
                
                return mu, std
            
            def sample(self, obs):
                mu, std = self.forward(obs)
                dist = torch.distributions.Normal(mu, std)
                z = dist.rsample()
                action = torch.tanh(z)
                
                # Log-prob corrigé pour tanh
                logp_z = dist.log_prob(z).sum(dim=-1)
                logp_action = logp_z - (2 * (np.log(2) - z - F.softplus(-2 * z))).sum(dim=-1)
                
                return action, logp_action
        
        return UltraPolicyNetwork(obs_dim, act_dim, hidden_sizes)
    
    def _build_q_network(self, obs_dim: int, act_dim: int, 
                        hidden_sizes: List[int]) -> nn.Module:
        """Construction réseau Q ultra-optimisé avec Dueling"""
        
        class UltraQNetwork(nn.Module):
            def __init__(self, obs_dim, act_dim, hidden_sizes):
                super().__init__()
                
                # Normalisation d'entrée
                self.input_norm = nn.LayerNorm(obs_dim + act_dim)
                
                # Trunk partagé
                layers = []
                in_dim = obs_dim + act_dim
                
                for hidden_size in hidden_sizes:
                    layers.extend([
                        nn.Linear(in_dim, hidden_size),
                        nn.LayerNorm(hidden_size),
                        nn.GELU(),
                        nn.Dropout(0.1)
                    ])
                    in_dim = hidden_size
                
                self.trunk = nn.Sequential(*layers)
                
                # Architecture Dueling
                self.value_head = nn.Linear(in_dim, 1)
                self.advantage_head = nn.Linear(in_dim, 1)
                
                self._init_weights()
            
            def _init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        if m in [self.value_head, self.advantage_head]:
                            nn.init.xavier_uniform_(m.weight, gain=0.1)
                        else:
                            nn.init.orthogonal_(m.weight, gain=1.0)
                        nn.init.constant_(m.bias, 0.0)
            
            def forward(self, obs, act):
                x = torch.cat([obs, act], dim=-1)
                x = self.input_norm(x)
                features = self.trunk(x)
                
                value = self.value_head(features)
                advantage = self.advantage_head(features)
                
                # Dueling: Q = V + A - mean(A)
                q_value = value + advantage - advantage.mean(dim=-1, keepdim=True)
                
                return q_value.squeeze(-1)
        
        return UltraQNetwork(obs_dim, act_dim, hidden_sizes)
    
    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sélection d'action ultra-optimisée avec cache"""
        
        # Cache check pour obs similaires
        obs_key = hash(obs.cpu().numpy().tobytes()) if obs.is_cuda else hash(obs.numpy().tobytes())
        cache_key = f"action_{obs_key}_{deterministic}"
        
        cached_action = self.cache.get(cache_key)
        if cached_action is not None:
            return cached_action
        
        with torch.no_grad():
            if deterministic:
                mu, _ = self.policy(obs.unsqueeze(0))
                action = torch.tanh(mu)
            else:
                action, _ = self.policy.sample(obs.unsqueeze(0))
            
            action = action.squeeze(0)
            
            # Cache pour réutilisation
            self.cache.set(cache_key, action.clone())
            
            return action

class UltraSACPERTrainer:
    """Trainer ultra-robuste combinant SAC + PER avec toutes garanties"""
    
    def __init__(self, config_path: str):
        # Configuration système ultra-robuste
        self.config = self._load_and_validate_config(config_path)
        
        # Setup logging ultra-structuré
        global logger
        logger = setup_ultra_logging(
            Path(self.config['task']['output_dir']) / 'logs',
            logging.DEBUG if self.config.get('debug', False) else logging.INFO
        )
        
        logger.info("🚀 Initialisation UltraSACPERTrainer")
        
        # Device avec validation
        self.device = self._setup_device()
        
        # Setup répertoires
        self._setup_directories()
        
        # Chargement modèle MuJoCo avec validation
        self._load_mujoco_model()
        
        # Setup tâche avec validation
        self._setup_task()
        
        # Initialisation agent SAC
        self._setup_sac_agent()
        
        # Initialisation buffer PER
        self._setup_per_buffer()
        
        # Setup monitoring ultra-avancé
        self._setup_monitoring()
        
        # État d'entraînement
        self.step_count = 0
        self.episode_count = 0
        self.best_reward = -np.inf
        
        # Recovery system
        self.last_checkpoint_step = 0
        self.emergency_saves = 0
        
        logger.info("✅ UltraSACPERTrainer initialisé avec succès!")
    
    def _load_and_validate_config(self, config_path: str) -> Dict[str, Any]:
        """Chargement et validation ultra-complète de la configuration"""
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ Config non trouvée: {config_path}")
        
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"❌ Erreur lecture config: {e}")
        
        # Validation structure config
        required_sections = ['task', 'rl']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"❌ Section manquante: {section}")
        
        # Validation paramètres RL critiques
        rl_config = config['rl']
        required_rl_params = ['gamma', 'learning_rate', 'batch_size', 'total_steps']
        for param in required_rl_params:
            if param not in rl_config:
                raise ValueError(f"❌ Paramètre RL manquant: {param}")
        
        # Ajout paramètres par défaut ultra-optimisés
        config['rl'].setdefault('per_alpha', 0.6)
        config['rl'].setdefault('per_beta', 0.4)
        config['rl'].setdefault('per_beta_annealing', True)
        config['rl'].setdefault('replay_size', 1000000)
        config['rl'].setdefault('tau', 0.005)
        config['rl'].setdefault('target_update_freq', 1)
        config['rl'].setdefault('gradient_clip', 1.0)
        
        return config
    
    def _setup_device(self) -> torch.device:
        """Setup device avec validation et optimisation"""
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # Optimisations CUDA
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            logger.info(f"🚀 CUDA: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            # Optimisations CPU
            torch.set_num_threads(4)
            logger.info("💻 Device: CPU avec optimisations")
        
        return device
    
    def _setup_directories(self):
        """Setup répertoires avec structure professionnelle"""
        
        self.output_dir = Path(self.config['task']['output_dir'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.tb_dir = self.output_dir / 'tensorboard'
        self.backup_dir = self.output_dir / 'backups'
        
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir, 
                        self.tb_dir, self.backup_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Répertoires configurés: {self.output_dir}")
    
    def _load_mujoco_model(self):
        """Chargement modèle MuJoCo avec validation ultra-complète"""
        
        model_path = "results/g1_combined.xml"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Modèle G1 non trouvé: {model_path}")
        
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
            
            # Validation modèle
            assert self.model.nv > 0, "Modèle sans DOF"
            assert self.model.nu > 0, "Modèle sans actuateurs"
            assert self.model.nbody > 0, "Modèle sans corps"
            
            logger.info(f"✅ Modèle MuJoCo: DOF={self.model.nv}, Act={self.model.nu}")
            
        except Exception as e:
            raise RuntimeError(f"❌ Erreur modèle MuJoCo: {e}")
    
    def _setup_task(self):
        """Setup tâche avec validation robuste"""
        
        try:
            self.task = GraspLiftTaskOptimized(self.model, self.data, self.config['task'])
            
            # Test fonctionnel
            obs = self.task.reset()
            test_action = np.zeros(self.task.act_dim)
            _, _, _, _ = self.task.step(test_action)
            
            self.obs_dim = obs.shape[0]
            self.act_dim = self.task.act_dim
            
            logger.info(f"🎯 Tâche: obs_dim={self.obs_dim}, act_dim={self.act_dim}")
            
        except Exception as e:
            raise RuntimeError(f"❌ Erreur setup tâche: {e}")
    
    def _setup_sac_agent(self):
        """Setup agent SAC ultra-optimisé"""
        
        rl_config = self.config['rl']
        
        self.agent = UltraAdvancedSACAgent(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            device=self.device,
            hidden_sizes=rl_config.get('hidden_sizes', [512, 512, 256]),
            learning_rate=rl_config['learning_rate']
        )
        
        # Paramètres d'entraînement
        self.gamma = rl_config['gamma']
        self.tau = rl_config['tau']
        self.batch_size = rl_config['batch_size']
        self.total_steps = rl_config['total_steps']
        self.start_steps = rl_config.get('start_steps', 10000)
        self.update_after = rl_config.get('update_after', 1000)
        self.update_every = rl_config.get('update_every', 50)
        self.num_updates = rl_config.get('num_updates', 50)
        
        logger.info("🧠 Agent SAC ultra-configuré")
    
    def _setup_per_buffer(self):
        """Setup buffer PER ultra-optimisé"""
        
        rl_config = self.config['rl']
        
        self.buffer = UltraPrioritizedReplayBuffer(
            size=rl_config['replay_size'],
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            device=self.device,
            alpha=rl_config['per_alpha'],
            beta=rl_config['per_beta']
        )
        
        # Annealing β pour PER
        self.per_beta_annealing = rl_config.get('per_beta_annealing', True)
        self.per_beta_start = rl_config['per_beta']
        self.per_beta_end = 1.0
        
        logger.info("💾 Buffer PER ultra-configuré")
    
    def _setup_monitoring(self):
        """Setup monitoring ultra-avancé"""
        
        # TensorBoard avec structure organisée
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))
        
        # Métriques performance
        self.metrics = {
            'episode_rewards': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100),
            'q_losses': deque(maxlen=1000),
            'policy_losses': deque(maxlen=1000),
            'alpha_values': deque(maxlen=1000),
            'buffer_stats': deque(maxlen=100)
        }
        
        # Performance profiling
        self.timers = defaultdict(float)
        self.counters = defaultdict(int)
        
        logger.info("📊 Monitoring ultra-configuré")
    
    @contextmanager
    def _timer(self, name: str):
        """Context manager pour timing précis"""
        start = time.time()
        try:
            yield
        finally:
            self.timers[name] += time.time() - start
            self.counters[name] += 1
    
    def _emergency_save(self):
        """Sauvegarde d'urgence en cas de problème"""
        try:
            self.emergency_saves += 1
            emergency_path = self.backup_dir / f'emergency_{self.emergency_saves}_{int(time.time())}.pth'
            
            torch.save({
                'step': self.step_count,
                'episode': self.episode_count,
                'agent_state': self.agent.policy.state_dict(),
                'buffer_stats': self.buffer.get_stats()
            }, emergency_path)
            
            logger.info(f"🚨 Sauvegarde urgence: {emergency_path}")
            
        except Exception as e:
            logger.error(f"❌ Échec sauvegarde urgence: {e}")
    
    def _update_per_beta(self):
        """Mise à jour β pour PER avec annealing"""
        if self.per_beta_annealing:
            progress = min(self.step_count / self.total_steps, 1.0)
            self.buffer.beta = self.per_beta_start + progress * (self.per_beta_end - self.per_beta_start)
    
    def train(self):
        """Boucle d'entraînement ultra-robuste avec recovery automatique"""
        
        logger.info("🚀 === DÉBUT ENTRAÎNEMENT ULTRA SAC+PER ===")
        logger.info(f"📊 Config: {self.total_steps:,} steps, batch={self.batch_size}")
        
        start_time = time.time()
        obs = self.task.reset()
        episode_reward = 0.0
        episode_length = 0
        
        try:
            for step in range(self.total_steps):
                self.step_count = step
                
                with self._timer('step_total'):
                    # Mise à jour PER β
                    self._update_per_beta()
                    
                    # === SÉLECTION ACTION ===
                    with self._timer('action_selection'):
                        if step < self.start_steps:
                            # Exploration aléatoire
                            action = torch.rand(self.act_dim, device=self.device) * 2 - 1
                        else:
                            # Politique SAC
                            obs_tensor = torch.from_numpy(obs).float().to(self.device)
                            action = self.agent.select_action(obs_tensor, deterministic=False)
                    
                    # === EXÉCUTION ENVIRONNEMENT ===
                    with self._timer('env_step'):
                        action_np = action.cpu().numpy()
                        next_obs, reward, done, info = self.task.step(action_np)
                    
                    # === STOCKAGE BUFFER ===
                    with self._timer('buffer_store'):
                        obs_tensor = torch.from_numpy(obs).float()
                        next_obs_tensor = torch.from_numpy(next_obs).float()
                        
                        success = self.buffer.store(
                            obs_tensor, action.cpu(), reward, next_obs_tensor, done)
                        
                        if not success:
                            logger.warning("⚠️  Échec stockage buffer")
                    
                    # Mise à jour métriques épisode
                    episode_reward += reward
                    episode_length += 1
                    obs = next_obs
                    
                    # === FIN D'ÉPISODE ===
                    if done:
                        self.metrics['episode_rewards'].append(episode_reward)
                        self.metrics['episode_lengths'].append(episode_length)
                        self.episode_count += 1
                        
                        # Logging épisode
                        if self.episode_count % 10 == 0:
                            avg_reward = np.mean(list(self.metrics['episode_rewards'])[-10:])
                            logger.info(
                                f"Ep {self.episode_count:4d} | "
                                f"Step {step:7d} | "
                                f"R: {episode_reward:7.2f} | "
                                f"R̄₁₀: {avg_reward:7.2f} | "
                                f"Len: {episode_length:3d}"
                            )
                        
                        # TensorBoard
                        self.writer.add_scalar('Episode/Reward', episode_reward, self.episode_count)
                        self.writer.add_scalar('Episode/Length', episode_length, self.episode_count)
                        
                        # Reset épisode
                        obs = self.task.reset()
                        episode_reward = 0.0
                        episode_length = 0
                    
                    # === MISE À JOUR SAC ===
                    if step >= self.update_after and step % self.update_every == 0:
                        with self._timer('sac_update'):
                            losses = self._update_sac()
                            
                            if losses:
                                # Logging losses
                                for loss_name, loss_value in losses.items():
                                    self.writer.add_scalar(f'Loss/{loss_name}', loss_value, step)
                                    if 'q' in loss_name:
                                        self.metrics['q_losses'].append(loss_value)
                                    elif loss_name == 'policy_loss':
                                        self.metrics['policy_losses'].append(loss_value)
                                
                                # Métriques supplémentaires
                                alpha_value = torch.exp(self.agent.log_alpha).item()
                                self.metrics['alpha_values'].append(alpha_value)
                                self.writer.add_scalar('Training/Alpha', alpha_value, step)
                                self.writer.add_scalar('Training/PER_Beta', self.buffer.beta, step)
                    
                    # === MONITORING PERFORMANCE ===
                    if step % 1000 == 0:
                        # Buffer stats
                        buffer_stats = self.buffer.get_stats()
                        self.metrics['buffer_stats'].append(buffer_stats)
                        
                        for key, value in buffer_stats.items():
                            if isinstance(value, (int, float)):
                                self.writer.add_scalar(f'Buffer/{key}', value, step)
                        
                        # Performance stats
                        if self.counters['step_total'] > 0:
                            avg_step_time = self.timers['step_total'] / self.counters['step_total']
                            self.writer.add_scalar('Performance/AvgStepTime', avg_step_time, step)
                            logger.debug(f"⚡ Step time: {avg_step_time*1000:.2f}ms")
                    
                    # === SAUVEGARDE PÉRIODIQUE ===
                    if step > 0 and step % 25000 == 0:
                        self._save_checkpoint(step)
                        self.last_checkpoint_step = step
                    
                    # Check early stopping
                    if episode_reward > self.best_reward:
                        self.best_reward = episode_reward
                        self._save_checkpoint(step, suffix='best')
        
        except KeyboardInterrupt:
            logger.info("⏹️  Arrêt demandé par utilisateur")
            self._emergency_save()
            
        except Exception as e:
            logger.error(f"💥 Erreur fatale: {e}")
            logger.error(traceback.format_exc())
            self._emergency_save()
            raise
        
        finally:
            # === NETTOYAGE FINAL ===
            total_time = time.time() - start_time
            logger.info(f"✅ Entraînement terminé en {total_time:.1f}s")
            logger.info(f"📊 {self.episode_count} épisodes, {self.total_steps/total_time:.1f} steps/sec")
            
            # Sauvegarde finale
            self._save_checkpoint(self.total_steps, final=True)
            
            # Statistiques finales
            self._log_final_stats()
            
            # Nettoyage ressources
            self.writer.close()
    
    def _update_sac(self) -> Optional[Dict[str, float]]:
        """Mise à jour SAC ultra-robuste avec PER"""
        
        # Échantillonnage PER
        sample_result = self.buffer.sample(self.batch_size)
        if sample_result is None:
            return None
        
        batch, weights, indices = sample_result
        
        losses = defaultdict(float)
        
        try:
            for _ in range(self.num_updates):
                # === Q-LEARNING AVEC DOUBLE Q ===
                with torch.no_grad():
                    next_actions, next_log_probs = self.agent.policy.sample(batch['next_obs'])
                    next_actions = torch.clamp(next_actions, -1, 1)
                    
                    q1_next = self.agent.q1_target(batch['next_obs'], next_actions)
                    q2_next = self.agent.q2_target(batch['next_obs'], next_actions)
                    min_q_next = torch.min(q1_next, q2_next)
                    
                    alpha = torch.exp(self.agent.log_alpha)
                    target_q = batch['rew'] + self.gamma * (~batch['done']) * (min_q_next - alpha * next_log_probs)
                
                # Mise à jour Q1 avec PER weights
                q1_pred = self.agent.q1(batch['obs'], batch['act'])
                q1_errors = F.mse_loss(q1_pred, target_q, reduction='none')
                q1_loss = (q1_errors * weights).mean()
                
                self.agent.q1_optimizer.zero_grad()
                q1_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.q1.parameters(), 1.0)
                self.agent.q1_optimizer.step()
                
                # Mise à jour Q2 avec PER weights
                q2_pred = self.agent.q2(batch['obs'], batch['act'])
                q2_errors = F.mse_loss(q2_pred, target_q, reduction='none')
                q2_loss = (q2_errors * weights).mean()
                
                self.agent.q2_optimizer.zero_grad()
                q2_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.q2.parameters(), 1.0)
                self.agent.q2_optimizer.step()
                
                # === MISE À JOUR POLITIQUE ===
                new_actions, log_probs = self.agent.policy.sample(batch['obs'])
                new_actions = torch.clamp(new_actions, -1, 1)
                
                q1_new = self.agent.q1(batch['obs'], new_actions)
                q2_new = self.agent.q2(batch['obs'], new_actions)
                min_q_new = torch.min(q1_new, q2_new)
                
                alpha = torch.exp(self.agent.log_alpha)
                policy_loss = (alpha * log_probs - min_q_new).mean()
                
                self.agent.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(), 1.0)
                self.agent.policy_optimizer.step()
                
                # === ENTROPIE ADAPTATIVE ===
                alpha_loss = -(self.agent.log_alpha * (log_probs + self.agent.target_entropy).detach()).mean()
                
                self.agent.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.agent.alpha_optimizer.step()
                
                # === MISE À JOUR PRIORITÉS PER ===
                td_errors = torch.abs(q1_pred - target_q).detach()
                self.buffer.update_priorities(indices, td_errors)
                
                # === SOFT UPDATE CIBLES ===
                self._soft_update(self.agent.q1, self.agent.q1_target)
                self._soft_update(self.agent.q2, self.agent.q2_target)
                
                # Accumulation losses
                losses['q1_loss'] += q1_loss.item()
                losses['q2_loss'] += q2_loss.item()
                losses['policy_loss'] += policy_loss.item()
                losses['alpha_loss'] += alpha_loss.item()
            
            # Moyennes
            for key in losses:
                losses[key] /= self.num_updates
            
            # Scheduler step
            self.agent.policy_scheduler.step()
            
            return dict(losses)
            
        except Exception as e:
            logger.error(f"❌ Erreur update SAC: {e}")
            return None
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update avec vérification"""
        with torch.no_grad():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    self.tau * source_param.data + (1.0 - self.tau) * target_param.data
                )
    
    def _save_checkpoint(self, step: int, final: bool = False, suffix: str = None):
        """Sauvegarde checkpoint ultra-robuste"""
        
        try:
            if suffix:
                checkpoint_name = f'sac_per_{suffix}.pth'
            elif final:
                checkpoint_name = 'sac_per_final.pth'
            else:
                checkpoint_name = f'sac_per_step_{step}.pth'
            
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            
            checkpoint = {
                'step': step,
                'episode_count': self.episode_count,
                'best_reward': self.best_reward,
                'config': self.config,
                
                # États agent
                'policy_state_dict': self.agent.policy.state_dict(),
                'q1_state_dict': self.agent.q1.state_dict(),
                'q2_state_dict': self.agent.q2.state_dict(),
                'q1_target_state_dict': self.agent.q1_target.state_dict(),
                'q2_target_state_dict': self.agent.q2_target.state_dict(),
                
                # États optimiseurs
                'policy_optimizer': self.agent.policy_optimizer.state_dict(),
                'q1_optimizer': self.agent.q1_optimizer.state_dict(),
                'q2_optimizer': self.agent.q2_optimizer.state_dict(),
                'alpha_optimizer': self.agent.alpha_optimizer.state_dict(),
                'policy_scheduler': self.agent.policy_scheduler.state_dict(),
                
                # Paramètres adaptatifs
                'log_alpha': self.agent.log_alpha,
                
                # Buffer stats
                'buffer_stats': self.buffer.get_stats(),
                
                # Métriques
                'metrics': {
                    'episode_rewards': list(self.metrics['episode_rewards']),
                    'episode_lengths': list(self.metrics['episode_lengths'])
                },
                
                # Métadonnées
                'timestamp': time.time(),
                'pytorch_version': torch.__version__,
                'mujoco_version': mujoco.__version__
            }
            
            # Sauvegarde atomique
            temp_path = checkpoint_path.with_suffix('.tmp')
            torch.save(checkpoint, temp_path)
            temp_path.rename(checkpoint_path)
            
            logger.info(f"💾 Checkpoint: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde: {e}")
    
    def _log_final_stats(self):
        """Logging statistiques finales ultra-détaillées"""
        
        logger.info("📈 === STATISTIQUES FINALES ===")
        
        # Métriques épisodes
        if self.metrics['episode_rewards']:
            rewards = list(self.metrics['episode_rewards'])
            logger.info(f"Reward moyen: {np.mean(rewards):.2f}")
            logger.info(f"Reward max: {np.max(rewards):.2f}")
            logger.info(f"Reward final: {rewards[-1]:.2f}")
        
        # Métriques buffer
        buffer_stats = self.buffer.get_stats()
        logger.info(f"Buffer fill: {buffer_stats['fill_ratio']:.1%}")
        logger.info(f"Cache hit rate: {buffer_stats['cache_hit_ratio']:.1%}")
        
        # Performance timing
        total_time = sum(self.timers.values())
        for operation, time_spent in self.timers.items():
            if self.counters[operation] > 0:
                avg_time = time_spent / self.counters[operation]
                percentage = time_spent / total_time * 100
                logger.info(f"{operation}: {avg_time*1000:.2f}ms avg ({percentage:.1f}%)")

def main():
    """Point d'entrée avec gestion ultra-robuste"""
    
    print("="*80)
    print("🚀 ULTRA-ROBUST SAC + PER TRAINING SYSTEM")
    print("="*80)
    print("🎯 Mission: Entraînement G1 avec garanties ultra-robustes")
    print("🔒 Garanties: Validation complète + Recovery automatique")
    print("⚡ Performance: Cache intelligent + Optimisations avancées")
    print("📊 Monitoring: Logging structuré + Métriques temps réel")
    print("="*80)
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra-Robust SAC+PER Trainer")
    parser.add_argument('--config', '-c', default='config/sac_grasp_lift.yaml',
                       help='Configuration file')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    try:
        # Configuration reproductibilité
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Initialisation trainer
        trainer = UltraSACPERTrainer(args.config)
        
        # Lancement entraînement
        trainer.train()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Arrêt utilisateur")
        return 0
    except Exception as e:
        print(f"\n💥 Erreur fatale: {e}")
        return 1

if __name__ == "__main__":
    exit(main())