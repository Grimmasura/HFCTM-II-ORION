"""
ORION Chiral Inversion Trainer
Implements chiral inversion mechanics for recursive optimization with anti-symmetric learning
and common-mode drift detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass
import hashlib
from prometheus_client import Counter, Gauge, Histogram
import logging

logger = logging.getLogger(__name__)

# Prometheus metrics
CHIRAL_CONSISTENCY = Gauge('orion_chiral_consistency', 'Chiral consistency metric C_χ')
COMMON_MODE_EIGENVAL = Gauge('orion_common_mode_eigenval', 'Top eigenvalue of correlation matrix')
GRADIENT_GATING_RATIO = Gauge('orion_gradient_gating_ratio', 'Ratio of gated common-mode gradients')
ANTI_SYM_LOSS = Histogram('orion_anti_sym_loss', 'Anti-symmetry loss values')

@dataclass
class ChiralConfig:
    """Configuration for chiral inversion mechanics"""
    lambda_anti: float = 0.1      # Anti-symmetric learning weight
    lambda_sym: float = 0.05      # Common-mode symmetry weight
    corr_threshold: float = 0.8   # Correlation threshold for gradient gating
    gating_alpha: float = 0.3     # Common-mode gating factor
    symmetry_op: str = "anti"     # "anti" for S=-I, "align" for S=+I
    distance_metric: str = "js"   # "js", "kl", "l2"

class ChiralInversion:
    """
    Implements chiral inversion operators J and action mapping M
    """
    
    def __init__(self, config: ChiralConfig):
        self.config = config
        self.involution_cache = {}
        
    def text_inversion(self, text: str) -> str:
        """Text domain chiral inversion: polarity flip, negation, temporal reversal"""
        # Polarity/valence flip
        polarity_map = {
            'good': 'bad', 'bad': 'good', 'positive': 'negative', 'negative': 'positive',
            'yes': 'no', 'no': 'yes', 'true': 'false', 'false': 'true',
            'before': 'after', 'after': 'before', 'first': 'last', 'last': 'first',
            'start': 'end', 'end': 'start', 'begin': 'finish', 'finish': 'begin'
        }
        
        # Simple word-level inversion (can be enhanced with semantic models)
        words = text.lower().split()
        inverted_words = []
        
        for word in words:
            if word in polarity_map:
                inverted_words.append(polarity_map[word])
            else:
                inverted_words.append(word)
        
        # Temporal flip: reverse sentence order
        sentences = ' '.join(inverted_words).split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        inverted_text = '. '.join(reversed(sentences))
        
        return inverted_text
    
    def tensor_inversion(self, x: torch.Tensor, mode: str = "parity") -> torch.Tensor:
        """Tensor domain chiral inversion: parity, time-reverse, Hodge dual"""
        if mode == "parity":
            # Parity/reflection across feature dimensions
            return torch.flip(x, dims=[-1])
        elif mode == "time_reverse":
            # Time reversal for sequential data
            if x.dim() >= 2:
                return torch.flip(x, dims=[1])  # Assume time is dim 1
            return x
        elif mode == "hodge_dual":
            # Simplified Hodge dual approximation
            return x.transpose(-2, -1) if x.dim() >= 2 else x
        else:
            raise ValueError(f"Unknown tensor inversion mode: {mode}")
    
    def __call__(self, x: Any, domain: str = "tensor") -> Any:
        """Apply chiral inversion J with J(J(x)) = x property"""
        cache_key = (str(x)[:50] if isinstance(x, str) else str(x.shape), domain)
        
        if cache_key in self.involution_cache:
            return self.involution_cache[cache_key]
        
        if domain == "text" and isinstance(x, str):
            result = self.text_inversion(x)
        elif domain == "tensor" and isinstance(x, torch.Tensor):
            result = self.tensor_inversion(x)
        else:
            raise ValueError(f"Unsupported domain {domain} for input type {type(x)}")
        
        self.involution_cache[cache_key] = result
        return result

class ActionMapping:
    """
    Implements action mapping M with M(M(a)) = a property
    """
    
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        
    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply action mapping M"""
        # Simple negation mapping for continuous actions
        return -actions
    
    def inverse(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply M^(-1)"""
        return -actions  # Self-inverse for negation

class ChiralTrainer:
    """
    Main trainer implementing chiral inversion mechanics with anti-symmetric learning
    """
    
    def __init__(self, model: nn.Module, config: ChiralConfig):
        self.model = model
        self.config = config
        self.chiral_inv = ChiralInversion(config)
        self.action_map = ActionMapping(getattr(model, 'action_dim', 64))
        
        # Gradient tracking
        self.grad_history = []
        self.correlation_history = []
        
    def compute_distance(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """Compute distance between probability distributions or action embeddings"""
        if self.config.distance_metric == "js":
            # Jensen-Shannon divergence
            m = 0.5 * (p1 + p2)
            return 0.5 * (F.kl_div(p1.log(), m, reduction='batchmean') + 
                         F.kl_div(p2.log(), m, reduction='batchmean'))
        elif self.config.distance_metric == "kl":
            return F.kl_div(p1.log(), p2, reduction='batchmean')
        elif self.config.distance_metric == "l2":
            return F.mse_loss(p1, p2)
        else:
            raise ValueError(f"Unknown distance metric: {self.config.distance_metric}")
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract internal features φ_θ(x) from model"""
        if hasattr(self.model, 'encoder'):
            return self.model.encoder(x)
        elif hasattr(self.model, 'features'):
            return self.model.features(x)
        else:
            # Fallback: use intermediate layer
            return x
    
    def compute_chiral_loss(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute full chiral loss with anti-symmetric and common-mode terms"""
        # Original batch
        policy_orig = self.model(x_batch)
        task_loss_orig = F.cross_entropy(policy_orig, y_batch)
        
        # Chiral inverted batch
        x_chiral = torch.stack([self.chiral_inv(x.unsqueeze(0), domain="tensor").squeeze(0) 
                               for x in x_batch])
        policy_chiral = self.model(x_chiral)
        
        # Action mapping for chiral consistency
        policy_chiral_mapped = self.action_map.inverse(policy_chiral)
        
        # Anti-symmetric alignment loss
        align_loss = self.compute_distance(F.softmax(policy_orig, dim=-1), 
                                         F.softmax(policy_chiral_mapped, dim=-1))
        
        # Feature symmetry loss
        feat_orig = self.extract_features(x_batch)
        feat_chiral = self.extract_features(x_chiral)
        
        if self.config.symmetry_op == "anti":
            # Features should anti-align (S = -I)
            sym_loss = torch.mean((feat_orig - feat_chiral) ** 2)
        else:
            # Features should align (S = +I)  
            sym_loss = torch.mean((feat_orig + feat_chiral) ** 2)
        
        # Total loss
        total_loss = (task_loss_orig + 
                     self.config.lambda_anti * align_loss + 
                     self.config.lambda_sym * sym_loss)
        
        # Update metrics
        chiral_consistency = 1.0 - align_loss.item()
        CHIRAL_CONSISTENCY.set(chiral_consistency)
        ANTI_SYM_LOSS.observe(sym_loss.item())
        
        return {
            'total_loss': total_loss,
            'task_loss': task_loss_orig,
            'align_loss': align_loss,
            'sym_loss': sym_loss,
            'chiral_consistency': chiral_consistency
        }
    
    def gradient_decomposition_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> Dict[str, Any]:
        """Perform gradient decomposition with anti/common-mode gating"""
        # Clear gradients
        self.model.zero_grad()
        
        # Compute loss for original batch
        policy_orig = self.model(x_batch)
        task_loss_orig = F.cross_entropy(policy_orig, y_batch)
        task_loss_orig.backward(retain_graph=True)
        
        # Capture gradients from original path
        g_plus = []
        for param in self.model.parameters():
            if param.grad is not None:
                g_plus.append(param.grad.clone())
            else:
                g_plus.append(torch.zeros_like(param))
        
        # Clear gradients and compute for chiral batch
        self.model.zero_grad()
        x_chiral = torch.stack([self.chiral_inv(x.unsqueeze(0), domain="tensor").squeeze(0) 
                               for x in x_batch])
        policy_chiral = self.model(x_chiral)
        task_loss_chiral = F.cross_entropy(policy_chiral, y_batch)
        task_loss_chiral.backward()
        
        # Capture gradients from chiral path
        g_minus = []
        for param in self.model.parameters():
            if param.grad is not None:
                g_minus.append(param.grad.clone())
            else:
                g_minus.append(torch.zeros_like(param))
        
        # Decompose gradients
        g_anti = []
        g_com = []
        
        for gp, gm in zip(g_plus, g_minus):
            g_anti.append(0.5 * (gp - gm))
            g_com.append(0.5 * (gp + gm))
        
        # Compute correlation for gating
        g_plus_flat = torch.cat([g.flatten() for g in g_plus])
        g_minus_flat = torch.cat([g.flatten() for g in g_minus])
        
        correlation = torch.corrcoef(torch.stack([g_plus_flat, g_minus_flat]))[0, 1]
        COMMON_MODE_EIGENVAL.set(abs(correlation.item()))
        
        # Apply correlation-aware gating
        gating_factor = 1.0
        if abs(correlation) > self.config.corr_threshold:
            gating_factor = self.config.gating_alpha
            logger.info(f"High correlation detected ({correlation:.3f}), applying gating factor {gating_factor}")
        
        GRADIENT_GATING_RATIO.set(gating_factor)
        
        # Apply gated update with telemetry
        self.model.zero_grad()
        cancellation_gain = 0.0
        
        if hasattr(self, '_last_loss'):
            # Estimate cancellation gain
            anti_norm = torch.cat([g.flatten() for g in g_anti]).norm().item()
            com_norm = torch.cat([g.flatten() for g in g_com]).norm().item()
            
            # Simplified gain estimate (would need actual loss evaluation for precision)
            if com_norm > 0 and anti_norm > 0:
                cancellation_ratio = anti_norm / (anti_norm + com_norm)
                cancellation_gain = cancellation_ratio * gating_factor
        
        for param, ga, gc in zip(self.model.parameters(), g_anti, g_com):
            if param.grad is not None:
                param.grad = ga + gating_factor * gc
            else:
                param.grad = ga + gating_factor * gc
        
        return {
            'correlation': correlation.item(),
            'gating_factor': gating_factor,
            'g_anti_norm': torch.cat([g.flatten() for g in g_anti]).norm().item(),
            'g_com_norm': torch.cat([g.flatten() for g in g_com]).norm().item(),
            'cancellation_gain': cancellation_gain
        }
    
    def training_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Full training step with chiral inversion mechanics"""
        # Standard loss computation
        loss_dict = self.compute_chiral_loss(x_batch, y_batch)
        
        # Gradient decomposition and gating
        grad_dict = self.gradient_decomposition_step(x_batch, y_batch)
        
        # Optimizer step
        optimizer.step()
        
        # Combine metrics
        metrics = {**loss_dict, **grad_dict}
        
        # Track history for analysis
        self.correlation_history.append(grad_dict['correlation'])
        if len(self.correlation_history) > 1000:  # Keep last 1000 steps
            self.correlation_history = self.correlation_history[-1000:]
        
        return metrics
    
    def get_chiral_consistency(self, test_batch: torch.Tensor) -> float:
        """Evaluate chiral consistency on test data"""
        self.model.eval()
        with torch.no_grad():
            policy_orig = F.softmax(self.model(test_batch), dim=-1)
            
            x_chiral = torch.stack([self.chiral_inv(x.unsqueeze(0), domain="tensor").squeeze(0) 
                                   for x in test_batch])
            policy_chiral = F.softmax(self.model(x_chiral), dim=-1)
            policy_chiral_mapped = self.action_map.inverse(policy_chiral)
            
            distance = self.compute_distance(policy_orig, policy_chiral_mapped)
            consistency = 1.0 - distance.item()
            
        self.model.train()
        return consistency

__all__ = [
    "ChiralConfig",
    "ChiralInversion",
    "ActionMapping",
    "ChiralTrainer",
]
