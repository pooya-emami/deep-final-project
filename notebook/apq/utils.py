import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np


def compute_output_loss(O: torch.Tensor, O_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute normalized Frobenius loss for output preservation.
    
    Args:
        O: Reference FP16 output
        O_hat: Quantized output
        eps: Small constant for numerical stability
    
    Returns:
        Normalized Frobenius loss
    """
    diff = O_hat - O
    return torch.norm(diff, p='fro')**2 / (torch.norm(O, p='fro')**2 + eps)


def compute_kl_loss(P: torch.Tensor, P_hat: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None, 
                    eps: float = 1e-8) -> torch.Tensor:
    """
    Compute KL divergence between attention distributions.
    
    Args:
        P: Reference attention distribution [batch, num_heads, seq_len, seq_len]
        P_hat: Quantized attention distribution [batch, num_heads, seq_len, seq_len]
        mask: Optional mask [batch, seq_len] for valid positions
        eps: Small constant for numerical stability
    
    Returns:
        KL divergence averaged over batch and heads
    """
    # KL(P || P_hat) = sum(P * log(P/P_hat)) over keys dimension
    kl = P * (torch.log(P + eps) - torch.log(P_hat + eps))
    
    # Sum over key dimension (last dim)
    kl_sum = kl.sum(dim=-1)  # [batch, num_heads, seq_len]
    
    # Apply mask if provided
    if mask is not None:
        # Expand mask to match dimensions
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq_len, 1]
            kl_sum = kl_sum * mask.squeeze(-1)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            kl = kl * mask
            kl_sum = kl.sum(dim=-1)
    
    # Average over all dimensions
    return kl_sum.mean()


def compute_entropy(P: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute entropy of attention distribution.
    
    Args:
        P: Attention distribution [batch, num_heads, seq_len, seq_len]
        eps: Small constant for numerical stability
    
    Returns:
        Entropy averaged over batch and heads
    """
    return -(P * torch.log(P + eps)).sum(dim=-1).mean()


def compute_entropy_loss(P: torch.Tensor, P_hat: torch.Tensor, 
                         eps: float = 1e-8) -> torch.Tensor:
    """
    Compute absolute difference in entropy between FP and quantized distributions.
    
    Args:
        P: Reference attention distribution
        P_hat: Quantized attention distribution
        eps: Small constant for numerical stability
    
    Returns:
        Absolute entropy difference
    """
    H = compute_entropy(P, eps)
    H_hat = compute_entropy(P_hat, eps)
    return torch.abs(H - H_hat)


def compute_joint_loss(O: torch.Tensor, O_hat: torch.Tensor,
                       P: torch.Tensor, P_hat: torch.Tensor,
                       lam: float, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute joint loss L_Joint = L_output + λ * L_KL
    
    Args:
        O: Reference output
        O_hat: Quantized output
        P: Reference attention distribution
        P_hat: Quantized attention distribution
        lam: Lambda weighting factor
        eps: Small constant for numerical stability
    
    Returns:
        Tuple of (total_loss, l_output, l_kl)
    """
    l_out = compute_output_loss(O, O_hat, eps)
    l_kl = compute_kl_loss(P, P_hat, eps=eps)
    return l_out + lam * l_kl, l_out, l_kl


def compute_joint_loss_with_entropy(O: torch.Tensor, O_hat: torch.Tensor,
                                   P: torch.Tensor, P_hat: torch.Tensor,
                                   lam: float, beta: float = 0.1,
                                   eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute joint loss with entropy preservation.
    
    Args:
        O: Reference output
        O_hat: Quantized output
        P: Reference attention distribution
        P_hat: Quantized attention distribution
        lam: Lambda weighting factor for KL
        beta: Beta weighting factor for entropy
        eps: Small constant for numerical stability
    
    Returns:
        Tuple of (total_loss, l_output, l_kl, l_entropy)
    """
    l_out = compute_output_loss(O, O_hat, eps)
    l_kl = compute_kl_loss(P, P_hat, eps=eps)
    l_ent = compute_entropy_loss(P, P_hat, eps)
    return l_out + lam * l_kl + beta * l_ent, l_out, l_kl, l_ent


def normalize_attention(P: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize attention distribution to sum to 1.
    
    Args:
        P: Attention distribution
        eps: Small constant for numerical stability
    
    Returns:
        Normalized attention distribution
    """
    return P / (P.sum(dim=-1, keepdim=True) + eps)


def smooth_attention(P: torch.Tensor, alpha: float = 0.1, eps: float = 1e-8) -> torch.Tensor:
    """
    Apply label smoothing to attention distribution.
    
    Args:
        P: Attention distribution
        alpha: Smoothing factor
        eps: Small constant for numerical stability
    
    Returns:
        Smoothed attention distribution
    """
    K = P.shape[-1]
    return (1 - alpha) * P + alpha / K


def top_k_attention(P: torch.Tensor, k: int) -> torch.Tensor:
    """
    Keep only top-k attention weights, zero out others.
    
    Args:
        P: Attention distribution
        k: Number of top values to keep
    
    Returns:
        Sparse attention distribution
    """
    values, indices = torch.topk(P, k, dim=-1)
    mask = torch.zeros_like(P)
    mask.scatter_(-1, indices, values)
    return mask / (mask.sum(dim=-1, keepdim=True) + 1e-8)


def compute_attention_entropy_stats(P: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics about attention entropy.
    
    Args:
        P: Attention distribution
    
    Returns:
        Dictionary of statistics
    """
    with torch.no_grad():
        H = compute_entropy(P)
        H_min = H.min().item()
        H_max = H.max().item()
        H_mean = H.mean().item()
        H_std = H.std().item()
        
        # Compute effective number of tokens attended to
        eff_tokens = torch.exp(H)
        
        return {
            'entropy_mean': H_mean,
            'entropy_std': H_std,
            'entropy_min': H_min,
            'entropy_max': H_max,
            'effective_tokens_mean': eff_tokens.mean().item(),
            'effective_tokens_std': eff_tokens.std().item(),
        }


class ActivationStats:
    """Track activation statistics for AWQ-style weighting"""
    
    def __init__(self):
        self.stats = {}
    
    def update(self, name: str, x: torch.Tensor):
        """Update statistics for a tensor"""
        with torch.no_grad():
            if name not in self.stats:
                self.stats[name] = {
                    'mean': [],
                    'std': [],
                    'max': [],
                    'min': [],
                }
            
            self.stats[name]['mean'].append(x.mean().item())
            self.stats[name]['std'].append(x.std().item())
            self.stats[name]['max'].append(x.max().item())
            self.stats[name]['min'].append(x.min().item())
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get aggregated statistics for a tensor"""
        if name not in self.stats:
            return {}
        
        s = self.stats[name]
        return {
            'mean': np.mean(s['mean']),
            'std': np.mean(s['std']),
            'max': np.max(s['max']),
            'min': np.min(s['min']),
        }
    
    def clear(self):
        """Clear all statistics"""
        self.stats.clear()


def compute_activation_weighted_loss(O: torch.Tensor, O_hat: torch.Tensor,
                                    P: torch.Tensor, P_hat: torch.Tensor,
                                    lam: float, weight: Optional[torch.Tensor] = None,
                                    eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute loss with activation-aware weighting (mini-AWQ style).
    
    Args:
        O: Reference output
        O_hat: Quantized output
        P: Reference attention distribution
        P_hat: Quantized attention distribution
        lam: Lambda weighting factor
        weight: Optional per-sample weights
        eps: Small constant for numerical stability
    
    Returns:
        Tuple of (total_loss, l_output, l_kl)
    """
    l_out = compute_output_loss(O, O_hat, eps)
    l_kl = compute_kl_loss(P, P_hat, eps=eps)
    
    if weight is not None:
        l_out = l_out * weight
        l_kl = l_kl * weight
    
    return l_out + lam * l_kl, l_out, l_kl