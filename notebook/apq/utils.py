import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np


def compute_output_loss(O_ref: torch.Tensor, O_hat: torch.Tensor) -> torch.Tensor:
    """
    Normalized Frobenius loss, as in the proposal:
    L_output = ||O_ref - O_hat||_F / (||O_ref||_F + eps)
    """
    eps = 1e-8
    diff = O_ref - O_hat
    num = torch.norm(diff, p='fro')
    denom = torch.norm(O_ref, p='fro') + eps
    return num / denom


def compute_kl_loss(P_ref: torch.Tensor, P_hat: torch.Tensor,
                    attn_mask: torch.Tensor = None) -> torch.Tensor:
    """
    KL(P_ref || P_hat), averaged over valid query positions only.
    P_ref, P_hat: [B, H, T, T]
    attn_mask: [B, T] with 1 for valid tokens, 0 for padding.
    """
    eps = 1e-8
    P_ref = P_ref.clamp(min=eps, max=1.0)
    P_hat = P_hat.clamp(min=eps, max=1.0)

    log_P_ref = torch.log(P_ref)
    log_P_hat = torch.log(P_hat)

    kl = (P_ref * (log_P_ref - log_P_hat))  # [B, H, T, T]

    if attn_mask is not None:
        # attn_mask: [B, T] → [B, 1, T, 1] to mask query positions
        q_mask = (attn_mask == 1).unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1]
        kl = kl * q_mask

        valid_queries = q_mask.sum()
        if valid_queries > 0:
            return kl.sum() / valid_queries
        else:
            return kl.mean()
    else:
        return kl.mean()


def compute_entropy_loss(P_ref: torch.Tensor, P_hat: torch.Tensor,
                         attn_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Optional entropy-preservation loss:
    |H(P_ref) - H(P_hat)|, averaged over valid query positions.
    """
    eps = 1e-8
    P_ref = P_ref.clamp(min=eps, max=1.0)
    P_hat = P_hat.clamp(min=eps, max=1.0)

    H_ref = -(P_ref * torch.log(P_ref)).sum(dim=-1)  # [B, H, T]
    H_hat = -(P_hat * torch.log(P_hat)).sum(dim=-1)  # [B, H, T]

    ent_diff = torch.abs(H_ref - H_hat)  # [B, H, T]

    if attn_mask is not None:
        q_mask = (attn_mask == 1).unsqueeze(1)  # [B, 1, T]
        ent_diff = ent_diff * q_mask

        valid_queries = q_mask.sum()
        if valid_queries > 0:
            return ent_diff.sum() / valid_queries
        else:
            return ent_diff.mean()
    else:
        return ent_diff.mean()


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
    l_out = compute_output_loss(O, O_hat)
    l_kl = compute_kl_loss(P, P_hat)
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
    l_out = compute_output_loss(O, O_hat)
    l_kl = compute_kl_loss(P, P_hat)
    l_ent = compute_entropy_loss(P, P_hat)
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
    l_out = compute_output_loss(O, O_hat)
    l_kl = compute_kl_loss(P, P_hat)
    
    if weight is not None:
        l_out = l_out * weight
        l_kl = l_kl * weight
    
    return l_out + lam * l_kl, l_out, l_kl