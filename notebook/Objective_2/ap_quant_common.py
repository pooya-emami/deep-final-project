"""
Shared utilities for AP-Quant -- used by BOTH Objective 1 and Objective 2.

IMPORTANT: everything from `set_seed` through `collect_layer_inputs` below is
copied verbatim from the merged notebook (ap-quant.ipynb, Objective 1's
code), NOT reimplemented -- so Objective 2 measures sensitivity using the exact
same quantized-attention machinery that Objective 1's calibration step uses.
Only `Config`, `load_model_and_tokenizer`, `load_calibration_data` at the bottom
are new, written to match that notebook's driver-cell conventions (model, data,
chunking) so both objectives run on identical inputs.

Do not hand-edit the ported section -- if the teammate's notebook changes
(new arch support, a loss fix, etc.), re-copy it from there instead.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os
import numpy as np

# ============================================================
# FIX: Set global seed for reproducibility
# ============================================================
def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

print("=" * 70)
print(f"AP-QUANT: JOINT CALIBRATION (ALL LAYERS, FIXED 4-BIT, DYNAMIC λ, QKV-ONLY + FP32 Wo, GPT2+LLaMA)")
print(f"SEED: {SEED}")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ============================================================
# 1. QUANTIZATION MODULES
# ============================================================

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scale, q_max):
        # scale: [1, out_features] (broadcast over in_features)
        scale_c = torch.clamp(scale, min=1e-8)
        w_scaled = weight / scale_c          # [in_features, out_features]
        w_clamped = torch.clamp(w_scaled, -q_max, q_max)
        w_round = torch.round(w_clamped)
        w_quant = w_round * scale_c

        ctx.save_for_backward(w_scaled, w_clamped, w_round, scale_c)
        ctx.q_max = q_max
        return w_quant

    @staticmethod
    def backward(ctx, grad_output):
        w_scaled, w_clamped, w_round, scale_c = ctx.saved_tensors
        q_max = ctx.q_max

        # STE for weight: pass gradient where not saturated
        not_clipped = (w_scaled.abs() <= q_max).float()
        grad_weight = grad_output * not_clipped

        # LSQ-style gradient wrt scale (per output channel)
        grad_scale_elem = torch.where(
            w_scaled > q_max, torch.full_like(w_scaled, float(q_max)),
            torch.where(
                w_scaled < -q_max, torch.full_like(w_scaled, -float(q_max)),
                w_round - w_scaled
            )
        )
        contrib = grad_output * grad_scale_elem  # [in_features, out_features]

        # reduce over rows → per-output-channel gradient, shape [1, out_features]
        grad_scale = contrib.sum(dim=0, keepdim=True)

        return grad_weight, grad_scale, None


class QuantizedLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None, bits: int = 4):
        super().__init__()
        # weight [in_features, out_features]
        self.register_buffer("weight_fp", weight.clone().detach())
        if bias is not None:
            self.register_buffer("bias_fp", bias.clone().detach())
        else:
            self.bias_fp = None

        self.bits = bits
        self.q_max = 2 ** (bits - 1) - 1

        init_scale = (
            weight.detach().abs().amax(dim=0, keepdim=True) / self.q_max
        ).clamp(min=1e-6)
        init_raw = torch.log(torch.expm1(init_scale).clamp_min(1e-8))
        self.raw_scale = nn.Parameter(init_raw)  # [1, out_features]

    @property
    def scale(self):
        return F.softplus(self.raw_scale) + 1e-8  # [1, out_features]

    def forward(self, x: torch.Tensor, force_fp: bool = False) -> torch.Tensor:
        # x: [batch, seq, in_features]
        if force_fp:
            w = self.weight_fp
        else:
            w = STEQuantize.apply(self.weight_fp, self.scale, self.q_max)  # [in, out]

        out = torch.matmul(x, w)  # [batch, seq, out_features]
        if self.bias_fp is not None:
            out = out + self.bias_fp
        return out
        
    def set_bits(self, bits: int):
        self.bits = bits
        self.q_max = 2 ** (bits - 1) - 1

        # RTN scale recomputed for this bit-width
        init_scale = (
            self.weight_fp.abs().amax(dim=0, keepdim=True) / self.q_max
        ).clamp(min=1e-6)

        init_raw = torch.log(torch.expm1(init_scale).clamp_min(1e-8))
        self.raw_scale.data = init_raw

# ============================================================
# 1a. GPT2 ATTENTION WRAPPER (QKV-ONLY + FP32 Wo)
# ============================================================

class QuantizedGPT2Attention(nn.Module):
    def __init__(self, original_attn: nn.Module, bits: int = 4):
        super().__init__()

        self.force_fp_mode = False

        self.original_attn = original_attn
        self.embed_dim = original_attn.embed_dim
        self.num_heads = original_attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.attn_dropout = original_attn.attn_dropout
        self.resid_dropout = original_attn.resid_dropout

        # ---- QKV (c_attn) ----
        W = original_attn.c_attn.weight.data
        b = original_attn.c_attn.bias.data if original_attn.c_attn.bias is not None else None

        w_q, w_k, w_v = W.chunk(3, dim=-1)
        b_q, b_k, b_v = b.chunk(3, dim=-1) if b is not None else (None, None, None)

        self.q_proj = QuantizedLinear(w_q, b_q, bits=bits)
        self.k_proj = QuantizedLinear(w_k, b_k, bits=bits)
        self.v_proj = QuantizedLinear(w_v, b_v, bits=bits)

        # ---- Keep original FP32 Wo ----
        self.o_proj_fp = original_attn.c_proj  # nn.Linear, untouched

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, T, self.embed_dim)

    def forward_components(self, hidden_states: torch.Tensor, use_quant: bool = True):
        B, T, _ = hidden_states.shape

        if use_quant:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        else:
            q = self.q_proj(hidden_states, force_fp=True)
            k = self.k_proj(hidden_states, force_fp=True)
            v = self.v_proj(hidden_states, force_fp=True)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale

        causal_mask = torch.triu(
            torch.ones((T, T), device=scores.device, dtype=torch.bool),
            diagonal=1
        )

        mask_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), mask_value)

        P = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        P = self.attn_dropout(P)

        O = torch.matmul(P, v)
        O_merged = self._merge_heads(O)

        return P, O_merged, causal_mask

    def forward(self, hidden_states, layer_past=None, past_key_values=None,
                attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, use_cache=False, output_attentions=False,
                **kwargs):
        if layer_past is None:
            layer_past = past_key_values
        if layer_past is not None:
            raise NotImplementedError("KV-cache not supported.")

        use_quant = not self.force_fp_mode
        P, O_merged, _ = self.forward_components(hidden_states, use_quant=use_quant)

        # Apply FP32 Wo (not quantized)
        attn_output = self.o_proj_fp(O_merged)  # FP32 projection
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output,)
        if use_cache:
            outputs = outputs + (None,)
        if output_attentions:
            outputs = outputs + (P,)
        if len(outputs) == 1:
            outputs = outputs + (None,)
        return outputs

# ============================================================
# 1b. LLaMA ATTENTION WRAPPER (QKV-ONLY + FP32 Wo)
# ============================================================

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states, n_rep):
    # [B, num_kv_heads, T, head_dim] -> [B, num_kv_heads * n_rep, T, head_dim]
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

class LlamaRotaryHelper(nn.Module):
    """RoPE cos/sin generator matching HF LlamaRotaryEmbedding, including
    llama3-style rope scaling used by Llama 3.1/3.2 models."""
    def __init__(self, head_dim, config=None, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

        rope_scaling = getattr(config, "rope_scaling", None) if config is not None else None
        if rope_scaling is not None and rope_scaling.get("rope_type") == "llama3":
            factor = rope_scaling["factor"]
            low_freq_factor = rope_scaling["low_freq_factor"]
            high_freq_factor = rope_scaling["high_freq_factor"]
            old_context_len = rope_scaling["original_max_position_embeddings"]

            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor
            wavelen = 2 * math.pi / inv_freq

            inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smoothed_inv_freq = smooth_factor * inv_freq_llama / factor + (1 - smooth_factor) * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
            inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().to(x.device)
        inv_freq_expanded = inv_freq_expanded.expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class QuantizedLlamaAttention(nn.Module):
    def __init__(self, original_attn, bits=4, rotary_emb=None):
        super().__init__()

        self.force_fp_mode = False

        self.original_attn = original_attn
        cfg = original_attn.config
        self.rotary_emb = rotary_emb

        self.embed_dim = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = getattr(original_attn, "head_dim", self.embed_dim // self.num_heads)
        self.rope_theta = getattr(cfg, "rope_theta", 10000.0)
        self.attn_dropout = nn.Dropout(getattr(original_attn, "attention_dropout", 0.0))

        # nn.Linear weight is [out_features, in_features] — transpose to [in, out]
        # to match QuantizedLinear's Conv1D-style convention (matmul(x, w))
        W_q = original_attn.q_proj.weight.data.T.contiguous()
        b_q = original_attn.q_proj.bias.data if original_attn.q_proj.bias is not None else None
        W_k = original_attn.k_proj.weight.data.T.contiguous()
        b_k = original_attn.k_proj.bias.data if original_attn.k_proj.bias is not None else None
        W_v = original_attn.v_proj.weight.data.T.contiguous()
        b_v = original_attn.v_proj.bias.data if original_attn.v_proj.bias is not None else None

        self.q_proj = QuantizedLinear(W_q, b_q, bits=bits)
        self.k_proj = QuantizedLinear(W_k, b_k, bits=bits)
        self.v_proj = QuantizedLinear(W_v, b_v, bits=bits)

        # ---- Keep original FP32 Wo ----
        self.o_proj_fp = original_attn.o_proj  # nn.Linear, untouched

        self.rotary = LlamaRotaryHelper(self.head_dim, config=cfg, base=self.rope_theta)

    def _split_heads(self, x, num_heads):
        B, T, _ = x.shape
        return x.view(B, T, num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        B, H, T, D = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, T, H * D)

    def forward_components(self, hidden_states, use_quant=True, position_embeddings=None):
        B, T, _ = hidden_states.shape

        if use_quant:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        else:
            q = self.q_proj(hidden_states, force_fp=True)
            k = self.k_proj(hidden_states, force_fp=True)
            v = self.v_proj(hidden_states, force_fp=True)

        q = self._split_heads(q, self.num_heads)     # [B, num_heads, T, head_dim]
        k = self._split_heads(k, self.num_kv_heads)   # [B, num_kv_heads, T, head_dim]
        v = self._split_heads(v, self.num_kv_heads)

        # RoPE — position_ids assumed 0..T-1 (chunks are contiguous, unpadded, like GPT2 path)
        if position_embeddings is not None:
            cos, sin = position_embeddings   # HF passes correct RoPE here
        elif self.rotary_emb is not None:
            # fallback for calibration path (hooks bypass full model forward)
            position_ids = torch.arange(T, device=hidden_states.device).unsqueeze(0).expand(B, -1)
            cos, sin = self.rotary_emb(hidden_states, position_ids)
        else:
            raise RuntimeError("No RoPE available")

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: expand K/V up to num_heads
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale

        causal_mask = torch.triu(
            torch.ones((T, T), device=scores.device, dtype=torch.bool), diagonal=1
        )

        mask_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), mask_value)

        P = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        P = self.attn_dropout(P)

        O = torch.matmul(P, v)
        O_merged = self._merge_heads(O)
        return P, O_merged, causal_mask

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False,
                position_embeddings=None, **kwargs):
        use_quant = not self.force_fp_mode
        P, O_merged, _ = self.forward_components(
            hidden_states,
            use_quant=use_quant,
            position_embeddings=position_embeddings,
        )
        
        # Apply FP32 Wo (not quantized)
        attn_output = self.o_proj_fp(O_merged)
        outputs = (attn_output, None)
        if output_attentions:
            outputs = outputs + (P,)
        return outputs

# ============================================================
# 2. IMPROVED ATTENTION-PRESERVING LOSS (WITH TEMP + MSE)
# ============================================================

def compute_ap_loss_improved(P_fp, O_fp, P_q, O_q, causal_mask,
                         lam=1.0, temp=2.0, eps=1e-8):
    diff = O_fp - O_q
    mse_raw = diff.pow(2).mean()

    l_output = mse_raw / (O_fp.pow(2).mean() + eps)

    P_fp_c = F.softmax(torch.log(P_fp.clamp(min=eps)) / temp, dim=-1)
    P_q_c = F.softmax(torch.log(P_q.clamp(min=eps)) / temp, dim=-1)

    kl_matrix = P_fp_c * (torch.log(P_fp_c + eps) - torch.log(P_q_c + eps))

    valid_mask = (~causal_mask).unsqueeze(0).unsqueeze(0)
    kl_valid = kl_matrix.masked_fill(~valid_mask, 0.0)

    kl_per_query = kl_valid.sum(dim=-1)
    query_valid = valid_mask.any(dim=-1).to(kl_per_query.dtype)
    l_kl = (kl_per_query * query_valid).sum() / query_valid.sum().clamp_min(1.0)

    l_joint = l_output + lam * l_kl
    return l_joint, l_output, l_kl, mse_raw

# ============================================================
# 3. GENERIC TRANSFORMER HELPERS (GPT2 + LLaMA)
# ============================================================

def detect_arch(model):
    if hasattr(model, "transformer"):
        return "gpt2"
    if hasattr(model, "model"):
        return "llama"
    raise ValueError("Unsupported model architecture")

def quantize_transformer_attn(model, layer_bits=None):
    arch = detect_arch(model)
    q_attn_blocks = []

    if arch == "gpt2":
        for layer_idx, block in enumerate(model.transformer.h):
            bits = layer_bits[layer_idx] if layer_bits is not None else 4
            q_attn = QuantizedGPT2Attention(block.attn, bits=bits).to(device)
            block.attn = q_attn
            q_attn_blocks.append(q_attn)
            print(f"  GPT2 layer {layer_idx:02d}: attention quantized to {bits}-bit (QKV-ONLY + FP32 Wo)")
    elif arch == "llama":
        shared_rotary_emb = model.model.rotary_emb
        for layer_idx, block in enumerate(model.model.layers):
            bits = layer_bits[layer_idx] if layer_bits is not None else 4
            q_attn = QuantizedLlamaAttention(
                block.self_attn,
                bits=bits,
                rotary_emb=shared_rotary_emb,
            ).to(device)
            block.self_attn = q_attn
            q_attn_blocks.append(q_attn)
            print(f"  LLaMA layer {layer_idx:02d}: attention quantized to {bits}-bit (QKV-ONLY + FP32 Wo)")
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    return arch, q_attn_blocks

# ============================================================
# FIXED: collect_layer_inputs - captures LN OUTPUT (post-normalized)
# No duplicate passes - just one forward pass through calibration data
# ============================================================
def collect_layer_inputs(model, arch: str, calib_tokens, device):
    """Collect layer inputs for calibration.
    
    IMPORTANT FIX: Captures the OUTPUT of LayerNorm (post-normalized tensor),
    which is what the attention module actually receives during inference.
    Previously was incorrectly capturing pre-LN residual (inp[0]).
    
    Also removed the 3x repetition since there's no stochasticity in eval mode.
    """
    if arch == "gpt2":
        num_layers = len(model.transformer.h)
    elif arch == "llama":
        num_layers = len(model.model.layers)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    layer_inputs = [[] for _ in range(num_layers)]

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            # FIX: Capture out (post-LayerNorm) NOT inp[0] (pre-LN residual)
            # This is what attention actually receives during forward pass
            layer_inputs[layer_idx].append(out.detach())
        return hook_fn

    handles = []
    if arch == "gpt2":
        for layer_idx, block in enumerate(model.transformer.h):
            h = block.ln_1.register_forward_hook(make_hook(layer_idx))
            handles.append(h)
    else:
        for layer_idx, block in enumerate(model.model.layers):
            h = block.input_layernorm.register_forward_hook(make_hook(layer_idx))
            handles.append(h)

    with torch.no_grad():
        # Single pass - no duplication since model is in eval mode
        for chunk in calib_tokens:
            model(chunk.to(device), use_cache=False)

    for h in handles:
        h.remove()

    return layer_inputs


def prepare_tokens(corpus, tokenizer, chunk_size=128, max_len_per_example=256):
    chunks = []
    for text in corpus:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=max_len_per_example)["input_ids"]
        for i in range(0, enc.size(1), chunk_size):
            chunk = enc[:, i:i+chunk_size]
            if chunk.size(1) >= 2:
                chunks.append(chunk)
    return chunks


# ============================================================
# Config + loaders (new -- matches ap-quant.ipynb's driver cell exactly:
# same dataset field ("instruction"), same chunk_size, same slicing)
# ============================================================

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # NOTE: this mirrors ap-quant.ipynb, where model_name is currently GPT-2
    # with Llama-3.2-1B commented out. If the team decides to finalize on
    # Llama (to match Quant_Baseline.ipynb's GPTQ/AWQ comparison), swap this
    # -- everything below already supports both via detect_arch().
    # model_id: str = "openai-community/gpt2"
    model_id: str = "unsloth/Llama-3.2-1B-Instruct"

    num_calib_examples: int = 256
    num_eval_examples: int = 256
    chunk_size: int = 128
    max_len_per_example: int = 256

    candidate_bits: List[int] = field(default_factory=lambda: [2, 3, 4, 8, 16])
    target_avg_bits: int = 4
    temp: float = 2.0            # KL temperature, matches compute_sensitivity_matrix default
    lambda_samples: int = 8      # matches compute_sensitivity_matrix default
    results_path: str = "objective2_results.json"


def load_model_and_tokenizer(cfg: Config):
    """Same loading pattern as ap-quant.ipynb's driver cell (plain
    from_pretrained + .to(device), no torch_dtype/device_map auto).
    token= is added so this also works unmodified if model_id is switched
    to a gated repo like Llama-3.2 -- harmless no-op for public models like GPT-2."""
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, attn_implementation="eager", token=hf_token
    ).to(device)
    return model, tokenizer


def load_calibration_data(cfg: Config, tokenizer):
    """Same as ap-quant.ipynb: Open-Platypus 'instruction' field, chunked with
    prepare_tokens. First `num_calib_examples` rows for calibration, the next
    `num_eval_examples` for evaluation (matches the notebook's [0:256] / [256:512] split)."""
    ds = load_dataset("garage-bAInd/Open-Platypus", split="train")

    calib_texts = [ds[i]["instruction"] for i in range(cfg.num_calib_examples)]
    calib_tokens = prepare_tokens(
        calib_texts, tokenizer, chunk_size=cfg.chunk_size, max_len_per_example=cfg.max_len_per_example
    )

    eval_start = cfg.num_calib_examples
    eval_end = cfg.num_calib_examples + cfg.num_eval_examples
    eval_texts = [ds[i]["instruction"] for i in range(eval_start, eval_end)]
    eval_tokens = prepare_tokens(
        eval_texts, tokenizer, chunk_size=cfg.chunk_size, max_len_per_example=cfg.max_len_per_example
    )

    return calib_tokens, eval_tokens
