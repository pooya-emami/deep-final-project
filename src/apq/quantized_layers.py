import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import detect_arch, get_position_embeddings, build_causal_mask

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scale, q_max):
        scale_c = torch.clamp(scale, min=1e-8)
        w_scaled = weight / scale_c
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
        not_clipped = (w_scaled.abs() <= q_max).float()
        grad_weight = grad_output * not_clipped
        grad_scale_elem = torch.where(
            w_scaled > q_max, torch.full_like(w_scaled, float(q_max)),
            torch.where(w_scaled < -q_max, torch.full_like(w_scaled, -float(q_max)), w_round - w_scaled)
        )
        grad_scale = (grad_output * grad_scale_elem).sum(dim=0, keepdim=True)
        return grad_weight, grad_scale, None

class QuantizedLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None, bits: int = 4):
        super().__init__()
        self.register_buffer("weight_fp", weight.clone().detach())
        if bias is not None:
            self.register_buffer("bias_fp", bias.clone().detach())
        else:
            self.bias_fp = None
        self.bits = bits
        self.q_max = 2 ** (bits - 1) - 1
        self.force_fp = False
        init_scale = (weight.detach().float().abs().amax(dim=0, keepdim=True) / self.q_max).clamp(min=1e-6)
        init_raw = torch.log(torch.expm1(init_scale).clamp_min(1e-8))
        self.raw_scale = nn.Parameter(init_raw.float())

    @property
    def scale(self):
        return F.softplus(self.raw_scale) + 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        if self.force_fp:
            w = self.weight_fp
            if orig_dtype in [torch.bfloat16, torch.float16]:
                w = w.to(orig_dtype)
                out = torch.matmul(x, w)
                if self.bias_fp is not None:
                    out = out + self.bias_fp.to(orig_dtype)
                return out
            else:
                out = torch.matmul(x, w)
                if self.bias_fp is not None:
                    out = out + self.bias_fp
                return out
        else:
            w = STEQuantize.apply(self.weight_fp, self.scale, self.q_max)
            if orig_dtype in [torch.bfloat16, torch.float16]:
                x_fp32 = x.float()
                w_fp32 = w.float()
                out = torch.matmul(x_fp32, w_fp32)
                if self.bias_fp is not None:
                    out = out + self.bias_fp.float()
                return out.to(orig_dtype)
            else:
                out = torch.matmul(x, w)
                if self.bias_fp is not None:
                    out = out + self.bias_fp
                return out

    def set_bits(self, bits: int):
        self.bits = bits
        self.q_max = 2 ** (bits - 1) - 1
        init_scale = (self.weight_fp.float().abs().amax(dim=0, keepdim=True) / self.q_max).clamp(min=1e-6)
        init_raw = torch.log(torch.expm1(init_scale).clamp_min(1e-8))
        self.raw_scale.data = init_raw.float()

class _FusedQKV(nn.Module):
    def __init__(self, q_proj, k_proj, v_proj):
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj

    def forward(self, x):
        return torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)

class NativeQuantizedGPT2Attention(nn.Module):
    def __init__(self, original_attn, bits: int = 4):
        super().__init__()
        self.force_fp_mode = False
        self.original_attn = original_attn
        W = original_attn.c_attn.weight.data
        b = original_attn.c_attn.bias.data if original_attn.c_attn.bias is not None else None
        w_q, w_k, w_v = W.chunk(3, dim=-1)
        b_q, b_k, b_v = b.chunk(3, dim=-1) if b is not None else (None, None, None)
        self.q_proj = QuantizedLinear(w_q, b_q, bits=bits)
        self.k_proj = QuantizedLinear(w_k, b_k, bits=bits)
        self.v_proj = QuantizedLinear(w_v, b_v, bits=bits)
        self.original_attn.c_attn = _FusedQKV(self.q_proj, self.k_proj, self.v_proj)
        W_o = original_attn.c_proj.weight.data
        b_o = original_attn.c_proj.bias.data if original_attn.c_proj.bias is not None else None
        self.o_proj = QuantizedLinear(W_o, b_o, bits=bits)
        original_attn.c_proj = self.o_proj

    def set_bits(self, bits: int):
        for p in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            p.set_bits(bits)

    def _set_force_fp(self, use_quant: bool):
        for p in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            p.force_fp = not use_quant

    def forward_components(self, hidden_states, use_quant: bool = True, position_embeddings=None):
        self._set_force_fp(use_quant)
        out = self.original_attn(hidden_states, use_cache=False, output_attentions=True)
        attn_output, P = out[0], out[1]
        T = hidden_states.shape[1]
        causal_mask = torch.triu(torch.ones((T, T), device=hidden_states.device, dtype=torch.bool), diagonal=1)
        return P, attn_output, causal_mask

    def forward(self, hidden_states, **kwargs):
        self._set_force_fp(use_quant=not self.force_fp_mode)
        return self.original_attn(hidden_states, **kwargs)

class NativeQuantizedLlamaAttention(nn.Module):
    def __init__(self, original_attn, bits: int = 4):
        super().__init__()
        self.force_fp_mode = False
        self.original_attn = original_attn
        W_q = original_attn.q_proj.weight.data.T.contiguous()
        b_q = original_attn.q_proj.bias.data if original_attn.q_proj.bias is not None else None
        W_k = original_attn.k_proj.weight.data.T.contiguous()
        b_k = original_attn.k_proj.bias.data if original_attn.k_proj.bias is not None else None
        W_v = original_attn.v_proj.weight.data.T.contiguous()
        b_v = original_attn.v_proj.bias.data if original_attn.v_proj.bias is not None else None
        self.q_proj = QuantizedLinear(W_q, b_q, bits=bits)
        self.k_proj = QuantizedLinear(W_k, b_k, bits=bits)
        self.v_proj = QuantizedLinear(W_v, b_v, bits=bits)
        self.original_attn.q_proj = self.q_proj
        self.original_attn.k_proj = self.k_proj
        self.original_attn.v_proj = self.v_proj
        W_o = original_attn.o_proj.weight.data.T.contiguous()
        b_o = original_attn.o_proj.bias.data if original_attn.o_proj.bias is not None else None
        self.o_proj = QuantizedLinear(W_o, b_o, bits=bits)
        original_attn.o_proj = self.o_proj

    def set_bits(self, bits: int):
        for p in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            p.set_bits(bits)

    def _set_force_fp(self, use_quant: bool):
        for p in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            p.force_fp = not use_quant

    def forward_components(self, hidden_states, use_quant=True, position_embeddings=None, attention_mask=None):
        if position_embeddings is None:
            raise ValueError("position_embeddings required")
        self._set_force_fp(use_quant)
        T = hidden_states.shape[1]
        if attention_mask is None:
            attention_mask = build_causal_mask(T, hidden_states.device, hidden_states.dtype)
        out = self.original_attn(hidden_states, attention_mask=attention_mask,
                                  position_embeddings=position_embeddings, output_attentions=True)
        attn_output, P = out[0], out[1]
        causal_mask = torch.triu(torch.ones((T, T), device=hidden_states.device, dtype=torch.bool), diagonal=1)
        return P, attn_output, causal_mask

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, position_embeddings=None, **kwargs):
        self._set_force_fp(use_quant=not self.force_fp_mode)
        return self.original_attn(hidden_states, attention_mask=attention_mask,
                                   position_ids=position_ids, past_key_value=past_key_value,
                                   output_attentions=output_attentions,
                                   position_embeddings=position_embeddings, **kwargs)

def quantize_transformer_attn(model, layer_bits=None, device="cuda"):
    arch = detect_arch(model)
    q_attn_blocks = []
    if arch == "gpt2":
        for layer_idx, block in enumerate(model.transformer.h):
            bits = layer_bits[layer_idx] if layer_bits is not None else 4
            q_attn = NativeQuantizedGPT2Attention(block.attn, bits=bits).to(device)
            block.attn = q_attn
            q_attn_blocks.append(q_attn)
    elif arch == "llama":
        for layer_idx, block in enumerate(model.model.layers):
            bits = layer_bits[layer_idx] if layer_bits is not None else 4
            q_attn = NativeQuantizedLlamaAttention(block.self_attn, bits=bits).to(device)
            block.self_attn = q_attn
            q_attn_blocks.append(q_attn)
    return arch, q_attn_blocks