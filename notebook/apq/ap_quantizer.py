import json
import math
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from .utils import compute_output_loss, compute_kl_loss, compute_entropy_loss


class QuantizedLinearSTE(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, 
                 bit_width: int = 8, per_channel: bool = True):
        super().__init__()
        
        self.weight_fp16 = nn.Parameter(weight.clone().detach(), requires_grad=False)
        self.bias_fp16 = nn.Parameter(bias.clone().detach(), requires_grad=False) if bias is not None else None
        
        self.bit_width = bit_width
        self.q_max = 2**(bit_width - 1) - 1
        self.per_channel = per_channel
        
        init_scale = self._calc_init_scale(weight)
        self.scale = nn.Parameter(init_scale)
    
    def _calc_init_scale(self, weight: torch.Tensor) -> torch.Tensor:
        if self.per_channel:
            max_abs = torch.max(torch.abs(weight), dim=1, keepdim=True).values
            max_abs = torch.clamp(max_abs, min=1e-8)
            return max_abs / self.q_max * 0.8
        else:
            max_abs = torch.max(torch.abs(weight))
            max_abs = max(max_abs, 1e-8)
            return torch.ones(1, 1, device=weight.device) * max_abs / self.q_max * 0.8

    def set_bit_width(self, bit_width: int):
        self.bit_width = bit_width
        self.q_max = 2**(bit_width - 1) - 1
        self.scale.data.copy_(self._calc_init_scale(self.weight_fp16))
    
    def quantize(self, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        scale_clamped = torch.clamp(scale, min=1e-8)
        
        w_scaled = weight / scale_clamped
        w_clamped = torch.clamp(w_scaled, -self.q_max, self.q_max)
        
        w_rounded = torch.round(w_clamped)
        w_quantized = w_clamped + (w_rounded - w_clamped).detach()
        
        return w_quantized * scale_clamped

    def forward(self, x: torch.Tensor, force_fp16: bool = False) -> torch.Tensor:
        if self.bit_width == 16 or force_fp16:
            if self.bias_fp16 is not None:
                return F.linear(x, self.weight_fp16, self.bias_fp16)
            else:
                return F.linear(x, self.weight_fp16)
        
        w_q = self.quantize(self.weight_fp16, self.scale)
        
        if self.bias_fp16 is not None:
            return F.linear(x, w_q, self.bias_fp16)
        else:
            return F.linear(x, w_q)


class QuantizedAttention(nn.Module):
    def __init__(self, original_attn: nn.Module, bit_width: int = 8, per_channel: bool = True):
        super().__init__()
        
        self.enabled = True
        self.bit_width = bit_width
        self.layer_idx = -1
        
        if hasattr(original_attn, "q_proj"):
            self.arch = "llama"
            self.num_heads = original_attn.num_heads
            self.head_dim = original_attn.head_dim
            self.embed_dim = original_attn.embed_dim
            
            self.q_proj = QuantizedLinearSTE(
                original_attn.q_proj.weight, original_attn.q_proj.bias, bit_width, per_channel
            )
            self.k_proj = QuantizedLinearSTE(
                original_attn.k_proj.weight, original_attn.k_proj.bias, bit_width, per_channel
            )
            self.v_proj = QuantizedLinearSTE(
                original_attn.v_proj.weight, original_attn.v_proj.bias, bit_width, per_channel
            )
            self.o_proj = QuantizedLinearSTE(
                original_attn.o_proj.weight, original_attn.o_proj.bias, bit_width, per_channel
            )
            
        elif hasattr(original_attn, "c_attn"):
            self.arch = "gpt"
            W = original_attn.c_attn.weight
            b = original_attn.c_attn.bias
            
            embed_dim = W.shape[0]
            qkv_dim = W.shape[1] // 3
            self.embed_dim = embed_dim
            self.num_heads = original_attn.num_heads
            self.head_dim = embed_dim // self.num_heads
            
            self.q_proj = QuantizedLinearSTE(
                W[:, :qkv_dim].T,
                b[:qkv_dim] if b is not None else None,
                bit_width,
                per_channel
            )
            self.k_proj = QuantizedLinearSTE(
                W[:, qkv_dim:2*qkv_dim].T,
                b[qkv_dim:2*qkv_dim] if b is not None else None,
                bit_width,
                per_channel
            )
            self.v_proj = QuantizedLinearSTE(
                W[:, 2*qkv_dim:].T,
                b[2*qkv_dim:] if b is not None else None,
                bit_width,
                per_channel
            )
            self.o_proj = QuantizedLinearSTE(
                original_attn.c_proj.weight.T,
                original_attn.c_proj.bias,
                bit_width,
                per_channel
            )
            
        else:
            raise ValueError(f"Unknown attention architecture: {type(original_attn)}")
        
        self._original_P = None
        self._original_O = None
        self._last_P = None
        self._last_O = None
        self._last_attention_output = None
        self._last_attn_output = None
    
    def set_bit_width(self, bit_width: int):
        self.bit_width = bit_width
        self.q_proj.set_bit_width(bit_width)
        self.k_proj.set_bit_width(bit_width)
        self.v_proj.set_bit_width(bit_width)
        self.o_proj.set_bit_width(bit_width)
        
        if bit_width == 16:
            self.enabled = False
        else:
            self.enabled = True
    
    def set_fp_reference(self, P: torch.Tensor, O: torch.Tensor):
        self._original_P = P.detach()
        self._original_O = O.detach()
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        
        if not self.enabled or self.bit_width == 16:
            q = F.linear(hidden_states, self.q_proj.weight_fp16, self.q_proj.bias_fp16)
            k = F.linear(hidden_states, self.k_proj.weight_fp16, self.k_proj.bias_fp16)
            v = F.linear(hidden_states, self.v_proj.weight_fp16, self.v_proj.bias_fp16)
        else:
            q = self.q_proj(hidden_states, force_fp16=False)
            k = self.k_proj(hidden_states, force_fp16=False)
            v = self.v_proj(hidden_states, force_fp16=False)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                attn_weights = attn_weights + attention_mask
            elif attention_mask.dim() == 2:
                extended_mask = attention_mask[:, None, None, :]
                extended_mask = (1.0 - extended_mask) * -10000.0
                attn_weights = attn_weights + extended_mask
        
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=attn_weights.device), 
            diagonal=1
        ).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        P = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(P, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        self._last_P = P
        self._last_attention_output = attn_output
        
        if not self.enabled or self.bit_width == 16:
            attn_output = F.linear(attn_output, self.o_proj.weight_fp16, self.o_proj.bias_fp16)
        else:
            attn_output = self.o_proj(attn_output, force_fp16=False)
        
        self._last_O = attn_output
        self._last_attn_output = attn_output
        
        return attn_output, P


class CalibrationDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return tokens.input_ids[0], tokens.attention_mask[0]


class AttentionPreservingQuantizer:
    def __init__(self, model, tokenizer, calibration_texts: List[str], config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.calibration_dataloader = self._prepare_calibration_data(calibration_texts)
        
        self._original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        
        self._replace_attention_modules()
        
        self._lambda = None
        self._sensitivity_matrix = None
        self.loss_history = []
        
    def _prepare_calibration_data(self, texts: List[str]) -> DataLoader:
        dataset = CalibrationDataset(texts, self.tokenizer)
        return DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            drop_last=True
        )
    
    def _replace_attention_modules(self):
        layer_idx = 0
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for layer in self.model.model.layers:
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    if not isinstance(attn, QuantizedAttention):
                        quantized_attn = QuantizedAttention(
                            attn, 
                            bit_width=self.config.initial_bit_width,
                            per_channel=getattr(self.config, 'per_channel', True)
                        )
                        quantized_attn.layer_idx = layer_idx
                        layer.self_attn = quantized_attn
                        layer_idx += 1
                elif hasattr(layer, 'attn'):
                    attn = layer.attn
                    if not isinstance(attn, QuantizedAttention):
                        quantized_attn = QuantizedAttention(
                            attn,
                            bit_width=self.config.initial_bit_width,
                            per_channel=getattr(self.config, 'per_channel', True)
                        )
                        quantized_attn.layer_idx = layer_idx
                        layer.attn = quantized_attn
                        layer_idx += 1
        
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            for block in self.model.transformer.h:
                if hasattr(block, 'attn'):
                    attn = block.attn
                    if not isinstance(attn, QuantizedAttention):
                        quantized_attn = QuantizedAttention(
                            attn,
                            bit_width=self.config.initial_bit_width,
                            per_channel=getattr(self.config, 'per_channel', True)
                        )
                        quantized_attn.layer_idx = layer_idx
                        block.attn = quantized_attn
                        layer_idx += 1
        
        self.num_layers = layer_idx
        print(f"Replaced {self.num_layers} attention layers")
    
    def _get_layers(self) -> List[QuantizedAttention]:
        layers = []
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for layer in self.model.model.layers:
                if hasattr(layer, 'self_attn') and isinstance(layer.self_attn, QuantizedAttention):
                    layers.append(layer.self_attn)
                elif hasattr(layer, 'attn') and isinstance(layer.attn, QuantizedAttention):
                    layers.append(layer.attn)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            for block in self.model.transformer.h:
                if hasattr(block, 'attn') and isinstance(block.attn, QuantizedAttention):
                    layers.append(block.attn)
        return layers
    
    def _set_quantization(self, enabled: bool, layer_indices: Optional[List[int]] = None):
        for attn in self._get_layers():
            if layer_indices is None or attn.layer_idx in layer_indices:
                if enabled and attn.bit_width == 16:
                    continue
                attn.enabled = enabled
    
    def _set_bit_width_for_layer(self, layer_idx: int, bit_width: int):
        for attn in self._get_layers():
            if attn.layer_idx == layer_idx:
                attn.set_bit_width(bit_width)
                break
    
    def _set_all_bit_widths(self, assignments: Dict[int, int]):
        for layer_idx, bit_width in assignments.items():
            for attn in self._get_layers():
                if attn.layer_idx == layer_idx:
                    attn.set_bit_width(bit_width)
                    break
    
    def _forward_with_attention(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs
    
    def _compute_reference(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        with torch.no_grad():
            captured = {}
            
            def make_hook(idx):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple) and len(output) >= 2:
                        captured[idx] = {
                            'P': output[1].detach(),
                            'O': output[0].detach()
                        }
                    elif hasattr(module, '_last_P') and hasattr(module, '_last_O'):
                        captured[idx] = {
                            'P': module._last_P.detach(),
                            'O': module._last_O.detach()
                        }
                return hook_fn
            
            hooks = []
            for attn in self._get_layers():
                handle = attn.register_forward_hook(make_hook(attn.layer_idx))
                hooks.append(handle)
            
            for attn in self._get_layers():
                attn.enabled = False
            
            _ = self._forward_with_attention(input_ids, attention_mask)
            
            for attn in self._get_layers():
                if attn.bit_width != 16:
                    attn.enabled = True
            
            for handle in hooks:
                handle.remove()
            
            for attn in self._get_layers():
                if attn.layer_idx in captured:
                    attn.set_fp_reference(
                        captured[attn.layer_idx]['P'],
                        captured[attn.layer_idx]['O']
                    )
                elif hasattr(attn, '_last_P') and hasattr(attn, '_last_O'):
                    attn.set_fp_reference(attn._last_P.detach(), attn._last_O.detach())
    
    def _compute_lambda(self, num_steps: int = 5) -> float:
        l_output_sum = 0.0
        l_kl_sum = 0.0
        count = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.calibration_dataloader):
                if batch_idx >= num_steps:
                    break
                
                input_ids, attention_mask = batch
                input_ids = input_ids.to(self.model.device)
                attention_mask = attention_mask.to(self.model.device)
                
                self._compute_reference(input_ids, attention_mask)
                
                outputs_q = self._forward_with_attention(input_ids, attention_mask)
                
                total_output_loss = 0.0
                total_kl_loss = 0.0
                num_layers = 0
                
                for idx, attn in enumerate(self._get_layers()):
                    if idx < len(outputs_q.attentions):
                        P_hat = outputs_q.attentions[idx]
                        P_ref = attn._original_P
                        O_ref = attn._original_O
                        
                        O_hat = getattr(attn, '_last_O', None)
                        if O_hat is None:
                            O_hat = torch.zeros_like(O_ref)
                        
                        l_output = compute_output_loss(O_ref, O_hat)
                        l_kl = compute_kl_loss(P_ref, P_hat)
                        
                        total_output_loss += l_output
                        total_kl_loss += l_kl
                        num_layers += 1
                
                if num_layers > 0:
                    l_output_sum += (total_output_loss / num_layers).item()
                    l_kl_sum += (total_kl_loss / num_layers).item()
                    count += 1
        
        if count > 0 and l_kl_sum > 1e-8:
            lambda_val = l_output_sum / (l_kl_sum + 1e-8)
            lambda_val = max(1e-3, min(lambda_val, 1e3))
            lambda_val = lambda_val * 0.01
            print(f"λ = {lambda_val:.4f}")
            return lambda_val
        else:
            return 1.0
    
    def compute_sensitivity_matrix(self, bit_candidates: List[int] = [2, 3, 4, 8, 16],
                                  num_batches: int = 5) -> Dict[int, Dict[int, float]]:
        print("\n" + "=" * 70)
        print("COMPUTING SENSITIVITY MATRIX")
        print("=" * 70)
        
        if self._lambda is None:
            self._lambda = self._compute_lambda()
        
        sensitivity = {}
        num_layers = len(self._get_layers())
        
        print(f"Layers: {num_layers}, Bits: {bit_candidates}, Batches: {num_batches}\n")
        
        for layer_idx in tqdm(range(num_layers), desc="Processing layers"):
            sensitivity[layer_idx] = {}
            
            for bit in bit_candidates:
                l_joint_total = 0.0
                count = 0
                
                for batch_idx, batch in enumerate(self.calibration_dataloader):
                    if batch_idx >= num_batches:
                        break
                    
                    l_joint = self._compute_loss_single_layer(batch, layer_idx, bit)
                    l_joint_total += l_joint
                    count += 1
                
                avg_l_joint = l_joint_total / count if count > 0 else float('inf')
                sensitivity[layer_idx][bit] = avg_l_joint
                print(f"Layer {layer_idx}, Bit {bit}: {avg_l_joint:.6f}")
            
            if 16 not in bit_candidates:
                sensitivity[layer_idx][16] = 0.0
            
            print()
        
        self._sensitivity_matrix = sensitivity
        
        print("=" * 70)
        print("SENSITIVITY MATRIX COMPLETE")
        print("=" * 70)
        
        self._save_sensitivity_matrix(sensitivity)
        
        return sensitivity
    
    def _compute_loss_single_layer(self, batch: Tuple[torch.Tensor, torch.Tensor], 
                                   layer_idx: int, bit_width: int) -> float:
        with torch.no_grad():
            input_ids, attention_mask = batch
            input_ids = input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)
            
            self._compute_reference(input_ids, attention_mask)
            
            for attn in self._get_layers():
                attn.enabled = False
            
            self._set_bit_width_for_layer(layer_idx, bit_width)
            
            if bit_width != 16:
                for attn in self._get_layers():
                    if attn.layer_idx == layer_idx:
                        attn.enabled = True
                        break
            
            outputs_q = self._forward_with_attention(input_ids, attention_mask)
            
            total_output_loss = 0.0
            total_kl_loss = 0.0
            num_layers = 0
            
            for idx, attn in enumerate(self._get_layers()):
                if idx < len(outputs_q.attentions):
                    P_hat = outputs_q.attentions[idx]
                    P_ref = attn._original_P
                    O_ref = attn._original_O
                    
                    O_hat = getattr(attn, '_last_O', None)
                    if O_hat is None:
                        O_hat = torch.zeros_like(O_ref)
                    
                    l_output = compute_output_loss(O_ref, O_hat)
                    l_kl = compute_kl_loss(P_ref, P_hat)
                    
                    total_output_loss += l_output
                    total_kl_loss += l_kl
                    num_layers += 1
            
            for attn in self._get_layers():
                attn.enabled = False
            
            if num_layers > 0:
                avg_output_loss = total_output_loss / num_layers
                avg_kl_loss = total_kl_loss / num_layers
                return avg_output_loss.item() + self._lambda * avg_kl_loss.item()
            else:
                return float('inf')
    
    def _save_sensitivity_matrix(self, sensitivity: Dict):
        save_path = "sensitivity_matrix.json"
        
        serializable = {}
        for layer_idx, bits in sensitivity.items():
            serializable[str(layer_idx)] = {str(bit): float(loss) for bit, loss in bits.items()}
        
        with open(save_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"\nSensitivity matrix saved to {save_path}")
    
    def load_sensitivity_matrix(self, path: str) -> Dict:
        with open(path, 'r') as f:
            data = json.load(f)
        
        sensitivity = {}
        for layer_idx, bits in data.items():
            sensitivity[int(layer_idx)] = {int(bit): float(loss) for bit, loss in bits.items()}
        
        return sensitivity
    
    def apply_mixed_precision(self, assignments: Dict[int, int]):
        print("\n" + "=" * 70)
        print("APPLYING MIXED PRECISION ASSIGNMENTS")
        print("=" * 70)
        
        self._set_all_bit_widths(assignments)
        
        for layer_idx, bit_width in assignments.items():
            print(f"Layer {layer_idx}: {bit_width} bits")
        
        print("\nMixed precision applied!")
        
        return self.model
    
    def _get_active_scale_params(self) -> List[nn.Parameter]:
        params = []
        for attn in self._get_layers():
            if attn.bit_width == 16:
                continue
            if attn.enabled:
                params.extend([
                    attn.q_proj.scale,
                    attn.k_proj.scale,
                    attn.v_proj.scale,
                    attn.o_proj.scale
                ])
        return params
    
    def _calibrate_jointly(self, num_iterations: int = 50, use_entropy: bool = False, 
                           beta: float = 0.1, gradient_clip: float = 1.0):
        print("\n" + "=" * 70)
        print("JOINT CALIBRATION")
        print("=" * 70)
        
        if self._lambda is None:
            self._lambda = self._compute_lambda()
        
        for attn in self._get_layers():
            if attn.bit_width == 16:
                attn.enabled = False
            else:
                attn.enabled = True
        
        params_to_optimize = self._get_active_scale_params()
        
        print(f"Number of scale parameters to optimize: {len(params_to_optimize)}")
        
        if len(params_to_optimize) == 0:
            print("WARNING: No active scale parameters found! (All layers at FP16?)")
            return
        
        optimizer = torch.optim.AdamW(params_to_optimize, lr=self.config.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations, eta_min=1e-6)
        
        self.loss_history = []
        
        for iter_idx in tqdm(range(num_iterations), desc="Joint Calibration"):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in self.calibration_dataloader:
                input_ids, attention_mask = batch
                input_ids = input_ids.to(self.model.device)
                attention_mask = attention_mask.to(self.model.device)
                
                self._compute_reference(input_ids, attention_mask)
                
                optimizer.zero_grad()
                
                outputs = self._forward_with_attention(input_ids, attention_mask)
                
                total_output_loss = 0.0
                total_kl_loss = 0.0
                total_ent_loss = 0.0
                num_layers = 0
                
                for idx, attn in enumerate(self._get_layers()):
                    if attn.bit_width == 16:
                        continue
                    if idx < len(outputs.attentions):
                        P_hat = outputs.attentions[idx]
                        P_ref = attn._original_P
                        O_ref = attn._original_O
                        
                        O_hat = getattr(attn, '_last_O', None)
                        if O_hat is None:
                            O_hat = torch.zeros_like(O_ref)
                        
                        l_output = compute_output_loss(O_ref, O_hat)
                        l_kl = compute_kl_loss(P_ref, P_hat)
                        
                        if use_entropy:
                            l_ent = compute_entropy_loss(P_ref, P_hat)
                            total_ent_loss += l_ent
                        
                        total_output_loss += l_output
                        total_kl_loss += l_kl
                        num_layers += 1
                
                if num_layers > 0:
                    avg_output_loss = total_output_loss / num_layers
                    avg_kl_loss = total_kl_loss / num_layers
                    
                    if iter_idx % 10 == 0:
                        print(f"  output={avg_output_loss.item():.6f}, KL={avg_kl_loss.item():.6f}, "
                              f"λ*KL={self._lambda * avg_kl_loss.item():.6f}")
                    
                    if use_entropy:
                        avg_ent_loss = total_ent_loss / num_layers
                        loss = avg_output_loss + self._lambda * avg_kl_loss + beta * avg_ent_loss
                    else:
                        loss = avg_output_loss + self._lambda * avg_kl_loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, gradient_clip)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
            
            scheduler.step()
            
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                self.loss_history.append(avg_loss)
                
                if iter_idx % 10 == 0:
                    print(f"Step {iter_idx}: Loss = {avg_loss:.6f}")
        
        print("\nJoint calibration complete!")
    
    def calibrate(self, num_iterations: int = 50, use_entropy: bool = False, 
                  beta: float = 0.1) -> nn.Module:
        if self._lambda is None:
            self._lambda = self._compute_lambda()
        
        self._calibrate_jointly(num_iterations, use_entropy, beta)
        return self.model
    
    def save_quantized_model(self, path: str):
        model_path = f"{path}_quantized.pth"
        torch.save(self.model.state_dict(), model_path)
        
        config_path = f"{path}_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'batch_size': self.config.batch_size,
                'lr': self.config.lr,
                'initial_bit_width': self.config.initial_bit_width,
                'per_channel': getattr(self.config, 'per_channel', True),
                'lambda': self._lambda,
                'loss_history': self.loss_history if hasattr(self, 'loss_history') else [],
                'sensitivity_matrix': self._sensitivity_matrix,
            }, f, indent=2)
        
        print(f"Model saved to {model_path}")
        print(f"Config saved to {config_path}")
    
    def reset_to_original(self):
        for name, param in self.model.named_parameters():
            if name in self._original_state_dict:
                param.data.copy_(self._original_state_dict[name])
        self._replace_attention_modules()