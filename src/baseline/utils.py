import torch
import torch.nn as nn

def replace_gpt2_qkv_conv1d_with_linear(model):
    from transformers.pytorch_utils import Conv1D
    replaced = 0
    for block in model.transformer.h:
        old_layer = block.attn.c_attn
        if not isinstance(old_layer, Conv1D):
            continue
        in_features = old_layer.weight.shape[0]
        out_features = old_layer.weight.shape[1]
        new_layer = nn.Linear(in_features, out_features, bias=old_layer.bias is not None)
        new_layer.weight.data.copy_(old_layer.weight.data.T)
        if old_layer.bias is not None:
            new_layer.bias.data.copy_(old_layer.bias.data)
        new_layer = new_layer.to(device=old_layer.weight.device, dtype=old_layer.weight.dtype)
        block.attn.c_attn = new_layer
        replaced += 1
    print(f"✓ Converted {replaced} GPT2 c_attn (QKV) layers to nn.Linear.")
    return model

def replace_gpt2_wo_conv1d_with_linear(model):
    from transformers.pytorch_utils import Conv1D
    replaced = 0
    for block in model.transformer.h:
        old_layer = block.attn.c_proj
        if not isinstance(old_layer, Conv1D):
            continue
        in_features = old_layer.weight.shape[0]
        out_features = old_layer.weight.shape[1]
        new_layer = nn.Linear(in_features, out_features, bias=old_layer.bias is not None)
        new_layer.weight.data.copy_(old_layer.weight.data.T)
        if old_layer.bias is not None:
            new_layer.bias.data.copy_(old_layer.bias.data)
        new_layer = new_layer.to(device=old_layer.weight.device, dtype=old_layer.weight.dtype)
        block.attn.c_proj = new_layer
        replaced += 1
    print(f"✓ Converted {replaced} GPT2 c_proj (Wo) layers to nn.Linear.")
    return model