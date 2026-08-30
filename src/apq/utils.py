import math
import torch
import numpy as np
import pickle
from collections import defaultdict
from transformers import AutoTokenizer
from datasets import load_dataset

class _StopForward(Exception):
    pass

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_model_precision(model):
    return next(model.parameters()).dtype

def should_use_amp(model):
    dtype = get_model_precision(model)
    return dtype in [torch.bfloat16, torch.float16]

def load_calibration_dataset(dataset_name, tokenizer, calib_samples=256, eval_samples=256, chunk_size=128, max_len_per_example=256):
    if dataset_name == "garage-bAInd/Open-Platypus":
        ds = load_dataset(dataset_name, split="train")
        text_col = "instruction"
    elif dataset_name == "Salesforce/wikitext":
        ds = load_dataset(dataset_name, "wikitext-2-raw-v1", split="train")
        text_col = "text"
    elif dataset_name == "allenai/c4":
        ds = load_dataset("brando/small-c4-dataset", split="train")
        text_col = "text"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    calib_texts = [ds[i][text_col] for i in range(min(calib_samples, len(ds)))]
    eval_start = calib_samples
    eval_end = min(eval_start + eval_samples, len(ds))
    eval_texts = [ds[i][text_col] for i in range(eval_start, eval_end)]
    
    calib_tokens = prepare_tokens(calib_texts, tokenizer, chunk_size, max_len_per_example)
    eval_tokens = prepare_tokens(eval_texts, tokenizer, chunk_size, max_len_per_example)
    return calib_tokens, eval_tokens

def prepare_tokens(corpus, tokenizer, chunk_size=128, max_len_per_example=256):
    chunks = []
    for text in corpus:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len_per_example)["input_ids"]
        for i in range(0, enc.size(1), chunk_size):
            chunk = enc[:, i:i+chunk_size]
            if chunk.size(1) >= 2:
                chunks.append(chunk)
    return chunks

def batch_tokens(calib_tokens, batch_size=8):
    groups = defaultdict(list)
    for t in calib_tokens:
        groups[t.shape[1]].append(t)
    batches = []
    for _, chunks in groups.items():
        for i in range(0, len(chunks), batch_size):
            batches.append(torch.cat(chunks[i:i+batch_size], dim=0))
    return batches

def build_causal_mask(T, device, dtype):
    mask_bool = torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)
    mask = torch.zeros((T, T), device=device, dtype=dtype)
    mask.masked_fill_(mask_bool, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0)

def evaluate_ppl(model, test_tokens, device):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for chunk in test_tokens:
            input_ids = chunk.to(device)
            outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            num_tokens = input_ids.shape[1] - 1
            total_nll += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
    return math.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')

def detect_arch(model):
    if hasattr(model, "transformer"):
        return "gpt2"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "llama"
    raise ValueError("Unsupported model architecture")

def get_position_embeddings(model, hidden_states):
    B, T, _ = hidden_states.shape
    position_ids = torch.arange(T, device=hidden_states.device).unsqueeze(0).expand(B, -1)
    return model.model.rotary_emb(hidden_states, position_ids)

def save_calibration_scales(model, filepath="calibration_scales.pkl"):
    arch = detect_arch(model)
    scales = {}
    if arch == "gpt2":
        blocks = model.transformer.h
        for layer_idx, block in enumerate(blocks):
            attn = block.attn
            scales[layer_idx] = {
                'q_raw': attn.q_proj.raw_scale.detach().float().cpu().numpy(),
                'k_raw': attn.k_proj.raw_scale.detach().float().cpu().numpy(),
                'v_raw': attn.v_proj.raw_scale.detach().float().cpu().numpy(),
                'o_raw': attn.o_proj.raw_scale.detach().float().cpu().numpy(),
            }
    elif arch == "llama":
        blocks = model.model.layers
        for layer_idx, block in enumerate(blocks):
            attn = block.self_attn
            scales[layer_idx] = {
                'q_raw': attn.q_proj.raw_scale.detach().float().cpu().numpy(),
                'k_raw': attn.k_proj.raw_scale.detach().float().cpu().numpy(),
                'v_raw': attn.v_proj.raw_scale.detach().float().cpu().numpy(),
                'o_raw': attn.o_proj.raw_scale.detach().float().cpu().numpy(),
            }
    with open(filepath, 'wb') as f:
        pickle.dump(scales, f)

def load_calibration_scales(model, filepath="calibration_scales.pkl"):
    arch = detect_arch(model)
    with open(filepath, 'rb') as f:
        scales = pickle.load(f)
    if arch == "gpt2":
        blocks = model.transformer.h
        for layer_idx, block in enumerate(blocks):
            if layer_idx in scales:
                attn = block.attn
                attn.q_proj.raw_scale.data = torch.tensor(scales[layer_idx]['q_raw'], device=attn.q_proj.raw_scale.device)
                attn.k_proj.raw_scale.data = torch.tensor(scales[layer_idx]['k_raw'], device=attn.k_proj.raw_scale.device)
                attn.v_proj.raw_scale.data = torch.tensor(scales[layer_idx]['v_raw'], device=attn.v_proj.raw_scale.device)
                attn.o_proj.raw_scale.data = torch.tensor(scales[layer_idx]['o_raw'], device=attn.o_proj.raw_scale.device)
    elif arch == "llama":
        blocks = model.model.layers
        for layer_idx, block in enumerate(blocks):
            if layer_idx in scales:
                attn = block.self_attn
                attn.q_proj.raw_scale.data = torch.tensor(scales[layer_idx]['q_raw'], device=attn.q_proj.raw_scale.device)
                attn.k_proj.raw_scale.data = torch.tensor(scales[layer_idx]['k_raw'], device=attn.k_proj.raw_scale.device)
                attn.v_proj.raw_scale.data = torch.tensor(scales[layer_idx]['v_raw'], device=attn.v_proj.raw_scale.device)
                attn.o_proj.raw_scale.data = torch.tensor(scales[layer_idx]['o_raw'], device=attn.o_proj.raw_scale.device)
    return model

