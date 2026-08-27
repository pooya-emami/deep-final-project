"""
Shared utilities for AP-Quant -- used by BOTH Objective 1 and Objective 2.

Mirrors the config/calibration conventions of Quant_Baseline.ipynb so that
the GPTQ/AWQ baselines and AP-Quant are calibrated on the *same* data.
Keep this file identical between teammates so P/O/loss are computed the
same way on both sides of the project.
"""
import os
from dataclasses import dataclass, field
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Config:
    # --- shared with Quant_Baseline.ipynb: keep these in sync with your teammate ---
    model_id: str = "unsloth/Llama-3.2-1B-Instruct"
    cal_dataset: str = "open_platypus"     # "open_platypus" | "ultrachat" | "c4_small" | "wikitext"
    num_calibration_samples: int = 128      # start small to verify things run; raise to 128 for final numbers
    max_seq_length: int = 512

    # --- Objective 2 specific ---
    candidate_bits: List[int] = field(default_factory=lambda: [2, 3, 4, 8, 16])
    target_avg_bits: int = 4
    lambda_ref_bits: int = 4               # bit-width used once to estimate lambda (eq. 10)
    results_path: str = "objective2_results.json"


def load_model_and_tokenizer(cfg: Config):
    """Same pattern as Quant_Baseline.ipynb, plus attn_implementation='eager'
    (needed so output_attentions actually returns P instead of None)."""
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager",
        token=os.environ.get("HF_TOKEN", None),
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_calibration_dataset(cfg: Config, tokenizer):
    """Copied from Quant_Baseline.ipynb (same function, same defaults) so AP-Quant
    calibrates on the exact same data as the GPTQ/AWQ baselines. If this changes
    in the baseline notebook, update it here too."""
    if cfg.cal_dataset is None:
        return None

    if cfg.cal_dataset == "open_platypus":
        ds = load_dataset("garage-bAInd/Open-Platypus", split="train")
        ds = ds.shuffle(seed=42).select(range(cfg.num_calibration_samples))

        def to_text(example):
            return {"text": example.get("question", "") + "\n" + example.get("answer", "")}

        ds = ds.map(to_text)

    elif cfg.cal_dataset == "ultrachat":
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        ds = ds.shuffle(seed=42).select(range(cfg.num_calibration_samples))

        def preprocess(example):
            return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

        ds = ds.map(preprocess)

    elif cfg.cal_dataset == "c4_small":
        ds = load_dataset("brando/small-c4-dataset", split="train")
        ds = ds.shuffle(seed=42).select(range(cfg.num_calibration_samples))

    elif cfg.cal_dataset == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        ds = ds.shuffle(seed=42).select(range(cfg.num_calibration_samples))

    else:
        raise ValueError(f"Unknown calibration dataset: {cfg.cal_dataset}")

    def tokenize(sample):
        return tokenizer(
            sample["text"], padding=False, max_length=cfg.max_seq_length,
            truncation=True, add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=[c for c in ds.column_names if c != "input_ids"])
    return ds


def calib_dataset_to_tensors(calib_ds) -> List[torch.Tensor]:
    """Turns the tokenized HF Dataset into a list of (1, seq_len) LongTensors,
    one per calibration example. Skips any example that tokenized to length 0."""
    tensors = []
    for row in calib_ds:
        ids = row["input_ids"]
        if len(ids) == 0:
            continue
        tensors.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0))
    return tensors


@torch.no_grad()
def get_P_and_O(model, input_ids):
    """One forward pass. Returns:
       - P: tuple of attention-probability tensors, one per layer  (== Softmax(S), eq. 3/6)
       - O: dict {layer_idx: tensor}, the input to o_proj           (== P @ V, eq. 3)
    Shared between Objective 1 and Objective 2 -- both must compute P, O the same way.
    """
    captured_O = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inp, out):
            captured_O[layer_idx] = inp[0].detach()
        return hook

    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.o_proj.register_forward_hook(make_hook(i)))

    device = next(model.parameters()).device
    out = model(input_ids.to(device), output_attentions=True)

    for h in hooks:
        h.remove()

    return out.attentions, captured_O


def joint_loss(P1, O1, P2, O2, lam, eps=1e-6):
    """Eq. (7)-(9): L_joint = L_output + lambda * L_KL.
    P1,O1 = full-precision reference; P2,O2 = quantized."""
    L_out = (O2 - O1).pow(2).sum() / (O1.pow(2).sum() + eps)
    p1 = P1.clamp(min=1e-9)
    p2 = P2.clamp(min=1e-9)
    L_kl = (p1 * (p1.log() - p2.log())).sum(dim=-1).mean()
    return L_out + lam * L_kl, L_out.item(), L_kl.item()
