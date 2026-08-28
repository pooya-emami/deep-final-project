"""
Objective 2: Full-Curve Sensitivity Allocation.

`compute_sensitivity_matrix` and `solve_mckp` below are copied from the
teammate's merged notebook (ap-quant.ipynb) -- NOT my earlier reimplementation.
compute_sensitivity_matrix depends on quantize_transformer_attn/collect_layer_inputs/
compute_ap_loss_improved from ap_quant_common.py, which are the same functions
Objective 1's calibration step uses -- so both objectives measure things the
same way and the bit_assignment this produces is meaningful input to that step.

Do not hand-edit compute_sensitivity_matrix or solve_mckp -- if the teammate's
notebook changes, re-copy from there instead.
"""
import json

import numpy as np
import torch

from ap_quant_common import (
    Config,
    device,
    load_model_and_tokenizer,
    load_calibration_data,
    detect_arch,
    quantize_transformer_attn,
    collect_layer_inputs,
    compute_ap_loss_improved,
)


@torch.no_grad()
def compute_sensitivity_matrix(
    model,
    calib_tokens,
    candidate_bits,
    temp: float = 2.0,
    device: str = "cuda",
    lambda_samples: int = 8,
):
    """
    Sensitivity matrix for mixed-precision AP-Quant.

    - Baseline: FP32 model vs RTN-quantized attention (no sequential calibration).
    - Inputs: post-LayerNorm activations collected from FP32 forward.
    - For each layer and bit-width, we measure AP loss between FP32 and quantized
      attention outputs, using fixed RTN scales for that bit-width.
    """

    # 1) Quantize attention blocks with RTN scales (QKV-only, FP32 Wo)
    arch, q_attn_blocks = quantize_transformer_attn(model, layer_bits=None)

    # 2) Collect FP32 layer inputs (post-LN) once
    for q_attn in q_attn_blocks:
        q_attn.force_fp_mode = True  # attention runs in FP32

    layer_inputs = collect_layer_inputs(model, arch, calib_tokens, device)

    for q_attn in q_attn_blocks:
        q_attn.force_fp_mode = False  # back to quantized mode when needed

    # 3) Estimate a global λ from a few layers, comparing FP32 vs 4-bit RTN
    lam_values = []
    valid_layers = [i for i in range(len(q_attn_blocks)) if len(layer_inputs[i]) > 0]
    if not valid_layers:
        raise RuntimeError("No layer inputs collected; cannot estimate lambda.")

    num_layers_to_use = min(3, len(valid_layers))
    layers_to_use = valid_layers[:num_layers_to_use]

    for layer_idx in layers_to_use:
        q_attn = q_attn_blocks[layer_idx]

        # ensure reference bit-width is 4-bit RTN
        q_attn.q_proj.set_bits(4)
        q_attn.k_proj.set_bits(4)
        q_attn.v_proj.set_bits(4)

        n_samples = min(lambda_samples, len(layer_inputs[layer_idx]))
        sample_indices = np.random.choice(len(layer_inputs[layer_idx]), n_samples, replace=False)

        for idx in sample_indices:
            inp = layer_inputs[layer_idx][idx].to(device)

            # FP32 path
            q_attn.force_fp_mode = True
            P_fp, O_fp, mask = q_attn.forward_components(inp, use_quant=False)

            # 4-bit RTN path
            q_attn.force_fp_mode = False
            P_q, O_q, _ = q_attn.forward_components(inp, use_quant=True)

            _, l_out_0, l_kl_0, _ = compute_ap_loss_improved(
                P_fp, O_fp, P_q, O_q, mask,
                lam=1.0,
                temp=temp,
            )

            lam_values.append(float(l_out_0 / (l_kl_0 + 1e-8)))

    lam = np.mean(lam_values) if lam_values else 1.0
    lam = min(max(lam, 0.01), 10.0)
    print(f"[Sensitivity] λ = {lam:.4f}")

    # 4) Build sensitivity matrix: per-layer, per-bit AP loss vs FP32
    sensitivity = {i: {} for i in range(len(q_attn_blocks))}

    for layer_idx, q_attn in enumerate(q_attn_blocks):
        if len(layer_inputs[layer_idx]) == 0:
            continue

        print(f"layer {layer_idx:02d}:", end=" ")

        for bits in candidate_bits:
            # RTN scales for this bit-width
            q_attn.q_proj.set_bits(bits)
            q_attn.k_proj.set_bits(bits)
            q_attn.v_proj.set_bits(bits)

            total_l_joint = 0.0
            n_samples = len(layer_inputs[layer_idx])

            for inp in layer_inputs[layer_idx]:
                inp = inp.to(device)

                # FP32 reference
                q_attn.force_fp_mode = True
                P_fp, O_fp, mask = q_attn.forward_components(inp, use_quant=False)

                # quantized at `bits`
                q_attn.force_fp_mode = False
                P_q, O_q, _ = q_attn.forward_components(inp, use_quant=True)

                l_joint, _, _, _ = compute_ap_loss_improved(
                    P_fp, O_fp, P_q, O_q, mask,
                    lam=lam,
                    temp=temp,
                )
                total_l_joint += l_joint.item()

            avg_loss = total_l_joint / max(n_samples, 1)
            sensitivity[layer_idx][bits] = avg_loss

            print(f"{bits}b={avg_loss:.4f}", end=", ")

        print()  # newline per layer

    return sensitivity, lam


def solve_mckp(sensitivity, cost, budget, candidate_bits):
    """Eq. (13): each block i picks exactly one bit-width from candidate_bits;
    minimize total loss subject to a memory budget. Exact 0/1-choice-per-item
    knapsack via dynamic programming -- O(n * budget * len(candidate_bits))."""
    n = len(sensitivity)
    INF = float("inf")
    dp = [INF] * (budget + 1)
    dp[0] = 0.0
    choice = [[None] * (budget + 1) for _ in range(n)]

    for i in range(n):
        new_dp = [INF] * (budget + 1)
        for c in range(budget + 1):
            if dp[c] == INF:
                continue
            for b in candidate_bits:
                c2 = c + cost[b]
                if c2 <= budget and dp[c] + sensitivity[i][b] < new_dp[c2]:
                    new_dp[c2] = dp[c] + sensitivity[i][b]
                    choice[i][c2] = (b, c)
        dp = new_dp

    best_c = min(range(budget + 1), key=lambda c: dp[c])
    assignment, c = {}, best_c
    for i in reversed(range(n)):
        b, prev_c = choice[i][c]
        assignment[i] = b
        c = prev_c
    return assignment, dp[best_c]


def save_results(path, assignment, sensitivity, lam):
    """Hand-off point for Objective 1's calibration step: this JSON says which
    bit-width each attention block should be trained at."""
    payload = {
        "lambda": lam,
        "bit_assignment": {str(k): v for k, v in assignment.items()},
        "sensitivity_table": {
            str(i): {str(b): v for b, v in bd.items()} for i, bd in sensitivity.items()
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_results(path):
    with open(path) as f:
        payload = json.load(f)
    assignment = {int(k): v for k, v in payload["bit_assignment"].items()}
    sensitivity = {
        int(i): {int(b): v for b, v in bd.items()} for i, bd in payload["sensitivity_table"].items()
    }
    return assignment, sensitivity, payload["lambda"]


def run_objective2(cfg: Config):
    """End-to-end, matching ap-quant.ipynb's driver cell: load model -> load
    calibration data -> compute_sensitivity_matrix -> solve_mckp -> save results."""
    model, tokenizer = load_model_and_tokenizer(cfg)
    calib_tokens, eval_tokens = load_calibration_data(cfg, tokenizer)
    print(f"{len(calib_tokens)} calibration chunks, {len(eval_tokens)} evaluation chunks")

    if hasattr(model.config, "n_layer"):
        n_layers = model.config.n_layer
    elif hasattr(model.config, "num_hidden_layers"):
        n_layers = model.config.num_hidden_layers
    else:
        raise ValueError("Unknown model config - cannot determine number of layers")

    sensitivity, lam = compute_sensitivity_matrix(
        model=model,
        calib_tokens=calib_tokens,
        candidate_bits=cfg.candidate_bits,
        temp=cfg.temp,
        device=device,
        lambda_samples=cfg.lambda_samples,
    )

    cost = {b: b for b in cfg.candidate_bits}
    budget = cfg.target_avg_bits * n_layers
    assignment, total_loss = solve_mckp(sensitivity, cost, budget, cfg.candidate_bits)
    achieved_avg = sum(cost[assignment[i]] for i in range(n_layers)) / n_layers

    print("lambda used:", lam)
    print("bit assignment:", assignment)
    print(f"total joint loss at this budget: {total_loss:.4f}")
    print(f"achieved average bit-width: {achieved_avg:.2f} (target: {cfg.target_avg_bits})")

    save_results(cfg.results_path, assignment, sensitivity, lam)
    print(f"saved bit assignment + sensitivity table -> {cfg.results_path}")
    return assignment, sensitivity, lam


if __name__ == "__main__":
    run_objective2(Config())
