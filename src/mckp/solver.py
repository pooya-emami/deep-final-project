import numpy as np
import torch
from src.apq.quantized_layers import quantize_transformer_attn
from src.apq.calibration import compute_ap_loss, collect_layer_inputs
from src.apq.utils import detect_arch, get_position_embeddings

@torch.no_grad()
def compute_sensitivity_matrix(model, calib_tokens, candidate_bits, temp=2.0, device="cuda", lambda_samples=8):
    arch, q_attn_blocks = quantize_transformer_attn(model, layer_bits=None)

    for q_attn in q_attn_blocks:
        q_attn.force_fp_mode = True
    layer_inputs = collect_layer_inputs(model, arch, calib_tokens, device)
    for q_attn in q_attn_blocks:
        q_attn.force_fp_mode = False

    lam_values = []
    valid_layers = [i for i in range(len(q_attn_blocks)) if len(layer_inputs[i]) > 0]
    if not valid_layers:
        raise RuntimeError("No layer inputs collected")

    for layer_idx in valid_layers[:3]:
        q_attn = q_attn_blocks[layer_idx]
        q_attn.q_proj.set_bits(4)
        q_attn.k_proj.set_bits(4)
        q_attn.v_proj.set_bits(4)
        q_attn.o_proj.set_bits(4)

        n_samples = min(lambda_samples, len(layer_inputs[layer_idx]))
        for idx in np.random.choice(len(layer_inputs[layer_idx]), n_samples, replace=False):
            inp = layer_inputs[layer_idx][idx].to(device)
            pos_emb = get_position_embeddings(model, inp) if arch == "llama" else None

            q_attn.force_fp_mode = True
            P_fp, O_fp, mask = q_attn.forward_components(inp, use_quant=False, position_embeddings=pos_emb)
            q_attn.force_fp_mode = False
            P_q, O_q, _ = q_attn.forward_components(inp, use_quant=True, position_embeddings=pos_emb)

            _, l_out, l_kl, _ = compute_ap_loss(P_fp, O_fp, P_q, O_q, mask, lam=1.0, temp=temp)
            lam_values.append(float(l_out / (l_kl + 1e-8)))

    lam = min(max(np.mean(lam_values) if lam_values else 1.0, 0.01), 10.0)

    sensitivity = {i: {} for i in range(len(q_attn_blocks))}
    for layer_idx, q_attn in enumerate(q_attn_blocks):
        if len(layer_inputs[layer_idx]) == 0:
            continue

        for bits in candidate_bits:
            q_attn.q_proj.set_bits(bits)
            q_attn.k_proj.set_bits(bits)
            q_attn.v_proj.set_bits(bits)
            q_attn.o_proj.set_bits(bits)

            total = 0.0
            for inp in layer_inputs[layer_idx]:
                inp = inp.to(device)
                pos_emb = get_position_embeddings(model, inp) if arch == "llama" else None

                q_attn.force_fp_mode = True
                P_fp, O_fp, mask = q_attn.forward_components(inp, use_quant=False, position_embeddings=pos_emb)
                q_attn.force_fp_mode = False
                P_q, O_q, _ = q_attn.forward_components(inp, use_quant=True, position_embeddings=pos_emb)

                l_j, _, _, _ = compute_ap_loss(P_fp, O_fp, P_q, O_q, mask, lam=lam, temp=temp)
                total += l_j.item()

            sensitivity[layer_idx][bits] = total / max(len(layer_inputs[layer_idx]), 1)

    return sensitivity, lam

def solve_mckp(sensitivity, cost, budget, candidate_bits):
    budget = int(budget)
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