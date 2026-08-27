"""
Objective 2: Full-Curve Sensitivity Allocation (AP-Quant proposal, eq. 11-13).

Measures a 5-point sensitivity curve per attention block, then allocates
bit-widths as a Multiple-Choice Knapsack Problem solved exactly via DP.
No training, no backprop -- everything here is forward passes + a
combinatorial optimizer.

Depends on ap_quant_common.py (same directory).
"""
import json

import torch

from ap_quant_common import (
    Config,
    load_model_and_tokenizer,
    load_calibration_dataset,
    calib_dataset_to_tensors,
    get_P_and_O,
    joint_loss,
)

PROJ_NAMES = ["q_proj", "k_proj", "v_proj"]


def fake_quantize(W, bits, per_channel=True):
    """Symmetric round-to-nearest -- ONLY for measuring sensitivity.
    The learned LSQ quantizer (Objective 1) is a separate, gradient-trained
    module; this one just needs to be a reasonable stand-in so the sensitivity
    ordering across blocks/bit-widths is meaningful."""
    qmax = 2 ** (bits - 1) - 1
    dim = 1 if per_channel else None
    max_val = W.abs().amax(dim=dim, keepdim=per_channel).clamp(min=1e-8)
    delta = max_val / qmax
    return torch.clamp(torch.round(W / delta), -qmax, qmax) * delta


@torch.no_grad()
def cache_reference(model, calib_tensors):
    """Runs the full-precision model once per calibration example and caches
    P, O for every layer. This is the (O, P) that every quantized variant is
    compared against."""
    ref_P, ref_O = [], []
    for ids in calib_tensors:
        P, O = get_P_and_O(model, ids)
        ref_P.append(P)
        ref_O.append(O)
    return ref_P, ref_O


@torch.no_grad()
def _quantize_block(model, layer_idx, bits):
    attn = model.model.layers[layer_idx].self_attn
    orig = {n: getattr(attn, n).weight.data.clone() for n in PROJ_NAMES}
    for n in PROJ_NAMES:
        getattr(attn, n).weight.data = fake_quantize(orig[n], bits=bits)
    return orig


@torch.no_grad()
def _restore_block(model, layer_idx, orig):
    attn = model.model.layers[layer_idx].self_attn
    for n in PROJ_NAMES:
        getattr(attn, n).weight.data = orig[n]


@torch.no_grad()
def measure_block_sensitivity(model, calib_tensors, layer_idx, bits, lam, ref_P, ref_O):
    """Quantizes ONLY q/k/v of `layer_idx` to `bits` (rest of the model stays
    full precision), runs the calibration set, and returns the average joint
    loss for that layer relative to the cached full-precision reference.
    Eq. (12): Li(b)."""
    orig = _quantize_block(model, layer_idx, bits)
    total = 0.0
    for ex, ids in enumerate(calib_tensors):
        P_q, O_q = get_P_and_O(model, ids)
        loss, _, _ = joint_loss(
            ref_P[ex][layer_idx], ref_O[ex][layer_idx],
            P_q[layer_idx], O_q[layer_idx], lam=lam,
        )
        total += loss.item()
    _restore_block(model, layer_idx, orig)
    return total / len(calib_tensors)


@torch.no_grad()
def estimate_lambda(model, calib_tensors, ref_P, ref_O, ref_bits=4):
    """Eq. (10): lambda = mean(L_output) / mean(L_KL), estimated ONCE at a
    reference bit-width and then held fixed for the entire sensitivity table
    -- otherwise L_i(b) values at different b would not be comparable and
    the MCKP allocation would be meaningless."""
    n_layers = len(model.model.layers)
    l_out_sum, l_kl_sum, n = 0.0, 0.0, 0
    for i in range(n_layers):
        orig = _quantize_block(model, i, ref_bits)
        for ex, ids in enumerate(calib_tensors):
            P_q, O_q = get_P_and_O(model, ids)
            _, l_out, l_kl = joint_loss(ref_P[ex][i], ref_O[ex][i], P_q[i], O_q[i], lam=1.0)
            l_out_sum += l_out
            l_kl_sum += l_kl
            n += 1
        _restore_block(model, i, orig)
    return (l_out_sum / n) / max(l_kl_sum / n, 1e-8)


@torch.no_grad()
def build_sensitivity_table(model, calib_tensors, candidate_bits, lam, ref_P, ref_O, verbose=True):
    n_layers = len(model.model.layers)
    sensitivity = {i: {} for i in range(n_layers)}
    for i in range(n_layers):
        for b in candidate_bits:
            sensitivity[i][b] = measure_block_sensitivity(
                model, calib_tensors, i, b, lam, ref_P, ref_O
            )
        if verbose:
            row = ", ".join(f"{b}b={sensitivity[i][b]:.4f}" for b in candidate_bits)
            print(f"layer {i:2d}: {row}")
    return sensitivity


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
    """Hand-off point for your teammate's Objective 1: this JSON says which
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
    """End-to-end: load model -> load calibration data -> estimate lambda ->
    build the 5-point sensitivity curve -> solve MCKP -> save results."""
    model, tokenizer = load_model_and_tokenizer(cfg)
    calib_ds = load_calibration_dataset(cfg, tokenizer)
    calib_tensors = calib_dataset_to_tensors(calib_ds)
    print(f"{len(calib_tensors)} calibration examples loaded from '{cfg.cal_dataset}'")

    n_layers = len(model.model.layers)
    ref_P, ref_O = cache_reference(model, calib_tensors)

    lam = estimate_lambda(model, calib_tensors, ref_P, ref_O, ref_bits=cfg.lambda_ref_bits)
    print(f"estimated lambda = {lam:.4f}")

    sensitivity = build_sensitivity_table(
        model, calib_tensors, cfg.candidate_bits, lam, ref_P, ref_O
    )

    cost = {b: b for b in cfg.candidate_bits}
    budget = cfg.target_avg_bits * n_layers
    assignment, total_loss = solve_mckp(sensitivity, cost, budget, cfg.candidate_bits)
    achieved_avg = sum(cost[assignment[i]] for i in range(n_layers)) / n_layers
    print(f"total joint loss at this budget: {total_loss:.4f}")
    print(f"achieved average bit-width: {achieved_avg:.2f} (target: {cfg.target_avg_bits})")

    save_results(cfg.results_path, assignment, sensitivity, lam)
    print(f"saved bit assignment + sensitivity table -> {cfg.results_path}")
    return assignment, sensitivity, lam


if __name__ == "__main__":
    run_objective2(Config())
