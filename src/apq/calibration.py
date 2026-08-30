import numpy as np
import torch
from .quantized_layers import quantize_transformer_attn
from .utils import detect_arch, get_position_embeddings, evaluate_ppl, _StopForward, should_use_amp, get_model_precision

def compute_ap_loss(P_fp, O_fp, P_q, O_q, causal_mask, lam=1.0, temp=2.0, eps=1e-8):
    diff = O_fp - O_q
    mse_raw = diff.pow(2).mean()
    l_output = mse_raw / (O_fp.pow(2).mean() + eps)
    
    P_fp_c = torch.softmax(torch.log(P_fp.clamp(min=eps)) / temp, dim=-1)
    P_q_c = torch.softmax(torch.log(P_q.clamp(min=eps)) / temp, dim=-1)
    kl_matrix = P_fp_c * (torch.log(P_fp_c + eps) - torch.log(P_q_c + eps))
    valid_mask = (~causal_mask).unsqueeze(0).unsqueeze(0)
    kl_valid = kl_matrix.masked_fill(~valid_mask, 0.0)
    kl_per_query = kl_valid.sum(dim=-1)
    query_valid = valid_mask.any(dim=-1).to(kl_per_query.dtype)
    l_kl = (kl_per_query * query_valid).sum() / query_valid.sum().clamp_min(1.0)
    return l_output + lam * l_kl, l_output, l_kl, mse_raw

def collect_layer_inputs(model, arch, calib_tokens, device, max_layer_idx=None, use_amp=None):
    if arch == "gpt2":
        num_layers = len(model.transformer.h)
    else:
        num_layers = len(model.model.layers)
    
    if use_amp is None:
        use_amp = should_use_amp(model) and device == "cuda"
    
    layer_inputs = [[] for _ in range(num_layers)]
    stop_at = num_layers - 1 if max_layer_idx is None else max_layer_idx

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            for row in out.detach().split(1, dim=0):
                layer_inputs[layer_idx].append(row.cpu())
            if layer_idx == stop_at:
                raise _StopForward()
        return hook_fn

    handles = []
    if arch == "gpt2":
        for layer_idx, block in enumerate(model.transformer.h):
            handles.append(block.ln_1.register_forward_hook(make_hook(layer_idx)))
    else:
        for layer_idx, block in enumerate(model.model.layers):
            handles.append(block.input_layernorm.register_forward_hook(make_hook(layer_idx)))

    with torch.no_grad():
        if use_amp and device == "cuda":
            model_dtype = get_model_precision(model)
            with torch.autocast(device_type="cuda", dtype=model_dtype):
                for chunk in calib_tokens:
                    try:
                        model(chunk.to(device), use_cache=False)
                    except _StopForward:
                        pass
        else:
            for chunk in calib_tokens:
                try:
                    model(chunk.to(device), use_cache=False)
                except _StopForward:
                    pass

    for h in handles:
        h.remove()
    return layer_inputs

def calibrate_ap_quant_sequential(model, calib_tokens, device, layer_bits=None,
                                  num_steps_per_layer=200, lr=1e-3, init_temp=2.0,
                                  lambda_update_every=10, batch_size=4, val_fraction=0.1,
                                  calibration_passes=2):
    model.eval()
    model.to(device)
    arch, q_attn_blocks = quantize_transformer_attn(model, layer_bits=layer_bits)

    for q_attn in q_attn_blocks:
        q_attn.force_fp_mode = True
    layer_inputs_fp = collect_layer_inputs(model, arch, calib_tokens, device)
    for q_attn in q_attn_blocks:
        q_attn.force_fp_mode = False

    for q_attn in q_attn_blocks:
        q_attn.q_proj.set_bits(4)
        q_attn.k_proj.set_bits(4)
        q_attn.v_proj.set_bits(4)
        q_attn.o_proj.set_bits(4)

    best_scales_overall = [(q_attn.q_proj.raw_scale.data.clone(),
                           q_attn.k_proj.raw_scale.data.clone(),
                           q_attn.v_proj.raw_scale.data.clone(),
                           q_attn.o_proj.raw_scale.data.clone()) for q_attn in q_attn_blocks]
    best_overall_loss = float('inf')

    for _ in range(calibration_passes):
        for layer_idx, q_attn in enumerate(q_attn_blocks):
            if len(layer_inputs_fp[layer_idx]) == 0:
                continue

            q_attn.force_fp_mode = True
            layer_inputs_quant = collect_layer_inputs(model, arch, calib_tokens, device, max_layer_idx=layer_idx)
            q_attn.force_fp_mode = False

            current_inputs = layer_inputs_quant[layer_idx] if len(layer_inputs_quant[layer_idx]) > 0 else layer_inputs_fp[layer_idx]
            n_samples = len(current_inputs)
            n_val = max(1, int(n_samples * val_fraction))
            shuffled = np.random.permutation(n_samples)
            train_inputs = [current_inputs[i] for i in shuffled[:-n_val]]
            val_inputs = [current_inputs[i] for i in shuffled[-n_val:]]

            params = [q_attn.q_proj.raw_scale, q_attn.k_proj.raw_scale,
                     q_attn.v_proj.raw_scale, q_attn.o_proj.raw_scale]
            optimizer = torch.optim.Adam(params, lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps_per_layer)

            lam_val = 1.0
            if len(train_inputs) > 0:
                lam_values = []
                for idx in np.random.choice(len(train_inputs), min(8, len(train_inputs)), replace=False):
                    inp = train_inputs[idx].to(device)
                    pos_emb = get_position_embeddings(model, inp) if arch == "llama" else None
                    q_attn.force_fp_mode = True
                    P_fp, O_fp, mask = q_attn.forward_components(inp, use_quant=False, position_embeddings=pos_emb)
                    q_attn.force_fp_mode = False
                    P_q, O_q, _ = q_attn.forward_components(inp, use_quant=True, position_embeddings=pos_emb)
                    _, l_out, l_kl, _ = compute_ap_loss(P_fp, O_fp, P_q, O_q, mask, lam=1.0, temp=init_temp)
                    lam_values.append(float(l_out / (l_kl + 1e-8)))
                lam_val = min(max(np.mean(lam_values), 0.01), 10.0)

            temp = init_temp
            best_loss = float("inf")
            best_scales = (q_attn.q_proj.raw_scale.data.clone(),
                          q_attn.k_proj.raw_scale.data.clone(),
                          q_attn.v_proj.raw_scale.data.clone(),
                          q_attn.o_proj.raw_scale.data.clone())

            def step_loss(inputs):
                if len(inputs) == 0:
                    return None
                idxs = torch.randint(0, len(inputs), (min(batch_size, len(inputs)),))
                total = 0.0
                for idx in idxs:
                    inp = inputs[idx.item()].to(device)
                    pos_emb = get_position_embeddings(model, inp) if arch == "llama" else None
                    q_attn.force_fp_mode = True
                    with torch.no_grad():
                        P_fp, O_fp, mask = q_attn.forward_components(inp, use_quant=False, position_embeddings=pos_emb)
                    q_attn.force_fp_mode = False
                    P_q, O_q, _ = q_attn.forward_components(inp, use_quant=True, position_embeddings=pos_emb)
                    l_j, _, _, _ = compute_ap_loss(P_fp, O_fp, P_q, O_q, mask, lam=lam_val, temp=temp)
                    total += l_j
                return total / len(idxs)

            def val_loss(inputs):
                if len(inputs) == 0:
                    return float('inf')
                total = 0.0
                for inp in inputs:
                    inp = inp.to(device)
                    pos_emb = get_position_embeddings(model, inp) if arch == "llama" else None
                    q_attn.force_fp_mode = True
                    with torch.no_grad():
                        P_fp, O_fp, mask = q_attn.forward_components(inp, use_quant=False, position_embeddings=pos_emb)
                    q_attn.force_fp_mode = False
                    with torch.no_grad():
                        P_q, O_q, _ = q_attn.forward_components(inp, use_quant=True, position_embeddings=pos_emb)
                    l_j, _, _, _ = compute_ap_loss(P_fp, O_fp, P_q, O_q, mask, lam=lam_val, temp=temp)
                    total += l_j.item()
                return total / len(inputs)

            for step in range(num_steps_per_layer):
                optimizer.zero_grad()
                l_joint = step_loss(train_inputs)
                if l_joint is None:
                    continue
                l_joint.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                optimizer.step()
                scheduler.step()

                if (step + 1) % lambda_update_every == 0:
                    lam_val = min(max(0.5 * lam_val + 0.5 * float(l_joint.item() / 1e-8), 0.01), 10.0)
                temp = max(1.0, temp * 0.999)

                if step % 10 == 0 or step == 0:
                    v_loss = val_loss(val_inputs)
                    if v_loss < best_loss:
                        best_loss = v_loss
                        best_scales = (q_attn.q_proj.raw_scale.data.clone(),
                                      q_attn.k_proj.raw_scale.data.clone(),
                                      q_attn.v_proj.raw_scale.data.clone(),
                                      q_attn.o_proj.raw_scale.data.clone())

            q_attn.q_proj.raw_scale.data = best_scales[0]
            q_attn.k_proj.raw_scale.data = best_scales[1]
            q_attn.v_proj.raw_scale.data = best_scales[2]
            q_attn.o_proj.raw_scale.data = best_scales[3]
            for p in params:
                p.requires_grad = False

        current_ppl = evaluate_ppl(model, calib_tokens, device)
        if current_ppl < best_overall_loss:
            best_overall_loss = current_ppl
            best_scales_overall = [(q_attn.q_proj.raw_scale.data.clone(),
                                   q_attn.k_proj.raw_scale.data.clone(),
                                   q_attn.v_proj.raw_scale.data.clone(),
                                   q_attn.o_proj.raw_scale.data.clone()) for q_attn in q_attn_blocks]

    for q_attn, scales in zip(q_attn_blocks, best_scales_overall):
        q_attn.q_proj.raw_scale.data = scales[0]
        q_attn.k_proj.raw_scale.data = scales[1]
        q_attn.v_proj.raw_scale.data = scales[2]
        q_attn.o_proj.raw_scale.data = scales[3]
    return model

def calibrate_ap_quant_joint(model, calib_tokens, device, layer_bits=None,
                             num_steps=300, lr=1e-3, init_temp=2.0,
                             lambda_update_every=10, batch_size=4, val_fraction=0.1):
    model.eval()
    model.to(device)
    arch, q_attn_blocks = quantize_transformer_attn(model, layer_bits=layer_bits)

    for q_attn in q_attn_blocks:
        q_attn.force_fp_mode = True
    layer_inputs = collect_layer_inputs(model, arch, calib_tokens, device)
    for q_attn in q_attn_blocks:
        q_attn.force_fp_mode = False

    train_inputs, val_inputs = {}, {}
    for layer_idx in range(len(q_attn_blocks)):
        n_samples = len(layer_inputs[layer_idx])
        if n_samples == 0:
            train_inputs[layer_idx], val_inputs[layer_idx] = [], []
            continue
        n_val = max(1, int(n_samples * val_fraction))
        shuffled = np.random.permutation(n_samples)
        train_inputs[layer_idx] = [layer_inputs[layer_idx][i] for i in shuffled[:-n_val]]
        val_inputs[layer_idx] = [layer_inputs[layer_idx][i] for i in shuffled[-n_val:]]

    params = []
    for q_attn in q_attn_blocks:
        params.extend([q_attn.q_proj.raw_scale, q_attn.k_proj.raw_scale,
                      q_attn.v_proj.raw_scale, q_attn.o_proj.raw_scale])
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    lam_val = 1.0
    valid_layers = [i for i in range(len(q_attn_blocks)) if len(train_inputs[i]) > 0]
    if valid_layers:
        lam_values = []
        for layer_idx in valid_layers[:3]:
            q_attn = q_attn_blocks[layer_idx]
            for idx in np.random.choice(len(train_inputs[layer_idx]), min(8, len(train_inputs[layer_idx])), replace=False):
                inp = train_inputs[layer_idx][idx].to(device)
                pos_emb = get_position_embeddings(model, inp) if arch == "llama" else None
                with torch.no_grad():
                    P_fp, O_fp, mask = q_attn.forward_components(inp, use_quant=False, position_embeddings=pos_emb)
                    P_q, O_q, _ = q_attn.forward_components(inp, use_quant=True, position_embeddings=pos_emb)
                    _, l_out, l_kl, _ = compute_ap_loss(P_fp, O_fp, P_q, O_q, mask, lam=1.0, temp=init_temp)
                    lam_values.append(float(l_out / (l_kl + 1e-8)))
        lam_val = min(max(np.mean(lam_values), 0.01), 10.0)

    temp = init_temp
    best_loss = float('inf')
    best_scales = [(q_attn.q_proj.raw_scale.data.clone(),
                   q_attn.k_proj.raw_scale.data.clone(),
                   q_attn.v_proj.raw_scale.data.clone(),
                   q_attn.o_proj.raw_scale.data.clone()) for q_attn in q_attn_blocks]

    for step in range(num_steps):
        optimizer.zero_grad()
        total_loss = 0.0
        count = 0

        for layer_idx, q_attn in enumerate(q_attn_blocks):
            if len(train_inputs[layer_idx]) == 0:
                continue
            idxs = torch.randint(0, len(train_inputs[layer_idx]), (min(batch_size, len(train_inputs[layer_idx])),))
            for idx in idxs:
                inp = train_inputs[layer_idx][idx.item()].to(device)
                pos_emb = get_position_embeddings(model, inp) if arch == "llama" else None
                with torch.no_grad():
                    P_fp, O_fp, mask = q_attn.forward_components(inp, use_quant=False, position_embeddings=pos_emb)
                P_q, O_q, _ = q_attn.forward_components(inp, use_quant=True, position_embeddings=pos_emb)
                l_j, _, _, _ = compute_ap_loss(P_fp, O_fp, P_q, O_q, mask, lam=lam_val, temp=temp)
                total_loss += l_j
                count += 1

        if count > 0:
            total_loss.backward()
            for q_attn in q_attn_blocks:
                torch.nn.utils.clip_grad_norm_(
                    [q_attn.q_proj.raw_scale, q_attn.k_proj.raw_scale,
                     q_attn.v_proj.raw_scale, q_attn.o_proj.raw_scale], 10.0)
            optimizer.step()
            scheduler.step()

            if (step + 1) % lambda_update_every == 0:
                lam_val = min(max(0.5 * lam_val + 0.5 * float(total_loss.item() / (count * 1e-8)), 0.01), 10.0)
            temp = max(1.0, temp * 0.999)

        if step % 10 == 0 or step == 0:
            val_loss = 0.0
            val_count = 0
            for layer_idx, q_attn in enumerate(q_attn_blocks):
                for inp in val_inputs.get(layer_idx, []):
                    inp = inp.to(device)
                    pos_emb = get_position_embeddings(model, inp) if arch == "llama" else None
                    with torch.no_grad():
                        P_fp, O_fp, mask = q_attn.forward_components(inp, use_quant=False, position_embeddings=pos_emb)
                        P_q, O_q, _ = q_attn.forward_components(inp, use_quant=True, position_embeddings=pos_emb)
                        l_j, _, _, _ = compute_ap_loss(P_fp, O_fp, P_q, O_q, mask, lam=lam_val, temp=temp)
                        val_loss += l_j.item()
                        val_count += 1
            avg_val_loss = val_loss / val_count if val_count > 0 else float('inf')

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_scales = [(q_attn.q_proj.raw_scale.data.clone(),
                               q_attn.k_proj.raw_scale.data.clone(),
                               q_attn.v_proj.raw_scale.data.clone(),
                               q_attn.o_proj.raw_scale.data.clone()) for q_attn in q_attn_blocks]

    for q_attn, scales in zip(q_attn_blocks, best_scales):
        q_attn.q_proj.raw_scale.data = scales[0]
        q_attn.k_proj.raw_scale.data = scales[1]
        q_attn.v_proj.raw_scale.data = scales[2]
        q_attn.o_proj.raw_scale.data = scales[3]
    return model