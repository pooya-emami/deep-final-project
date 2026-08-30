import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

from src.apq.utils import set_seed, evaluate_ppl, prepare_tokens, detect_arch
from .utils import replace_gpt2_qkv_conv1d_with_linear, replace_gpt2_wo_conv1d_with_linear

def build_gptq_recipe(arch, include_wo=True, num_bits=4):
    if arch == "gpt2":
        targets = [r"re:transformer\.h\.\d+\.attn\.c_attn$"]
        if include_wo:
            targets.append(r"re:transformer\.h\.\d+\.attn\.c_proj$")
    elif arch == "llama":
        targets = [
            r"re:model\.layers\.\d+\.self_attn\.q_proj$",
            r"re:model\.layers\.\d+\.self_attn\.k_proj$",
            r"re:model\.layers\.\d+\.self_attn\.v_proj$",
        ]
        if include_wo:
            targets.append(r"re:model\.layers\.\d+\.self_attn\.o_proj$")
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    return GPTQModifier(
        config_groups={
            "qkv_wo" if include_wo else "qkv_only": {
                "targets": targets,
                "weights": {
                    "num_bits": num_bits,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "channel",
                },
            }
        }
    )

def run_gptq_quantization(MODEL_ID, calib_corpus, eval_corpus, output_dir, include_wo=True, num_bits=4, seed=42):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    arch = detect_arch(model)
    if arch == "gpt2":
        model = replace_gpt2_qkv_conv1d_with_linear(model)
        if include_wo:
            model = replace_gpt2_wo_conv1d_with_linear(model)

    eval_tokens = prepare_tokens(eval_corpus, tokenizer, chunk_size=128)

    fp_ppl = evaluate_ppl(model, eval_tokens, device)
    print(f"FP32 baseline PPL ({arch}): {fp_ppl:.2f}")

    calib_dataset = Dataset.from_dict({"text": calib_corpus})
    recipe = build_gptq_recipe(arch, include_wo=include_wo, num_bits=num_bits)

    oneshot(
        model=model,
        dataset=calib_dataset,
        recipe=recipe,
        output_dir=output_dir,
        max_seq_length=128,
        num_calibration_samples=min(256, len(calib_dataset)),
    )

    model.eval()
    ppl = evaluate_ppl(model, eval_tokens, device)
    label = "QKV+Wo" if include_wo else "QKV-Only"
    print(f"\n[Result]: Perplexity of GPTQ {label} model ({arch}, {num_bits}-bit): {ppl:.2f}")

    return model, fp_ppl, ppl