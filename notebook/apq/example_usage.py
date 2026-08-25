import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ap_quantizer import AttentionPreservingQuantizer, Config
import json

def main():
    # Configuration
    class Config:
        def __init__(self):
            self.batch_size = 2
            self.lr = 1e-3
            self.initial_bit_width = 8
            self.per_channel = True
            self.num_calibration_batches = 10
    
    # Load model
    model_name = "HuggingFaceTB/SmolLM2-135M"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare calibration data
    calibration_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Natural language processing enables computers to understand human language.",
        "Transformers have revolutionized deep learning.",
        "Quantization helps deploy large models on edge devices.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Deep learning models require large amounts of data.",
        "Artificial intelligence continues to advance rapidly.",
        "Neural networks are inspired by the human brain.",
        "Training large models requires significant computational resources.",
    ] * 5  # Repeat for more data
    
    # Initialize quantizer
    config = Config()
    quantizer = AttentionPreservingQuantizer(
        model, tokenizer, calibration_texts, config
    )
    
    # Run calibration with KL only
    print("=" * 50)
    print("Calibrating with KL only...")
    print("=" * 50)
    quantizer.calibrate(num_iterations=50, use_entropy=False)
    
    # Evaluate
    results_kl = quantizer.evaluate_attention_error()
    print(f"KL only results: {results_kl}")
    
    # Reset and calibrate with entropy
    quantizer.reset_to_original()
    print("\n" + "=" * 50)
    print("Calibrating with KL + Entropy...")
    print("=" * 50)
    quantizer.calibrate(num_iterations=50, use_entropy=True, beta=0.1)
    
    # Evaluate
    results_entropy = quantizer.evaluate_attention_error()
    print(f"KL + Entropy results: {results_entropy}")
    
    # Save the best model
    quantizer.save_quantized_model("ap_quant_model")
    
    # Print comparison
    print("\n" + "=" * 50)
    print("Comparison:")
    print("=" * 50)
    print(f"KL only - KL Divergence: {results_kl['avg_kl_divergence']:.4f}")
    print(f"KL + Entropy - KL Divergence: {results_entropy['avg_kl_divergence']:.4f}")
    print(f"KL only - Entropy Error: {results_kl['avg_entropy_error']:.4f}")
    print(f"KL + Entropy - Entropy Error: {results_entropy['avg_entropy_error']:.4f}")

if __name__ == "__main__":
    main()