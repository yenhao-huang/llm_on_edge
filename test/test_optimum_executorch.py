from optimum.executorch import ExecuTorchModelForCausalLM
from transformers import AutoTokenizer

# Load and export the model on-the-fly
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
model = ExecuTorchModelForCausalLM.from_pretrained(
    model_id,
    recipe="xnnpack",
    attn_implementation="custom_sdpa",  # Use custom SDPA implementation for better performance
    use_custom_kv_cache=True,  # Use custom KV cache for better performance
    **{"qlinear": "8da4w", "qembedding": "8w"},  # Quantize linear and embedding layers
)

# Generate text right away
tokenizer = AutoTokenizer.from_pretrained(model_id)
generated_text = model.text_generation(
    tokenizer=tokenizer,
    prompt="Once upon a time",
    max_seq_len=128,
)
print(generated_text)
