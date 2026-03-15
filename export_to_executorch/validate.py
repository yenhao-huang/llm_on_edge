from optimum.executorch import ExecuTorchModelForCausalLM
from transformers import AutoTokenizer

base_model = "/Users/yenhaohuang/Desktop/side_project/edge_devices/llm-on-iphone/qwen3_0.6b_executorch"
# Load the exported model
model = ExecuTorchModelForCausalLM.from_pretrained(base_model)

# Initialize tokenizer and generate text
tokenizer = AutoTokenizer.from_pretrained(base_model)
generated_text = model.text_generation(
    tokenizer=tokenizer,
    prompt="say hello",
    max_seq_len=128
)
print(generated_text)