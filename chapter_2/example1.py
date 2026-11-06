from transformers import AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B"


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto"
)

config = model.config

print("\n=== Model Configuration Parameters ===")

print("\nArchitecture Parameters:")

print(f"Hidden size: {config.hidden_size}")

print(f"Number of layers: {config.num_hidden_layers}")

print(f"Number of attention heads: {config.num_attention_heads}")

print(f"Intermediate size: {config.intermediate_size}")

print("\nTokenizer Parameters:")

print(f"Vocabulary size: {config.vocab_size}")

print(f"Maximum position embeddings:{config.max_position_embeddings}")

print("\nModel-specific Parameters:")

for key, value in config.to_dict().items():
    if key not in ['architectures', 'model_type', 'torch_dtype']:
        print(f"{key}: {value}")

#####################################################################

print("\nModel Structure:")
def print_module_structure(module, prefix="", max_depth=4, depth=0):
    if depth > max_depth:
        return
    for name, child in module.named_children():
        print(f"{prefix}{name}: {type(child).__name__}")

        if type(child).__name__ == "Qwen2Attention":
            print(f"{prefix}  -> head_dim: {child.head_dim}")
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                p = getattr(child, proj)
                print(f"{prefix}  -> {proj}: {p.in_features} -> {p.out_features}")

        print_module_structure(child, prefix + "  ", max_depth, depth + 1)

print_module_structure(model)


