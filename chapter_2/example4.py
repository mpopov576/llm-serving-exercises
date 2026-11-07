import time

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = (
    AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True
    ).to("cuda" if torch.cuda.is_available() else "cpu")
)

prompt = """The history of human … … … How might the next wave of
communication tools shape our relationships, societies, and sense
of identity?"""

max_new_tokens = 100

idx = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

start_time = total_time = time.time()

times = []

for _ in range(max_new_tokens):
    idx_cond = idx

    with torch.no_grad():
        outputs = model(idx_cond)
        logits = outputs.logits

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probas, num_samples=1)

        time_cost = time.time() - start_time
        times.append(time_cost)

        idx = torch.cat((idx, idx_next), dim=1)

        if idx_next.item() == tokenizer.eos_token_id:
            print("\n[Generation completed - EOS token reached]")
            break

generated_text = tokenizer.decode(idx[0], skip_special_tokens=True)

print(generated_text)
print(times)
