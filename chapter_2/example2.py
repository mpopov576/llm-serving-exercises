from transformers import AutoModelForCausalLM, AutoTokenizer
from bertviz import head_view
import torch

text = "write a short introduction about the US capital city"
model_name = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name, output_attentions=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    output_attentions=True
).eval().cuda()

inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

attention = outputs.attentions

tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
html = head_view(attention, tokens, html_action="return")
with open("head_view.html", "w", encoding="utf-8") as f:
    f.write(html.data)
print("Saved head_view.html")