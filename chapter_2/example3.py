from transformers import pipeline

generator = pipeline('text-generation', model='Qwen/Qwen2.5-0.5B')
prompt = "Write a short introduction about the US capital city."

generated_text = generator(prompt, max_length=50, num_return_sequences=1)

print(generated_text[0]['generated_text'])