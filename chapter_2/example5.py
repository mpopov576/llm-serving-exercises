import time
from vllm import LLM, SamplingParams

model_name = "Qwen/Qwen2.5-0.5B"

llm = LLM(model=model_name, dtype="float16")

prompt = """You are an expert AI historian writing a
detailed chapter for a book titled "The Evolution of
Human-AI Collaboration." … … Write in a formal tone,
with rich detail and examples in each era."""

inference_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

outputs = llm.generate([prompt], inference_params=inference_params)

for output in outputs:
    print(f"Generated text: {output}")


