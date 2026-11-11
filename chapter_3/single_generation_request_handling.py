import torch
import logging
import multiprocessing as mp
import uuid
from dataclasses import dataclass
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


@dataclass
class Sequence:
    id: str
    prompt: str
    a: object = None
    b: object = None


class WorkloadManager:
    pass


class LLMEngine:
    def __init__(self):
        self.model_executor = ModelExecutor()
        self.workload_manager = WorkloadManager()
        self.max_tokens = 20
        self.model_executor.setup_worker("facebook/opt-125m")

    def basic_generate(self, prompt: str) -> str:
        sequence = Sequence(str(uuid.uuid4()), prompt, None, None)
        results = self.model_executor.execute_batch(sequence)
        return results[0]['generated_text']

class ModelExecutor:
    def __init__(self):
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

    def setup_worker(self, model_name: str):
        self.worker_process = mp.Process(
            target=ModelWorker.run,
            args=(model_name, self.task_queue, self.result_queue)
        )
        self.worker_process.start()

    def execute_batch(self, prompt):
        self.task_queue.put(prompt)
        status, result = self.result_queue.get()
        return [result]

class ModelWorker:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = ModelManager().load_model(model_name)
        self.model.to(self.device)

    def generate(self, prompt_data):
        inputs = self.tokenizer(prompt_data.prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=20)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            'request_id': prompt_data.id,
            'generated_text': generated_text
        }

    @staticmethod
    def run(model_name: str, task_queue: mp.Queue, result_queue: mp.Queue):
        worker = ModelWorker(model_name)
        while True:
            logger.debug("Waiting for batch from queue...")
            request = task_queue.get()
            result_queue.put(('complete', worker.generate(request)))


class ModelManager:
    def load_model(self, model_name: str = "facebook/opt-125m"):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return model, tokenizer


class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    generated_text: str


app = FastAPI()
llm_engine = None

def get_llm():
    global llm_engine
    if llm_engine is None:
        llm_engine = LLMEngine()
    return llm_engine


@app.post("/basic_generate", response_model=GenerateResponse)
async def basic_generate(request: GenerateRequest,
                         llm: LLMEngine = Depends(get_llm)):
    generated_text = llm.basic_generate(request.prompt)
    return GenerateResponse(generated_text=generated_text)

