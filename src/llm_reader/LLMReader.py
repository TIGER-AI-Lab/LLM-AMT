import os
import openai
from openai import OpenAI
import re
import random
from tqdm import tqdm
import time
import json
from vllm import LLM, SamplingParams
import transformers


class LLMReader:
    def __init__(self, model_name, model_type, api_key=None):
        self.model_name = model_name
        self.model_type = model_type
        self.api_key = api_key
        self.llm = None
        self.sampling_params = None
        self.tokenizer = None
        self.client = None
        self.create_client()

    def create_client(self):
        if self.model_type == "local":
            llm = LLM(model=self.model_name, gpu_memory_utilization=0.8,
                      tensor_parallel_size=1,
                      max_model_len=4096,
                      trust_remote_code=True)
            sampling_params = SamplingParams(temperature=0, max_tokens=2048, stop=[])
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.llm = llm
            self.sampling_params = sampling_params
            self.tokenizer = tokenizer
        elif self.model_type == "api":
            openai.api_key = self.api_key
            self.client = openai
        else:
            print("unsupported model type, please implement it manually!")

    def inference_single(self, instruction, input_text):
        if self.model_type == "local":
            outputs = self.llm.generate([instruction + input_text], self.sampling_params)
        elif self.model_type == "api":
            message_text = [{"role": "user", "content": instruction + input_text}]
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_text,
                temperature=0,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            outputs = completion.choices[0].message.content
        else:
            print("unsupported model type, please implement it manually!")
            return None
        return outputs


if __name__ == "__main__":
    llm_reader = LLMReader("/gpfs/public/01/models/hf_models/Llama-2-7b-hf", "local")
    test_result = llm_reader.inference_single("hello, assistant!\n", "compute 1 + 2 for me")
    print("test result:", test_result)


















