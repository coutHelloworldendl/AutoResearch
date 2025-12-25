import os
from pathlib import Path
from openai import OpenAI
from typing import List, Optional, Union
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel

class API:
    def __init__(self, model_name: str = "deepseek"):
        self.model_name = model_name
    
    def forward(self, prompt: str, file: str = None) -> str:
        if self.model_name == "deepseek":
            return self.deepseek_forward(prompt, file)
        elif self.model_name == "kimi":
            return self.kimi_forward(prompt, file)
        else:
            raise ValueError(f"Model {self.model_name} not supported.")

    def deepseek_forward(self, prompt: str, file: str = None) -> str:
        assert file is None, "File input not supported for DeepSeek model."
        client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        
        return response.choices[0].message.content
    
    def kimi_forward(self, prompt: str, file: str = None) -> str:
        client = OpenAI(
            api_key=os.environ.get('MOONSHOT_API_KEY'),
            base_url="https://api.moonshot.cn/v1"
        )

        messages = [
            {
                "role": "system",
                "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。",
            }
        ]

        if file:
            path = Path(file)
            if not path.exists():
                raise FileNotFoundError(f"file {file} not found")

            # create file object - 支持 pdf/doc/images
            file_obj = client.files.create(file=path, purpose="file-extract")

            try:
                file_content = client.files.content(file_id=file_obj.id).text
            except Exception:
                file_content = client.files.retrieve_content(file_id=file_obj.id)

            # 把文件内容放进 system message，靠近前面以便模型参考
            messages.append({"role": "system", "content": file_content})

        # 添加用户请求
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="kimi-k2-turbo-preview",
            messages=messages,
            temperature=0.6,
            stream=False,
        )

        return response.choices[0].message.content


class Embed:

    def __init__(self, model_path: str):
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.max_length = 32768

    def encode(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, max_length=self.max_length)
        return embeddings.cpu().numpy()