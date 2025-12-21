import os
from pathlib import Path
from openai import OpenAI
from typing import List, Optional, Union

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception:
    # Defer import errors until the Embed class is used
    torch = None
    AutoTokenizer = None
    AutoModel = None

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


class Embed:
    """Embed wrapper for a local Hugging Face model (e.g. nvidia/NV-Embed-v2).

    Usage:
      e = Embed(model_dir='/model/squirrel/NV-Embed-v2')
      vec = e.encode('some text')
    """

    def __init__(self, model_dir: str = '/model/squirrel/NV-Embed-v2', device: Optional[str] = None):
        if AutoModel is None or AutoTokenizer is None or torch is None:
            raise RuntimeError('Required packages not found. Please install transformers and torch.')

        self.model_dir = model_dir
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        except Exception as e:
            raise RuntimeError(f'加载嵌入模型失败: {e}')

        self.model.to(self.device)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> Union[List[float], List[List[float]]]:
        """Encode a single string or a list of strings to embedding vectors (list of floats).

        Returns a single vector for a single input string, or a list of vectors for multiple inputs.
        """
        single = False
        if isinstance(texts, str):
            texts = [texts]
            single = True

        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        input_ids = enc['input_ids'].to(self.device)
        attention_mask = enc['attention_mask'].to(self.device)

        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            emb = self._mean_pooling(out, attention_mask)
            if normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        emb = emb.cpu().numpy()
        if single:
            return emb[0].tolist()
        return [v.tolist() for v in emb]
    
    def kimi_forward(self, prompt: str, file: str = None) -> str:
        client = OpenAI(
            api_key=os.environ.get('KIMI_API_KEY'),
            base_url="https://api.kimi.com"
        )

        messages = [
            {
                "role": "system",
                "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。",
            }
        ]

        # 如果提供了文件，上传并读取内容放入 system message
        if file:
            path = Path(file)
            if not path.exists():
                raise FileNotFoundError(f"file {file} not found")

            # create file object - 支持 pdf/doc/images
            file_obj = client.files.create(file=path, purpose="file-extract")

            # Kimi/OpenAI 客户端有两种方式获取文件内容：
            # 旧版: retrieve_content(file_id)
            # 新版: files.content(file_id).text
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
    
