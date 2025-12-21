import os
from pathlib import Path
# lazy import OpenAI to avoid failing when openai package is not installed
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
from typing import List, Optional, Union

try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
except Exception:
    # Defer import errors until the Embed class is used
    torch = None
    F = None
    np = None
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
    """Embed helper: load a pretrained embed model and provide an `encode` method.

    Behavior:
    - If the loaded model exposes an `encode` method, it will be used directly.
    - Otherwise the class will tokenize inputs and run the model, mean-pooling
      the `last_hidden_state` with the attention mask.

    Returns a NumPy array of shape (N, D). Optionally normalizes to unit L2.
    """

    def __init__(self, model_name: str = "nvidia/NV-Embed-v2", device: Optional[str] = None, max_length: int = 32768, batch_size: int = 8, trust_remote_code: bool = True, multi_gpu: bool = False):
        if torch is None or AutoModel is None or AutoTokenizer is None:
            raise ImportError("torch and transformers are required to use Embed. Please install them.")

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code

        self.multi_gpu = bool(multi_gpu)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)

        # If multiple GPUs requested and available, wrap with DataParallel for inference
        try:
            if self.multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
                device_ids = list(range(torch.cuda.device_count()))
                self.model.to(torch.device('cuda'))
                self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
            else:
                self.model.to(self.device)
        except Exception:
            # best-effort placement; continue on exception
            pass

    def _batch_iter(self, items):
        for i in range(0, len(items), self.batch_size):
            yield items[i : i + self.batch_size]

    def encode(self, texts: Union[str, List[str]], instruction: Optional[str] = None, max_length: Optional[int] = None, normalize: bool = True) -> "np.ndarray":
        """Encode one or multiple texts and return embeddings as a NumPy array.

        Args:
            texts: a single string or list of strings.
            instruction: optional instruction string (passed to model.encode if available).
            max_length: optional override for max token length.
            normalize: whether to L2-normalize embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        if max_length is None:
            max_length = self.max_length

        # If the model exposes an `encode` helper (like NV-Embed), prefer it
        model_for_call = self.model
        # If wrapped by DataParallel, the real module is under .module
        if hasattr(self.model, "module"):
            model_for_call = self.model.module

        if hasattr(model_for_call, "encode"):
            # model.encode may return torch.Tensor or numpy array
            emb = model_for_call.encode(texts, instruction=(instruction or ""), max_length=max_length, batch_size=self.batch_size)
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            return self._maybe_normalize(emb, normalize)

        # Otherwise use tokenizer + forward + mean pooling
        all_embs = []
        for batch in self._batch_iter(texts):
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
            # DataParallel will scatter tensors across devices; ensure tensors are on primary device
            target_device = self.device
            if self.multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
                # DataParallel uses primary CUDA device (cuda:0)
                target_device = torch.device('cuda:0')
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs, return_dict=True)
                last_hidden = out.last_hidden_state  # (B, L, H)

            # move to cpu for pooling
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
                summed = (last_hidden * mask).sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / lengths
            else:
                pooled = last_hidden.mean(dim=1)

            pooled = pooled.detach().cpu().numpy()
            all_embs.append(pooled)

        emb = np.vstack(all_embs)
        return self._maybe_normalize(emb, normalize)

    def _maybe_normalize(self, emb: "np.ndarray", normalize: bool) -> "np.ndarray":
        if not normalize or np is None:
            return emb
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return emb / norms
    
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