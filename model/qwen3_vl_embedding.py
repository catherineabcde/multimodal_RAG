from typing import List, Dict, Any
import torch
import json
from langchain_core.embeddings import Embeddings

from .scripts_qwen3_vl_embedding import Qwen3VLEmbedder


class Qwen3VLEmbeddings(Embeddings):
    """
    LangChain Embeddings wrapper for Qwen3-VL-Embedding model.

    支援文字和圖片的多模態 embedding。
    模型會自動從 HuggingFace Hub 下載並快取到 ~/.cache/huggingface/hub/。

    用法：
        # 使用預設 Hub ID（自動下載）
        model = Qwen3VLEmbeddings()

        # 使用指定 Hub ID 或本地路徑
        model = Qwen3VLEmbeddings("Qwen/Qwen3-VL-Embedding-2B")
        model = Qwen3VLEmbeddings("./models/Qwen3-VL-Embedding-2B")
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-Embedding-2B", device: str = "cuda"):
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"載入模型: {model_name}")

        self.model = Qwen3VLEmbedder(
            model_name_or_path=model_name,
            min_pixels=384 * 384,
            max_pixels=1024 * 1024,
            device=device,
        )

    def _parse_text_to_dict(self, text: str) -> Dict[str, Any]:
        """將字串解析為多模態輸入 dict（支援 JSON 格式的 image/text/video）。"""
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and any(k in parsed for k in ['text', 'image', 'video']):
                return parsed
            return {"text": text}
        except (json.JSONDecodeError, TypeError):
            return {"text": text}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多個文件（LangChain 介面）。"""
        embeddings = []
        for t in texts:
            processed_input = self._parse_text_to_dict(t)
            embed = self.model.process([processed_input])
            """
            NumPy 原生無法直接接收 PyTorch 的 bfloat16 張量格式
            """
            # embeddings.append(embed[0].detach().cpu().numpy().tolist())
            embeddings.append(embed[0].detach().float().cpu().numpy().tolist())

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入單個查詢（LangChain 介面）。"""
        embeddings = self.model.process([{"text": text}])
        """
        NumPy 原生無法直接接收 PyTorch 的 bfloat16 張量格式
        """
        # return embeddings[0].detach().cpu().numpy().tolist()
        return embeddings[0].detach().float().cpu().numpy().tolist()

    def embed_image(self, uris: List[str]) -> List[List[float]]:
        """嵌入多張圖片，回傳 embedding list。"""
        embeddings = []
        for image in uris:
            embed = self.model.process([{"image": image}])
            embeddings.append(embed[0].detach().cpu().numpy().tolist())
        return embeddings
