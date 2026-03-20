from langchain_experimental.open_clip import OpenCLIPEmbeddings
from model.qwen3_vl_embedding import Qwen3VLEmbeddings


def get_embedding_model(model_key: str, model_path: str = None):
    """
    根據 model_key 回傳對應的 LangChain Embeddings 物件。

    Args:
        model_key:  "clip" 或 "qwen"
        model_path: 模型路徑或 HuggingFace Hub ID（CLIP 不需要）

    用法：
        model = get_embedding_model("clip")
        model = get_embedding_model("qwen", "Qwen/Qwen3-VL-Embedding-2B")
        model = get_embedding_model("qwen")  # 使用預設 Hub ID
    """
    if model_key == "clip":
        print("載入 CLIP 模型 (OpenCLIP via LangChain)...")
        return OpenCLIPEmbeddings()

    elif model_key == "qwen":
        # model_path 可以是：
        #   - HuggingFace Hub ID: "Qwen/Qwen3-VL-Embedding-2B" (自動下載到 ~/.cache/huggingface/)
        #   - 本地路徑: "./models/Qwen3-VL-Embedding-2B"
        #   - None: 使用 Qwen3VLEmbeddings class 自帶的預設值
        hub_or_path = model_path or "Qwen/Qwen3-VL-Embedding-2B"
        print(f"載入 Qwen3-VL-Embedding 模型: {hub_or_path}")
        return Qwen3VLEmbeddings(hub_or_path)

    raise ValueError(f"不支援的模型: {model_key}，目前支援: clip, qwen")