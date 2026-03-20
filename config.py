import os
import configparser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config(config_file):
    """
    讀取 .ini 設定檔，回傳一個 dict-like 的 config 物件。

    用法：
        config = load_config("configs/exp-clip.ini")
        model_name = config.get("model", "name")       # → "clip"
        db_path    = config.get("database", "db_path")  # → "./db/chroma_clip"
    """
    config = configparser.ConfigParser()
    config.read(config_file, encoding="utf-8")
    return config


def get_experiment_settings(config_file):
    """
    把 .ini 設定檔讀取後，轉成一個簡單好用的 dict。
    這樣在程式裡就可以用 settings["model_name"] 存取。
    """
    config = load_config(config_file)

    settings = {
        # [model] section — visual embedding model
        "model_name":  config.get("model", "name"),
        "model_path":  config.get("model", "model_path", fallback=None) or None,

        # [database] section — image DB（existing）+ text DB（new）
        "db_path":         config.get("database", "db_path"),
        "collection_name": config.get("database", "collection_name", fallback="page_screenshots"),
        "text_db_path":    config.get("database", "text_db_path", fallback="./db/chroma_text"),
        "text_collection": config.get("database", "text_collection", fallback="text_store"),

        # [data] section
        "image_folder": config.get("data", "image_folder",
                                   fallback=os.path.join(BASE_DIR, "data", "figures", "pages")),
        "text_data_path": config.get("data", "text_data_path", fallback=None) or None,

        # [embedding] section — text embedding（new）
        "embed_model":    config.get("embedding", "model", fallback="dengcao/Qwen3-Embedding-4B:Q4_K_M"), # bge-m3:latest
        "embed_base_url": config.get("embedding", "base_url", fallback="http://10.5.16.143:11434"),

        # [retrieval] section（new）
        "retrieval_mode": config.get("retrieval", "mode", fallback="image"),

        # [evaluation] section（可選）
        "testset":  config.get("evaluation", "testset", fallback=None),
        "top_k":    config.getint("evaluation", "top_k", fallback=3),

        # [generation] section（可選）
        "vlm_model":    config.get("generation", "vlm_model",    fallback="qwen3-vl:8b-instruct"),
        "vlm_num_ctx":  config.getint("generation", "vlm_num_ctx", fallback=65536),
        "vlm_base_url": config.get("generation", "vlm_base_url", fallback="http://10.5.16.143:11434"),
        "output_dir":   config.get("generation", "output_dir",   fallback="./results"),

        # [reranker] section（可選）
        "reranker_model":    config.get("reranker", "model",      fallback="Qwen/Qwen3-Reranker-2B"),
        "rerank_top_k":      config.getint("reranker", "top_k",   fallback=None),
    }

    # 把相對路徑轉成絕對路徑
    for key in ["db_path", "text_db_path", "image_folder", "text_data_path", "testset", "output_dir"]:
        val = settings[key]
        if val is not None and not os.path.isabs(val):
            settings[key] = os.path.join(BASE_DIR, val)

    # model_path 只在看起來像本地路徑時才轉（有 . 或 / 開頭）
    # HuggingFace Hub ID（如 "Qwen/Qwen3-VL-Embedding-2B"）不做轉換
    mp = settings["model_path"]
    if mp is not None and mp.startswith((".", "/")) and not os.path.isabs(mp):
        settings["model_path"] = os.path.join(BASE_DIR, mp)

    return settings