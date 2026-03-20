"""
main_retrieve.py — 多模式檢索評估 Pipeline

支援三種檢索模式（透過 config [retrieval] mode 切換）：
  image : 圖片向量庫檢索
  text  : 純文字向量庫檢索
  mix   : 文字 + 圖片 混合檢索

流程：
  1. 讀取 config.ini
  2. 根據 retrieval_mode 載入對應的 embedding model → 連接 ChromaDB
  3. 載入測試集 (FAQ JSON)
  4. 對每筆 user_input 做檢索
  5. 計算 Precision / Hit Rate 指標
  6. 輸出 JSON

用法：
  python main_retrieve.py --config configs/exp-clip.ini     # image mode
  python main_retrieve.py --config configs/exp-qwen.ini     # image mode
  python main_retrieve.py --config configs/exp-mix.ini      # mix mode
  python main_retrieve.py --config configs/exp-mix.ini --top_k 5
"""

import argparse
import json
import os

from config import get_experiment_settings
from src.models import get_embedding_model
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from src.retrieval import retrieve_pages, retrieve_texts, retrieve_mix


# ── helpers ──────────────────────────────────────────────


def load_testset(testset_path: str) -> list:
    """載入測試集 JSON，回傳 list of dict。"""
    with open(testset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📋 載入測試集: {len(data)} 筆 from {testset_path}")
    return data
from src.metrics import compute_metrics


# ── main ─────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="用測試集查詢 ChromaDB，評估 Embedding Model 檢索效能（支援 image / text / mix）"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="設定檔路徑，例如 configs/exp-clip.ini"
    )
    parser.add_argument(
        "--top_k", type=int, default=None,
        help="覆蓋 config 中的 top_k 值（可選）"
    )
    args = parser.parse_args()

    # ── 1. 讀取設定檔 ──
    settings = get_experiment_settings(args.config)

    # CLI --top_k 覆蓋 config 值
    if args.top_k is not None:
        settings["top_k"] = args.top_k

    model_name = settings["model_name"]
    top_k = settings["top_k"]
    testset_path = settings["testset"]
    retrieval_mode = settings.get("retrieval_mode", "image")

    print(f"\n{'='*50}")
    print(f"實驗設定:")
    for k, v in settings.items():
        print(f"  {k}: {v}")
    print(f"{'='*50}\n")

    if testset_path is None:
        raise ValueError("設定檔缺少 [evaluation] testset 欄位")

    # ── 2. 根據 retrieval_mode 載入 embedding model + ChromaDB ──
    db_image = None
    db_text = None

    # Image DB（image / mix 模式需要）→ 用 [model] section
    if retrieval_mode in ("image", "mix"):
        model = get_embedding_model(settings["model_name"], settings["model_path"])
        db_image = Chroma(
            collection_name=settings["collection_name"],
            embedding_function=model,
            persist_directory=settings["db_path"],
        )
        print(f"📦 Image DB: {settings['collection_name']} "
              f"({db_image._collection.count()} 筆)")

    # Text DB（text / mix 模式需要）→ 用 [embedding] section
    if retrieval_mode in ("text", "mix"):
        text_embeddings = OllamaEmbeddings(
            model=settings["embed_model"],
            base_url=settings["embed_base_url"],
        )
        db_text = Chroma(
            collection_name=settings.get("text_collection", "text_store"),
            embedding_function=text_embeddings,
            persist_directory=settings.get("text_db_path", "./db/chroma_text"),
        )
        print(f"📖 Text DB: {settings.get('text_collection', 'text_store')} "
              f"({db_text._collection.count()} 筆)")

    # ── 3. 載入測試集 ──
    testset = load_testset(testset_path)

    # ── 4. 逐筆檢索 ──
    all_results = []
    for i, entry in enumerate(testset):
        query = entry["user_input"]
        gt_page = entry["page"]

        if retrieval_mode == "image":
            # ── Image-only retrieval ──
            retrieved = retrieve_pages(db_image, query, top_k)
            pages_output = [{"page": r["page"], "score": r["score"]} for r in retrieved]

        elif retrieval_mode == "text":
            # ── Text-only retrieval ──
            retrieved = retrieve_texts(db_text, query, top_k)
            pages_output = [{"page": r["page"], "score": r["score"]} for r in retrieved]

        elif retrieval_mode == "mix":
            # ── Mixture retrieval ──
            mix_data = retrieve_mix(db_text, db_image, query, top_k)
            text_pages = [{"page": r["page"], "score": r["score"], "type": "text"}
                          for r in mix_data["text_results"]]
            image_pages = [{"page": r["page"], "score": r["score"], "type": "image"}
                           for r in mix_data["image_results"]]
            pages_output = text_pages + image_pages

        else:
            raise ValueError(f"不支援的 retrieval_mode: {retrieval_mode}")

        result_entry = {
            "input": query,
            "ref": [entry.get("reference", "")],
            "page": gt_page,
            "doc": entry.get("doc", ""),
            "retrieval_mode": retrieval_mode,
            "pages": pages_output,
        }
        all_results.append({
            **result_entry,
            "ground_truth_page": gt_page,
            "retrieved": pages_output,  # 給 compute_metrics 用
        })

        # 進度顯示
        if (i + 1) % 50 == 0 or (i + 1) == len(testset):
            print(f"  進度: {i+1}/{len(testset)}")

    # ── 5. 計算指標 ──
    metrics = compute_metrics(all_results)

    print(f"\n{'='*50}")
    print(f"📊 評估結果 — {model_name.upper()} [{retrieval_mode}]")
    print(f"{'='*50}")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f} ({value*100:.2f}%)")
    print(f"{'='*50}\n")

    # ── 6. 輸出結果 JSON ──
    output_data = []
    for r in all_results:
        output_entry = {
            "input": r["input"],
            "ref": r["ref"],
            "page": r["page"],
            "doc": r["doc"],
            "retrieval_mode": r["retrieval_mode"],
            "pages": r["pages"],
        }
        output_data.append(output_entry)

    output_wrapper = {
        "model": model_name,
        "retrieval_mode": retrieval_mode,
        "metrics": metrics,
        "total_queries": len(testset),
        "top_k": top_k,
        "results": output_data,
    }

    output_filename = f"Vulcan_Training_{model_name}_{retrieval_mode}_retrieval_result.json"
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_wrapper, f, ensure_ascii=False, indent=2)

    print(f"💾 結果已儲存: {output_path}")


if __name__ == "__main__":
    main()