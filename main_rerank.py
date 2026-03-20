"""
main_rerank.py — Two-Stage Retrieval Pipeline: Retrieve + Rerank

流程：
  1. 讀取 config.ini
  2. 載入 embedding model → 連接 Image ChromaDB
  3. 載入 Qwen3VLReranker 模型
  4. 載入測試集 (FAQ JSON)
  5. 對每筆 user_input：
     a. retrieve_pages() 做初步 image 檢索取 top-k 候選
     b. Qwen3VLReranker 對候選圖片重新評分
     c. 依 reranker 分數重新排序
  6. 計算 rerank 後的 Precision / Hit Rate 指標
  7. 輸出 JSON

用法：
  python main_rerank.py --config configs/exp-qwen.ini
  python main_rerank.py --config configs/exp-qwen.ini --top_k 5
  python main_rerank.py --config configs/exp-qwen.ini --rerank_top_k 3
"""
import argparse
import json
import os
import time

import torch
from config import get_experiment_settings
from src.models import get_embedding_model
from src.retrieval import retrieve_pages
from src.metrics import compute_metrics
from langchain_community.vectorstores import Chroma
from model.qwen3_vl_reranker import Qwen3VLReranker


# ── helpers ──────────────────────────────────────────────


def load_testset(testset_path: str) -> list:
    """載入測試集 JSON，回傳 list of dict。"""
    with open(testset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loading testset: {len(data)} samples from {testset_path}")
    return data


def rerank_with_scores(
    reranker: Qwen3VLReranker,
    query: str,
    candidates: list,
    image_folder: str,
) -> list:
    """
    用 Qwen3VLReranker 對候選頁面圖片做 reranking。

    Args:
        reranker:     Qwen3VLReranker 實例
        query:        使用者查詢文字
        candidates:   retrieve_pages() 回傳的 list of dict
                      [{"page": int, "score": float, "source": str}, ...]
        image_folder: 圖片資料夾路徑

    Returns:
        reranked list of dict，依 reranker 分數由高到低排序：
        [{"page": int, "retrieval_score": float, "rerank_score": float, "source": str}, ...]
    """
    # 組裝 reranker 需要的 document list
    documents = []
    valid_candidates = []

    for cand in candidates:
        img_path = cand.get("source", "")
        # 若 source 是空的，嘗試從 image_folder 用 page number 組路徑
        if not img_path or not os.path.isfile(img_path):
            img_path = os.path.join(image_folder, f"page_{cand['page']}.png")

        if os.path.isfile(img_path):
            documents.append({"text": None, "image": img_path})
            valid_candidates.append(cand)
        else:
            print(f"找不到圖片: {img_path}，跳過 page {cand['page']}")

    if not documents:
        return []

    # 呼叫 reranker
    scores = reranker.process({
        "query": {"text": query, "image": None},
        "documents": documents,
    })

    # 合併結果
    reranked = []
    for cand, rerank_score in zip(valid_candidates, scores):
        reranked.append({
            "page": cand["page"],
            "retrieval_score": cand["score"],
            "rerank_score": float(rerank_score),
            "source": cand.get("source", ""),
        })

    # 依 rerank_score 由高到低排序
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked


# ── main ─────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Retrieval: Image Retrieve + Qwen3VL Rerank")
    parser.add_argument("--config", type=str, required=True, help="設定檔路徑，例如 configs/exp-qwen.ini")
    parser.add_argument("--top_k", type=int, default=None, help="初步檢索 top-k（覆蓋 config 值）")
    parser.add_argument("--rerank_top_k", type=int, default=None, help="Rerank 後取前幾名來計算指標（預設 = top_k）")
    parser.add_argument("--reranker_model", type=str, default=None, help="Reranker 模型名稱或路徑（覆蓋 config 值）")
    args = parser.parse_args()

    # ── 1. 讀取設定檔 ──
    settings = get_experiment_settings(args.config)

    if args.top_k is not None:
        settings["top_k"] = args.top_k

    model_name = settings["model_name"]
    top_k = settings["top_k"]
    testset_path = settings["testset"]
    image_folder = settings["image_folder"]
    output_dir = settings.get("output_dir", "./results")

    # Reranker 設定
    reranker_model = (
        args.reranker_model
        or settings.get("reranker")
        or "Qwen/Qwen3-VL-Reranker-2B"
    )
    rerank_top_k = args.rerank_top_k or top_k

    print(f"\n{'='*60}")
    print(f"實驗設定:")
    for k, v in settings.items():
        print(f"  {k}: {v}")
    print(f"  reranker_model: {reranker_model}")
    print(f"  rerank_top_k: {rerank_top_k}")
    print(f"{'='*60}\n")

    if testset_path is None:
        raise ValueError("設定檔缺少 [evaluation] testset 欄位")

    # ── 2. 載入 embedding model + Image ChromaDB ──
    model = get_embedding_model(settings["model_name"], settings["model_path"])
    db_image = Chroma(
        collection_name=settings["collection_name"],
        embedding_function=model,
        persist_directory=settings["db_path"],
    )
    print(f"Image DB: {settings['collection_name']} "
          f"({db_image._collection.count()} 筆)")

    # ── 3. 載入 Reranker ──
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: ", num_gpus)
    reranker_device = "cuda:1" if num_gpus > 1 else "cuda:0"
    print(f"Loading Reranker: {reranker_model} (device: {reranker_device}, 可用 GPU: {num_gpus})")
    reranker = Qwen3VLReranker(model_name_or_path=reranker_model, device=reranker_device)
    print(f"Reranker loaded")

    # ── 4. 載入測試集 ──
    testset = load_testset(testset_path)

    # ── 5. 逐筆：檢索 → Rerank ──
    all_results = []
    total = len(testset)
    start_time = time.time()

    for i, entry in enumerate(testset):
        query = entry["user_input"]
        gt_page = entry["page"]

        try:
            # Stage 1: Retrieve
            retrieved = retrieve_pages(db_image, query, top_k)

            # Stage 2: Rerank
            reranked = rerank_with_scores(reranker, query, retrieved, image_folder)

            # 取 rerank_top_k 名
            reranked_topk = reranked[:rerank_top_k]

            pages_output = [{
                "page": r["page"],
                "retrieval_score": r["retrieval_score"],
                "rerank_score": r["rerank_score"],
            } for r in reranked_topk]

            result_entry = {
                "input": query,
                "ref": [entry.get("reference", "")],
                "page": gt_page,
                "doc": entry.get("doc", ""),
                "pages": pages_output,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ❌ 失敗 (#{i+1}): {e}")
            result_entry = {
                "input": query,
                "ref": [entry.get("reference", "")],
                "page": gt_page,
                "doc": entry.get("doc", ""),
                "pages": [],
            }
            reranked_topk = []

        all_results.append({
            **result_entry,
            "ground_truth_page": gt_page,
            "retrieved": [{"page": r["page"], "score": r["rerank_score"]}
                          for r in reranked_topk],
        })

        # 進度顯示
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (total - i - 1)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] ⏱ {elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining")

    # ── 6. 計算指標 ──
    metrics = compute_metrics(all_results)

    print(f"\n{'='*60}")
    print(f" Rerank 評估結果 — {model_name.upper()}")
    print(f"   Reranker: {reranker_model}")
    print(f"   Retrieve top_k={top_k} → Rerank top_k={rerank_top_k}")
    print(f"{'='*60}")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f} ({value*100:.2f}%)")
    print(f"{'='*60}\n")

    # ── 7. 輸出 JSON ──
    os.makedirs(output_dir, exist_ok=True)

    output_data = []
    for r in all_results:
        output_data.append({
            "input": r["input"],
            "ref": r["ref"],
            "page": r["page"],
            "doc": r["doc"],
            "pages": r["pages"],
        })

    reranker_short = os.path.basename(reranker_model)
    output_filename = (
        f"Vulcan_Training_{model_name}_topk{top_k}"
        f"_rerank_{reranker_short}_rerank_k{rerank_top_k}_result.json"
    )
    output_path = os.path.join(output_dir, output_filename)

    output_wrapper = {
        "model": model_name,
        "reranker": reranker_model,
        "metrics": metrics,
        "total_queries": len(testset),
        "retrieve_top_k": top_k,
        "rerank_top_k": rerank_top_k,
        "results": output_data,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_wrapper, f, ensure_ascii=False, indent=2)

    elapsed_total = time.time() - start_time
    print(f"Done! Total {total} entries, elapsed {elapsed_total:.1f}s")
    print(f" 結果已儲存: {output_path}")


if __name__ == "__main__":
    main()