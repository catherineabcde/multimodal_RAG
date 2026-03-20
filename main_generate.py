"""
main_generate.py — RAG + VLM Answer Generation Pipeline

支援三種檢索模式（透過 config [retrieval] mode 切換）：
  image : 圖片檢索 → VLM 生成（原有）
  text  : 純文字檢索 → VLM 生成
  mix   : 文字 + 圖片 混合檢索 → VLM 生成

流程：
  1. 讀取 config.ini
  2. 載入 embedding model → 連接 ChromaDB
  3. 載入測試集 (FAQ JSON)
  4. 對每筆 user_input：
     a. 根據 retrieval_mode 檢索
     b. 構建 prompt (text / image / mix)
     c. ChatOllama (VLM) 生成答案
  5. 輸出 JSON

用法：
  python main_generate.py --config configs/exp-qwen.ini                   # image mode
  python main_generate.py --config configs/exp-mix.ini                    # mix mode
  python main_generate.py --config configs/exp-mix.ini --top_k 2
  python main_generate.py --config configs/exp-mix.ini --top_k 5 --rerank --reranker_model Qwen/Qwen3-VL-Reranker-2B --rerank_top_k 3
"""

import argparse
import base64
import json
import os
import time

from config import get_experiment_settings
from src.models import get_embedding_model
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ChatPromptTemplate
from src.retrieval import retrieve_pages, retrieve_texts, retrieve_mix
from main_rerank import rerank_with_scores
from model.qwen3_vl_reranker import Qwen3VLReranker
import torch


# ── helpers ──────────────────────────────────────────────


def load_testset(testset_path: str) -> list:
    """載入測試集 JSON，回傳 list of dict。"""
    with open(testset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📋 載入測試集: {len(data)} 筆 from {testset_path}")
    return data


def image_to_base64(image_path: str) -> str:
    """讀取圖片並轉成 base64 字串。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def output_format():
    attribution_prompt = ChatPromptTemplate.from_template(
        """
        For each fact or claim in your answer, iclude a citation using [1], [2], etc. that refers to the source. Include a numbered reference list at the end.
        Question: {query}
        Your anser:
        Sources:
        {sources}
        """
    )
# ── prompt building ──────────────────────────────────────


def build_vlm_message_image(query: str, image_paths: list) -> HumanMessage:
    """
    構建 image-only prompt：圖片 + 文字 query。
    """
    content = []

    for img_path in image_paths:
        if os.path.isfile(img_path):
            b64 = image_to_base64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        else:
            print(f"Image not found: {img_path}")

    content.append({
        "type": "text",
        "text": (
            "Based on the provided page image(s), please answer the following question. "
            "Answer in the same language as the question. Be concise and accurate.\n\n"
            f"Question: {query}"
        ),
    })

    return HumanMessage(content=content)


def build_vlm_message_text(query: str, text_context: str) -> HumanMessage:
    """
    構建 text-only prompt：文字 context + query。
    """
    content = [{
        "type": "text",
        "text": (
            "Based on the following context, please answer the question. "
            "Answer in the same language as the question. Be concise and accurate.\n\n"
            f"Context:\n{text_context}\n\n"
            f"Question: {query}"
        ),
    }]

    return HumanMessage(content=content)


def build_vlm_message_mix(
    query: str,
    text_context: str,
    image_paths: list,
) -> HumanMessage:
    """
    構建 mix prompt：文字 context + 圖片 + query。
    """
    content = []

    # 先加入圖片
    for img_path in image_paths:
        if os.path.isfile(img_path):
            b64 = image_to_base64(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        else:
            print(f"Image not found: {img_path}")

    # 加入文字 context + question
    content.append({
        "type": "text",
        "text": (
            "Based on the provided images and text context, please answer the following question. "
            "Answer in the same language as the question. Be concise and accurate.\n\n"
            f"Text Context:\n{text_context}\n\n"
            f"Question: {query}"
        ),
    })

    return HumanMessage(content=content)


# ── generation functions ─────────────────────────────────


def generate_answer_image(llm: ChatOllama, query: str, image_paths: list) -> str:
    """用 VLM 生成答案（image mode）。"""
    msg = build_vlm_message_image(query, image_paths)
    response = llm.invoke([msg])
    return response.content


def generate_answer_text(llm: ChatOllama, query: str, text_context: str) -> str:
    """用 VLM 生成答案（text mode）。"""
    msg = build_vlm_message_text(query, text_context)
    response = llm.invoke([msg])
    return response.content


def generate_answer_mix(
    llm: ChatOllama,
    query: str,
    text_context: str,
    image_paths: list,
) -> str:
    """用 VLM 生成答案（mix mode）。"""
    msg = build_vlm_message_mix(query, text_context, image_paths)
    response = llm.invoke([msg])
    return response.content


# ── main ─────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="RAG + VLM 答案生成：支援 image / text / mix 檢索模式")
    parser.add_argument("--config", type=str, required=True, help="設定檔路徑，例如 configs/exp-mix.ini")
    parser.add_argument("--top_k", type=int, default=None, help="覆蓋 config 中的 top_k 值（可選）")
    parser.add_argument("--rerank", action="store_true", help="activate reranker")
    parser.add_argument("--rerank_top_k", type=int, default=None, help="retrieve top_k after reranking for VLM（default= top_k）")
    parser.add_argument("--reranker_model", type=str, default=None, help="Reranker 模型名稱或路徑")  
    args = parser.parse_args()

    # Compute the time taken for the entire process
    process_start_time = time.time()

    # ── 1. 讀取設定檔 ──
    settings = get_experiment_settings(args.config)

    # CLI --top_k 覆蓋 config 值
    if args.top_k is not None:
        settings["top_k"] = args.top_k

    top_k = settings["top_k"]
    testset_path = settings["testset"]
    vlm_model = settings["vlm_model"]
    vlm_num_ctx = settings["vlm_num_ctx"]
    vlm_base_url = settings["vlm_base_url"]
    output_dir = settings["output_dir"]
    retrieval_mode = settings.get("retrieval_mode", "image")


    # Rerank settings
    use_rerank = args.rerank
    reranker_model = (
        args.reranker_model
        or settings.get("reranker")
        or "Qwen/Qwen3-VL-Reranker-2B"
    )
    rerank_top_k = (
        args.rerank_top_k
        or top_k
    )
    image_folder = settings.get("image_folder", "")

    print(f"\n{'='*60}")
    print(f"實驗設定:")
    for k, v in settings.items():
        print(f"  {k}: {v}")
    if use_rerank:
        print(f"  rerank: True")
        print(f"  reranker_model: {reranker_model}")
        print(f"  rerank_top_k: {rerank_top_k}")
    print(f"{'='*60}\n")

    if testset_path is None:
        raise ValueError("設定檔缺少 [evaluation] testset 欄位")

    # ── 2. 載入 embedding model + ChromaDB ──
    db_image = None
    db_text = None

    # Image DB（image / mix 模式需要）
    if retrieval_mode in ("image", "mix"):
        model = get_embedding_model(settings["model_name"], settings["model_path"])
        db_image = Chroma(
            collection_name=settings["collection_name"],
            embedding_function=model,
            persist_directory=settings["db_path"],
        )
        db_count = db_image._collection.count()
        print(f"Image DB: {settings['collection_name']} ({db_count} 筆)")

    # Text DB（text / mix 模式需要）
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
        db_text_count = db_text._collection.count()
        print(f"Text DB: {settings.get('text_collection', 'text_store')} ({db_text_count} 筆)")

    # ── 3. 初始化 VLM ──
    print(f"初始化 VLM: {vlm_model} (num_ctx={vlm_num_ctx})")
    llm = ChatOllama(
        model=vlm_model,
        num_ctx=vlm_num_ctx,
        base_url=vlm_base_url,
    )
    # ── 3.1 Loading Reranker ──
    reranker = None
    if use_rerank:
        print(f"Loading Reranker: {reranker_model} ...")
        nums_gpus = torch.cuda.device_count()
        reranker_device = "cuda:1" if nums_gpus > 1 else "cuda:0"
        reranker = Qwen3VLReranker(model_name_or_path=reranker_model, device=reranker_device)
        print(f"Reranker loaded completely!")

    # ── 4. 載入測試集 ──
    testset = load_testset(testset_path)

    # ── 5. 逐筆檢索 + 生成 ──
    all_results = []
    total = len(testset)
    start_time = time.time()

    for i, entry in enumerate(testset):
        query = entry["user_input"]
        gt_page = entry["page"]

        try:
            if retrieval_mode == "image":
                # ── Image-only retrieval ──
                retrieved = retrieve_pages(db_image, query, top_k)

                if use_rerank and reranker is not None:
                    # Rerank 後取 rerank_top_k 名
                    reranked = rerank_with_scores(reranker, query, retrieved, image_folder)
                    reranked_topk = reranked[:rerank_top_k]
                    image_paths = [r["source"] for r in reranked_topk if r["source"]]
                    pages_output = [{
                        "page": r["page"],
                        "retrieval_score": r["retrieval_score"],
                        "rerank_score": r["rerank_score"],
                    } for r in reranked_topk]
                else:
                    image_paths = [r["source"] for r in retrieved if r["source"]]
                    pages_output = [{"page": r["page"], "retrieval_score": r["retrieval_score"]} for r in retrieved]
                
                vlm_answer = generate_answer_image(llm, query, image_paths)
                result_entry = {
                    "input": query,
                    "ref": [entry.get("reference", "")],
                    "page": gt_page,
                    "doc": entry.get("doc", ""),
                    "retrieval_mode": "image",
                    "rerank": use_rerank,
                    "pages": pages_output,
                    "vlm_answer": vlm_answer,
                }

            elif retrieval_mode == "text":
                # ── Text-only retrieval ──
                text_results = retrieve_texts(db_text, query, top_k)
                text_context = "\n\n".join(
                    f"[Page {r['page']}] {r['text']}" for r in text_results
                )
                vlm_answer = generate_answer_text(llm, query, text_context)
                pages_output = [{"page": r["page"], "score": r["score"]} for r in text_results]
                result_entry = {
                    "input": query,
                    "ref": [entry.get("reference", "")],
                    "page": gt_page,
                    "doc": entry.get("doc", ""),
                    "retrieval_mode": "text",
                    "pages": pages_output,
                    "vlm_answer": vlm_answer,
                }

            elif retrieval_mode == "mix":
                # ── Mixture retrieval ──
                mix_data = retrieve_mix(
                    db_text, db_image, query, top_k,
                )
                vlm_answer = generate_answer_mix(
                    llm, query,
                    mix_data["text_context"],
                    mix_data["image_paths"],
                )
                text_pages = [{"page": r["page"], "score": r["score"], "type": "text"}
                              for r in mix_data["text_results"]]
                image_pages = [{"page": r["page"], "score": r["score"], "type": "image"}
                               for r in mix_data["image_results"]]
                result_entry = {
                    "input": query,
                    "ref": [entry.get("reference", "")],
                    "page": gt_page,
                    "doc": entry.get("doc", ""),
                    "retrieval_mode": "mix",
                    "text_pages": text_pages,
                    "image_pages": image_pages,
                    "all_pages": mix_data["all_pages"],
                    "vlm_answer": vlm_answer,
                }

            else:
                raise ValueError(f"不支援的 retrieval_mode: {retrieval_mode}")

        except Exception as e:
            print(f"生成失敗 (#{i+1}): {e}")
            result_entry = {
                "input": query,
                "ref": [entry.get("reference", "")],
                "page": gt_page,
                "doc": entry.get("doc", ""),
                "retrieval_mode": retrieval_mode,
                "vlm_answer": f"[ERROR] {str(e)}",
            }

        all_results.append(result_entry)

        # 進度顯示
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (total - i - 1)
        print(f"  [{i+1}/{total}] ⏱ {elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining")

        process_end_time = time.time()
        print(f"  Total elapsed for current entry: {process_end_time - process_start_time:.1f}s")

    # ── 6. 儲存 JSON ──
    os.makedirs(output_dir, exist_ok=True)

    model_name = settings["model_name"]
    if use_rerank:
        reranker_short = os.path.basename(reranker_model).replace("/", "-")
        output_filename = (
            f"Vulcan_Training_{model_name}_topk{top_k}"
            f"_{retrieval_mode}"
            f"_rerank-{reranker_short}_rk{rerank_top_k}"
            f"_generate_result.json"
        )
    else:
        output_filename = f"Vulcan_Training_{model_name}_topk{top_k}_{retrieval_mode}_generate_result.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    elapsed_total = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done! Total {total} entries, elapsed {elapsed_total:.1f}s")
    print(f"Retrieval mode: {retrieval_mode}")
    print(f"Result saved at: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
