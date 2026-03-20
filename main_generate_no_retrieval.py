"""
main_generate_no_retrieval.py — VLM No-Retrieval Baseline

流程：
  1. 讀取 config.ini
  2. 初始化 VLM (ChatOllama)
  3. 載入測試集 (FAQ JSON)
  4. 對每筆 user_input：
     a. 構建純文字 prompt（不附圖片）
     b. ChatOllama (VLM) 生成答案
  5. 輸出 JSON

目的：
  測試 VLM 在「沒有 RAG 檢索結果」的情況下，
  能否僅靠模型本身的知識回答問題。
  作為 baseline 與 RAG 管線進行比較。

用法：
  python main_generate_no_retrieval.py --config configs/exp-qwen.ini
"""

import argparse
import json
import os
import time

from config import get_experiment_settings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


# ── helpers ──────────────────────────────────────────────


def load_testset(testset_path: str) -> list:
    """載入測試集 JSON，回傳 list of dict。"""
    with open(testset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📋 載入測試集: {len(data)} 筆 from {testset_path}")
    return data


def build_text_only_message(query: str) -> HumanMessage:
    """
    構建純文字的 HumanMessage，不包含任何圖片。
    直接把 query 傳給 VLM，不加額外 prompt。
    """
    return HumanMessage(content=query)


def generate_answer(llm: ChatOllama, query: str) -> str:
    """用 VLM 生成答案（純文字，無圖片）。"""
    msg = build_text_only_message(query)
    response = llm.invoke([msg])
    return response.content


# ── main ─────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="VLM No-Retrieval Baseline：不使用 RAG 檢索，直接用 VLM 回答問題"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="設定檔路徑，例如 configs/exp-qwen.ini"
    )
    args = parser.parse_args()

    # ── 1. 讀取設定檔 ──
    settings = get_experiment_settings(args.config)

    testset_path = settings["testset"]
    vlm_model    = settings["vlm_model"]
    vlm_num_ctx  = settings["vlm_num_ctx"]
    vlm_base_url = settings["vlm_base_url"]
    output_dir   = settings["output_dir"]

    print(f"\n{'='*60}")
    print(f"🔧 No-Retrieval Baseline 實驗設定:")
    print(f"  vlm_model:    {vlm_model}")
    print(f"  vlm_num_ctx:  {vlm_num_ctx}")
    print(f"  vlm_base_url: {vlm_base_url}")
    print(f"  testset:      {testset_path}")
    print(f"  output_dir:   {output_dir}")
    print(f"{'='*60}\n")

    if testset_path is None:
        raise ValueError("設定檔缺少 [evaluation] testset 欄位")

    # ── 2. 初始化 VLM（不需要 embedding model 和 ChromaDB）──
    print(f"🤖 初始化 VLM: {vlm_model} (num_ctx={vlm_num_ctx})")
    llm = ChatOllama(
        model=vlm_model,
        num_ctx=vlm_num_ctx,
        base_url=vlm_base_url,
    )

    # ── 3. 載入測試集 ──
    testset = load_testset(testset_path)

    # ── 4. 逐筆生成答案（無 retrieval）──
    all_results = []
    total = len(testset)
    start_time = time.time()

    for i, entry in enumerate(testset):
        query = entry["user_input"]

        # 直接問 VLM，不檢索、不給圖片
        try:
            vlm_answer = generate_answer(llm, query)
        except Exception as e:
            print(f"  ❌ VLM 生成失敗 (#{i+1}): {e}")
            vlm_answer = f"[ERROR] {str(e)}"

        # 組裝輸出（pages 為空，因為沒有 retrieval）
        result_entry = {
            "input": query,
            "ref": [entry.get("reference", "")],
            "page": entry.get("page", 0),
            "doc": entry.get("doc", ""),
            "pages": [],
            "vlm_answer": vlm_answer,
        }
        all_results.append(result_entry)

        # 進度顯示
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (total - i - 1)
        print(f"  [{i+1}/{total}] ⏱ {elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining")

    # ── 5. 儲存 JSON ──
    os.makedirs(output_dir, exist_ok=True)

    output_filename = "Vulcan_Training_no_retrieval_result.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    elapsed_total = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Done! Total {total} entries, elapsed {elapsed_total:.1f}s")
    print(f"Result saved at: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
