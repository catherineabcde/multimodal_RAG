"""
app.py — Multimodal RAG Demo (Gradio)

啟動方式：
  python app.py
  python app.py --share    # 產生公開連結
"""

import argparse
import glob
import os
import time

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont

from config import get_experiment_settings
from src.models import get_embedding_model
from src.retrieval import retrieve_pages, retrieve_texts, retrieve_mix
from main_generate import (
    generate_answer_image,
    generate_answer_text,
    generate_answer_mix,
)
from main_rerank import rerank_with_scores
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama


# ── 字型設定 ──────────────────────────────────────────────

# 如果有 VIA Sans 字型檔，放在專案根目錄的 fonts/ 資料夾下
# 例如: fonts/VIASans-Regular.ttf
_FONT_SEARCH_PATHS = [
    os.path.join(os.path.dirname(__file__), "fonts", "VIASans-Regular.ttf"),
    os.path.join(os.path.dirname(__file__), "fonts", "VIA Sans.ttf"),
    os.path.join(os.path.dirname(__file__), "fonts", "VIASans.ttf"),
]

_font_path = None
for p in _FONT_SEARCH_PATHS:
    if os.path.isfile(p):
        _font_path = p
        break


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """取得指定大小的字型，優先使用 VIA Sans。"""
    if _font_path:
        return ImageFont.truetype(_font_path, size)
    # Fallback: 嘗試系統字型
    for fallback in ["DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(fallback, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ── 全域資源（啟動時載入一次）─────────────────────────────

CONFIG_MAP = {
    "Qwen (image)": "configs/exp-qwen.ini",
    "CLIP (image)": "configs/exp-clip.ini",
    "Mix (image + text)": "configs/exp-mix.ini",
}


def load_testset_queries() -> list[str]:
    """從預設 config 的 testset 讀取所有查詢作為 Dropdown 選項。"""
    import json
    default_config = list(CONFIG_MAP.values())[0]
    settings = get_experiment_settings(default_config)
    testset_path = settings.get("testset")
    if not testset_path or not os.path.isfile(testset_path):
        return []
    with open(testset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [entry["user_input"] for entry in data if "user_input" in entry]


TESTSET_QUERIES = load_testset_queries()

_cache = {}


def get_resources(config_label: str):
    """根據 config 載入或取得快取的 DB / VLM / settings。"""
    if config_label in _cache:
        return _cache[config_label]

    config_path = CONFIG_MAP[config_label]
    settings = get_experiment_settings(config_path)
    retrieval_mode = settings.get("retrieval_mode", "image")

    db_image = None
    db_text = None

    if retrieval_mode in ("image", "mix"):
        model = get_embedding_model(settings["model_name"], settings["model_path"])
        db_image = Chroma(
            collection_name=settings["collection_name"],
            embedding_function=model,
            persist_directory=settings["db_path"],
        )

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

    llm = ChatOllama(
        model=settings["vlm_model"],
        num_ctx=settings["vlm_num_ctx"],
        base_url=settings["vlm_base_url"],
    )

    res = {
        "settings": settings,
        "db_image": db_image,
        "db_text": db_text,
        "llm": llm,
    }
    _cache[config_label] = res
    return res


_reranker = None


def get_reranker():
    """載入 reranker（lazy，第一次呼叫時才載入）。"""
    global _reranker
    if _reranker is None:
        from model.qwen3_vl_reranker import Qwen3VLReranker

        num_gpus = torch.cuda.device_count()
        device = "cuda:1" if num_gpus > 1 else "cuda:0"
        print(f"Loading Reranker on {device} (GPUs available: {num_gpus})")
        _reranker = Qwen3VLReranker(
            model_name_or_path="Qwen/Qwen3-VL-Reranker-2B",
            device=device,
        )
    return _reranker


# ── 工具函式 ──────────────────────────────────────────────


def find_page_image(image_folder: str, page: int) -> str | None:
    """根據 page number 找到對應的截圖路徑。"""
    pattern = os.path.join(image_folder, f"page_{page}_*.png")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def resolve_image_path(result: dict, image_folder: str) -> str | None:
    """從檢索結果中取得圖片路徑。"""
    img_path = result.get("source", "")
    if img_path and os.path.isfile(img_path):
        return img_path
    return find_page_image(image_folder, result["page"])


def format_score_lines(result: dict, prefix: str = "") -> list[str]:
    """格式化分數為多行文字。"""
    lines = []
    if prefix:
        lines.append(prefix)
    lines.append(f"Page {result['page']}")
    if "rerank_score" in result:
        lines.append(f"Rerank: {result['rerank_score']:.4f}")
        lines.append(f"Retrieval: {result['retrieval_score']:.4f}")
    elif "score" in result:
        lines.append(f"Score: {result.get('score', 0):.4f}")
    return lines


def annotate_image(img_path: str, lines: list[str]) -> Image.Image:
    """
    在圖片底部加上半透明黑色 bar + 白色文字標註。
    回傳新的 PIL Image。
    """
    img = Image.open(img_path).convert("RGBA")
    w, h = img.size

    font_size = max(14, w // 30)
    font = get_font(font_size)
    line_height = font_size + 4
    padding = 8
    bar_height = padding * 2 + line_height * len(lines)

    # 建立半透明黑色 overlay
    overlay = Image.new("RGBA", (w, bar_height), (0, 0, 0, 180))
    draw = ImageDraw.Draw(overlay)

    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=(255, 255, 255, 255), font=font)
        y += line_height

    # 合成到原圖底部
    result = Image.new("RGBA", (w, h + bar_height))
    result.paste(img, (0, 0))
    result.paste(overlay, (0, h))

    return result.convert("RGB")


# ── 核心推理函式 ──────────────────────────────────────────


def run_rag(query: str, config_label: str, top_k: int, use_rerank: bool, rerank_top_k: int):
    """
    執行 RAG pipeline，回傳 (gallery_images, answer, info)。
    """
    if not query.strip():
        return [], "Please enter a question.", ""

    try:
        res = get_resources(config_label)
        settings = res["settings"]
        llm = res["llm"]
        db_image = res["db_image"]
        db_text = res["db_text"]
        retrieval_mode = settings.get("retrieval_mode", "image")
        image_folder = settings["image_folder"]

        top_k = int(top_k)
        rerank_top_k = int(rerank_top_k)

        # ── Retrieval ──
        t0 = time.time()
        gallery_items = []
        answer = ""

        if retrieval_mode == "image":
            retrieved = retrieve_pages(db_image, query, top_k)

            if use_rerank:
                reranker = get_reranker()
                reranked = rerank_with_scores(reranker, query, retrieved, image_folder)
                display_results = reranked[:rerank_top_k]
            else:
                display_results = retrieved

            for r in display_results:
                img_path = resolve_image_path(r, image_folder)
                if img_path:
                    annotated = annotate_image(img_path, format_score_lines(r))
                    gallery_items.append(annotated)

            retrieval_time = time.time() - t0

            t1 = time.time()
            image_paths = [resolve_image_path(r, image_folder) for r in display_results]
            image_paths = [p for p in image_paths if p]
            answer = generate_answer_image(llm, query, image_paths)

        elif retrieval_mode == "text":
            text_results = retrieve_texts(db_text, query, top_k)

            for r in text_results:
                img_path = find_page_image(image_folder, r["page"])
                if img_path:
                    annotated = annotate_image(img_path, format_score_lines(r))
                    gallery_items.append(annotated)

            retrieval_time = time.time() - t0

            t1 = time.time()
            text_context = "\n\n".join(
                f"[Page {r['page']}] {r['text']}" for r in text_results
            )
            answer = generate_answer_text(llm, query, text_context)

        elif retrieval_mode == "mix":
            mix_data = retrieve_mix(db_text, db_image, query, top_k)

            for r in mix_data["image_results"]:
                img_path = resolve_image_path(r, image_folder)
                if img_path:
                    annotated = annotate_image(img_path, format_score_lines(r, prefix="[Image]"))
                    gallery_items.append(annotated)
            for r in mix_data["text_results"]:
                img_path = find_page_image(image_folder, r["page"])
                if img_path:
                    annotated = annotate_image(img_path, format_score_lines(r, prefix="[Text]"))
                    gallery_items.append(annotated)

            retrieval_time = time.time() - t0

            t1 = time.time()
            answer = generate_answer_mix(
                llm, query,
                mix_data["text_context"],
                mix_data["image_paths"],
            )

        gen_time = time.time() - t1

        info = (
            f"Mode: {retrieval_mode}"
            f"{' + rerank' if use_rerank else ''} | "
            f"Retrieval: {retrieval_time:.2f}s | "
            f"Generation: {gen_time:.2f}s"
        )

        return gallery_items, answer, info

    except Exception as e:
        import traceback
        error_msg = (
            f"**Error:** `{type(e).__name__}`\n\n"
            f"```\n{traceback.format_exc()}\n```"
        )
        return [], error_msg, ""


# ── Custom CSS ────────────────────────────────────────────

CUSTOM_CSS = """
@font-face {
    font-family: 'VIA Sans';
    src: url('/file=fonts/VIASans-Regular.ttf') format('truetype');
}
* {
    font-family: 'VIA Sans', 'Helvetica Neue', Arial, sans-serif !important;
}

/* 加上按鈕的專屬顏色設定 */
#custom-submit-btn {
    background-color: #FF5733 !important; /* 背景顏色：改成你要的色碼 */
    color: #FFFFFF !important;            /* 文字顏色：白色 */
    border: none !important;              /* 移除預設邊框 */
}

/* 選擇性：設定滑鼠移上去時的顏色變化 */
#custom-submit-btn:hover {
    background-color: #E04D2D !important; 
}
"""


# ── Gradio UI ─────────────────────────────────────────────


# ── Gradio UI ─────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="Multimodal RAG Demo", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
        # 1. 標題與說明文字置中 (使用 HTML 語法)
        gr.Markdown("<h1 style='text-align: center;'>Multimodal RAG Demo</h1>")
        gr.Markdown("<p style='text-align: center;'>Retrieve relevant pages from training documents and generate answers with VLM.</p>")

        with gr.Row():
            # 左側：設定
            with gr.Column(scale=1):
                config_dd = gr.Dropdown(
                    choices=list(CONFIG_MAP.keys()),
                    value="Qwen (image)",
                    label="Config",
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1,
                    label="Top K (retrieval)",
                )
                rerank_cb = gr.Checkbox(label="Enable Reranking", value=False)
                rerank_top_k_slider = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1,
                    label="Top K (after rerank)",
                    visible=False,
                )

                rerank_cb.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=rerank_cb,
                    outputs=rerank_top_k_slider,
                )

            # 右側：查詢
            with gr.Column(scale=3):
                # 2. 改用 Dropdown 並開啟 allow_custom_value=True，實現可選也可輸入
                query_input = gr.Dropdown(
                    choices=TESTSET_QUERIES,
                    label="Question",
                    value="",
                    allow_custom_value=True,
                )
                submit_btn = gr.Button("Submit", variant="primary")

        # 結果區
        with gr.Row():
            gallery = gr.Gallery(
                label="Retrieved Pages",
                columns=5,
                height="auto",
                object_fit="contain",
            )

        answer_box = gr.Textbox(
            label="Generated Answer",
            interactive=False,
            lines=8,
        )

        gr.Markdown("---")
        info_box = gr.Textbox(label="Info", interactive=False)

        # 綁定事件
        submit_btn.click(
            fn=run_rag,
            inputs=[query_input, config_dd, top_k_slider, rerank_cb, rerank_top_k_slider],
            outputs=[gallery, answer_box, info_box],
        )

    return demo


# ── 入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    args = parser.parse_args()

    demo = build_ui()
    
    # 取得 fonts 資料夾的絕對路徑
    font_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "fonts"))
    
    # 在 launch 加上 allowed_paths 參數
    demo.launch(
        server_name="0.0.0.0", 
        server_port=args.port, 
        share=True,
        allowed_paths=[font_dir] 
    )
