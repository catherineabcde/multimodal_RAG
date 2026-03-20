"""
src/retrieval.py — 共用檢索函式

提供三種檢索模式：
  - retrieve_pages : 圖片向量庫檢索
  - retrieve_texts : 文字向量庫檢索
  - retrieve_mix   : 混合檢索（文字 + 圖片）
"""

from langchain_community.vectorstores import Chroma


def retrieve_pages(db: Chroma, query: str, top_k: int) -> list:
    """
    從 ChromaDB 檢索 top-k 頁面（image mode）。
    回傳 list of dict: [{"page": int, "score": float, "source": str}, ...]
    """
    results = db.similarity_search_with_score(query, k=top_k)
    retrieved = []
    for doc, score in results:
        retrieved.append({
            "page": doc.metadata.get("page", 0),
            "score": float(score),
            "source": doc.metadata.get("source", ""),
        })
    return retrieved


def retrieve_texts(db_text: Chroma, query: str, top_k: int) -> list:
    """
    從文字 ChromaDB 檢索 top-k text chunks（text mode）。
    回傳 list of dict: [{"page": int, "score": float, "text": str}, ...]
    """
    results = db_text.similarity_search_with_score(query, k=top_k)
    retrieved = []
    for doc, score in results:
        retrieved.append({
            "page": doc.metadata.get("page", 0),
            "score": float(score),
            "text": doc.page_content,
        })
    return retrieved


def retrieve_mix(
    db_text: Chroma,
    db_image: Chroma,
    query: str,
    top_k: int,
) -> dict:
    """
    混合檢索：分別從文字 DB 和圖片 DB 檢索，合併結果。

    回傳 dict:
      {
        "text_results": [...],
        "image_results": [...],
        "text_context": str,
        "image_paths": [str, ...],
        "all_pages": [int, ...]
      }
    """
    text_results = retrieve_texts(db_text, query, top_k)
    image_results = retrieve_pages(db_image, query, top_k)

    all_pages = sorted(set(
        [r["page"] for r in text_results] +
        [r["page"] for r in image_results]
    ))

    text_context = "\n\n".join(
        f"[Page {r['page']}] {r['text']}" for r in text_results
    )

    image_paths = [r["source"] for r in image_results if r["source"]]

    return {
        "text_results": text_results,
        "image_results": image_results,
        "text_context": text_context,
        "image_paths": image_paths,
        "all_pages": all_pages,
    }
