"""
main_build_db.py — 建立向量資料庫

支援三種模式 (--mode):
  image  : 建立圖片向量資料庫（預設，使用 CLIP / Qwen3-VL-Embedding）
  text   : 建立文字向量資料庫（從 _merged.json chunk 後寫入）
  all    : 同時建立圖片 + 文字資料庫

用法：
  python main_build_db.py --config configs/exp-qwen.ini                  # image mode
  python main_build_db.py --config configs/exp-mix.ini --mode text       # text only
  python main_build_db.py --config configs/exp-mix.ini --mode all        # both
"""

import argparse
from config import get_experiment_settings
from src.data_loader import get_image_data
from src.vector_store import ImageVectorStoreBuilder, TextVectorStoreBuilder
from src.models import get_embedding_model


def build_image_db(settings):
    """建立圖片向量資料庫（existing）"""
    model = get_embedding_model(settings["model_name"], settings["model_path"])
    image_uris, metadatas = get_image_data(settings["image_folder"])

    print(f"📦 建立圖片向量資料庫: {settings['collection_name']}")
    builder = ImageVectorStoreBuilder(
        embedding_model=model,
        persist_dir=settings["db_path"],
        collection_name=settings["collection_name"],
    )
    builder.add_images_to_db(image_uris, metadatas)


def build_text_db(settings):
    """建立文字向量資料庫（new）"""
    from langchain_community.embeddings import OllamaEmbeddings

    text_data_path = settings.get("text_data_path")
    if not text_data_path:
        raise ValueError("設定檔缺少 [data] text_data_path 欄位（_merged.json 路徑）")

    embed_model = settings.get("embed_model", "dengcao/Qwen3-Embedding-4B:Q4_K_M") # bge-m3:latest
    embed_base_url = settings.get("embed_base_url", "http://10.5.16.143:11434")

    print(f"📖 載入 text embedding model: {embed_model} @ {embed_base_url}")
    text_embeddings = OllamaEmbeddings(model=embed_model, base_url=embed_base_url)

    text_db_path = settings.get("text_db_path", "./db/chroma_text")
    text_collection = settings.get("text_collection", "text_store")

    print(f"📦 建立文字向量資料庫: {text_collection} → {text_db_path}")
    builder = TextVectorStoreBuilder(
        embedding_model=text_embeddings,
        persist_dir=text_db_path,
        collection_name=text_collection,
    )
    builder.add_text_from_merged_json(text_data_path)


def main():
    parser = argparse.ArgumentParser(
        description="建立向量資料庫（支援 image / text / all 模式）"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="設定檔路徑，例如 configs/exp-mix.ini",
    )
    parser.add_argument(
        "--mode", type=str, default="image",
        choices=["image", "text", "all"],
        help="建置模式：image（圖片 DB）、text（文字 DB）、all（兩者都建）",
    )
    args = parser.parse_args()

    # 1. 讀取設定檔
    settings = get_experiment_settings(args.config)
    print(f"\n{'='*50}")
    print(f"實驗設定:")
    for k, v in settings.items():
        print(f"  {k}: {v}")
    print(f"{'='*50}\n")

    # 2. 根據 mode 建置
    if args.mode in ("image", "all"):
        build_image_db(settings)

    if args.mode in ("text", "all"):
        build_text_db(settings)

    print(f"\n✅ 建置完成 (mode={args.mode})")


if __name__ == "__main__":
    main()
