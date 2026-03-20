import json
import uuid
from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ImageVectorStoreBuilder:
    """
    用 LangChain 的 Chroma wrapper 建立圖片向量資料庫。

    用法：
        builder = ImageVectorStoreBuilder(
            embedding_model=model,       # LangChain Embeddings 物件
            persist_dir="./db/chroma_clip",
            collection_name="clip_images"
        )
        builder.add_images_to_db(image_uris, metadatas)
    """

    def __init__(self, embedding_model, persist_dir: str, collection_name: str = "page_screenshots"):
        self.embed_model = embedding_model
        self.persist_dir = persist_dir

        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embed_model,
            persist_directory=self.persist_dir,
        )

    def add_images_to_db(self, image_uris: List[str], metadatas: List[Dict]):
        """將圖片 embed 後存入 Chroma 向量資料庫。"""
        uuids = [str(uuid.uuid4()) for _ in range(len(image_uris))]
        self.db.add_images(uris=image_uris, metadatas=metadatas, ids=uuids)
        print(f"✅ 成功寫入 {len(image_uris)} 張圖片到 {self.persist_dir}")


class TextVectorStoreBuilder:
    """
    從 _merged.json 讀取頁面文字，chunk 後存入 ChromaDB。

    用法：
        from langchain_community.embeddings import OllamaEmbeddings
        embed = OllamaEmbeddings(model="dengcao/Qwen3-Embedding-4B:Q4_K_M", base_url="http://...") # bge-m3:latest
        builder = TextVectorStoreBuilder(
            embedding_model=embed,
            persist_dir="./db/chroma_text",
            collection_name="text_store"
        )
        builder.add_text_from_merged_json("./data/Vulcan_merged.json")
    """

    def __init__(self, embedding_model, persist_dir: str, collection_name: str = "text_store"):
        self.embed_model = embedding_model
        self.persist_dir = persist_dir

        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embed_model,
            persist_directory=self.persist_dir,
        )

    def add_text_from_merged_json(
        self,
        merged_json_path: str,
        chunk_size: int = 400,
        chunk_overlap: int = 40,
    ):
        """
        讀取 _merged.json → 按頁面拆 chunk → 寫入 ChromaDB。

        _merged.json 結構：
        {
          "content": {
            "pages": [
              {"page": 1, "text": "..."},
              {"page": 2, "text": "..."},
              ...
            ]
          }
        }
        """
        with open(merged_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pages = data.get("content", {}).get("pages", [])
        if not pages:
            raise ValueError(f"在 {merged_json_path} 中找不到 content.pages 資料")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks = []
        all_metadatas = []

        for page_data in pages:
            page_num = page_data["page"]
            text = page_data.get("text", "").strip()
            if not text:
                continue

            chunks = splitter.split_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadatas.append({
                    "page": page_num,
                    "type": "text",
                })

        if not all_chunks:
            print("⚠️  沒有產生任何 text chunk")
            return

        uuids = [str(uuid.uuid4()) for _ in range(len(all_chunks))]
        self.db.add_texts(texts=all_chunks, metadatas=all_metadatas, ids=uuids)
        print(f"✅ 成功寫入 {len(all_chunks)} 個 text chunks（來自 {len(pages)} 頁）到 {self.persist_dir}")
