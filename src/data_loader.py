import os
import re


def get_image_data(image_folder: str):
    """
    掃描 image_folder 中的所有 .png / .jpg 圖片，
    回傳 (image_uris, metadatas) 兩個 list。

    metadata 包含：
      - source: 圖片完整路徑
      - page:   從檔名 page_XX 提取的頁碼
      - type:   固定為 "page_screenshot"
    """
    image_uris = []
    metadatas = []

    if not os.path.isdir(image_folder):
        raise FileNotFoundError(f"圖片資料夾不存在: {image_folder}")

    for filename in sorted(os.listdir(image_folder)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            filepath = os.path.join(image_folder, filename)
            image_uris.append(filepath)

            match = re.search(r"page_(\d+)", filename)
            page_num = int(match.group(1)) if match else 0

            metadatas.append({
                "source": filepath,
                "page": page_num,
                "type": "page_screenshot"
            })

    print(f"📁 從 {image_folder} 找到 {len(image_uris)} 張圖片")
    return image_uris, metadatas