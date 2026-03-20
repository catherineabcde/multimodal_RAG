"""
src/metrics.py — 共用評估指標函式
"""


def compute_metrics(results: list) -> dict:
    """
    根據每筆 query 的檢索結果計算 Precision 和 Hit Rate。

    每筆 result 需包含:
      - ground_truth_page: int
      - retrieved:         list of {"page": int, "score": float}
    """
    n = len(results)
    if n == 0:
        return {}

    max_k = max(len(r["retrieved"]) for r in results)
    metrics = {}

    for k in [1, min(3, max_k)]:
        hits = 0        # Hit Rate: 至少一個命中
        precision_sum = 0.0  # Precision: 命中數量 / k

        for r in results:
            gt_page = r["ground_truth_page"]
            top_k_pages = [item["page"] for item in r["retrieved"][:k]]

            match_count = top_k_pages.count(gt_page)
            if match_count > 0:
                hits += 1
            precision_sum += match_count / k

        metrics[f"hit_rate@{k}"] = hits / n
        metrics[f"precision@{k}"] = precision_sum / n

    return metrics
