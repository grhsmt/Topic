import os
from pathlib import Path

import pandas as pd

def ensure_dir(file_path):
    """确保文件路径的父目录存在"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def save_lda_topics(lda, output_path, topn=20):
    """保存LDA主题词到CSV"""
    ensure_dir(output_path)
    topn = max(1, topn)
    records = []
    for topic_id in range(lda.num_topics):
        topic = lda.show_topic(topic_id, topn=topn)
        for rank, (word, weight) in enumerate(topic, start=1):
            records.append({
                "topic_id": topic_id,
                "rank": rank,
                "word": word,
                "weight": weight,
            })
    pd.DataFrame(records, columns=["topic_id", "rank", "word", "weight"]).to_csv(output_path, index=False, encoding="utf-8")