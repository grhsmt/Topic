from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import random
import logging
import numpy as np

def compute_lda_metrics(
    texts,
    topic_range,
    passes=10,
    no_below=15,
    no_above=0.6,
    chunksize=2000,
    update_every=1,
    iterations=50,
    eval_every=None,
    coherence_texts_size=30000,
    random_state=42
):
    """
    针对 ~20,000 条短文本的稳定版 LDA 主题数选择
    """

    # 固定随机种子，保证可复现
    random.seed(random_state)

    # 构建词典
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # ⭐ 抽样 texts 计算 coherence
    if len(texts) > coherence_texts_size:
        coherence_texts = random.sample(texts, coherence_texts_size)
    else:
        coherence_texts = texts

    results = []

    for k in topic_range:
        logging.info(f"----------------------------------------------------------Training LDA with {k} topics...-----------------------------------------------------------------------------")

        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            chunksize=chunksize,
            update_every=update_every,
            passes=passes,
            iterations=iterations,
            eval_every=eval_every,
            alpha="auto", # 50/k
            eta=0.01,
            random_state=random_state
        )
        log_perplexity = lda.log_perplexity(corpus)
        perplexity = np.exp(-log_perplexity)

        coherence_model = CoherenceModel(
            model=lda,
            texts=coherence_texts,
            dictionary=dictionary,
            coherence='c_v',
            topn=10,
            processes=1
        )

        coherence = coherence_model.get_coherence()

        results.append({
            "num_topics": k,
            "perplexity": perplexity,
            "coherence": coherence
        })

    return results

def select_best_topic(metrics, weight_coherence=0.6, weight_perplexity=0.4):
    """根据 perplexity 与 coherence 综合评分选择最优主题数"""
    if not metrics:
        raise ValueError("metrics list is empty")

    weight_sum = weight_coherence + weight_perplexity
    if weight_sum == 0:
        raise ValueError("weight_coherence and weight_perplexity cannot both be zero")

    # 归一化 coherence（值越大越好）
    coherences = np.array([m["coherence"] for m in metrics], dtype=float)
    coh_min, coh_max = coherences.min(), coherences.max()
    if np.isclose(coh_max, coh_min):
        norm_coherence = np.ones_like(coherences)
    else:
        norm_coherence = (coherences - coh_min) / (coh_max - coh_min)

    # 归一化 perplexity（值越小越好，取反向分数）
    perplexities = np.array([m["perplexity"] for m in metrics], dtype=float)
    perp_min, perp_max = perplexities.min(), perplexities.max()
    if np.isclose(perp_max, perp_min):
        norm_perplexity = np.ones_like(perplexities)
    else:
        norm_perplexity = (perp_max - perplexities) / (perp_max - perp_min)

    norm_coherence *= weight_coherence / weight_sum
    norm_perplexity *= weight_perplexity / weight_sum
    scores = norm_coherence + norm_perplexity

    best_index = int(scores.argmax())
    return metrics[best_index]["num_topics"]