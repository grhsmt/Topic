from src import config
from src.clustering import kmeans_cluster, birch_cluster, spectral_cluster, hierarchical_cluster
from src.data_loader import load_data
from src.lda_model import train_lda
from src.lda_selection import compute_lda_metrics, select_best_topic
from src.plot_utils import plot_kmeans_cluster_counts, plot_lda_metrics, plot_cluster_time_span, plot_cluster_hot_scores
from src.preprocess_raw_csv import standardize_csv
from src.stopwords import load_stopwords, remove_stopwords
from src.tfidf_model import build_tfidf
from src.tokenizer import tokenize
from src.utils import save_lda_topics
from src.visualization import *
from collections import Counter, defaultdict
import logging
import pandas as pd
import os
from os import system
from logging.handlers import RotatingFileHandler
import multiprocessing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('simulation.log', mode='w', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)

import argparse
import numpy as np
from scipy.sparse import issparse

p = argparse.ArgumentParser()
p.add_argument("-l", "--language", default="English", choices=["English", "Chinese"], help="Language of the text data")
p.add_argument("--cluster_method", default="kmeans", choices=["kmeans", "birch", "spectral"], help="Clustering method to use")
p.add_argument("--LDA", type=bool, default=False, help="Whether to perform LDA topic modeling")
p.add_argument("--n_clusters", type=int, default=config.NUM_CLUSTERS, help="Number of clusters for methods that need it")
# Birch params
p.add_argument("--birch_threshold", type=float, default=0.5, help="threshold for Birch clustering")
# Spectral params
p.add_argument("--spectral_affinity", type=str, default='rbf', help="affinity for SpectralClustering: rbf|nearest_neighbors|precomputed")
args = p.parse_args()
lang = args.language
cluster_method = args.cluster_method

if __name__ == "__main__":
    # 1. 读取数据
    if lang == "Chinese":
        input_path = config.RAW_DATA_PATH + "/20251112-20251209.csv"
        comment_col = "微博正文"
        time_col = "发布时间"
        attitudes_col = "点赞数"
        platform_name = "weibo"
    if lang == "English":
        input_path = config.RAW_DATA_PATH + "/english_youtube_processed.csv"
        comment_col = "名称"
        time_col = "发布时间"
        attitudes_col = "likes"
        platform_name = "youtube"

    output_path = config.RAW_DATA_PATH + f"/comments_{lang}.csv"

    # 1.1 整理数据
    standardize_csv(input_csv=input_path,
        output_csv=output_path,
        comment_col=comment_col,
        time_col=time_col,
        attitudes_col=attitudes_col,
        platform_name=platform_name
    )
    df = load_data(output_path)
    logging.info(f"Loaded {len(df)} comments for language: {lang}") # 26247

    # 2. 分词
    df['tokens'] = df['comment'].apply(tokenize)
    # output_path = config.CLEAN_DATA_PATH + f"/tokenized_comments_{lang}.csv"
    # df.to_csv(output_path, index=False, encoding="utf-8")
    token_sizes = df['tokens'].apply(len)
    logging.info(token_sizes.describe())
    logging.info("Max tokens: %d | Min tokens: %d", token_sizes.max(), token_sizes.min())
    """
        count    26247.000000
        mean       171.741266
        std        271.100827
        min          0.000000
        25%         29.000000
        50%         70.000000
        75%        174.000000
        max       2491.000000
        Name: tokens, dtype: float64
    """

    # 3. 停用词
    stopwords = load_stopwords(
        "stopwords/cn_stopwords.txt",
        "stopwords/en_stopwords.txt",
        "stopwords/custom_stopwords.txt"
    )
    df['clean_tokens'] = df['tokens'].apply(
        lambda x: remove_stopwords(x, stopwords)
    )
    clean_sizes = df['clean_tokens'].apply(len)
    logging.info(clean_sizes.describe())
    logging.info("Max clean tokens: %d | Min clean tokens: %d", clean_sizes.max(), clean_sizes.min())
    """
        count    26247.000000
        mean       114.614699
        std        187.265343
        min          0.000000
        25%         17.000000
        50%         45.000000
        75%        114.000000
        max       1619.000000
        Name: clean_tokens, dtype: float64
    """

    cleaned_output_path = config.CLEAN_DATA_PATH + f"/cleaned_comments_{lang}.csv"
    df.to_csv(cleaned_output_path, index=False, encoding="utf-8")

    # 4. TF-IDF
    texts = df['clean_tokens'].apply(lambda x: " ".join(x))
    tfidf_matrix, vectorizer = build_tfidf(
        texts=texts,
        max_df=config.TFIDF_MAX_DF,
        min_df=config.TFIDF_MIN_DF,
        max_features=config.TFIDF_MAX_FEATURES
    )
    logging.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # 5. LDA
    # 5.1 LDA主题数选择
    multiprocessing.freeze_support()
    metrics = compute_lda_metrics(
        texts=df['clean_tokens'].tolist(),
        topic_range=range(1, 20),
        passes=config.LDA_PASSES
    )

    for m in metrics:
        logging.info(m)

    plot_lda_metrics(metrics, title=f"LDA Topic Number Selection ({lang})")

    best_k = select_best_topic(metrics)
    logging.info(f"自动选择的最优主题数为: {best_k}")
    # best_k = 9
    # 5.2 获取 LDA 主题-文档分布矩阵
    config.NUM_TOPICS = best_k
    lda, dictionary = train_lda(texts=df['clean_tokens'], num_topics=config.NUM_TOPICS, passes=config.LDA_PASSES)
    corpus = [dictionary.doc2bow(tokens) for tokens in df['clean_tokens']]
    doc_topics = lda.get_document_topics(corpus, minimum_probability=0.0)
    doc_topics_matrix = np.array([[prob for _, prob in topic] for topic in doc_topics])
    logging.info(f"Document-Topic matrix shape: {doc_topics_matrix.shape}")

    lda_topics_output = config.TABLE_OUTPUT_PATH + f"/{lang}_lda_topics{' (LDA)' if args.LDA else ''}.csv"
    save_lda_topics(lda=lda, output_path=lda_topics_output, topn=config.LDA_TOPIC_TOP_WORDS)
    logging.info(f"LDA topic table saved to: {lda_topics_output}")

    # 6. 聚类（支持多种方法）
    args.n_clusters = best_k  # 设置聚类数为最优主题数
    input_matrix = doc_topics_matrix if args.LDA else tfidf_matrix
    logging.info(f"Clustering using method: {cluster_method} on {'LDA topic distribution' if args.LDA else 'TF-IDF vectors'}")
    if cluster_method == "kmeans":
        model, labels = kmeans_cluster(input_matrix=input_matrix, n_clusters=args.n_clusters)
        # try to expose cluster_centers_ if available
        cluster_centers = getattr(model, "cluster_centers_", None)
    elif cluster_method == "birch":
        model, labels = birch_cluster(input_matrix=input_matrix, n_clusters=args.n_clusters, threshold=args.birch_threshold)
        # Birch exposes subcluster_centers_ or cluster_centers_
        cluster_centers = getattr(model, "subcluster_centers_", None)
        if cluster_centers is None:
            cluster_centers = getattr(model, "cluster_centers_", None)
    elif cluster_method == "spectral":
        model, labels = spectral_cluster(input_matrix=input_matrix, n_clusters=args.n_clusters, affinity=args.spectral_affinity)
        cluster_centers = None
    elif cluster_method == "hirerarchical":
        model, labels = hierarchical_cluster(input_matrix=input_matrix, n_clusters=args.n_clusters, linkage='ward')
        cluster_centers = None
    else:
        raise ValueError(f"Unsupported clustering method: {cluster_method}")

    df['cluster'] = labels
    logging.info(f"聚类结果分布:\n{Counter(labels)}")
    # 输出聚类计数（按实际标签集合）
    def count_clusters_from_labels(labels):
        counter = Counter(labels)
        # sort by label
        keys = sorted(counter.keys())
        return [counter[k] for k in keys], keys

    cluster_counts, cluster_keys = count_clusters_from_labels(labels)
    plot_kmeans_cluster_counts(cluster_counts=cluster_counts, title=f"Cluster counts ({lang}) - {cluster_method}" + f"{' (LDA)' if args.LDA else ''}")

    cluster_time_span = (df.groupby("cluster")["created_at"].agg(start="min", end="max").reset_index())
    plot_cluster_time_span(cluster_time_span=cluster_time_span, title=f"Documents Cluster Time Span ({lang}) - {cluster_method}" + f"{' (LDA)' if args.LDA else ''}")

    # 6.1 提取主题词：通用方法
    terms = vectorizer.get_feature_names_out()

    def get_topic_words_for_labels(labels, topn, tfidf_matrix):
        cluster_to_docs = defaultdict(list)
        for doc_idx, label in enumerate(labels):
            if label != -1:
                cluster_to_docs[label].append(doc_idx)

        uniq = sorted(cluster_to_docs.keys())
        rows = []
        for lab in uniq:
            idx = cluster_to_docs[lab]
            if not idx:
                logging.warning(f"Cluster {lab} is empty")
                rows.append([])
                continue

            cluster_matrix = tfidf_matrix[idx]
            if issparse(cluster_matrix):
                mean_vector = cluster_matrix.mean(axis=0).A1
            else:
                mean_vector = np.asarray(cluster_matrix.mean(axis=0)).ravel()

            sorted_indices = mean_vector.argsort()[::-1]
            words = []
            for term_idx in sorted_indices:
                if mean_vector[term_idx] <= 0:
                    break
                words.append(terms[term_idx])
                if len(words) >= topn:
                    break

            if not words:
                logging.warning(f"Cluster {lab} has no non-zero TF-IDF features")
            rows.append(words)

        return uniq, rows

    keyword_topn = max(1, min(config.TOPIC_KEYWORDS_PER_CLUSTER, len(terms)))
    cluster_ids, cluster_keywords = get_topic_words_for_labels(labels, topn=keyword_topn, tfidf_matrix=tfidf_matrix)
    def get_topic_words(i_index):
        # i_index is cluster index in range(len(cluster_ids))
        return cluster_keywords[i_index]
    
    # 打印每个聚类的主题词（基于实际簇 id 列表）
    for idx, cid in enumerate(cluster_ids):
        print(f"Cluster {cid}:", get_topic_words(idx))

    # 6.2 对所有聚类的主题词进行计数
    topic_counter = Counter([word for words in cluster_keywords for word in words])

    # 打印出现次数最多的几个主题词
    print(f"Top {keyword_topn} hot topics:")
    for word, count in topic_counter.most_common(keyword_topn):
        print("%s: %d" % (word, count))

    topic_rows = []
    for idx, cid in enumerate(cluster_ids):
        topic_rows.append({
            "cluster": int(cid),
            "keywords": ",".join(cluster_keywords[idx])
        })
    os.makedirs(os.path.dirname(config.TABLE_OUTPUT_PATH) or "output/tables", exist_ok=True)
    pd.DataFrame(topic_rows).to_csv(config.TABLE_OUTPUT_PATH + f"/{lang}_topic_keywords_{cluster_method}{' (LDA)' if args.LDA else ''}.csv", index=False, encoding='utf-8')
    
    # 6.3 计算热度得分
    def calculate_hot_score(cluster):
        # 获取该聚类的所有微博
        cluster_tweets = df[df['cluster'] == cluster]
    
        # 计算话题的出现频次
        frequency = len(cluster_tweets)
    
        # 计算相关微博的总评论数和总点赞数
        total_comments = int(pd.to_numeric(cluster_tweets['comment'], errors='coerce')  # 强制转换，非数字→NaN
            .fillna(0)  # NaN→0
            .sum()  # 求和
        )
        total_attitudes = int(pd.to_numeric(cluster_tweets['attitudes'], errors='coerce')  # 强制转换，非数字→NaN
            .fillna(0)  # NaN→0
            .sum()  # 求和
        )
    
        # 返回一个得分，这个得分是频次、评论数和点赞数的加权平均
        # 这里假设所有因素的权重都是1，你可以根据实际需要调整权重
        return (frequency + total_comments + total_attitudes) / 3
    
    # 计算每个聚类的热度得分（按实际簇 id 列表）
    hot_scores = [calculate_hot_score(cid) for cid in cluster_ids]

    # 打印每个聚类的热度得分
    for idx, score in enumerate(hot_scores):
        cid = cluster_ids[idx]
        print("Cluster %s:" % str(cid), get_topic_words(idx))
        print("Hot score: %f" % score)

    plot_cluster_hot_scores(hot_scores=hot_scores, title=f"Cluster Hot Scores ({lang}) - {cluster_method}" + f"{' (LDA)' if args.LDA else ''}")

    # 7. 文本分类准备
    # Use cluster id as the label (multi-class classification)
    df['label'] = df['cluster'].astype(int)

    # Save labeled CSV for downstream use (e.g., text classification)
    labeled_output = config.CLEAN_DATA_PATH + f"/labeled_comments_{lang}.csv"
    df.to_csv(labeled_output, index=False, encoding='utf-8')

    logging.info("Pipeline completed successfully.")

    # 8. 可视化
    logging.info("Generating visualizations...")
    draw_wordcloud(
        words=sum(df['clean_tokens'].tolist(), []),
        font_path=config.FONT_PATH,
        output=config.FIG_OUTPUT_PATH + f"/{lang}_wordcloud{' (LDA)' if args.LDA else ''}.png",
        shape_path="star.png"
    )
    logging.info("Word cloud generated.")
    topic_wc_output_dir = os.path.join(
        config.FIG_OUTPUT_PATH,
        f"{lang}_topic_wordclouds" + ("_lda" if args.LDA else "")
    )
    draw_topic_wordclouds(
        lda=lda,
        font_path=config.FONT_PATH,
        output_dir=topic_wc_output_dir,
        shape_path="star.png"
    )
    logging.info("Per-topic word clouds generated.")
    draw_topic_trend(df=df, output=config.FIG_OUTPUT_PATH + f"/{lang}_topic_trend{' (LDA)' if args.LDA else ''}.png")
    logging.info("Topic trend over time generated.")
    draw_sankey(lda=lda, output=config.FIG_OUTPUT_PATH + f"/{lang}_topic_sankey{' (LDA)' if args.LDA else ''}.html")
    logging.info("Sankey diagram generated.")
