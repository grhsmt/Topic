# 主运行脚本示例

示例：

1) 运行整个预处理 + LDA + 聚类 + 可视化管线（默认文件路径基于 `src.config` 配置）：

    ```bash
    python main.py --language Chinese/English --cluster_method kmeans --n_clusters 10
    # 或者：
    python main.py -l Chinese --cluster_method birch/spectral
    ```

2) 仅运行文本分类（XGBoost + SVM），假设已生成 `data/processed/labeled_comments.csv`：

    ```bash
    python -m src.text_classification --data data/processed/labeled_comments_Chinese/English.csv --text_col comment --label_col label --out_dir output/models --language Chinese/English
    # 支持短参数: -d (data), -t (text_col), -l (label_col), -o (out_dir)
    ```

3) 安装依赖（在虚拟环境中）：

    ```bash
    pip install -r requirements.txt
    ```

4) 快速说明：

- **聚类方法**: `--cluster_method` 支持 `kmeans`, `dbscan`, `single_pass`。生成的图/表文件名会包含所用聚类方法。
- **标签含义**: 输出 CSV (`data/processed/labeled_comments.csv`) 中的 `label` 列是聚类 ID（整数），用于之后的文本分类训练。
- **Tokenizer 兼容性**: 如果本地未安装 `jieba`，`src/tokenizer.py` 会降级为简单字符/词切分以保证流程能跑通；推荐在处理中文时安装 `jieba` 以获得更好分词效果。

## 输出位置（默认）

- 图像与交互式可视化: `output/figures/`（例: `topic_sankey.html`, `topic_trend_kmeans.png`）
- 表格与关键词: `output/tables/topic_keywords.csv`（文件名包含聚类方法）
- 模型与训练报告: `output/models/`（包含保存的模型、`vectorizer.joblib` 与 `training_report_<language>.json`）

## 文件结构

```txt
weibo_facebook_topic_analysis/
│
├── data/
│   ├── raw/
│   │   └── comments.csv              # 原始评论数据
│   ├── processed/
│   │   ├── cleaned_comments.csv               # 清洗后数据
│   │   └── labeled_comments.csv
│
├── stopwords/
│   ├── cn_stopwords.txt               # 中文停用词
│   ├── en_stopwords.txt               # 英文停用词
│   └── custom_stopwords.txt           # 自定义停用词
│
├── src/
│   ├── config.py                      # 全局参数配置
│   ├── data_loader.py                 # 数据读取与时间处理
│   ├── tokenizer.py                   # jieba 分词
│   ├── stopwords.py                   # 停用词加载
│   ├── tfidf_model.py                 # TF-IDF 关键词提取
│   ├── lda_model.py                   # OLDA 主题建模
│   ├── clustering.py                  # KMeans 聚类
│   ├── visualization.py               # 词云 / 桑基图 / 时间分布图
│   └── plot_utils.py                  # 画图函数
│
├── output/
│   ├── figures/
│   │   ├── wordcloud.png
│   │   ├── topic_trend.png
│   │   └── topic_sankey.html
│   ├── tables/
│   │   └── topic_keywords.csv
│
├── main.py                            # 主程序（老师重点看）
├── requirements.txt
├── simulation.log                     # 运行日志
├── star.png                           # 词云形状
└── README.md                          # 作业说明
```
