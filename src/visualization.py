import os
from datetime import datetime

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pyecharts import options as opts
from pyecharts.charts import Sankey
from wordcloud import WordCloud

def draw_wordcloud(words, font_path, output, shape_path=None, width=4000, height=2400, dpi=300):
    # 如果提供了形状图片，加载并预处理为掩码
    mask = None
    if shape_path:
        shape_image = Image.open(shape_path)
        # 转换为灰度图（确保只有黑白）
        shape_array = np.array(shape_image.convert('L'))
        # 二值化处理：非黑色区域设为可填充
        mask = np.where(shape_array < 10, 255, 0)

    wc = WordCloud(
        font_path=font_path,
        background_color="white",
        mask=mask,
        width=width,
        height=height
    ).generate(" ".join(words))

    # 将 WordCloud 导出为 PIL.Image，以便保存时指定 DPI（提高分辨率）
    img = wc.to_image()
    try:
        img.save(output, dpi=(dpi, dpi))
    except TypeError:
        # 某些环境下 PIL 可能不支持 dpi 参数，作为回退直接保存
        img.save(output)


def draw_topic_wordclouds(lda, font_path, output_dir, topn=30, shape_path=None, width=4000, height=2400, dpi=300, background_color="white"):
    os.makedirs(output_dir, exist_ok=True)

    mask = None
    if shape_path:
        shape_image = Image.open(shape_path)
        shape_array = np.array(shape_image.convert('L'))
        mask = np.where(shape_array < 10, 255, 0)

    for topic_id in range(lda.num_topics):
        frequencies = dict(lda.show_topic(topic_id, topn))
        if not frequencies:
            continue

        wc = WordCloud(
            font_path=font_path,
            background_color=background_color,
            mask=mask,
            width=width,
            height=height
        ).generate_from_frequencies(frequencies)

        output_path = os.path.join(output_dir, f"topic_{topic_id}.png")
        img = wc.to_image()
        try:
            img.save(output_path, dpi=(dpi, dpi))
        except TypeError:
            img.save(output_path)

def draw_topic_trend(df, output):
    # Ensure date column is datetime
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Group by date and cluster, then reindex to full date range
    trend = df.groupby(['date', 'cluster']).size().unstack(fill_value=0)

    # Define fixed x-axis limits
    start_limit = datetime(2025, 11, 12)
    end_limit = datetime(2025, 12, 1)
    all_dates = pd.date_range(start=start_limit, end=end_limit, freq='D')

    trend = trend.reindex(all_dates, fill_value=0)

    # Set Chinese font for Windows (falls back silently if not available)
    mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    ax = trend.plot(figsize=(12, 6))
    ax.set_title("热门话题随时间变化")

    # Set x-axis ticks every 3 days and format as YYYY-MM-DD
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_xlim(start_limit, end_limit)
    plt.setp(ax.get_xticklabels(), rotation=30)

    plt.tight_layout()
    plt.savefig(output)

def draw_sankey(lda, output):
    nodes, links = [], []
    seen = set()
    for i in range(lda.num_topics):
        topic = f"主题{i}"
        if topic not in seen:
            nodes.append({"name": topic})
            seen.add(topic)
        for word, w in lda.show_topic(i, 5):
            if word not in seen:
                nodes.append({"name": word})
                seen.add(word)
            links.append({
                "source": topic,
                "target": word,
                "value": round(w * 100, 2)
            })

    # Use Chinese font for labels and title text
    sankey = Sankey(init_opts=opts.InitOpts(width="1000px", height="600px"))
    sankey.add(
        "",
        nodes,
        links,
        label_opts=opts.LabelOpts(font_family="Microsoft YaHei")
    ).set_global_opts(
        title_opts=opts.TitleOpts(title="主题词Sankey", title_textstyle_opts=opts.TextStyleOpts(font_family="Microsoft YaHei"))
    )
    sankey.render(output)
