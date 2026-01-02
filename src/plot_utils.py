import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def plot_lda_metrics(metrics, title):
    """
    绘制 Perplexity + Coherence 双轴图
    """
    ks = [m["num_topics"] for m in metrics]
    perplexities = [m["perplexity"] for m in metrics]
    coherences = [m["coherence"] for m in metrics]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("Number of Topics")
    ax1.set_ylabel("Perplexity", color='red')
    ax1.plot(ks, perplexities, marker='o', color='red')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Coherence (c_v)", color='blue')
    ax2.plot(ks, coherences, marker='s', color='blue')
    ax2.tick_params(axis='y')

    plt.title(title)

    fig.tight_layout()
    plt.savefig(f"output/figures/{title}.png", dpi=300)

def plot_kmeans_cluster_counts(cluster_counts, title):
    clusters = [f"C{i}" for i in range(len(cluster_counts))]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(clusters, cluster_counts)

    plt.xlabel("Cluster")
    plt.ylabel("Document Number")
    plt.title(title)

    # 在柱子上标数字
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(f"output/figures/{title}.png", dpi=300)

def plot_cluster_time_span(cluster_time_span, title):
    cluster_time_span = cluster_time_span.sort_values("cluster")

    clusters = [f"C{c}" for c in cluster_time_span["cluster"]]
    start_dates = cluster_time_span["start"]
    durations = (cluster_time_span["end"] - start_dates).dt.days

    plt.figure(figsize=(10, 5))

    # Convert start dates to matplotlib float representation
    start_nums = mdates.date2num(start_dates.dt.to_pydatetime())
    widths = durations.values

    plt.barh(clusters, widths, left=start_nums)

    ax = plt.gca()
    ax.invert_yaxis()

    # Set x-axis to show ticks every 7 days between 2025-11-12 and 2025-12-09
    start_limit = datetime(2025, 11, 12)
    end_limit = datetime(2025, 12, 9)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_xlim(mdates.date2num(start_limit), mdates.date2num(end_limit))
    plt.xticks(rotation=30)

    plt.xlabel("Time")
    plt.ylabel("Cluster")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(f"output/figures/{title}.png", dpi=300)

def plot_cluster_hot_scores(hot_scores, title):
    clusters = [f"C{i}" for i in range(len(hot_scores))]

    plt.figure(figsize=(8, 5))
    
    max_idx = hot_scores.index(max(hot_scores))

    colors = ["orange" if i == max_idx else "steelblue"
            for i in range(len(hot_scores))]

    bars = plt.bar(clusters, hot_scores, color=colors)

    plt.xlabel("Cluster")
    plt.ylabel("Hot Score")
    plt.title("Hot Score for Each Cluster")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(f"output/figures/{title}.png", dpi=300)
