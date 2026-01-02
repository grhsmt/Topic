import pandas as pd

def standardize_csv(
    input_csv,
    output_csv,
    comment_col,
    time_col,
    attitudes_col,
    platform_name=None
):
    """
    将任意 CSV 转为标准字段：
    comment | created_at | attitudes | platform
    """

    df = pd.read_csv(input_csv)

    df = df.rename(columns={
        comment_col: "comment",
        time_col: "created_at",
        attitudes_col: "attitudes"
    })

    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df = df.dropna(subset=['comment', 'created_at', 'attitudes'])

    if platform_name:
        df['platform'] = platform_name

    df = df[['comment', 'created_at', 'attitudes'] + (['platform'] if platform_name else [])]

    df.to_csv(output_csv, index=False, encoding="utf-8")

if __name__ == "__main__":
    input_path = "data/raw/20251112-20251201.csv"
    output_path = "data/raw/comments.csv"
    standardize_csv(
        input_csv=input_path,
        output_csv=output_path,
        comment_col="微博正文",
        time_col="发布时间",
        attitudes_col="点赞数",
        platform_name="weibo"
    )
