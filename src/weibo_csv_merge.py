import pandas as pd

# 场景1：按行合并（纵向拼接，结构相同）两个 CSV 列名相同，直接上下拼接

# 读取两个CSV文件
df1 = pd.read_csv('data/raw/20251112-20251201.csv')
df2 = pd.read_csv('data/raw/20251201-20251209.csv')

# 合并
merged_df = pd.concat([df1, df2], ignore_index=True)

# 保存结果
merged_df.to_csv('data/raw/20251112-20251209.csv', index=False)