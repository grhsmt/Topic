import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    return df
