import numpy as np
import pandas as pd

FEATURES = [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour_sin", "hour_cos"]

def load_data(path="../data/raw/creditcard.csv"):
    df = pd.read_csv(path)
    df = df.sort_values("Time").reset_index(drop=True)
    df["log_amount"] = np.log1p(df["Amount"])
    hour = (df["Time"] // 3600) % 24
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return df

def time_split(df, fractions=(0.6, 0.8)):
    n = len(df)
    a, b = int(n * fractions[0]), int(n * fractions[1])
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]   # train, val, test