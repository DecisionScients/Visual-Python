# %%
import pandas as pd


def read():
    df = pd.read_csv("./data/test.csv", encoding="Latin-1", low_memory=False)
    return(df)
