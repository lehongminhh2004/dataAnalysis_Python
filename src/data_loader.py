import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("-", "_")
    )
    return df
