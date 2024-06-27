import sys
import pandas as pd


def convert(data_path: str) -> None:
    df = pd.read_excel(data_path, sheet_name="dataset")
    print("Original data shape: ", df.shape)
    df.to_csv(data_path.replace(".xlsx", ".csv"), index=False)
    print("Subset of data shape: ", df.iloc[:1000, :].shape, " this will be saved with `_postfix`")
    df.iloc[:1000, :].to_csv(data_path.replace(".xlsx", "_small.csv"), index=False)
    return

if __name__ == "__main__":
    convert(sys.argv[1])