import pandas as pd


def main():
    file_path = "train_test_2025.csv"
    df = pd.read_csv(file_path)
    print(df)

if __name__ == "__main__":
    main()