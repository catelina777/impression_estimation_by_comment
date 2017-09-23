import pandas as pd
import glob


def add_average_colum(file_path):
    df = pd.read_csv(file_path, index_col=0)
    series = df.mean()
    series.name = "平均"
    df = df.append(series)
    df.to_csv(file_path)


def main():
    files = glob.glob('./result/' + '*result.csv')
    for file in files:
        add_average_colum(file)

    print("success!")


if __name__ == '__main__':
    main()
