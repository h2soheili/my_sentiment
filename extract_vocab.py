from bv_news_crawler import clean_text0

if __name__ == '__main__':
    import polars as pl

    df = pl.read_csv('./stocks.csv')
    # stocks = dict()
    # for r in df.rows(named=True):
    #     stocks[r["Ticker"]] = None
    #
    # print(list(stocks.keys()))
    def x2(x):
        return f"#{clean_text0(x[0])}"
    print(1)
    new_col = df.select(pl.col('Ticker').alias("Name2")).apply(x2)
    df = df.with_columns(new_col)

    df.write_csv('./stocks2.csv')
