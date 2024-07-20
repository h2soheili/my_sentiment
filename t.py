import polars as pl

df = pl.read_csv('./bv_news.csv')


def x(i):
    return len(i.split(' '))


df = df.with_columns(pl.col('text').alias("count").apply(x))
df.write_csv('./bv_news.csv')
