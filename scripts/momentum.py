import yfinance as yf
import pandas as pd

# 获取AAPL的历史数据
ticker = yf.Ticker("TSLA")

# 获取日K数据
df = ticker.history(period="max", interval="1d")

# 保存到CSV文件
df.to_csv('test-data/tsla/tsla_1d.csv')


