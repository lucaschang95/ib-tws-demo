import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna
from typing import List, Dict
import matplotlib.pyplot as plt

class TradingStrategy:
    def __init__(self, rsi_upper: float = 70, rsi_lower: float = 30):
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号
        1 表示买入信号
        -1 表示卖出信号
        0 表示持仓不变
        """
        signals = pd.Series(0, index=df.index)
        
        # RSI超卖买入
        signals[df['momentum_rsi'] < self.rsi_lower] = 1
        # RSI超买卖出
        signals[df['momentum_rsi'] > self.rsi_upper] = -1
        
        return signals

class Backtester:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.positions = pd.Series()
        self.portfolio = pd.Series()
        
    def run(self, df: pd.DataFrame, signals: pd.Series) -> Dict:
        """执行回测"""
        # 计算持仓
        self.positions = self.initial_capital * signals.cumsum()
        
        # 计算每日市值
        self.portfolio = self.initial_capital + (self.positions * df['Close'].pct_change()).cumsum()
        
        # 计算回测指标
        returns = self.portfolio.pct_change()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        max_drawdown = (self.portfolio / self.portfolio.cummax() - 1).min()
        
        return {
            'Final Value': self.portfolio.iloc[-1],
            'Total Return': (self.portfolio.iloc[-1] - self.initial_capital) / self.initial_capital * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown * 100
        }
    
    def plot_results(self, df: pd.DataFrame):
        """绘制回测结果"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 绘制股价和RSI
        ax1.plot(df.index, df['Close'], label='Price')
        ax1.set_title('Stock Price')
        ax1.legend()
        
        # 绘制投资组合价值
        ax2.plot(self.portfolio.index, self.portfolio, label='Portfolio Value')
        ax2.set_title('Portfolio Value')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

# 获取AAPL的历史数据
ticker = yf.Ticker("TSLA")

# 获取日K数据
# df = ticker.history(period="max", interval="1d")

# 保存到CSV文件
# df.to_csv('test-data/tsla/tsla_1d.csv')

# Load datas
df = pd.read_csv('test-data/tsla/tsla_1d.csv', sep=',')

# 确保数据不为空
if df.empty:
    raise ValueError("数据文件为空")

# 处理日期列，添加 utc=True 解决时区警告
df['Date'] = pd.to_datetime(df['Date'], utc=True)
df.set_index('Date', inplace=True)

# 确保所有必需的列都存在
required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"缺少必需的列: {missing_columns}")

# 删除包含 NaN 的行
df = df.dropna(subset=required_columns)

# 确保数据框不为空
if len(df) == 0:
    raise ValueError("清理后的数据为空")

# 添加技术指标
try:
    df = add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume",
        fillna=True  # 填充可能的 NaN 值
    )
except Exception as e:
    print(f"添加技术指标时出错: {str(e)}")
    raise

# 再次确保 RSI 列存在
if 'momentum_rsi' not in df.columns:
    raise ValueError("RSI 指标未能成功添加到数据中")

# 创建策略实例
strategy = TradingStrategy(rsi_upper=70, rsi_lower=30)
signals = strategy.generate_signals(df)

# 执行回测
backtester = Backtester(initial_capital=100000)
results = backtester.run(df, signals)

# 打印回测结果
print("\n=== Backtest Results ===")
for metric, value in results.items():
    print(f"{metric}: {value:.2f}")

# 绘制回测结果
backtester.plot_results(df)