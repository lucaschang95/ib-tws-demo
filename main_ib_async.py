from ib_async import *
import pandas as pd
# util.startLoop()  # uncomment this line when in a notebook

ib = IB()
ib.connect('127.0.0.1', 4002, clientId = 1)

##################
# ib.reqMarketDataType(4)  # Use free, delayed, frozen data
# contract = Forex('EURUSD')
# bars = ib.reqHistoricalData(
#     contract, endDateTime='', durationStr='30 D',
#     barSizeSetting='1 hour', whatToShow='MIDPOINT', useRTH=True)
##################

##################
ib.reqMarketDataType(4)  # Use free, delayed, frozen data
contract = Stock('AAPL', 'SMART', 'USD')
bars = ib.reqHistoricalData(
    contract, endDateTime='', durationStr='3 D',
    barSizeSetting='10 mins', whatToShow='TRADES', useRTH=True)
##################

##################
# convert to pandas dataframe (pandas needs to be installed):
# df = util.df(bars)
# print(df)
# Save bars data to local file
##################

##################
# # Save to csv file
# filename = 'aapl_bars.csv'
# df2 = util.df(bars)  # Convert to dataframe
# df2.to_csv(filename)
##################

##################
# df3 = pd.DataFrame(bars)
# filename = 'aapl_bars3.csv'
# df3.to_csv(filename)
##################


##################
print("bars的类型:", type(bars))
print("单个bar的类型:", type(bars[0]))
print("\n第一个bar的详细信息:")
print(vars(bars[0]))
##################

##################
# filename = 'aapl_bars4.txt'
# with open(filename, 'w') as f:
#     for bar in bars:
#         f.write(f'{bar}\n')
##################


# positions = ib.reqPositions()
# print(positions)