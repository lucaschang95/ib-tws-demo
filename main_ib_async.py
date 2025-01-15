from ib_async import *
# util.startLoop()  # uncomment this line when in a notebook

ib = IB()
ib.connect('127.0.0.1', 4001, clientId = 1)

# ib.reqMarketDataType(4)  # Use free, delayed, frozen data
# contract = Forex('EURUSD')
# bars = ib.reqHistoricalData(
#     contract, endDateTime='', durationStr='30 D',
#     barSizeSetting='1 hour', whatToShow='MIDPOINT', useRTH=True)

positions = ib.reqPositions()
print(positions)

# convert to pandas dataframe (pandas needs to be installed):
# df = util.df(bars)
# print(df)