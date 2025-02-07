from ib_async import *
import pandas as pd

ib = IB()
ib.connect('127.0.0.1', 4001, clientId = 1)
ib.reqMarketDataType(4)  # Use free, delayed, frozen data
contract = Stock('AAPL', 'SMART', 'USD')

def get_historical_data(ib, contract, filename='aapl_bars5.csv', num_batches=3):
    end_date = ''  # Start with current time
    temp_bars = []  # Temporary list to store batches

    for i in range(num_batches):  # Get multiple sets of data
        print(f'Requesting historical data batch {i+1}...')
        bars = ib.reqHistoricalData(
            contract, endDateTime=end_date, durationStr='3 D',
            barSizeSetting='1 min', whatToShow='TRADES', useRTH=True)
        print(f'Batch {i+1} received.')
        
        # Add the new bars to our temporary list
        temp_bars.append(bars)
        
        # Update end_date to get next batch of historical data
        if bars and len(bars) > 0:
            # Use the earliest timestamp from current batch as next end time
            end_date = bars[0].date.strftime('%Y%m%d %H:%M:%S')
        
        # Sleep briefly to avoid overwhelming the server
        ib.sleep(5)

    # Reverse the order and combine all bars
    all_bars = []
    for batch in reversed(temp_bars):
        all_bars.extend(batch)

    # Convert all collected bars to a single dataframe
    bars = all_bars

    # Save to csv file
    df2 = util.df(bars)  # Convert to dataframe
    df2.to_csv(filename)
    
    return bars

# get_historical_data(ib, contract)