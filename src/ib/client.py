from ib_async import IB, Contract, util
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IBClient:
    def __init__(self, host='127.0.0.1', port=4001, client_id=1):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        
    def connect(self):
        self.ib.connect('127.0.0.1', 4001, clientId = 1)
        self.ib.reqMarketDataType(4)
            
    async def disconnect(self):
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB")

    async def getHistoricalData(self, contract = 'AAPL', duration='1 D', bar_size='1 min', what_to_show='TRADES'):
        try:
            if not self.ib.isConnected():
                logger.warning("Not connected to IB, attempting to connect...")
                if not self.connect():
                    raise ConnectionError("Failed to connect to IB")
                    
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',  # 空字符串表示当前时间
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,
                formatDate=1
            )
            
            if not bars:
                logger.warning(f"No historical data received for {contract}")
                return None
                
            logger.info(f"Successfully retrieved {len(bars)} bars")
            return bars
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None

    def get_real_historical_data(self, contract, num_batches=20, filename='aapl_bars13.csv'):
        end_date = '20240604 09:30:00'  # Start with current time
        temp_bars = []  # Temporary list to store batches

        for i in range(num_batches):  # Get multiple sets of data
            print(f'Requesting historical data batch {i+1}...')
            bars = self.ib.reqHistoricalData(
                contract, endDateTime=end_date, durationStr='3 D',
                barSizeSetting='1 min', whatToShow='TRADES', useRTH=True)
            logger.info(f'Batch {i+1} received.')
            
            # Add the new bars to our temporary list
            temp_bars.append(bars)
            
            # Update end_date to get next batch of historical data
            if bars and len(bars) > 0:
                # Use the earliest timestamp from current batch as next end time
                end_date = bars[0].date.strftime('%Y%m%d %H:%M:%S')
                logger.info(f'Next end date: {end_date}')
            
            # Sleep briefly to avoid overwhelming the server
            self.ib.sleep(5)

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
    
