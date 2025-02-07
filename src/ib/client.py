from ib_async import IB
import logging
import os
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IBClient:
    def __init__(self, host='127.0.0.1', port=4001, client_id=1):
        """
        初始化IB客户端并自动连接
        
        Args:
            host: IB Gateway/TWS的主机地址
            port: IB Gateway/TWS的端口 (4001 for paper trading, 4002 for live trading)
            client_id: 客户端ID
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        # self.connect()
        
    def connect(self):
        """连接到IB Gateway/TWS并设置市场数据类型"""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            # 使用免费的延迟数据
            self.ib.reqMarketDataType(4)
            logger.info(f"Successfully connected to IB on {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IB: {str(e)}")
            return False
            
    def disconnect(self):
        """断开IB连接"""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB")

    async def getHistoricalData(self, contract = 'AAPL', duration='1 D', bar_size='1 min', what_to_show='TRADES'):
        # Get project root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        data_path = os.path.join(root_dir, 'test-data', 'aapl_bars4.csv')
        
        logger.info(f"Loading test data from: {data_path}")
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Successfully loaded {len(df)} rows from test data")
            return df
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            return None


        # try:
        #     if not self.ib.isConnected():
        #         logger.warning("Not connected to IB, attempting to connect...")
        #         if not self.connect():
        #             raise ConnectionError("Failed to connect to IB")
                    
        #     bars = await self.ib.reqHistoricalDataAsync(
        #         contract,
        #         endDateTime='',  # 空字符串表示当前时间
        #         durationStr=duration,
        #         barSizeSetting=bar_size,
        #         whatToShow=what_to_show,
        #         useRTH=True,
        #         formatDate=1
        #     )
            
        #     if not bars:
        #         logger.warning(f"No historical data received for {contract.symbol}")
        #         return None
                
        #     logger.info(f"Successfully retrieved {len(bars)} bars for {contract.symbol}")
        #     return bars
            
        # except Exception as e:
        #     logger.error(f"Error getting historical data: {str(e)}")
        #     return None