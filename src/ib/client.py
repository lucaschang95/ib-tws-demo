from ib_async import IB, Contract
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IBClient:
    def __init__(self, host='127.0.0.1', port=4001, client_id=1):
        """
        初始化IB客户端并自动连接
        
        Args:
            host: IB Gateway/TWS的主机地址
            port: IB Gateway/TWS的端口 (4001 for live trading, 4002 for paper trading)
            client_id: 客户端ID
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        # 移除自动连接，改为在异步上下文中手动连接
        
    async def connect(self):
        """异步连接到 IB"""
        try:
            await self.ib.connect(self.host, self.port, clientId=self.client_id)
            await self.ib.reqMarketDataTypeAsync(4)
            logger.info(f"Successfully connected to IB on {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IB: {str(e)}")
            return False
            
    async def disconnect(self):
        if self.ib.isConnected():
            await self.ib.disconnect()
            logger.info("Disconnected from IB")

    async def getHistoricalData(self, contract = 'AAPL', duration='1 D', bar_size='1 min', what_to_show='TRADES'):
        try:
            if not self.ib.isConnected():
                logger.warning("Not connected to IB, attempting to connect...")
                if not await self.connect():
                    raise ConnectionError("Failed to connect to IB")
                    
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',  # 空字符串表示当前时间
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,
                formatDate=1
            )
            
            if not bars:
                logger.warning(f"No historical data received for {symbol}")
                return None
                
            logger.info(f"Successfully retrieved {len(bars)} bars")
            return bars
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None

    async def fetch_historical_bars(self, contract, start_date, end_date, bar_size='1 min'):
        """
        分段获取历史数据
        """
        all_bars = []
        current_end = end_date
        
        while current_end > start_date:
            # 对于分钟数据，每次获取1天
            duration = '1 D'
            
            # 将时间转换为字符串格式 'YYYYMMDD HH:mm:ss'
            end_time_str = current_end.strftime('%Y%m%d %H:%M:%S')
            
            try:
                bars = await self.ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=end_time_str,
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1
                )
                
                if bars:
                    all_bars.extend(bars)
                    
                # 向前移动一天
                # current_end = current_end - timedelta(days=1)
                
                # 可以添加适当的延时，避免请求过于频繁
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error fetching data for {end_time_str}: {str(e)}")
                await asyncio.sleep(5)  # 出错时等待更长时间

        return all_bars