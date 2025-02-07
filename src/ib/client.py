from ib_async import IB
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IBClient:
    def __init__(self, host='127.0.0.1', port=4001, client_id=1):
        """
        初始化IB客户端
        
        Args:
            host: IB Gateway/TWS的主机地址
            port: IB Gateway/TWS的端口 (4001 for paper trading, 4002 for live trading)
            client_id: 客户端ID
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        
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
