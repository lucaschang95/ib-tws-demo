from utils import logger
from utils.args import load_args
from utils.logger import setup_logger
from utils.config import load_config
from ib.client import IBClient
import asyncio
from utils.preprocess import preprocess
from models.lstm import create_model_with_config
from models.trainer import LSTMTrainer
import logging

logger = logging.getLogger(__name__)

async def main():
    setup_logger(clear_log=True)
    args = load_args()

    config = load_config(args.config)

    client = IBClient(
        host=args.host,
        port=args.port,
        client_id=args.client_id
    )
    
    await client.connect()


    # data = await client.getHistoricalData()
    
    # 完成后断开连接
    # await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
