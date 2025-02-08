import os
from models.lstm import create_model_with_config, evaluate_models
from utils.args import load_args
from utils.logger import setup_logger
from utils.config import load_config
from ib.client import IBClient
from utils.preprocess import preprocess
from models.lstm import create_model_with_config
from ib_async import *
from models.trainer import LSTMTrainer

def main():
    setup_logger(clear_log=True)
    args = load_args()

    config = load_config(args.config)

    client = IBClient(
        host=args.host,
        port=args.port,
        client_id=args.client_id
    )
    client.connect()
    contract = Stock('AAPL', 'SMART', 'USD')

    data = client.get_real_historical_data(contract)
    
    # 完成后断开连接
    # await client.disconnect()

if __name__ == "__main__":
    main()
