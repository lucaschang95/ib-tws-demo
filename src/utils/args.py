import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='IB 客户端程序')
    parser.add_argument('--host', default='127.0.0.1',
                        help='IB Gateway 主机地址 (默认: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=4001,
                        help='IB Gateway 端口 (默认: 4001)')
    parser.add_argument('--client-id', type=int, default=1,
                        help='客户端 ID (默认: 1)')
    parser.add_argument('--paper-trading', action='store_true',
                        help='使用纸面交易模式')
    parser.add_argument('--config', type=str, default='config.json',
                        help='配置文件路径 (默认: config.json)')
    return parser.parse_args()


def load_args():
    """加载并处理命令行参数"""
    args = parse_args()
    
    # 如果使用纸面交易模式，强制使用 4001 端口
    if args.paper_trading:
        args.port = 4001
    
    return args