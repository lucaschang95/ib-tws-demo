from business import TestApp

def main():
    """
    主程序入口函数
    """
    print("程序开始运行...")
    
    # 创建 TestApp 实例并连接
    app = TestApp()
    app.connect("127.0.0.1", 4002, clientId=1)
    app.run()

if __name__ == "__main__":
    main()