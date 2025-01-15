from ibapi import wrapper, client, contract

class TestWrapper(wrapper.EWrapper):
    def nextValidId(self, orderId):
        print(f'Connected. Next valid order ID: {orderId}')
        self.reqSpxData()

    def tickPrice(self, reqId, field, price, attrib):
        print(f'Tick price: Field={field}, Price={price}')
        if reqId == 1 and field == 4:
            print(f'AAPL当前值: {price}')

    def reqSpxData(self):
        aapl_contract = contract.Contract()
        aapl_contract.symbol = "AAPL"
        aapl_contract.secType = "STK"
        aapl_contract.currency = "USD"
        aapl_contract.exchange = "SMART"
        aapl_contract.primaryExchange = "NASDAQ"
        self.reqMarketDataType(3)
        self.reqMktData(1, aapl_contract, "", False, False, [])

class TestClient(client.EClient):
    def __init__(self, wrapper):
        client.EClient.__init__(self, wrapper)

class TestApp(TestWrapper, TestClient):
    def __init__(self):
        TestWrapper.__init__(self)
        TestClient.__init__(self, wrapper=self)

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson, idk):
        print("Error {} {} {}".format(reqId, errorCode,errorString))