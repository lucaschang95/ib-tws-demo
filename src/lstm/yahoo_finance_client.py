import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_stock_data(symbol: str, start_date: str = None, end_date: str = None, 
                    period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        ticker = yf.Ticker(symbol)
        
        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date, interval=interval)
        else:
            df = ticker.history(period=period, interval=interval)
            
        if df.empty:
            logger.error(f"No data found for symbol {symbol}")
            return None
            
        logger.info(f"Successfully fetched data for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None
    

def fetch_ticker_history(
    symbol: str,
    period: str = None,
    interval: str = "1d",
    start: str = None,
    end: str = None,
    prepost: bool = False,
    auto_adjust: bool = True,
    repair: bool = False,
    keepna: bool = False,
    timeout: float = 10
) -> pd.DataFrame:
    """
    Fetch historical data for a given ticker using yfinance
    
    Args:
        symbol: Stock ticker symbol
        period: Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval: Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        prepost: Include Pre and Post market data
        auto_adjust: Adjust all OHLC automatically
        repair: Attempt to repair 100x currency unit mixups
        keepna: Keep NaN rows returned by Yahoo
        timeout: Timeout in seconds for the request
        
    Returns:
        pandas DataFrame with historical data or None if error occurs
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Build kwargs based on provided parameters
        kwargs = {
            'interval': interval,
            'prepost': prepost,
            'auto_adjust': auto_adjust,
            'repair': repair,
            'keepna': keepna,
            'timeout': timeout
        }
        
        # Use either period or start/end dates
        if start is not None and end is not None:
            kwargs['start'] = start
            kwargs['end'] = end
        elif period is not None:
            kwargs['period'] = period
        else:
            kwargs['period'] = '1y'  # default to 1 year if neither is specified
            
        df = ticker.history(**kwargs)
        
        if df.empty:
            logger.error(f"No data found for symbol {symbol}")
            return None
            
        logger.info(f"Successfully fetched history data for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching history data for {symbol}: {str(e)}")
        return None
    


if __name__ == "__main__":
    # Example usage
    symbol = "AAPL"
    df = fetch_stock_data(symbol, interval="1d")
    print(df)