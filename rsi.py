import ccxt
import time
from datetime import datetime
import pandas as pd
import numpy as np
import talib

# Exchange and trading pair configuration
exchange = ccxt.binance({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'  # Use this for futures trading
    }
})

symbol = 'BTC/USDT'
timeframe = '1h'

# Strategy parameters
short_window = 10
long_window = 30
rsi_period = 14
rsi_overbought = 70
rsi_oversold = 30

def fetch_ohlcv_data(symbol, timeframe, limit):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_signals(df):
    # Calculate moving averages
    df['short_ma'] = df['close'].rolling(window=short_window).mean()
    df['long_ma'] = df['close'].rolling(window=long_window).mean()
    
    # Calculate RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_period)
    
    # Generate signals
    df['ma_signal'] = 0
    df.loc[df['short_ma'] > df['long_ma'], 'ma_signal'] = 1
    df.loc[df['short_ma'] < df['long_ma'], 'ma_signal'] = -1
    
    df['rsi_signal'] = 0
    df.loc[df['rsi'] < rsi_oversold, 'rsi_signal'] = 1
    df.loc[df['rsi'] > rsi_overbought, 'rsi_signal'] = -1
    
    # Combine signals
    df['signal'] = 0
    df.loc[(df['ma_signal'] == 1) & (df['rsi_signal'] >= 0), 'signal'] = 1
    df.loc[(df['ma_signal'] == -1) & (df['rsi_signal'] <= 0), 'signal'] = -1
    
    return df

def calculate_position_size(balance, risk_per_trade, current_price, stop_loss_percent):
    risk_amount = balance * risk_per_trade
    stop_loss_amount = current_price * stop_loss_percent
    position_size = risk_amount / stop_loss_amount
    return position_size

def place_order(side, amount, symbol, stop_loss_percent):
    try:
        order = exchange.create_market_order(symbol, side, amount)
        print(f"Order placed: {order}")
        
        # Place stop loss order
        stop_price = order['price'] * (1 - stop_loss_percent) if side == 'buy' else order['price'] * (1 + stop_loss_percent)
        stop_loss_order = exchange.create_stop_market_order(symbol, 'sell' if side == 'buy' else 'buy', amount, stop_price)
        print(f"Stop loss order placed: {stop_loss_order}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

def run_bot():
    risk_per_trade = 0.01  # 1% risk per trade
    stop_loss_percent = 0.02  # 2% stop loss
    
    while True:
        try:
            print(f"Fetching data for {symbol} at {datetime.now()}")
            df = fetch_ohlcv_data(symbol, timeframe, long_window + rsi_period)
            df = calculate_signals(df)
            
            current_position = 0  # Assume no position to start
            last_signal = df['signal'].iloc[-2]
            current_signal = df['signal'].iloc[-1]
            
            if last_signal != current_signal:
                balance = exchange.fetch_balance()['USDT']['free']
                current_price = df['close'].iloc[-1]
                position_size = calculate_position_size(balance, risk_per_trade, current_price, stop_loss_percent)
                
                if current_signal == 1 and current_position <= 0:
                    print("Buy signal")
                    place_order('buy', position_size, symbol, stop_loss_percent)
                    current_position = 1
                elif current_signal == -1 and current_position >= 0:
                    print("Sell signal")
                    place_order('sell', position_size, symbol, stop_loss_percent)
                    current_position = -1
            
            time.sleep(3600)  # Wait for 1 hour before next iteration
        
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(60)  # Wait for 1 minute before retrying

if __name__ == "__main__":
    run_bot()

