import ccxt
import time
from datetime import datetime
import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Exchange and trading pair configuration
exchange = ccxt.binance({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future'
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
macd_fast = 12
macd_slow = 26
macd_signal = 9
atr_period = 14

def fetch_ohlcv_data(symbol, timeframe, limit):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_indicators(df):
    df['short_ma'] = df['close'].rolling(window=short_window).mean()
    df['long_ma'] = df['close'].rolling(window=long_window).mean()
    df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_period)
    macd, signal, _ = talib.MACD(df['close'], fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
    df['macd'] = macd
    df['macd_signal'] = signal
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
    df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = talib.BBANDS(df['close'], timeperiod=20)
    return df

def generate_features(df):
    df['ma_signal'] = np.where(df['short_ma'] > df['long_ma'], 1, -1)
    df['rsi_signal'] = np.where(df['rsi'] < rsi_oversold, 1, np.where(df['rsi'] > rsi_overbought, -1, 0))
    df['macd_signal'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
    df['bollinger_signal'] = np.where(df['close'] < df['bollinger_lower'], 1, np.where(df['close'] > df['bollinger_upper'], -1, 0))
    df['atr_percent'] = df['atr'] / df['close']
    return df

def train_model(df):
    features = ['ma_signal', 'rsi_signal', 'macd_signal', 'bollinger_signal', 'atr_percent']
    X = df[features]
    y = np.where(df['close'].shift(-1) > df['close'], 1, -1)  # Simple future returns as target
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled[:-1], y[:-1])  # Exclude the last row as we don't have a target for it
    
    joblib.dump(model, 'trading_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    return model, scaler

def predict_signal(model, scaler, current_features):
    scaled_features = scaler.transform([current_features])
    prediction = model.predict(scaled_features)
    return prediction[0]

def calculate_position_size(balance, risk_per_trade, current_price, atr):
    risk_amount = balance * risk_per_trade
    position_size = risk_amount / (atr * 2)  # Use 2 * ATR as initial stop loss
    return position_size

def place_order(side, amount, symbol, stop_loss_price, take_profit_price):
    try:
        order = exchange.create_market_order(symbol, side, amount)
        print(f"Order placed: {order}")
        
        # Place stop loss order
        stop_loss_order = exchange.create_stop_loss_order(symbol, 'sell' if side == 'buy' else 'buy', amount, stop_loss_price)
        print(f"Stop loss order placed: {stop_loss_order}")
        
        # Place take profit order
        take_profit_order = exchange.create_take_profit_order(symbol, 'sell' if side == 'buy' else 'buy', amount, take_profit_price)
        print(f"Take profit order placed: {take_profit_order}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

def run_bot():
    risk_per_trade = 0.01  # 1% risk per trade
    take_profit_atr_multiple = 3  # Take profit at 3 * ATR
    
    # Load or train the model
    try:
        model = joblib.load('trading_model.joblib')
        scaler = joblib.load('scaler.joblib')
        print("Loaded existing model and scaler")
    except:
        print("Training new model...")
        df = fetch_ohlcv_data(symbol, timeframe, 1000)  # Fetch more data for training
        df = calculate_indicators(df)
        df = generate_features(df)
        model, scaler = train_model(df)
        print("Model trained and saved")
    
    while True:
        try:
            print(f"Fetching data for {symbol} at {datetime.now()}")
            df = fetch_ohlcv_data(symbol, timeframe, 100)
            df = calculate_indicators(df)
            df = generate_features(df)
            
            current_features = df.iloc[-1][['ma_signal', 'rsi_signal', 'macd_signal', 'bollinger_signal', 'atr_percent']].values
            predicted_signal = predict_signal(model, scaler, current_features)
            
            balance = exchange.fetch_balance()['USDT']['free']
            current_price = df['close'].iloc[-1]
            current_atr = df['atr'].iloc[-1]
            
            position_size = calculate_position_size(balance, risk_per_trade, current_price, current_atr)
            stop_loss_price = current_price - (current_atr * 2) if predicted_signal == 1 else current_price + (current_atr * 2)
            take_profit_price = current_price + (current_atr * take_profit_atr_multiple) if predicted_signal == 1 else current_price - (current_atr * take_profit_atr_multiple)
            
            if predicted_signal == 1:
                print("Buy signal")
                place_order('buy', position_size, symbol, stop_loss_price, take_profit_price)
            elif predicted_signal == -1:
                print("Sell signal")
                place_order('sell', position_size, symbol, stop_loss_price, take_profit_price)
            else:
                print("No clear signal")
            
            time.sleep(3600)  # Wait for 1 hour before next iteration
        
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(60)  # Wait for 1 minute before retrying

if __name__ == "__main__":
    run_bot()
