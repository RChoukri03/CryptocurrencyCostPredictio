import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from common.model import build_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.dataLoader import  load_config, DataLoader

def get_binance_data(symbol, interval, lookback):
    url = f'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': lookback
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
    df[['Open', 'High', 'Low', 'Close', 'Volume', 'Number of Trades', 'Taker Buy Base Asset Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Number of Trades', 'Taker Buy Base Asset Volume']].astype(float)
    return df

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

def preprocess_and_evaluate(symbol, model, config):
    interval = '1m'
    lookback = 1440  # Number of minutes in 24 hours

    # Fetch data
    df = get_binance_data(symbol, interval, lookback)
    
    # Preprocess data
    data = {symbol: df}
    data_loader = DataLoader(data, config)
    
    # Generate sequences
    X, y_true = zip(*list(data_loader.create_sequences(df)))
    X = np.array(X)
    y_true = np.array(y_true)
    print(X.shape)
    
    # Predict
    probabilities = model.predict(X)
    y_pred = (probabilities > 0.5).astype(int)
    
    # Evaluate
    accuracy, precision, recall, f1 = evaluate_model(y_true, y_pred)
    print(f'Evaluation for {symbol}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Load the trained model
    model = tf.keras.models.load_model('crypto_trading_model.h5')
    
    # Symbol of the cryptocurrency to evaluate
    symbol = 'ETHUSDT'
    
    # Evaluate model on the fetched data
    preprocess_and_evaluate(symbol, model, config)
