import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def compute_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, window, k=2):
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + k * std
    lower_band = sma - k * std
    return sma, upper_band, lower_band

def compute_cci(df, window):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(window=window).mean()
    md = tp.rolling(window=window).apply(lambda x: np.fabs(x - x.mean()).mean())
    cci = (tp - ma) / (0.015 * md)
    return cci

def calculate_technical_indicators(df, seq_length, window_sma, window_ema, window_rsi, window_bb=20, k_bb=2, window_cci=20):
    min_window = max(1, int(seq_length * 0.1))
    window_sma = min(window_sma, min_window)
    window_ema = min(window_ema, min_window)
    window_rsi = min(window_rsi, min_window)
    window_bb = min(window_bb, min_window)
    window_cci = min(window_cci, min_window)

    df['SMA'] = df['Close'].rolling(window=window_sma).mean()
    df['EMA'] = df['Close'].ewm(span=window_ema, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'], window=window_rsi)
    df['BB_Middle'], df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'], window=window_bb, k=k_bb)
    df['CCI'] = compute_cci(df, window=window_cci)
    
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    return df

def create_sequences(data, seq_length, pred_window, include_technical_indicators, window_sma, window_ema, window_rsi, window_bb, k_bb, window_cci):
    X, y = [], []

    min_seq_length = seq_length if not include_technical_indicators else max(seq_length, window_sma, window_ema, window_rsi, window_bb, window_cci)

    for i in tqdm(range(0, len(data) - min_seq_length - pred_window, 20), desc='Processing sequences'):
        sequence = data[i:i+min_seq_length]

        if include_technical_indicators:
            df = pd.DataFrame(sequence, columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Number of Trades', 'Taker Buy Base Asset Volume'])
            df = calculate_technical_indicators(df, seq_length, window_sma, window_ema, window_rsi, window_bb, k_bb, window_cci)
            sequence_with_indicators = df.values
            if len(sequence_with_indicators) == min_seq_length:
                current_close = data[i+seq_length-1][3]  
                future_close = data[i+seq_length+pred_window-1][3] 
                X.append(sequence_with_indicators[:seq_length])  
                y.append(1 if future_close > current_close else 0)
        else:
            if len(sequence) == seq_length:  
                current_close = data[i+seq_length-1][3]  
                future_close = data[i+seq_length+pred_window-1][3] 
                X.append(sequence)  
                y.append(1 if future_close > current_close else 0)

    return np.array(X), np.array(y)

def preprocess_data(data, config):
    seq_length = config['model']['sequence_length']
    pred_window = config['model']['prediction_window']
    include_technical_indicators = config['technical_indicators']['include']
    window_sma = config['technical_indicators']['window_sma']
    window_ema = config['technical_indicators']['window_ema']
    window_rsi = config['technical_indicators']['window_rsi']
    window_bb = config['technical_indicators']['window_bb']
    k_bb = config['technical_indicators']['k_bb']
    window_cci = config['technical_indicators']['window_cci']

    all_X, all_y = [], []

    scaler = MinMaxScaler()

    for coin, df in data.items():
        print(f'Processing coin: {coin}')
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Number of Trades', 'Taker Buy Base Asset Volume']].values
        X, y = create_sequences(df, seq_length, pred_window, include_technical_indicators, window_sma, window_ema, window_rsi, window_bb, k_bb, window_cci)

        if len(X) == 0:
            print(f'No valid sequences for {coin}')
            continue

        X_shape = X.shape
        X = X.reshape(-1, X_shape[-1])
        X = scaler.fit_transform(X)
        X = X.reshape(X_shape)

        all_X.append(X)
        all_y.append(y)

    if len(all_X) == 0:
        raise ValueError('No valid sequences found for any coin.')

    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)

    return X_train, y_train
