import numpy as np
import tensorflow as tf
from utils.dataLoader import load_data, load_config
from utils.preprocessing import preprocess_data
from common.model import build_model
from common.performer import PerformerLayer

def create_sequences(data, seq_length, pred_window):
    X, y = [], []

    # Iterate over the dataset to create sequences
    for i in range(len(data) - seq_length - pred_window):
        # Extract the sequence of length `seq_length`
        sequence = data[i:i+seq_length]

        # Extract the closing price at the end of the sequence
        current_close = data[i+seq_length-1][3]  # Close price at the end of the sequence

        # Extract the closing price after the prediction window
        future_close = data[i+seq_length+pred_window-1][3]  # Close price after pred_window

        # Append the sequence and label
        X.append(sequence)
        y.append(1 if future_close > current_close else 0)

    return np.array(X)


if __name__ == "__main__":
    config = load_config()
    data = load_data(config)
    data = preprocess_data(data, config)
    
    seq_length = config['model']['sequence_length']
    pred_window = config['model']['prediction_window']
    
    model = tf.keras.models.load_model('crypto_trading_model.h5', custom_objects={'PerformerLayer': PerformerLayer})
    
    for coin, df in data.items():
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Number of Trades', 'Taker Buy Base Asset Volume']]
        X = create_sequences(df.values, seq_length, pred_window)
        probabilities = model.predict(X)
        predictions = (probabilities > 0.5).astype(int)  # Convert probabilities to binary labels
        print(f'Predictions for {coin}: {predictions}')