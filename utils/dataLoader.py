import os
import pandas as pd
import yaml

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.preprocessing import calculate_technical_indicators

def load_config():
    with open('config.yml', 'r') as file:
        return yaml.safe_load(file)

def load_data(config):
    data_dir = config['data']['directory']
    included_coins = config['data']['included_coins']
    all_data = {}

    for folder_name in os.listdir(data_dir):
        coin_name = folder_name.replace("USDT", "")
        if included_coins == ['all'] or coin_name in included_coins:
            csv_files = [f for f in os.listdir(os.path.join(data_dir, folder_name)) if f.endswith('.csv')]
            df_list = [pd.read_csv(os.path.join(data_dir, folder_name, csv_file),sep=";") for csv_file in csv_files]
            all_data[coin_name] = pd.concat(df_list, ignore_index=True)
    
    return all_data



class DataLoader:
    def __init__(self, data, config):
        self.data = data
        self.config = config

    def create_sequences(self, data):
        seq_length = self.config['model']['sequence_length']
        pred_window = self.config['model']['prediction_window']
        include_technical_indicators = self.config['technical_indicators']['include']
        window_sma = self.config['technical_indicators']['window_sma']
        window_ema = self.config['technical_indicators']['window_ema']
        window_rsi = self.config['technical_indicators']['window_rsi']
        window_bb = self.config['technical_indicators']['window_bb']
        k_bb = self.config['technical_indicators']['k_bb']
        window_cci = self.config['technical_indicators']['window_cci']

        min_seq_length = seq_length if not include_technical_indicators else max(seq_length, window_sma, window_ema, window_rsi, window_bb, window_cci)
        scaler = MinMaxScaler()

        # Select only relevant columns
        data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Number of Trades', 'Taker Buy Base Asset Volume']]

        for i in range(0, len(data) - min_seq_length - pred_window, 20):
            sequence = data.iloc[i:i+min_seq_length]
            if include_technical_indicators:
                df = pd.DataFrame(sequence, columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Number of Trades', 'Taker Buy Base Asset Volume'])
                df = calculate_technical_indicators(df, seq_length, window_sma, window_ema, window_rsi, window_bb, k_bb, window_cci)
                sequence_with_indicators = df.values
                if len(sequence_with_indicators) == min_seq_length:
                    current_close = data.iloc[i+seq_length-1]['Close']
                    future_close = data.iloc[i+seq_length+pred_window-1]['Close']
                    X = sequence_with_indicators[:seq_length]
                    X = scaler.fit_transform(X)
                    y = 1 if future_close > current_close else 0
                    yield X, y
            else:
                if len(sequence) == seq_length:
                    current_close = data.iloc[i+seq_length-1]['Close']
                    future_close = data.iloc[i+seq_length+pred_window-1]['Close']
                    X = sequence.values
                    X = scaler.fit_transform(X)
                    y = 1 if future_close > current_close else 0
                    yield X, y

    def data_generator(self):
        for coin, df in self.data.items():
            print(f'Processing coin: {coin}')
            for X, y in self.create_sequences(df):
                yield X, y

    def generate_tf_dataset(self, batch_size, num_epochs):
        first_df = next(iter(self.data.values()))
        sample_X, _ = next(self.create_sequences(first_df))
        dataset = tf.data.Dataset.from_generator(
            self.data_generator,
            output_signature=(
                tf.TensorSpec(shape=(self.config['model']['sequence_length'], sample_X.shape[-1]), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            )
        )
        dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat(num_epochs)
        return dataset