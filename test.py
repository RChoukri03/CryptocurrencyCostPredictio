from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import tensorflow as tf
from utils.dataLoader import load_data, load_config, DataLoader
from common.model import build_model

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    # Load configuration and data
    config = load_config()
    data = load_data(config)

    # Initialize DataLoader
    data_loader = DataLoader(data, config)
    
    # Define batch size and number of epochs
    batch_size = config['training']['batch_size']
    
    # Generate tf.data.Dataset
    dataset = data_loader.generate_tf_dataset(batch_size, 1) 
    # Define input shape based on sequence length and dataset
    seq_length = config['model']['sequence_length']
    sample_X, _ = next(data_loader.create_sequences(next(iter(data.values()))))
    input_shape = (seq_length, sample_X.shape[-1])
    
    # Load the trained model
    model = tf.keras.models.load_model('crypto_trading_model.h5')
    
    # Create lists to store true and predicted labels
    all_y_true, all_y_pred = [], []
    
    for coin, df in data.items():
        print(f'Evaluating coin: {coin}')
        
        # Preprocess the data for evaluation
        X, y_true = zip(*list(data_loader.create_sequences(df)))
        X = np.array(X)
        y_true = np.array(y_true)
        
        # Predict the probabilities
        probabilities = model.predict(X)
        y_pred = (probabilities > 0.5).astype(int)
        
        # Store true and predicted labels
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        
        # Evaluate model performance for the current coin
        accuracy, precision, recall, f1 = evaluate_model(y_true, y_pred)
        print(f'Evaluation for {coin}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')
    
    # Evaluate overall performance
    overall_accuracy, overall_precision, overall_recall, overall_f1 = evaluate_model(all_y_true, all_y_pred)
    print(f'Overall Evaluation: Accuracy={overall_accuracy:.4f}, Precision={overall_precision:.4f}, Recall={overall_recall:.4f}, F1={overall_f1:.4f}')
