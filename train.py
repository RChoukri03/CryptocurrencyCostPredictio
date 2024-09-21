import numpy as np
from utils.dataLoader import load_data, load_config,DataLoader
from common.model import build_model
from utils.callbacks import get_callbacks

if __name__ == "__main__":
    # Load configuration and data
    config = load_config()
    data = load_data(config)
    
    # Initialize DataLoader
    data_loader = DataLoader(data, config)
    
    # Define batch size and number of epochs
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['epochs']
    
    # Generate tf.data.Dataset with a specified number of epochs
    dataset = data_loader.generate_tf_dataset(batch_size, num_epochs)
    
    # Define input shape based on sequence length and dataset
    seq_length = config['model']['sequence_length']
    sample_X, _ = next(data_loader.create_sequences(next(iter(data.values()))))
    input_shape = (seq_length, sample_X.shape[-1])
    
    # Build the model
    model = build_model(input_shape, config['model']['num_heads'], config['model']['d_model'], config['model']['ff_dim'], config['model']['num_transformer_blocks'])
    model.summary()
    # Get callbacks from config
    callbacks_config = config['training']['callbacks']
    callbacks = get_callbacks(callbacks_config)

    # Calculate the steps per epoch
    steps_per_epoch = sum(1 for _ in data_loader.create_sequences(next(iter(data.values())))) // batch_size
    
    # Train the model
    model.fit(dataset, epochs=num_epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
    
    # Save the model
    model.save('crypto_ETH_trading_model.h5')
