import tensorflow as tf
import os
import numpy as np

class TerminateOnNaN(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if logs.get('loss') is not None and np.isnan(logs['loss']):
            print('NaN loss encountered. Stopping training.')
            self.model.stop_training = True

def get_callbacks(callbacks_config):
    callbacks = []

    for callback in callbacks_config:
        if 'ModelCheckpoint' in callback:
            params = callback['ModelCheckpoint']
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                filepath=params['filepath'],
                save_best_only=params['save_best_only'],
                monitor=params.get('monitor', 'val_loss'),
                mode=params.get('mode', 'min'),
                save_weights_only=params.get('save_weights_only', False),
                verbose=params.get('verbose', 1)
            ))
        elif 'EarlyStopping' in callback:
            params = callback['EarlyStopping']
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor=params.get('monitor', 'val_loss'),
                patience=params.get('patience', 20),
                verbose=params.get('verbose', 1)
            ))
        elif 'TensorBoard' in callback:
            params = callback['TensorBoard']
            log_dir = os.path.join(params['log_dir'])
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
        elif 'TerminateOnNaN' in callback:
            callbacks.append(TerminateOnNaN())
        else:
            raise ValueError(f"Unknown callback: {callback}")
    
    return callbacks
