import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout, LSTM, Bidirectional, Input, MultiHeadAttention, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait)
        ait = tf.expand_dims(ait, axis=-1)
        output = x * ait
        return tf.reduce_sum(output, axis=1)

def build_transformer_block(inputs, num_heads, d_model, ff_dim):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attention_output = Dropout(0.1)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(0.1)(ffn_output)
    
    # Ensure dimensions match before adding
    if out1.shape[-1] != ffn_output.shape[-1]:
        ffn_output = Dense(out1.shape[-1])(ffn_output)
    
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

def build_model(input_shape, num_heads, d_model, ff_dim, num_transformer_blocks):
    inputs = Input(shape=input_shape)
    
    x = inputs

    # Adding multiple LSTM and Bidirectional LSTM layers with Dropout
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    
    # Adding Transformer blocks
    for _ in range(num_transformer_blocks):
        x = build_transformer_block(x, num_heads, d_model, ff_dim)
    
    # Adding Attention layer
    x = AttentionLayer()(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Using an adaptive learning rate
    initial_learning_rate = 0.001
    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True
    )
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])
    return model


