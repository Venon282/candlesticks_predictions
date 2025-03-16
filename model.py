import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os
from tensorflow import keras

# ----------------------------
# Positional Encoding
# ----------------------------

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_encoding = self.compute_positional_encoding(seq_len, d_model)

    def compute_positional_encoding(self, seq_len, d_model):
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "seq_len": self.seq_len,
            "d_model": self.d_model
        })
        return config

# ----------------------------
# Custom Learning Rate Schedule (Noam schedule)
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super(CustomSchedule, self).__init__(**kwargs)
        # Store d_model as an integer for serialization
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        d_model = tf.cast(self.d_model, tf.float32)
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
        }


# ----------------------------
# Encoder Layer
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        attn_output = self.mha(x, x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate
        })
        return config


# ----------------------------
# Encoder (Projects continuous inputs via Dense)
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_seq_len, dropout_rate=0.1, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.input_seq_len = input_seq_len
        self.dropout_rate = dropout_rate

        # Project continuous candlestick data into a d_model–dimensional space.
        self.input_projection = Dense(d_model)
        self.pos_encoding = PositionalEncoding(input_seq_len, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        x = self.input_projection(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for layer in self.enc_layers:
            x = layer(x, training=training, mask=mask)
        return x

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "input_seq_len": self.input_seq_len,
            "dropout_rate": self.dropout_rate,
            # Assuming all EncoderLayers share same num_heads and dff
            "num_heads": self.enc_layers[0].num_heads if self.enc_layers else None,
            "dff": self.enc_layers[0].ffn.layers[0].units if self.enc_layers else None,
        })
        return config

# ----------------------------
# Decoder Layer
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        # Self-attention with look-ahead mask (for autoregressive decoding)
        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)
        # Encoder-decoder attention
        attn2 = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        # Feed-forward network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)

    def get_config(self):
        config = super(DecoderLayer, self).get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate
        })
        return config

# ----------------------------
# Decoder (For continuous outputs)
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_seq_len, dropout_rate=0.1, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.target_seq_len = target_seq_len
        self.dropout_rate = dropout_rate

        # Project target candlestick features into d_model dimensions.
        self.input_projection = Dense(d_model)
        self.pos_encoding = PositionalEncoding(target_seq_len, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rate)

    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        x = self.input_projection(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for layer in self.dec_layers:
            x = layer(x, enc_output, training=training,
                      look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
        return x

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "target_seq_len": self.target_seq_len,
            "dropout_rate": self.dropout_rate,
            "num_heads": self.dec_layers[0].num_heads if self.dec_layers else None,
            "dff": self.dec_layers[0].ffn.layers[0].units if self.dec_layers else None,
        })
        return config

# ----------------------------
# Transformer Forecaster (Encoder-Decoder Model)
# ----------------------------


# Assuming Encoder and Decoder are defined similarly in your model
# For example:
# from your_module import Encoder, Decoder
@tf.keras.utils.register_keras_serializable() #@tf.keras.utils.register_keras_serializable(package="Custom", name="TransformerForecaster")
class TransformerForecaster(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 input_seq_len, target_seq_len, num_features,
                 dropout_rate=0.1, **kwargs):
        # Pass extra keyword arguments to the parent class
        super(TransformerForecaster, self).__init__(**kwargs)

        # Save initialization parameters for serialization
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_seq_len = input_seq_len
        self.target_seq_len = target_seq_len
        self.num_features = num_features
        self.dropout_rate = dropout_rate

        # Build the model components (Encoder and Decoder must be defined accordingly)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_seq_len, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_seq_len, dropout_rate)
        self.final_layer = Dense(num_features)

    def call(self, inputs, training=False):
        # Expecting inputs as a tuple: (encoder_input, decoder_input)
        enc_input, dec_input = inputs
        enc_output = self.encoder(enc_input, training=training)
        # Create look-ahead mask based on decoder sequence length
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(dec_input)[1])
        dec_output = self.decoder(dec_input, enc_output, training=training, look_ahead_mask=look_ahead_mask)
        final_output = self.final_layer(dec_output)
        return final_output

    def create_look_ahead_mask(self, size):
        # Create a lower triangular matrix mask for causal decoding
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def get_config(self):
        # Retrieve the base configuration from the parent class
        base_config = super(TransformerForecaster, self).get_config()
        # Custom configuration for our parameters
        config = {
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "input_seq_len": self.input_seq_len,
            "target_seq_len": self.target_seq_len,
            "num_features": self.num_features,
            "dropout_rate": self.dropout_rate,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        # if "dtype" in config and isinstance(config["dtype"], str):
        #    config["dtype"] = tf.keras.mixed_precision.Policy(config["dtype"])
        if "dtype" in config and isinstance(config["dtype"], dict):
            d_config = config["dtype"]
            if d_config.get("class_name") == "DTypePolicy":
                policy_name = d_config.get("config", {}).get("name", None)
                if policy_name is not None:
                    config["dtype"] = tf.keras.mixed_precision.Policy(policy_name)
        return cls(**config)

class DirectionalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='directional_accuracy', **kwargs):
        super(DirectionalAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros', dtype=tf.float32)
        self.correct = self.add_weight(name='correct', initializer='zeros', dtype=tf.float32)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Extract the close price (assuming 4th feature: index 3)
        y_true_close = y_true[..., 3]
        y_pred_close = y_pred[..., 3]
        # Calculate differences between consecutive time steps along axis=1 (time axis)
        y_true_diff = y_true_close[:, 1:] - y_true_close[:, :-1]
        y_pred_diff = y_pred_close[:, 1:] - y_pred_close[:, :-1]
        # Compute the sign (direction) of these differences
        y_true_sign = tf.math.sign(y_true_diff)
        y_pred_sign = tf.math.sign(y_pred_diff)
        # Compare directional predictions: correct if both signs are equal
        correct = tf.cast(tf.equal(y_true_sign, y_pred_sign), tf.float32)
        # Update the counts
        self.correct.assign_add(tf.reduce_sum(correct))
        self.total.assign_add(tf.cast(tf.size(correct), tf.float32))
    
    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)
    
    def reset_states(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)
        
# ----------------------------
# Inference: Autoregressive Forecasting
# ----------------------------
@tf.function
def autoregressive_forecast(model, encoder_input, start_token, target_seq_len):
    """
    Generates predictions in an autoregressive manner using a compiled loop.

    Args:
      model: The trained TransformerForecaster.
      encoder_input: Past candlestick data of shape (batch, input_seq_len, num_features).
      start_token: A tensor of shape (num_features,) used as the initial decoder input.
      target_seq_len: The number of future steps to predict.

    Returns:
      A tensor of shape (batch, target_seq_len, num_features) representing the forecast.
    """
    batch_size = tf.shape(encoder_input)[0]
    # Create the initial decoder input by repeating the start token for each batch sample.
    decoder_input = tf.tile(tf.expand_dims(start_token, 0), [batch_size, 1, 1])
    
    # Initialize a TensorArray to collect predictions for each time step.
    predictions_array = tf.TensorArray(dtype=tf.float32, size=target_seq_len)

    # Define the loop condition: iterate until i == target_seq_len.
    def condition(i, decoder_input, predictions_array):
        return i < target_seq_len

    # Define the loop body: predict the next token and update the decoder input.
    def loop_body(i, decoder_input, predictions_array):
        # Generate predictions using the current decoder input.
        predictions = model((encoder_input, decoder_input), training=False)
        # Extract the last token prediction (shape: (batch, 1, num_features)).
        next_token = predictions[:, -1:, :]
        # Write the next token (squeezed to remove the time dimension) into the TensorArray.
        predictions_array = predictions_array.write(i, tf.squeeze(next_token, axis=1))
        # Append the next token to the decoder input.
        decoder_input = tf.concat([decoder_input, next_token], axis=1)
        return i + 1, decoder_input, predictions_array

    # Execute the while loop.
    i, decoder_input, predictions_array = tf.while_loop(
        condition,
        loop_body,
        loop_vars=[tf.constant(0), decoder_input, predictions_array]
    )
    
    # Stack the TensorArray and transpose to get shape: (batch, target_seq_len, num_features)
    predictions_stacked = predictions_array.stack()
    predictions_stacked = tf.transpose(predictions_stacked, perm=[1, 0, 2])
    
    return predictions_stacked

def callbacks():
    # Stop training if the validation loss doesn't improve for 10 epochs
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True,
        verbose=1
    )

    # Save the model weights when the validation loss improves
    model_checkpoint = ModelCheckpoint(
        filepath='saved_model/transformer_best.keras',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    # Reduce the learning rate if the validation loss plateaus
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-8,
        verbose=1
    )

    return [early_stopping, model_checkpoint, reduce_lr]

# Example usage for inference:
# start_token = tf.zeros((num_features,))  # Alternatively, use a learned start token.
# predicted_sequence = autoregressive_forecast(transformer, inputs_test, start_token, target_seq_len)
# print(predicted_sequence.shape)  # Expected: (batch, target_seq_len, num_features)
if __name__ == '__main__':
    import tensorflow as tf
    import joblib

    inputs_train = joblib.load(f'./datas/split/inputs_train.pkl')
    inputs_val = joblib.load(f'./datas/split/inputs_val.pkl')
    inputs_test = joblib.load(f'./datas/split/inputs_test.pkl')
    outputs_train = joblib.load(f'./datas/split/outputs_train.pkl')
    outputs_val = joblib.load(f'./datas/split/outputs_val.pkl')
    outputs_test = joblib.load(f'./datas/split/outputs_test.pkl')


    # ----------------------------
    # Hyperparameters
    # ----------------------------
    num_layers      = 4    # Increased depth for higher model capacity
    d_model         = 128  # Embedding dimension
    num_heads       = 8    # Number of attention heads
    dff             = 512  # Feed-forward network inner dimension
    dropout_rate    = 0.1

    input_seq_len   = 30   # Number of past candlesticks
    target_seq_len  = 5    # Number of future candlesticks to predict
    num_features    = 5    # Candlestick features: open, high, low, close, volume

    # ----------------------------
    # Instantiate & Compile the Model
    # ----------------------------
    transformer = TransformerForecaster(num_layers, d_model, num_heads, dff,
                                        input_seq_len, target_seq_len, num_features,
                                        dropout_rate)

    # Use the custom Noam learning rate schedule
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                        beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    transformer.compile(optimizer=optimizer, loss='mse',
                        metrics=[
                            'mse',
                            tf.keras.metrics.MeanAbsoluteError(name='mae'),
                            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                            DirectionalAccuracy(name='directional_accuracy')
                        ])

    # Build the model by running a dummy input through it
    dummy_encoder = tf.random.uniform((1, input_seq_len, num_features))
    dummy_decoder = tf.random.uniform((1, target_seq_len, num_features))
    _ = transformer((dummy_encoder, dummy_decoder), training=False)
    transformer.summary()

    # Define the number of features (here 5: open, high, low, close, volume)
    num_features = outputs_train.shape[-1]

    # Creating a start token for the decoder
    start_token = tf.zeros((num_features,), dtype=tf.float32)

    # Construction of the decoder input for training:
    # For each sequence of the target, we prefix with a start token and we shift (we remove the last element)
    decoder_input_train = tf.concat([
        tf.broadcast_to(tf.expand_dims(start_token, 0), [tf.shape(outputs_train)[0], 1, num_features]),
        outputs_train[:, :-1, :]
    ], axis=1)

    # Construction of the decoder input for validation
    decoder_input_val = tf.concat([
        tf.broadcast_to(tf.expand_dims(start_token, 0), [tf.shape(outputs_val)[0], 1, num_features]),
        outputs_val[:, :-1, :]
    ], axis=1)

    # Training the model with training and validation data
    history = transformer.fit(
        x=(inputs_train, decoder_input_train),  # Inputs: (encoder_input, decoder_input)
        y=outputs_train,                        # Target: sequence to predict
        validation_data=((inputs_val, decoder_input_val), outputs_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks(),
        verbose=2
    )

    # Preparing the decoder input for testing (similar to training)
    decoder_input_test = tf.concat([
        tf.broadcast_to(tf.expand_dims(start_token, 0), [tf.shape(outputs_test)[0], 1, num_features]),
        outputs_test[:, :-1, :]
    ], axis=1)

    # Evaluation on the test set
    test_loss = transformer.evaluate(
        x=(inputs_test, decoder_input_test),
        y=outputs_test,
        batch_size=64
    )
    print("Test Loss:", test_loss)

    # --- Saving the trained model ---
    os.makedirs('saved_model', exist_ok=True)
    transformer.save('saved_model/transformer_forecaster.keras')
