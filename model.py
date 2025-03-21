import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

from py_libraries.ml.model import TransformerForecaster
from py_libraries.ml.optimizer.schedule import Noam

@tf.keras.utils.register_keras_serializable()
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

@tf.keras.utils.register_keras_serializable()
class CustomCandleLoss(tf.keras.losses.Loss):
    def __init__(self, 
                 penalty_direction_weight=1.0, 
                 penalty_open_weight=1.0, 
                 penalty_close_weight=1.0,
                 penalty_size_weight=1.0, 
                 penalty_body_weight=1.0, 
                 name="custom_candle_loss", **kwargs):
        """
        Initializes the custom candle loss.

        Args:
          penalty_direction_weight: Multiplier for the direction mismatch penalty.
          penalty_open_weight: Multiplier for the open error penalty (applied always).
          penalty_close_weight: Multiplier for the close error penalty (applied always).
          penalty_size_weight: Multiplier for the penalty on the difference in overall candle size (high - low).
          penalty_body_weight: Multiplier for the penalty on the difference in candle body size (abs(open - close)).
          name: Name for the loss function.
          **kwargs: Additional keyword arguments.
        """
        super(CustomCandleLoss, self).__init__(name=name, **kwargs)
        self.penalty_direction_weight = penalty_direction_weight
        self.penalty_open_weight = penalty_open_weight
        self.penalty_close_weight = penalty_close_weight
        self.penalty_size_weight = penalty_size_weight
        self.penalty_body_weight = penalty_body_weight

    def call(self, y_true, y_pred, sample_weight=None):
        """
        Computes the loss as the sum of several components on an elementwise (per timestep) basis.
        This version assumes that y_true and y_pred have shape (batch, T, features) where the first 4 features 
        correspond to [open, high, low, close]. Optionally, sample_weight (of shape (batch, T)) can be provided to 
        mask out padded timesteps.

        The loss components are:
          - MSE over the four candlestick features.
          - Ordering penalties ensuring predicted high is the max and predicted low is the min.
          - A penalty for overall direction mismatch.
          - A penalty on the open error.
          - A penalty on the close error.
          - A penalty for the difference in overall candle size (high - low).
          - A penalty for the difference in candle body size (abs(open - close)).
        """
        # --- Compute elementwise MSE over open, high, low, close ---
        mse_elem = tf.square(y_true[..., :4] - y_pred[..., :4])  # shape: (batch, T, 4)
        mse = tf.reduce_mean(mse_elem, axis=-1)  # shape: (batch, T)

        # --- Ordering penalties ---
        prices_pred = y_pred[..., :4]  # [open, high, low, close]
        max_price_pred = tf.reduce_max(prices_pred, axis=-1)  # shape: (batch, T)
        min_price_pred = tf.reduce_min(prices_pred, axis=-1)  # shape: (batch, T)
        penalty_high_elem = tf.square(max_price_pred - y_pred[..., 1])
        penalty_low_elem  = tf.square(y_pred[..., 2] - min_price_pred)
        penalty_order_elem = penalty_high_elem + penalty_low_elem  # shape: (batch, T)

        # --- Directional penalty (categorical) ---
        true_direction = tf.sign(y_true[..., 3] - y_true[..., 0])  # (batch, T)
        pred_direction = tf.sign(y_pred[..., 3] - y_pred[..., 0])      # (batch, T)
        direction_mismatch = tf.cast(tf.not_equal(true_direction, pred_direction), tf.float32)  # (batch, T)
        penalty_direction_elem = self.penalty_direction_weight * direction_mismatch  # (batch, T)

        # --- Open and close error penalties (always applied) ---
        error_open_elem = tf.square(y_true[..., 0] - y_pred[..., 0])   # (batch, T)
        error_close_elem = tf.square(y_true[..., 3] - y_pred[..., 3])  # (batch, T)
        penalty_open_elem = self.penalty_open_weight * error_open_elem
        penalty_close_elem = self.penalty_close_weight * error_close_elem

        # --- Candle size penalty (high - low) ---
        true_size = y_true[..., 1] - y_true[..., 2]   # (batch, T)
        pred_size = y_pred[..., 1] - y_pred[..., 2]     # (batch, T)
        penalty_size_elem = self.penalty_size_weight * tf.square(true_size - pred_size)

        # --- Candle body penalty (abs(open - close)) ---
        true_body = tf.abs(y_true[..., 3] - y_true[..., 0])  # (batch, T)
        pred_body = tf.abs(y_pred[..., 3] - y_pred[..., 0])    # (batch, T)
        penalty_body_elem = self.penalty_body_weight * tf.square(true_body - pred_body)

        # --- Combine elementwise losses for each timestep ---
        loss_per_timestep = (
            mse +
            penalty_order_elem +
            penalty_direction_elem +
            penalty_open_elem +
            penalty_close_elem +
            penalty_size_elem +
            penalty_body_elem
        )  # shape: (batch, T)

        # --- If sample_weight (mask) is provided, compute the masked average loss ---
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, loss_per_timestep.dtype)  # shape: (batch, T)
            total_loss = tf.reduce_sum(loss_per_timestep * sample_weight)
            normalizer = tf.reduce_sum(sample_weight) + 1e-8
            loss = total_loss / normalizer
        else:
            loss = tf.reduce_mean(loss_per_timestep)

        return loss

    def get_config(self):
        config = super(CustomCandleLoss, self).get_config()
        config.update({
            "penalty_direction_weight": self.penalty_direction_weight,
            "penalty_open_weight": self.penalty_open_weight,
            "penalty_close_weight": self.penalty_close_weight,
            "penalty_size_weight": self.penalty_size_weight,
            "penalty_body_weight": self.penalty_body_weight
        })
        return config
        
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
    # reduce_lr = ReduceLROnPlateau(
    #     monitor='val_loss',
    #     factor=0.5,
    #     patience=5,
    #     min_lr=1e-8,
    #     verbose=1
    # )

    return [early_stopping, model_checkpoint] # , reduce_lr

# Example usage for inference:
# start_token = tf.zeros((num_features,))  # Alternatively, use a learned start token.
# predicted_sequence = autoregressive_forecast(transformer, inputs_test, start_token, target_seq_len)
# print(predicted_sequence.shape)  # Expected: (batch, target_seq_len, num_features)
if __name__ == '__main__':
    import tensorflow as tf
    import joblib

    inputs_train    = joblib.load(f'./datas/split/30_5_None_rsi_macd_bollinger/9_07_03/inputs_train.pkl')
    inputs_val      = joblib.load(f'./datas/split/30_5_None_rsi_macd_bollinger/9_07_03/inputs_val.pkl')
    inputs_test     = joblib.load(f'./datas/split/30_5_None_rsi_macd_bollinger/9_07_03/inputs_test.pkl')
    outputs_train   = joblib.load(f'./datas/split/30_5_None_rsi_macd_bollinger/9_07_03/outputs_train.pkl')
    outputs_val     = joblib.load(f'./datas/split/30_5_None_rsi_macd_bollinger/9_07_03/outputs_val.pkl')
    outputs_test    = joblib.load(f'./datas/split/30_5_None_rsi_macd_bollinger/9_07_03/outputs_test.pkl')


    # ----------------------------
    # Hyperparameters
    # ----------------------------
    num_layers      = 4    # Increased depth for higher model capacity
    d_model         = 128  # Embedding dimension
    num_heads       = 8    # Number of attention heads
    dff             = 512  # Feed-forward network inner dimension
    dropout_rate    = 0.1

    max_input_seq_len   = 100   # Number of past candlesticks
    max_target_seq_len  = 30    # Number of future candlesticks to predict
    num_features        = outputs_train.shape[-1]    # Candlestick features: open, high, low, close, volume

    # ----------------------------
    # Instantiate & Compile the Model
    # ----------------------------
    transformer = TransformerForecaster(num_layers, d_model, num_heads, dff,
                                        max_input_seq_len, max_target_seq_len, num_features,
                                        dropout_rate)

    # Use the custom Noam learning rate schedule
    learning_rate = Noam(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate) # beta_1=0.9, beta_2=0.98, epsilon=1e-9
    transformer.compile(optimizer=optimizer, loss=CustomCandleLoss(penalty_direction_weight=2.0),
                        metrics=[
                            'mse',
                            tf.keras.metrics.MeanAbsoluteError(name='mae'),
                            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                            DirectionalAccuracy(name='directional_accuracy')
                        ])


    transformer.summary()

    # Define the number of features (here 5: open, high, low, close, volume)
    # num_features = outputs_train.shape[-1]

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
        sample_weight=mask,                     # Masking sequence padding (0 if padded, 1 otherwise)
        validation_data=((inputs_val, decoder_input_val), outputs_val),
        epochs=10, #200,
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
