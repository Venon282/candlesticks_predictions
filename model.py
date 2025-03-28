import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #type: ignore
import os
import numpy as np
from pathlib import Path
import re

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
        y_true_close = tf.cast(y_true[..., 3], tf.float32)
        y_pred_close = tf.cast(y_pred[..., 3], tf.float32)
        # Calculate differences between consecutive time steps along axis=1 (time axis)
        y_true_diff = y_true_close[:, 1:] - y_true_close[:, :-1]
        y_pred_diff = y_pred_close[:, 1:] - y_pred_close[:, :-1]
        # Compute the sign (direction) of these differences
        y_true_sign = tf.math.sign(y_true_diff)
        y_pred_sign = tf.math.sign(y_pred_diff)
        # Compare directional predictions: correct if both signs are equal
        correct = tf.cast(tf.equal(y_true_sign, y_pred_sign), tf.float32)
        # Update the counts
        if sample_weight is not None:
            # Ensure sample_weight is cast to correct type and adjust shape: drop first timestep.
            sample_weight = tf.cast(sample_weight, correct.dtype)
            sample_weight = sample_weight[:, 1:]
            correct *= sample_weight
            self.total.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.total.assign_add(tf.cast(tf.size(correct), tf.float32))
            
        self.correct.assign_add(tf.reduce_sum(correct))
    
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


if __name__ == '__main__':
    import tensorflow as tf
    import joblib

    save_folder_path = Path(r'./saved_model')
    split_folder_path = Path(r'./datas/split')
    split_datas_str = '(5-100)_(1-30)_stepNone_scTrue_rsi_macd_bollinger'
    split_str = '7_15_15'
    model_file_name = 'transformer.keras'
    
    split_path = split_folder_path / split_datas_str / split_str
    save_path  = save_folder_path  / split_datas_str / split_str
    
    
    # Load datas
    inputs_train    = joblib.load(split_path / 'inputs_train.pkl' , mmap_mode='r')
    inputs_val      = joblib.load(split_path / 'inputs_val.pkl'   , mmap_mode='r')
    inputs_test     = joblib.load(split_path / 'inputs_test.pkl'  , mmap_mode='r')
    outputs_train   = joblib.load(split_path / 'outputs_train.pkl', mmap_mode='r')
    outputs_val     = joblib.load(split_path / 'outputs_val.pkl'  , mmap_mode='r')
    outputs_test    = joblib.load(split_path / 'outputs_test.pkl' , mmap_mode='r')
    
    # Load mask
    inputs_mask_train    = joblib.load(split_path / 'inputs_mask_train.pkl' , mmap_mode='r')
    inputs_mask_val      = joblib.load(split_path / 'inputs_mask_val.pkl'   , mmap_mode='r')
    inputs_mask_test     = joblib.load(split_path / 'inputs_mask_test.pkl'  , mmap_mode='r')
    outputs_mask_train   = joblib.load(split_path / 'outputs_mask_train.pkl', mmap_mode='r')
    outputs_mask_val     = joblib.load(split_path / 'outputs_mask_val.pkl'  , mmap_mode='r')
    outputs_mask_test    = joblib.load(split_path / 'outputs_mask_test.pkl' , mmap_mode='r')

    # ----------------------------
    # Hyperparameters
    # ----------------------------
    num_layers      = 1 #4    # Increased depth for higher model capacity
    d_model         = 1 #128  # Embedding dimension
    num_heads       = 1 #8    # Number of attention heads
    dff             = 1 #512  # Feed-forward network inner dimension
    dropout_rate    = 0.1
    batch_size      = 64
    es_patience     = 10
    
    # Implement id model
    id = f'nl{num_layers}_dm{d_model}_nh{num_heads}_dff{dff}_dr{dropout_rate}_bs{batch_size}_esp{es_patience}'.replace('.', ',')
    save_path = save_path / id
    save_path.mkdir(parents=True, exist_ok=True)

    # Get datas bounds
    bounds_pattern = r'\((\d+)-(\d+)\)_\((\d+)-(\d+)\)_'
    match = re.search(bounds_pattern, split_datas_str)
    [input_seq_len_min, input_seq_len_max, target_seq_len_min, target_seq_len_max] = [int(x) for x in match.groups()]
    
    num_features        = outputs_train.shape[-1]    # Candlestick features: open, high, low, close, volume
    
    # Display infos
    print(f'{num_features=}')
    print(f'{inputs_train.shape=}')
    print(f'{inputs_val.shape=}')
    print(f'{inputs_test.shape=}')
    print(f'{outputs_train.shape=}')
    print(f'{outputs_val.shape=}')
    print(f'{outputs_test.shape=}')

    # ----------------------------
    # Instantiate & Compile the Model
    # ----------------------------
    transformer = TransformerForecaster(num_layers, d_model, num_heads, dff,
                                        input_seq_len_max, target_seq_len_max, num_features,
                                        dropout_rate)
    transformer.summary()

    # Use the custom Noam learning rate schedule
    learning_rate = Noam(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    transformer.compile(optimizer=optimizer, loss=CustomCandleLoss(penalty_direction_weight=2.0),
                        metrics=[
                            'mse',
                            tf.keras.metrics.MeanAbsoluteError(name='mae'),
                            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                            DirectionalAccuracy(name='directional_accuracy')
                        ],
                        weighted_metrics=[DirectionalAccuracy(name='directional_accuracy')])

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
    
    
    train_dataset = tf.data.Dataset.from_tensor_slices((
        (np.array(inputs_train), np.array(decoder_input_train), {"encoder": np.array(inputs_mask_train), "decoder": np.array(outputs_mask_train)}),
        np.array(outputs_train), np.array(outputs_mask_train)
    )).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        (np.array(inputs_val), np.array(decoder_input_val), {"encoder": np.array(inputs_mask_val), "decoder": np.array(outputs_mask_val)}),
        np.array(outputs_val), np.array(outputs_mask_val)
    )).batch(batch_size)
    
    callbacks = [
        # Stop training if the validation loss doesn't improve for 10 epochs
        EarlyStopping(
            monitor='val_loss', 
            patience=es_patience, 
            restore_best_weights=True,
            verbose=1
        ),

        # Save the model weights when the validation loss improves
        ModelCheckpoint(
            filepath=save_path / model_file_name,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]

    # Training the model with training and validation data
    history = transformer.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=2, #200,
        callbacks=callbacks,
        verbose=2
    )
    
    print('Evaluate')
    test_loss = transformer.evaluate(
        x=(inputs_test, {'encoder': inputs_mask_test, 'decoder': outputs_mask_test }),
        y=outputs_test,
        batch_size=batch_size,
        verbose=1
    )

    
    