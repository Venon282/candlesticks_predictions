import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #type: ignore
import os
import numpy as np
from pathlib import Path
import re
import joblib

from py_libraries.ml.model import TransformerForecaster
from py_libraries.ml.optimizer.schedule import Noam

@tf.keras.utils.register_keras_serializable()
class DirectionalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='candle_direction_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total   = self.add_weight(name='total',   initializer='zeros', dtype=tf.float32)
        self.correct = self.add_weight(name='correct', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # assume feature 0 = open, feature 3 = close
        true_open  = tf.cast(y_true[..., 0], tf.float32)
        true_close = tf.cast(y_true[..., 3], tf.float32)
        pred_open  = tf.cast(y_pred[..., 0], tf.float32)
        pred_close = tf.cast(y_pred[..., 3], tf.float32)

        # +1 => bullish, -1 => bearish (0 => flat candle, counts as mismatch by default)
        true_dir = tf.sign(true_close - true_open)
        pred_dir = tf.sign(pred_close - pred_open)

        # compare across all batch × time steps
        correct = tf.cast(tf.equal(true_dir, pred_dir), tf.float32)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, correct.dtype)
            # broadcast/resize if necessary:
            #sample_weight = tf.broadcast_to(sample_weight, tf.shape(correct))
            correct *= sample_weight
            self.total.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.total.assign_add(tf.cast(tf.size(correct), tf.float32))

        self.correct.assign_add(tf.reduce_sum(correct))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_states(self):
        self.total.assign(0.)
        self.correct.assign(0.)

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
        super(CustomCandleLoss, self).__init__(name=name, reduction=tf.keras.losses.Reduction.NONE, **kwargs) 
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
        mse_elem = tf.square(y_true - y_pred)  # shape: (batch, T, n_features)
        mse = tf.reduce_mean(mse_elem, axis=-1)  # shape: (batch, T)

        # --- Ordering penalties ---
        prices_pred = y_pred[..., :4]  # [open, high, low, close]
        max_price_pred = tf.reduce_max(prices_pred, axis=-1)  # shape: (batch, T)
        min_price_pred = tf.reduce_min(prices_pred, axis=-1)  # shape: (batch, T)
        penalty_high_elem = tf.square(max_price_pred - y_pred[..., 1])
        penalty_low_elem  = tf.square(y_pred[..., 2] - min_price_pred)
        penalty_order_elem = penalty_high_elem + penalty_low_elem  # shape: (batch, T)

        # --- Directional penalty (categorical) ---
        # true_direction = tf.sign(y_true[..., 3] - y_true[..., 0])  # (batch, T)
        # pred_direction = tf.sign(y_pred[..., 3] - y_pred[..., 0])      # (batch, T)
        # direction_mismatch = tf.cast(tf.not_equal(true_direction, pred_direction), tf.float32)  # (batch, T)
        # penalty_direction_elem = self.penalty_direction_weight * direction_mismatch  # (batch, T)
        true_direction = tf.sign(y_true[...,3] - y_true[...,0])
        pred_margin    = y_pred[...,3] - y_pred[...,0]
        direction_loss  = tf.nn.relu(- true_direction * pred_margin) # 0 if true and pred same direction else margin value
        penalty_direction_elem = self.penalty_direction_weight * direction_loss

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
            total_loss = tf.reduce_sum(loss_per_timestep * sample_weight, axis=1)         # (batch,)
            normalizer = tf.reduce_sum(sample_weight, axis=1) + 1e-8             # (batch,)
            loss = total_loss / normalizer
        else:
            loss = tf.reduce_mean(loss_per_timestep, axis=1)              # (batch,)

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
    
def model(save_folder_path, split_folder_path, # paths
          split_datas_str, split_str, model_file_name, # datas
          num_layers      = 4,    # Number of layers in the encoder and decoder (BERT ou GPT-2 utilisent généralement de 12 à 24 couches)
          d_model         = 128,  # Latent space size (BERT de taille "base" utilise 768, et BERT "large" utilise 1024.)
          num_heads       = 8,    # Number of attention heads (d_model = 768 est num_heads = 12)
          dff             = 512,  # Feed-forward network inner dimension. The larger the dimension of dff, the more complex nonlinear transformations the model can learn. (2048 et 4096)
          dropout_rate    = 0.1,
          batch_size      = 32,
          epochs          = 50,
          warmup_rate     = 0.1,
          es_patience_rate= 0.1,
          penalty_direction_weight  = 1.0, 
          penalty_open_weight       = 1.0, 
          penalty_close_weight      = 1.0,
          penalty_size_weight       = 1.0, 
          penalty_body_weight       = 1.0
        ):
    save_folder_path, split_folder_path = Path(save_folder_path), Path(split_folder_path)
    
    split_path = split_folder_path / split_datas_str / split_str
    save_path  = save_folder_path  / split_datas_str / split_str
    
    N = 1000
    
    # Load datas
    inputs_train    = joblib.load(split_path / 'inputs_train.pkl' , mmap_mode='r')[:N]
    inputs_val      = joblib.load(split_path / 'inputs_val.pkl'   , mmap_mode='r')[:N]
    inputs_test     = joblib.load(split_path / 'inputs_test.pkl'  , mmap_mode='r')[:N]
    outputs_train   = joblib.load(split_path / 'outputs_train.pkl', mmap_mode='r')[:N]
    outputs_val     = joblib.load(split_path / 'outputs_val.pkl'  , mmap_mode='r')[:N]
    outputs_test    = joblib.load(split_path / 'outputs_test.pkl' , mmap_mode='r')[:N]
    
    # Load mask
    inputs_mask_train    = joblib.load(split_path / 'inputs_mask_train.pkl' , mmap_mode='r')[:N]
    inputs_mask_val      = joblib.load(split_path / 'inputs_mask_val.pkl'   , mmap_mode='r')[:N]
    inputs_mask_test     = joblib.load(split_path / 'inputs_mask_test.pkl'  , mmap_mode='r')[:N]
    outputs_mask_train   = joblib.load(split_path / 'outputs_mask_train.pkl', mmap_mode='r')[:N]
    outputs_mask_val     = joblib.load(split_path / 'outputs_mask_val.pkl'  , mmap_mode='r')[:N]
    outputs_mask_test    = joblib.load(split_path / 'outputs_mask_test.pkl' , mmap_mode='r')[:N]

    # ----------------------------
    # Hyperparameters
    # ----------------------------
    es_patience     = max(10, int(epochs * es_patience_rate))
    num_steps       = int(len(inputs_train) / batch_size)
    total_steps     = epochs * num_steps
    warmup_steps    = int(total_steps * warmup_rate)
    num_features    = outputs_train.shape[-1]
    
    # Implement id model
    id_ = f'nl{num_layers}_dm{d_model}_nh{num_heads}_dff{dff}_dr{dropout_rate}_bs{batch_size}_e{epochs}_esp{es_patience}_wu{warmup_steps}_pdw{penalty_direction_weight}_pow{penalty_open_weight}_pcw{penalty_close_weight}_psw{penalty_size_weight}_pbw{penalty_body_weight}'.replace('.', ',')
    
    save_path = save_path / id_
    save_path.mkdir(parents=True, exist_ok=True)

    # Get datas bounds
    bounds_pattern = r'\((\d+)-(\d+)\)_\((\d+)-(\d+)\)_'
    match = re.search(bounds_pattern, split_datas_str)
    [input_seq_len_min, input_seq_len_max, target_seq_len_min, target_seq_len_max] = [int(x) for x in match.groups()]
    
    # Display infos
    print(f'{id_=}')
    print(f'{num_features=}')
    print(f'{inputs_train.shape=}')
    print(f'{inputs_val.shape=}')
    print(f'{inputs_test.shape=}')
    print(f'{outputs_train.shape=}')
    print(f'{outputs_val.shape=}')
    print(f'{outputs_test.shape=}')
    print(f'{outputs_mask_train.shape=}')
    print(f'{outputs_mask_val.shape=}')
    print(f'{outputs_mask_test.shape=}')
    print(f'{num_layers=}')  
    print(f'{d_model=}')     
    print(f'{num_heads=}')   
    print(f'{dff=}')         
    print(f'{dropout_rate=}')
    print(f'{es_patience=}') 
    print(f'{batch_size=}')  
    print(f'{epochs=}')      
    print(f'{warmup_rate=}') 
    print(f'{num_steps=}')   
    print(f'{total_steps=}') 
    print(f'{warmup_steps=}')
    print(f'{num_features=}')
    print(f'{penalty_direction_weight=}')  
    print(f'{penalty_open_weight=}')       
    print(f'{penalty_close_weight=}')      
    print(f'{penalty_size_weight=}')       
    print(f'{penalty_body_weight=}')       

    # ----------------------------
    # Instantiate & Compile the Model
    # ----------------------------
    transformer = TransformerForecaster(num_layers, d_model, num_heads, dff,
                                        input_seq_len_max, target_seq_len_max, num_features,
                                        dropout_rate)
    # transformer.summary()
    # transformer.print_layers()

    # Use the custom Noam learning rate schedule
    learning_rate = Noam(d_model, warmup_steps=warmup_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    transformer.compile(optimizer=optimizer, loss=CustomCandleLoss(
                                                    penalty_direction_weight  = penalty_direction_weight, 
                                                    penalty_open_weight       = penalty_open_weight, 
                                                    penalty_close_weight      = penalty_close_weight,
                                                    penalty_size_weight       = penalty_size_weight, 
                                                    penalty_body_weight       = penalty_body_weight
                                            ),
                        metrics=[
                            'mse',
                            tf.keras.metrics.MeanAbsoluteError(name='mae'),
                            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                            DirectionalAccuracy(name='directional_accuracy_normal')
                        ],
                        weighted_metrics=[DirectionalAccuracy(name='directional_accuracy')])

    decoder_input_train = outputs_train
    decoder_input_val = outputs_val
    
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
    
    x_train = {
        'encoder_input':inputs_train, 
        'decoder_input':decoder_input_train, 
        'mask':{"encoder": inputs_mask_train, "decoder": outputs_mask_train}
    }
    x_val = {
        'encoder_input':inputs_val, 
        'decoder_input':decoder_input_val, 
        'mask':{"encoder": inputs_mask_val, "decoder": outputs_mask_val}
    }
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, outputs_train, outputs_mask_train)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    dataset_val = tf.data.Dataset.from_tensor_slices((x_val, outputs_val, outputs_mask_val)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    

    # Training the model with training and validation data
    history = transformer.fit(
        dataset_train,
        #sample_weight = np.array(outputs_mask_train),
        validation_data=dataset_val,
        epochs=epochs, 
        callbacks=callbacks,
        verbose=2
    )
    
    print('Evaluate:')
    test_loss = transformer.evaluate(
        x={'encoder_input':inputs_test, 'mask':{'encoder': inputs_mask_test, 'decoder': outputs_mask_test }},
        y=outputs_test,
        batch_size=batch_size,
        verbose=2
    )
    
    print('Decoder output')
    print(outputs_mask_test[0])
    
    print('Prediction:')
    pred = transformer(encoder_input=inputs_test[0], mask={'encoder': inputs_mask_test[0], 'decoder': outputs_mask_test[0]})
    print(pred)
    
    print('True:')
    print(outputs_test[0])
    
if __name__ == '__main__':
    tf.keras.backend.clear_session()
    # for split_datas_str in  [
    #                             '(40-40)_(1-1)_step=1_sc=True_drop(date_time_tickvol_spread)',
    #                             '(40-40)_(1-1)_step=1_sc=True_drop(date_time_tickvol_vol)',
    #                             '(40-40)_(1-1)_step=1_sc=True_drop(date_time_tickvol_vol_spread)',
    #                             '(40-40)_(1-1)_step=1_sc=True_drop(date_time_vol_spread)',
    #                             '(40-40)_(1-1)_step=1_sc=True_bollinger_drop(date_time_tickvol_vol_spread)',
    #                             '(40-40)_(1-1)_step=1_sc=True_rsi_drop(date_time_tickvol_vol_spread)',
    #                             '(40-40)_(1-1)_step=1_sc=True_macd_drop(date_time_tickvol_vol_spread)',
    #                             '(40-40)_(1-1)_step=1_sc=True_macd_bollinger_drop(date_time_tickvol_vol_spread)',
    #                             '(40-40)_(1-1)_step=1_sc=True_rsi_bollinger_drop(date_time_tickvol_vol_spread)',
    #                             '(40-40)_(1-1)_step=1_sc=True_rsi_macd_drop(date_time_tickvol_vol_spread)'
    #                             '(40-40)_(1-1)_step=1_sc=True_rsi_macd_bollinger_drop(date_time_tickvol_vol_spread)',
    #                         ]:
    #print(f'{epochs=} {split_datas_str=}')
    
    model(
        save_folder_path = r'E:\csp\saved_model',
        split_folder_path = r'E:\csp\split',
        
        split_datas_str = '(40-40)_(1-1)_step=1_sc=True_drop(date_time_tickvol_vol_spread)_sgs',
        split_str = '935_05_015',
        model_file_name = 'transformer.keras',
        num_layers      = 1, #4    # Number of layers in the encoder and decoder (BERT ou GPT-2 utilisent généralement de 12 à 24 couches)
        d_model         = 1, #128  # Latent space size (BERT de taille "base" utilise 768, et BERT "large" utilise 1024.)
        num_heads       = 1, #8    # Number of attention heads (d_model = 768 est num_heads = 12)
        dff             = 1, #512  # Feed-forward network inner dimension. The larger the dimension of dff, the more complex nonlinear transformations the model can learn. (2048 et 4096)
        dropout_rate    = 0.1,
        batch_size      = 64,
        epochs          = 2, #200
        warmup_rate     = 0.1,
        es_patience_rate= 0.1,
        penalty_direction_weight  = 3.0, 
        penalty_open_weight       = 0.5, 
        penalty_close_weight      = 0.5,
        penalty_size_weight       = 0.2, 
        penalty_body_weight       = 0.5
    )