import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

# Import custom objects from model.py if available to ensure proper model loading
try:
    from model import PositionalEncoding, EncoderLayer, Encoder, DecoderLayer, Decoder, TransformerForecaster, CustomSchedule
    custom_objects = {
        'PositionalEncoding': PositionalEncoding,
        'EncoderLayer': EncoderLayer,
        'Encoder': Encoder,
        'DecoderLayer': DecoderLayer,
        'Decoder': Decoder,
        'TransformerForecaster': TransformerForecaster,
        'CustomSchedule': CustomSchedule,
    }
except ImportError:
    custom_objects = {}

def load_data(data_path='data.npz'):
    """
    Loads test data from a .npz file.
    The file should contain: inputs_test and outputs_test.
    """
    data = np.load(data_path)
    inputs_test = data['inputs_test']
    outputs_test = data['outputs_test']
    return inputs_test, outputs_test

def prepare_decoder_input(outputs):
    """
    Prepares the decoder input by adding a start token (a zero vector)
    at the beginning and shifting the target sequence to the right.
    This is used for both training and evaluation in teacher forcing mode.
    """
    num_samples, seq_len, num_features = outputs.shape
    start_token = np.zeros((num_samples, 1, num_features), dtype=outputs.dtype)
    decoder_input = np.concatenate([start_token, outputs[:, :-1, :]], axis=1)
    return decoder_input

def compute_metrics(model, inputs_test, outputs_test):
    """
    Evaluates the model on the test set and prints the loss (MSE).
    """
    decoder_input_test = prepare_decoder_input(outputs_test)
    loss = model.evaluate((inputs_test, decoder_input_test), outputs_test, batch_size=64, verbose=0)
    print("Test Loss (MSE):", loss)
    return loss

def plot_candlestick(ax, candles, x_offset=0, alpha=1.0):
    """
    Plots candlesticks on the provided axis.

    Parameters:
      - candles: Array of shape (n, 5) representing [open, high, low, close, volume].
      - x_offset: Horizontal offset to correctly position the sequence.
      - alpha: Transparency for the plotted candlesticks.

    A candle is colored green if the close price is greater or equal to the open price,
    otherwise red.
    """
    for i, candle in enumerate(candles):
        x = x_offset + i
        open_, high, low, close = candle[:4]
        color = 'green' if close >= open_ else 'red'
        # Plot the wick (vertical line from low to high)
        ax.plot([x, x], [low, high], color=color, linewidth=1, alpha=alpha)
        # Plot the body (rectangle between open and close)
        body_low = min(open_, close)
        body_high = max(open_, close)
        rect = plt.Rectangle((x - 0.3, body_low), 0.6, body_high - body_low,
                             facecolor=color, edgecolor=color, alpha=alpha)
        ax.add_patch(rect)

def plot_example(example_idx, inputs_test, outputs_test, predictions, save_dir='plot'):
    """
    Creates and saves a plot for a given example index.

    The plot displays:
      - Historical candles (inputs_test) in the past period.
      - The forecast area where:
          • the real future candles (outputs_test) are shown with high opacity,
          • the predicted candles (predictions) are overlaid with lower opacity.

    The x-axis represents time (candle indices) and the forecast area is offset
    to clearly distinguish it from historical data.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Historical candles from inputs_test (e.g., shape: (30, 5))
    historical = inputs_test[example_idx]
    # Real future candles (outputs_test) and predicted candles (predictions)
    real_forecast = outputs_test[example_idx]
    pred_forecast = predictions[example_idx]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot historical candles from time 0 to 29
    plot_candlestick(ax, historical, x_offset=0, alpha=1.0)

    # Set the starting x position for the forecast area (after historical data)
    x_start = historical.shape[0]

    # Plot real forecast candles with high opacity
    plot_candlestick(ax, real_forecast, x_offset=x_start, alpha=0.8)

    # Overlay the predicted forecast candles with transparency
    plot_candlestick(ax, pred_forecast, x_offset=x_start, alpha=0.5)

    ax.set_title(f'Example {example_idx}: Historical and Forecast Candlesticks')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.grid(True)
    plt.tight_layout()

    # Save the figure to the folder "plot"
    fig.savefig(os.path.join(save_dir, f'example_{example_idx}.png'))
    plt.close(fig)

def main():
    # Load the saved model from the specified folder
    model_path = 'saved_model/transformer_best.keras'
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects) #, custom_objects=custom_objects
    print("Model loaded from:", model_path)

    # Load the test data from the file "data.npz"
    inputs_test = joblib.load(f'./datas/split/30_5_None_rsi_macd_bollinger/9_07_03/inputs_test.pkl')
    outputs_test = joblib.load(f'./datas/split/30_5_None_rsi_macd_bollinger/9_07_03/outputs_test.pkl')

    # Evaluate the model on the test set
    compute_metrics(model, inputs_test, outputs_test)

    # Prepare the decoder input for prediction (teacher forcing mode)
    decoder_input_test = prepare_decoder_input(outputs_test)

    # Generate predictions on the test set
    predictions = model.predict((inputs_test, decoder_input_test), batch_size=64)

    # Save plots for a few examples to visualize performance (up to 5 examples)
    num_examples = min(5, inputs_test.shape[0])
    for i in range(num_examples):
        plot_example(i, inputs_test, outputs_test, predictions, save_dir='plot')

    print("Performance plots have been saved in the 'plot' folder.")

if __name__ == '__main__':
    main()
