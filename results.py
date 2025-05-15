import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import pandas as pd
from matplotlib.patches import Patch

from py_libraries import visualize
from py_libraries.number import formated

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
    
import numpy as np

def metrics(true, pred, mask):
    """
    Compute an extensive set of evaluation metrics for predicted candlestick sequences.

    Args:
        true: array-like, shape (batch, seq_len, 4) with columns: open, high, low, close
        pred: same shape as true
        mask: binary array-like, shape (batch, seq_len), where 1 = keep this candle

    Returns:
        dict of metrics
    """
    # --- 1) préparation des données ---
    # Convert TensorFlow tensors to numpy
    for arr in (true, pred, mask):
        if hasattr(arr, 'numpy'):
            arr = arr.numpy()

    true = np.asarray(true)
    pred = np.asarray(pred)
    mask = np.asarray(mask).astype(bool)

    assert true.shape == pred.shape, "true/pred shape mismatch"
    assert mask.shape == true.shape[:2], "mask shape mismatch"

    # aplatissement
    true_flat = true[mask, :4]
    pred_flat = pred[mask, :4]
    n = len(true_flat)

    # unpack sous des noms explicites
    true_open, true_high, true_low, true_close = true_flat.T
    pred_open, pred_high, pred_low, pred_close = pred_flat.T

    # erreurs
    err = pred_flat - true_flat
    abs_err = np.abs(err)
    sq_err = err ** 2

    # --- 2) metrics globales et par feature ---
    def aggregate(arr):
        return {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'p25': float(np.percentile(arr, 25)),
            'p75': float(np.percentile(arr, 75)),
            'max': float(np.max(arr)),
        }

    report = {'n_samples': n}

    # Itération sur chaque feature
    for idx, name in enumerate(['open', 'high', 'low', 'close']):
        ae = abs_err[:, idx]
        se = sq_err[:, idx]
        yt = true_flat[:, idx]
        yp = pred_flat[:, idx]

        mse = np.mean(se)
        rmse = np.sqrt(mse)
        mae = np.mean(ae)
        ss_res = np.sum(se)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        mape = np.mean(np.abs(err[:, idx] / (yt + 1e-8))) * 100
        smape = np.mean(2 * ae / (np.abs(yt) + np.abs(yp) + 1e-8)) * 100

        report[f'mae_{name}'] = mae
        report[f'mse_{name}'] = mse
        report[f'rmse_{name}'] = rmse
        report[f'r2_{name}'] = r2
        report[f'mape_{name}'] = mape
        report[f'smape_{name}'] = smape
        for k, v in aggregate(ae).items():
            report[f'ae_{name}_{k}'] = v

    # Erreurs globales moyennées sur les 4 features
    report['mae'] = np.mean([report[f'mae_{f}'] for f in ['open','high','low','close']])
    report['mse'] = np.mean([report[f'mse_{f}'] for f in ['open','high','low','close']])
    report['mse'] = np.mean([report[f'rmse_{f}'] for f in ['open','high','low','close']])
    report['mse'] = np.mean([report[f'r2_{f}'] for f in ['open','high','low','close']])
    report['mse'] = np.mean([report[f'mape_{f}'] for f in ['open','high','low','close']])
    report['mse'] = np.mean([report[f'smape_{f}'] for f in ['open','high','low','close']])

    # --- 3) metrics spécifiques aux chandeliers ---
    # Direction (bull/bear)
    dir_true = np.sign(true_close - true_open)
    dir_pred = np.sign(pred_close - pred_open)
    report['accuracy_direction'] = float(np.mean(dir_true == dir_pred))

    # Volatilité & range
    true_vol = true_high - true_low
    pred_vol = pred_high - pred_low
    report['volatility_mae'] = float(np.mean(np.abs(pred_vol - true_vol)))
    report['volatility_ratio'] = float(np.mean(pred_vol / (true_vol + 1e-8)))

    true_range = true_close - true_open
    pred_range = pred_close - pred_open
    report['range_mae'] = float(np.mean(np.abs(pred_range - true_range)))
    report['range_bias'] = float(np.mean(pred_range - true_range))

    # Position high / low correcte
    true_highest = (true_high > true_open) & (true_high > true_low) & (true_high > true_close)
    pred_highest = (pred_high > pred_open) & (pred_high > pred_low) & (pred_high > pred_close)
    report['high_position_accuracy'] = float(np.mean(true_highest == pred_highest))

    true_lowest = (true_low < true_open) & (true_low < true_low) & (true_low < true_close)
    pred_lowest = (pred_low < pred_open) & (pred_low < pred_low) & (pred_low < pred_close)
    report['low_position_accuracy'] = float(np.mean(true_lowest == pred_lowest))

    # OHLC order check
    order_true = np.argsort(true_flat, axis=1)
    order_pred = np.argsort(pred_flat, axis=1)
    report['ohlc_order_accuracy'] = float(np.mean((order_true == order_pred).all(axis=1)))

    return report


def plotCandles(i, inputs, outputs, predictions, x_offset=0, alpha=1.0, save_dir='./plots'):
    save_path = Path(save_dir) / f'candles_{i}.png'
    if save_path.exists():
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    visualize.candles(inputs, x_offset=x_offset, alpha=alpha, ax=ax, fig=fig, show=False)
    
    if np.isscalar(x_offset):
        x_start = inputs.shape[0] + x_offset
    else:
        x_start = max(x_offset) + 1

    # Plot real forecast candles with high opacity
    visualize.candles(outputs, x_offset=x_start, open_legend='True_Buy', close_legend='True_Sell', alpha=0.5, ax=ax, fig=fig, show=False, legend_ax=True)
    # Overlay the predicted forecast candles with lower opacity
    visualize.candles(predictions, x_offset=x_start, open_color='blue', close_color='orange', open_legend='Pred_Buy', close_legend='Pred_Sell',alpha=0.5, ax=ax, fig=fig, 
                     title=f'True and pred candlesticks', xlabel='Time', ylabel='Price', 
                     show=False, path=save_path)
    
    # legend_elements = [
    #     Patch(facecolor='green', edgecolor='green', label='True Buy Candle'),
    #     Patch(facecolor='red', edgecolor='red', label='True Sell Candle'),
    #     Patch(facecolor='blue', edgecolor='blue', label='Predicted Buy Candle'),
    #     Patch(facecolor='orange', edgecolor='orange', label='Predicted Sell Candle'),
    # ]

    
    # fig.legend(handles=legend_elements, loc='upper left')
    # ax.legend()
    # fig.legend()
    
def main():
    # Load the saved model from the specified folder
    models_path = Path(r'E:\csp\saved_model')
    datas_path = Path(r'E:\csp\split')
    save_path = Path(r'E:\csp\plot')
    
    indentation = ' ' * 4
    models_paths = list(models_path.rglob('*.keras'))
    nb_models = len(models_paths)
    header = None
    models_metrics = []
    for model_idx, model_file in enumerate(models_paths, start=1):
        model_name = model_file.name
        print(f'{model_name} {model_idx}/{nb_models}', ' ' * 30)
        
        model_folder = model_file.parent
        split_folder = model_folder.parent
        datas_folder = split_folder.parent

        transformer = tf.keras.models.load_model(model_file, custom_objects=custom_objects) #, custom_objects=custom_objects
        
        model_datas_path = datas_path / datas_folder.name / split_folder.name

        # Load the test data from the file "data.npz"
        print(f'{indentation}Loading test data...')
        inputs_test = joblib.load(model_datas_path / 'inputs_test.pkl')
        outputs_test = joblib.load(model_datas_path / 'outputs_test.pkl')
        inputs_mask_test = joblib.load(model_datas_path / 'inputs_mask_test.pkl').astype(bool)
        outputs_mask_test = joblib.load(model_datas_path / 'outputs_mask_test.pkl').astype(bool)
        
        n = 50
        inputs_test, outputs_test = inputs_test[:n], outputs_test[:n]
        inputs_mask_test, outputs_mask_test = inputs_mask_test[:n], outputs_mask_test[:n]

        print(f'{indentation}Predicting...')
        predictions_auto_reg = transformer(encoder_input=inputs_test, mask={'encoder':inputs_mask_test, 'decoder':outputs_mask_test})
        predictions_decoder = transformer(encoder_input=inputs_test, decoder_input=outputs_test, mask={'encoder':inputs_mask_test, 'decoder':outputs_mask_test})
        
        model_metrics = metrics(outputs_test, predictions_auto_reg, outputs_mask_test)
        if header is None:
            header = ['parameters', 'split', 'datas', 'model'] + list(model_metrics.keys())
        models_metrics.append([model_folder.name, split_folder.name, datas_folder.name, model_file.stem] + list(model_metrics.values()))

        # Save plots 
        model_save_path = save_path / datas_folder.name / split_folder.name / model_folder.name
        os.makedirs(model_save_path, exist_ok=True)
        for i in range(len(inputs_test)):
            print(f'{indentation}Plotting example {i+1}/{len(inputs_test)}...', ' ' * 30, end='\r')
            plotCandles(f'{i}_a', inputs_test[i][inputs_mask_test[i], :4], outputs_test[i][outputs_mask_test[i], :4], np.array(predictions_auto_reg[i])[outputs_mask_test[i], :4], save_dir=str(model_save_path))
            plotCandles(f'{i}_d', inputs_test[i][inputs_mask_test[i], :4], outputs_test[i][outputs_mask_test[i], :4], np.array(predictions_decoder[i])[outputs_mask_test[i], :4], save_dir=str(model_save_path))
        print()
        
    df = pd.DataFrame(models_metrics, columns=header)
    df.sort_values(['accuracy_direction', 'mae'], ascending=[False, True],inplace=True)
    header = df.columns
    df = df[list(header[:4]) + sorted(header[4:], key=str)] # sort the cols
    for col in df.columns[4:]:
        df[col] = df[col].apply(lambda x: formated(x))
    df.to_csv(save_path / 'resume.csv', sep=';', encoding='utf-8-sig', index=False)
    
    
if __name__ == '__main__':
    main()
