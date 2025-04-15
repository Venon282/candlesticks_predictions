import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from py_libraries import visualize

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
    visualize.candles(outputs, x_offset=x_start, alpha=0.5, ax=ax, fig=fig, show=False)
    # Overlay the predicted forecast candles with lower opacity
    visualize.candles(predictions, x_offset=x_start, open_color='blue', close_color='orange', alpha=0.5, ax=ax, fig=fig, 
                     title=f'True and pred candlesticks', xlabel='Time', ylabel='Price', 
                     show=False, path=save_path)
    
def main():
    # Load the saved model from the specified folder
    models_path = Path(r'./saved_model')
    datas_path = Path(r'./datas/split')
    save_path = Path(r'./plot')
    
    indentation = ' ' * 4
    models_paths = list(models_path.rglob('*.keras'))
    nb_models = len(models_paths)
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

        # Save plots 
        model_save_path = save_path / datas_folder.name / split_folder.name / model_folder.name
        os.makedirs(model_save_path, exist_ok=True)
        for i in range(len(inputs_test)):
            print(f'{indentation}Plotting example {i}/{len(inputs_test)}...', ' ' * 30, end='\r')
            plotCandles(f'{i}_a', inputs_test[i][inputs_mask_test[i], :4], outputs_test[i][outputs_mask_test[i], :4], np.array(predictions_auto_reg[i])[outputs_mask_test[i], :4], save_dir=str(model_save_path))
            plotCandles(f'{i}_d', inputs_test[i][inputs_mask_test[i], :4], outputs_test[i][outputs_mask_test[i], :4], np.array(predictions_decoder[i])[outputs_mask_test[i], :4], save_dir=str(model_save_path))
        print()
    
if __name__ == '__main__':
    main()
