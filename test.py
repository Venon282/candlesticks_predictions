import tensorflow as tf
import numpy as np
from itertools import combinations
from matplotlib import pyplot as plt

from py_libraries.ml.preprocessing.StandardGlobalScaler import StandardGlobalScaler

def StandardGlobalScalerTest():
    # Create a sample dataset
    data = np.array([[1, 2], [3, 4], [5, 6]]) * 1000000
    
    # Initialize the StandardGlobalScaler
    scaler = StandardGlobalScaler()
    
    # Fit the scaler to the data
    scaler.fit(data)
    
    # Transform the data
    transformed_data = scaler.transform(data)
    
    # Inverse transform the data
    inverse_transformed_data = scaler.inverse_transform(transformed_data)
    
    # Check if the original data is recovered after inverse transformation
    assert np.allclose(data, inverse_transformed_data), "Inverse transform did not recover original data."
    
    print('Mean:', scaler.mean_, transformed_data.mean())
    print('Scale:', scaler.scale_, transformed_data.std())
    print('Min:',  transformed_data.min())
    print('Max:',  transformed_data.max())
    print(transformed_data)
    
    
    print("StandardGlobalScaler test passed.")

def numpyMaskTest():
    mask = np.array([[True, True, False], 
                    [True, False, True], 
                    [False, True, True]])
    # 3d array
    data = np.array([[[1, 2, 3], [4, 5, 6], [4, 5, 6]], 
                    [[7, 8, 9], [10, 11, 12], [10, 11, 12]], 
                    [[13, 14, 15], [16, 17, 18], [16, 17, 18]]])

    print(data[mask].shape)
    print(data[mask])

    data[~mask] = [0, 0, 0]

    print(data.shape)
    print(data)
    
def enumerateToDict():
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    enumeration = enumerate(seasons)
    print(dict(enumeration))
    
def productTest():
    indicators = ['rsi', 'macd', 'bollinger']
    combined_indicators = list(combinations(indicators, 2)) + list(combinations(indicators, 3))
    print(combined_indicators)
    
def plotNoam():
    from py_libraries.ml.optimizer.schedule import Noam
    
    plt.figure(figsize=(8,4))
    
    n_inputs_train = 2603825
    epochs = 50
    batch_size = 32
    num_steps       = int(n_inputs_train / batch_size)
    max_steps       = epochs * num_steps
    for d_model in [512, 1024, 2048]:
        for warmup_steps_ration in [0.05, 0.1, 0.2]:
            warmup_steps = int(max_steps * warmup_steps_ration)
            opt = Noam(d_model, warmup_steps)
            steps, lrs = opt.plot(max_steps=max_steps)
            plt.plot(steps.numpy(), lrs.numpy(), linewidth=2, label=f'{d_model=} {warmup_steps=} ({warmup_steps_ration})')
         
    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.title(f"Noam Schedule, {epochs=} {batch_size=} {n_inputs_train=}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.legend()

    # plt.show()
    from pathlib import Path
    path = r'C:\Users\ET281306\Desktop\folders\gtw\candlesticks_predictions\img'
    plot_name = f'{n_inputs_train=}_{epochs=}_{batch_size=}.png'
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch(exist_ok=True)
    plt.savefig(str(file_path / plot_name))
    plt.close()
    
plotNoam()