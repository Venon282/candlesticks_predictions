from tqdm import tqdm
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

def sequence(datas, n_candle_input_min, n_candle_input_max, n_candle_output_min, n_candle_output_max, size_coherence=True, step=None):
    """
    Attribute dinamically the number of candles for the input and output
    """ 
    data = datas.values # Convert DataFrame to a NumPy array for efficient slicing
    fake_candle = np.zeros(datas.shape[1])                   
    x, y, mask_x, mask_y = [], [], [], []
    i = 0
    
    # Create sequences
    while i < len(datas) - n_candle_input_min - n_candle_output_min:                                # While we can create a sequence
        margin = len(datas) - i                                                                     # Remaining candles
        
        # Determine the number of input candles
        input_max = min(n_candle_input_max, margin - n_candle_output_min)                           # Maximum number of candles for the input
        n_candle_input = random.randint(n_candle_input_min, input_max)                              # Random number of candles for the input   
        n_fake_candle_input = n_candle_input_max - n_candle_input                                   # Number of fake candles to add to the input                   
        
        # Determine the number of output candles
        output_max = min(n_candle_output_max, margin - n_candle_input)
        if size_coherence:
            n_candle_output = random.randint(n_candle_output_min, min(n_candle_input, output_max))
        else:
            n_candle_output = random.randint(n_candle_output_min, output_max)
        n_fake_candle_output = n_candle_output_max - n_candle_output                                # Number of fake candles to add to the output
        
        step_current = n_candle_input + n_candle_output if step is None else step  
        
        x_sequence = np.concatenate((np.tile(fake_candle, (n_fake_candle_input, 1)), data[i: i + n_candle_input]))
        y_sequence = np.concatenate((data[i + n_candle_input: i + n_candle_input + n_candle_output], np.tile(fake_candle, (n_fake_candle_output, 1))))
        mask_x_sequence = np.concatenate((np.zeros(n_fake_candle_input), np.ones(n_candle_input)))
        mask_y_sequence = np.concatenate((np.ones(n_candle_output), np.zeros(n_fake_candle_output)))  
        mask_x.append(mask_x_sequence)
        mask_y.append(mask_y_sequence)    
        x.append(x_sequence)
        y.append(y_sequence)
        
        i += step_current

    return x, y, mask_x, mask_y

def transformDfsDictToDatasDictItem(args):
    key, df, n_candle_input_min, n_candle_input_max, n_candle_output_min, n_candle_output_max, size_coherence, step = args
    inputs, outputs, inputs_mask, outputs_mask = sequence(df,  n_candle_input_min, n_candle_input_max, n_candle_output_min, n_candle_output_max, size_coherence=size_coherence, step=step)
    return key, {'inputs':inputs, 'outputs':outputs, 'inputs_mask':inputs_mask, 'outputs_mask':outputs_mask}

def transformDfsDictToDatasDict(dfs_dict, n_candle_input_min, n_candle_input_max, 
                                        n_candle_output_min, n_candle_output_max,
                                        size_coherence=True, step=None, 
                                        max_workers=None):
    # Define the number of candles for each data set
    datas_dict = {}
    
    # Prepare the arguments for parallel processing
    args_list = [(key, df, n_candle_input_min, n_candle_input_max, n_candle_output_min, n_candle_output_max, size_coherence, step) for key, df in dfs_dict.items()]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(transformDfsDictToDatasDictItem, args) for args in args_list]

        total_jobs = len(args_list)
        for future in tqdm(as_completed(futures), total=total_jobs, mininterval=1):
            key, data = future.result()
            datas_dict[key] = data

    return datas_dict

def dfsDtToDatas(dfs_dict, n_candle_input_min, n_candle_input_max, 
                           n_candle_output_min, n_candle_output_max,
                           size_coherence=True, step=None, 
                           max_workers=None, seed=42):
    
    print('\nTransform datas to sequences')
    random.seed(seed)
    datas_dict = transformDfsDictToDatasDict(dfs_dict, n_candle_input_min, n_candle_input_max, 
                                                       n_candle_output_min, n_candle_output_max,
                                                       size_coherence=size_coherence, step=step,
                                                       max_workers=max_workers)
    
    return datas_dict