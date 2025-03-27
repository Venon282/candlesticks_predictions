import time
import pandas as pd
from pathlib import Path
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import random
from tqdm import tqdm

# internal
from py_libraries.ml.preprocessing import MinMaxGlobalScaler, trainTestSplit
from py_libraries.lst import flatten
from TechnicalIndicators import TechnicalIndicators

def dfsDict(path, sep=',', encoding='utf-8-sig', n_cols=None, dtype=None):
    dfs_dict = {}
    files_path = list(Path(path).glob('*.csv'))
    for file_path in tqdm(files_path, total=len(files_path)):

        df = pd.read_csv(file_path, sep=sep, encoding=encoding, dtype=str)
        if n_cols is None:
            n_cols = len(df)
        
        if len(df.columns) != n_cols:
            for separator in (r'\t', '  ', ',', ';'):
                df = pd.read_csv(file_path, encoding=encoding, sep=separator, on_bad_lines='skip')
                if len(df.columns) == n_cols:
                    df.to_csv(file_path, sep=sep, index=False)
                    break
            else:
                raise ValueError(f'{file_path.stem} have {len(df.columns)} cols instead of {n_cols}')
        dfs_dict[file_path.stem] = improveDf(df, dtype)
    return dfs_dict

def improveDf(df, dtype, cols_to_drop=['<DATE>', '<TIME>']):

    df.drop(cols_to_drop, axis=1, inplace=True)
    df = df.map(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].astype(float)
    if dtype is not None:
        df = df.astype({key: value for key, value in dtype.items() if key in df} )
    df.rename(columns=lambda x: re.sub(r'[<>]','',x.lower()), inplace=True)
   
    return df


def sequence(datas, n_candle_input_min, n_candle_input_max, n_candle_output_min, n_candle_output_max, size_coherence=True, step=None):
    """
    Attribute dinamically the number of candles for the input and output
    """ 
    data = datas.values # Convert DataFrame to a NumPy array for efficient slicing
    # Create sequences
    step = n_candle_input_max + n_candle_output_max if step is None else step    
    fake_candle = np.zeros(datas.shape[1])                   
    x, y, mask_x, mask_y = [], [], [], []
    i = 0
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
    
    args_list = [(key, df,  n_candle_input_min, n_candle_input_max, n_candle_output_min, n_candle_output_max, size_coherence, step) for key, df in dfs_dict.items()]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(transformDfsDictToDatasDictItem, args) for args in args_list]

        total_jobs = len(args_list)
        for future in tqdm(as_completed(futures), total=total_jobs):
            key, data = future.result()
            datas_dict[key] = data

    return datas_dict

def splitDict(datas_dict, train=0.7, val=0.15, test=0.15, seed=42):
    split_dict = {}
    for key, values in tqdm(datas_dict.items(), total=len(datas_dict)):
        
        inputs_train, inputs_temp, outputs_train, outputs_temp, inputs_mask_train, inputs_mask_temp, outputs_mask_train, outputs_mask_temp = trainTestSplit(values['inputs'], values['outputs'], values['inputs_mask'], values['outputs_mask'], test_size=val+test, random_state=seed)
        inputs_val, inputs_test, outputs_val, outputs_test, inputs_mask_val, inputs_mask_test, outputs_mask_val, outputs_mask_test = trainTestSplit(inputs_temp, outputs_temp, inputs_mask_temp, outputs_mask_temp, test_size=test / (val+test), random_state=seed)
        split_dict[key] = {'inputs':{'train':inputs_train, 'val':inputs_val, 'test':inputs_test},
                           'outputs':{'train':outputs_train, 'val':outputs_val, 'test':outputs_test},
                           'inputs_mask':{'train':inputs_mask_train, 'val':inputs_mask_val, 'test':inputs_mask_test},
                           'outputs_mask':{'train':outputs_mask_train, 'val':outputs_mask_val, 'test':outputs_mask_test}}
    return split_dict

def scale(split_dict, cols_to_group_for_scale, columns_idx, path):
    def transformData(values, scaler, cols_idx):
        """Apply the respective scaler to each slice and concatenate"""
        values = np.array(values)  # Ensure conversion
        if len(values) > 0:
            values[:, :, cols_idx] = scaler.transform(values[:, :, cols_idx])
        return values
    path = Path(path)
    inputs_train, inputs_val, inputs_test = [], [], []
    outputs_train, outputs_val, outputs_test = [], [], []
    inputs_mask_train, inputs_mask_val, inputs_mask_test = [], [], []
    outputs_mask_train, outputs_mask_val, outputs_mask_test = [], [], []
    for key, values in tqdm(split_dict.items(), total=len(split_dict)):
        market = key.split('_')[0].lower()
        for cols in cols_to_group_for_scale:
            scaler_path = path / (market + '_' + '_'.join(cols) + '.pkl')
            inputs_train_array = np.array(values['inputs']['train'])
            outputs_train_array = np.array(values['outputs']['train'])
            cols_idx = [columns_idx[col] for col in cols]
            
            if not scaler_path.is_file():
                scaler = MinMaxGlobalScaler()
                scaler.fit([
                    inputs_train_array[:, :, cols_idx],  # First 4 cols
                    outputs_train_array[:, :, cols_idx]
                ])
                joblib.dump(scaler, scaler_path)
            else:
                scaler = joblib.load(scaler_path)
                
            for entry, vals in values.items():
                if 'mask' in entry:
                    continue
                for set_type, val in vals.items():
                    values[entry][set_type] = transformData(val, scaler, cols_idx=cols_idx)
  
        inputs_train.extend(values['inputs']['train'])
        outputs_train.extend(values['outputs']['train'])

        inputs_val.extend(values['inputs']['val'])
        outputs_val.extend(values['outputs']['val'])

        inputs_test.extend(values['inputs']['test'])
        outputs_test.extend(values['outputs']['test'])
        
        inputs_mask_train.extend(values['inputs_mask']['train'])
        outputs_mask_train.extend(values['outputs_mask']['train'])

        inputs_mask_val.extend(values['inputs_mask']['val'])
        outputs_mask_val.extend(values['outputs_mask']['val'])

        inputs_mask_test.extend(values['inputs_mask']['test'])
        outputs_mask_test.extend(values['outputs_mask']['test'])

    return inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test, inputs_mask_train, inputs_mask_val, inputs_mask_test, outputs_mask_train, outputs_mask_val, outputs_mask_test

def dfFirstValidIndex(df):
    first_valid_index = 0
    for col_name, data in df.items():
        first_valid_index = max(first_valid_index, data.first_valid_index())
    return first_valid_index

def addIndicatorsToDataSets(dfs_dict, indicators_to_add=None, params=None):
    tech_ind = TechnicalIndicators(indicators=indicators_to_add, params=params)
    for key, df in dfs_dict.items():
        tech_ind.add_indicators(df, inplace=True)
        
        # remove firsts indexs with nan. Insure not nan drop in the middle of the data set.
        first_idx = dfFirstValidIndex(df)
        dfs_dict[key] = df.loc[first_idx:].reset_index(drop=True)
        
def colsToGroupForScale(columns, base=[['open', 'high', 'low', 'close']]):
    similar_cols = {}
    base_flatten = flatten(base)

    for col in columns:
        if col in base_flatten:
            continue
        head = col.split('_')[0]
        if head in similar_cols:
            similar_cols[head].append(col)
        else:
            similar_cols[head] = [col]

    base.extend(list(similar_cols.values()))
    return base
    
def main(n_candle_input_min = 5, n_candle_input_max = 100, 
         n_candle_output_min = 1, n_candle_output_max = 5, 
         step=None, size_coherence=True,  split_rates=[0.7, 0.15, 0.15],
         indicators_to_add=['rsi', 'macd', 'bollinger'],
         sep_file = '_'):
    """
        size_coherence if true, n candle output can't be greater than n candle input in the random
    """
    
    # Verify split rates
    if not np.isclose(sum(split_rates), 1.0):
        raise ValueError(f'split_rates sum must be 1')
    
    # prepare split name folder
    split_str = sep_file.join(map(str, split_rates))
    split_str = split_str.replace('0.', '')
    
    # prepare id name folder
    id = f'({n_candle_input_min}-{n_candle_input_max}){sep_file}({n_candle_output_min}-{n_candle_output_max}){sep_file}step{step}{sep_file}sc{size_coherence}'
    if len(indicators_to_add) > 0:
        id +=  sep_file + sep_file.join(indicators_to_add)
        
    # Prepare datas set
    print('Prepare datas set')
    dfs_dict = dfsDict('./datas/raw', sep=';', n_cols=9, 
                       dtype={'<DATE>':str, '<TIME>':str, '<OPEN>':'float', '<HIGH>':'float', '<LOW>':'float', '<CLOSE>':'float', '<TICKVOL>':'int', '<VOL>':'int', '<SPREAD>':'int'})
    addIndicatorsToDataSets(dfs_dict, indicators_to_add = indicators_to_add) 
    
    # Prepare similar columns to have the same/different scaler
    columns = list(dfs_dict.values())[0].columns.to_list()
    columns_idx = {col: i for i, col in enumerate(columns)}
    cols_to_group_for_scale = colsToGroupForScale(columns, base=[['open', 'high', 'low', 'close']])
    
    print('Transform datas to sequences')
    datas_dict = transformDfsDictToDatasDict(dfs_dict, n_candle_input_min, n_candle_input_max, 
                                                       n_candle_output_min, n_candle_output_max,
                                                       size_coherence=size_coherence, step=step,
                                                       max_workers=2)

    print('Split datas set')
    split_dict = splitDict(datas_dict, train=split_rates[0], val=split_rates[1], test=split_rates[2])

    # Datas quantity check
    print(*(f'{key}, {ke}, {k}, {np.array(v).shape}' for key, val in split_dict.items() for ke, val in val.items() for k, v in val.items()), sep='\n')
    
    # Scale and prepare datas
    print('Scaler and Shuffle datas set')
    scaler_path = Path(f'./datas/scaler/{id}/{split_str}')
    scaler_path.mkdir(parents=True, exist_ok=True)
    inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test, inputs_mask_train, inputs_mask_val, inputs_mask_test, outputs_mask_train, outputs_mask_val, outputs_mask_test = scale(split_dict, cols_to_group_for_scale, columns_idx, path=scaler_path)
    
    inputs_train, outputs_train = sklearn.utils.shuffle(inputs_train, outputs_train)
    inputs_val, outputs_val = sklearn.utils.shuffle(inputs_val, outputs_val)
    inputs_test, outputs_test = sklearn.utils.shuffle(inputs_test, outputs_test)
    inputs_mask_train, outputs_mask_train = sklearn.utils.shuffle(inputs_mask_train, outputs_mask_train)
    inputs_mask_val, outputs_mask_val = sklearn.utils.shuffle(inputs_mask_val, outputs_mask_val)
    inputs_mask_test, outputs_mask_test = sklearn.utils.shuffle(inputs_mask_test, outputs_mask_test)

    # Save datas
    print('Save datas')
    datas_dict = {'inputs_train':inputs_train, 'inputs_val':inputs_val, 'inputs_test':inputs_test, 'outputs_train':outputs_train, 'outputs_val':outputs_val, 'outputs_test':outputs_test, 'inputs_mask_train':inputs_mask_train, 'inputs_mask_val':inputs_mask_val, 'inputs_mask_test':inputs_mask_test, 'outputs_mask_train':outputs_mask_train, 'outputs_mask_val':outputs_mask_val, 'outputs_mask_test':outputs_mask_test}
    save_path = Path(f'./datas/split/{id}/{split_str}')
    save_path.mkdir(parents=True, exist_ok=True)
    resum = ''
    for key, values in tqdm(datas_dict.items(), total=len(datas_dict)):
        if len(values) > 0:
            resum += f'{key}, {np.array(values).shape}, {np.min(values)}, {np.max(values)}\n'
            joblib.dump(np.array(values),save_path  / f'{key}.pkl')
    print(resum)



if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f'Time: {time.time() - start_time}ms')
