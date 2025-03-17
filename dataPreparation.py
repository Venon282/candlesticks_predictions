import time
import pandas as pd
from pathlib import Path
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

# internal
from py_libraries.ml.preprocessing import MinMaxGlobalScaler, trainTestSplit
from py_libraries.lst import flatten
from TechnicalIndicators import TechnicalIndicators

def dfsDict(path, sep=',', encoding='utf-8-sig', n_cols=None):
    dfs_dict = {}
    for file_path in Path(path).glob('*.csv'):
        df = pd.read_csv(file_path, sep=sep, encoding=encoding)
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
        dfs_dict[file_path.stem] = df
    return dfs_dict

def improveDataSets(dfs_dict):
    for key, df in dfs_dict.items():
        df.drop(['<DATE>', '<TIME>'], axis=1, inplace=True)
        df.rename(columns=lambda x: re.sub(r'[<>]','',x.lower()), inplace=True)
        df = df.map(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
        for dtype, col in zip(df.dtypes, df.columns):
            if dtype == 'object':
                df[col] = df[col].astype(float)
        dfs_dict[key] = df
    return dfs_dict


def sequence(datas, n_candle_input, n_candle_output, step=None):
    if step is None:
        step = n_candle_input + n_candle_output

    # Create sequences
    x, y = [], []
    for i in range(0, len(datas) - n_candle_input - n_candle_output, step):
        # -1 because the last one is include and not exclude as normal python
        x.append(datas.loc[i: i + n_candle_input -1].to_numpy())
        y.append(datas.loc[i + n_candle_input: i + n_candle_input + n_candle_output - 1].to_numpy())

    return x, y

def transformDfsDictToDatasDictItem(args):
    key, df, n_candle_input, n_candle_output, step = args
    inputs, outputs = sequence(df, n_candle_input, n_candle_output, step=step)
    return key, {'inputs':inputs, 'outputs':outputs}

def transformDfsDictToDatasDict(dfs_dict, n_candle_input, n_candle_output, step=None):
    datas_dict = {}
    args_list = [(key, df, n_candle_input, n_candle_output, step) for key, df in dfs_dict.items()]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(transformDfsDictToDatasDictItem, args) for args in args_list]

        total_jobs = len(args_list)
        for i, future in enumerate(as_completed(futures), start=1):
            key, data = future.result()
            datas_dict[key] = data
            print(f'{i}/{total_jobs}', end='\r')

    return datas_dict

def splitDict(datas_dict, train=0.7, val=0.15, test=0.15, seed=42):
    split_dict = {}
    for key, values in datas_dict.items():
        
        inputs_train, inputs_temp, outputs_train, outputs_temp = trainTestSplit(values['inputs'], values['outputs'], test_size=val+test, random_state=seed)
        inputs_val, inputs_test, outputs_val, outputs_test = trainTestSplit(inputs_temp, outputs_temp, test_size=test / (val+test), random_state=seed)
        split_dict[key] = {'inputs':{'train':inputs_train, 'val':inputs_val, 'test':inputs_test},
                           'outputs':{'train':outputs_train, 'val':outputs_val, 'test':outputs_test}}
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
    for key, values in split_dict.items():
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
                for set_type, val in vals.items():
                    values[entry][set_type] = transformData(val, scaler, cols_idx=cols_idx)
  
        inputs_train.extend(values['inputs']['train'])
        outputs_train.extend(values['outputs']['train'])

        inputs_val.extend(values['inputs']['val'])
        outputs_val.extend(values['outputs']['val'])

        inputs_test.extend(values['inputs']['test'])
        outputs_test.extend(values['outputs']['test'])

    return inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test

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
    
def main(n_candle_input = 30, n_candle_output = 5, step=None, 
         split_rates=[0.9, 0.1, 0.0],
         indicators_to_add=['rsi', 'macd', 'bollinger']):
    
    if not np.isclose(sum(split_rates), 1.0):
        raise ValueError(f'split_rates sum must be 1')
    split_str = '_'.join(map(str, split_rates))
    split_str = split_str.replace('0.', '')
    id = '_'.join([str(n_candle_input), str(n_candle_output), str(step)] + indicators_to_add)
    dfs_dict = dfsDict('./datas/raw', sep=';', n_cols=9)
    dfs_dict = improveDataSets(dfs_dict)
    addIndicatorsToDataSets(dfs_dict, indicators_to_add = indicators_to_add) 
    
    columns = list(dfs_dict.values())[0].columns.to_list()
    columns_idx = {col: i for i, col in enumerate(columns)}
    cols_to_group_for_scale = colsToGroupForScale(columns, base=[['open', 'high', 'low', 'close']])
 
    datas_dict = transformDfsDictToDatasDict(dfs_dict, n_candle_input, n_candle_output, step=step)

    split_dict = splitDict(datas_dict, train=split_rates[0], val=split_rates[1], test=split_rates[2])

    for key, val in split_dict.items():
        for ke, val in val.items():
            for k, v in val.items():
                print(key, ke, k, np.array(v).shape)
    scaler_path = Path(f'./datas/scaler/{id}/{split_str}')
    scaler_path.mkdir(parents=True, exist_ok=True)
    inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test = scale(split_dict, cols_to_group_for_scale, columns_idx, path=scaler_path)
    inputs_train, outputs_train = sklearn.utils.shuffle(inputs_train, outputs_train)
    inputs_val, outputs_val = sklearn.utils.shuffle(inputs_val, outputs_val)
    inputs_test, outputs_test = sklearn.utils.shuffle(inputs_test, outputs_test)

   
    datas_dict = {'inputs_train':inputs_train, 'inputs_val':inputs_val, 'inputs_test':inputs_test, 'outputs_train':outputs_train, 'outputs_val':outputs_val, 'outputs_test':outputs_test}
    save_path = Path(f'./datas/split/{id}/{split_str}')
    save_path.mkdir(parents=True, exist_ok=True)
    for key, values in datas_dict.items():
        if len(values) > 0:
            print(key, np.array(values).shape, np.min(values), np.max(values))
            joblib.dump(np.array(values),save_path  / f'{key}.pkl')



if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f'Time: {time.time() - start_time}ms')
