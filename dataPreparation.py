import time
import pandas as pd
from pathlib import Path
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import sklearn
import numpy as np
import joblib

# internal
from py_libraries.ml.preprocessing import MinMaxGlobalScaler

def dfsDict(path, sep=',', encoding='utf-8-sig'):
    return {file_path.stem: pd.read_csv(file_path, sep=sep, encoding=encoding) for file_path in Path(path).glob('*.csv')}

def improveDataSets(dfs_dict):
    for key, df in dfs_dict.items():
        df.drop(['<DATE>', '<TIME>'], axis=1, inplace=True)
        df.rename(columns=lambda x: re.sub(r'[<>]','',x.lower()), inplace=True)

def sequence(datas, n_candle_input, n_candle_output, input_cols, output_cols, step=None):
    if step is None:
        step = n_candle_input + n_candle_output

    # Create sequences
    x, y = [], []
    for i in range(0, len(datas) - n_candle_input - n_candle_output, step):
        # -1 because the last one is include and not exclude as normal python
        x.append(datas.loc[i: i + n_candle_input -1, input_cols].to_numpy())
        y.append(datas.loc[i + n_candle_input: i + n_candle_input + n_candle_output - 1, output_cols].to_numpy())

    return x, y

def transformDfsDictToDatasDictItem(args):
    key, df, n_candle_input, n_candle_output, input_cols, output_cols, step = args
    inputs, outputs = sequence(df, n_candle_input, n_candle_output, input_cols, output_cols, step=step)
    return key, {'inputs':inputs, 'outputs':outputs}

def transformDfsDictToDatasDict(dfs_dict, n_candle_input, n_candle_output, input_cols, output_cols, step=None):
    datas_dict = {}
    args_list = [(key, df, n_candle_input, n_candle_output, input_cols, output_cols, step) for key, df in dfs_dict.items()]

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
        inputs_train, inputs_temp, outputs_train, outputs_temp = sklearn.model_selection.train_test_split(values['inputs'], values['outputs'], test_size=val+test, random_state=seed)
        inputs_val, inputs_test, outputs_val, outputs_test = sklearn.model_selection.train_test_split(inputs_temp, outputs_temp, test_size=test / (val+test), random_state=seed)
        split_dict[key] = {'inputs':{'train':inputs_train, 'val':inputs_val, 'test':inputs_test},
                           'outputs':{'train':outputs_train, 'val':outputs_val, 'test':outputs_test}}
    return split_dict

def scale(split_dict, path):
    path = Path(path)
    inputs_train, inputs_val, inputs_test = [], [], []
    outputs_train, outputs_val, outputs_test = [], [], []
    for key, values in split_dict.items():
        market = key.split('_')[0].lower()
        scaler_path = path / f'{market}.pkl'
        if not scaler_path.is_file():
            scaler = MinMaxGlobalScaler()
            scaler.fit([values['inputs']['train'], values['outputs']['train']])
            joblib.dump(scaler, scaler_path)
        else:
            scaler = joblib.load(scaler_path)

        inputs_train.extend(scaler.transform(values['inputs']['train']))
        outputs_train.extend(scaler.transform(values['outputs']['train']))

        inputs_val.extend(scaler.transform(values['inputs']['val']))
        outputs_val.extend(scaler.transform(values['outputs']['val']))

        inputs_test.extend(scaler.transform(values['inputs']['test']))
        outputs_test.extend(scaler.transform(values['outputs']['test']))
    return inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test

def main():
    dfs_dict = dfsDict('./datas/raw', sep=';')
    improveDataSets(dfs_dict)

    n_candle_input = 30
    n_candle_output = 5

    datas_dict = transformDfsDictToDatasDict(dfs_dict, n_candle_input, n_candle_output,
                                             input_cols=['open', 'high', 'low', 'close', 'tickvol'],
                                             output_cols=['open', 'high', 'low', 'close', 'tickvol'],
                                             step=None)

    split_dict = splitDict(datas_dict, train=0.7, val=0.15, test=0.15)

    for key, val in split_dict.items():
        for ke, val in val.items():
            for k, v in val.items():
                print(key, ke, k, np.array(v).shape)

    inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test = scale(split_dict, path='./datas/scaler')
    inputs_train, outputs_train = sklearn.utils.shuffle(inputs_train, outputs_train)
    inputs_val, outputs_val = sklearn.utils.shuffle(inputs_val, outputs_val)
    inputs_test, outputs_test = sklearn.utils.shuffle(inputs_test, outputs_test)

    datas_dict = {'inputs_train':inputs_train, 'inputs_val':inputs_val, 'inputs_test':inputs_test, 'outputs_train':outputs_train, 'outputs_val':outputs_val, 'outputs_test':outputs_test}
    for key, values in datas_dict.items():
        print(key, np.array(values).shape, np.min(values), np.max(values))

if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f'Time: {time.time() - start_time}ms')
