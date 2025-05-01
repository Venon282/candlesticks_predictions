import sklearn
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib

# internal
from py_libraries.ml.preprocessing import StandardGlobalScaler

def computeGroups(values, rows, path, cols_to_group_for_scale, market, columns_idx, fake_candle):
    for cols in cols_to_group_for_scale:
        scaler_path = path / (market + '_' + '_'.join(cols) + '.pkl')
        inputs_train_array = np.array(values['inputs']['train'])       # Shape: [N, T, D]
        outputs_train_array = np.array(values['outputs']['train'])     # Shape: [N, T, D]
        inputs_mask = np.array(values['inputs_mask']['train'])         # Shape: [N, T]
        outputs_mask = np.array(values['outputs_mask']['train'])       # Shape: [N, T]
        cols_idx = [columns_idx[col] for col in cols]
        is_price_group = 'open' in cols
        
        
        scaler = computeFit(scaler_path, inputs_train_array, outputs_train_array, inputs_mask, outputs_mask, cols_idx)
        values, rows = computeTransformation(values, rows, scaler, fake_candle, cols_idx, is_price_group)
            
    return values, rows

def computeFit(scaler_path, inputs, outputs, inputs_mask, outputs_mask, cols_idx):
    if not scaler_path.is_file():
        scaler = StandardGlobalScaler()
        scaler.fit([
            inputs[:, :, cols_idx][inputs_mask.astype(bool)],  # First 4 cols without the mask
            outputs[:, :, cols_idx][outputs_mask.astype(bool)]
        ])
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
    return scaler

def computeTransformation(values, rows, scaler, fake_candle, cols_idx, is_price_group):
    mins, maxs, means, stds, counts = {}, {}, {}, {}, {}
    for entry, vals in values.items():
        if 'mask' in entry:
            continue
        
        for set_type, val in vals.items():
            mask = np.array(values[f"{entry}_mask"][set_type], bool)
            values[entry][set_type] = transformData(val, scaler, mask, fake_candle, cols_idx=cols_idx)
            
            if is_price_group: # only for the price candles
                valid_vals = values[entry][set_type][:, :, cols_idx][mask]
                mins, maxs, means, stds, counts = collectStats(valid_vals, set_type, mask, mins, maxs, means, stds, counts)
                
                
    if is_price_group: # only for the price candles
        rows = computeStats(rows, mins, maxs, means, stds, counts)
    
    return values, rows

def computeStats(rows, mins, maxs, means, stds, counts):
    for set_type in mins: # no need weight as all "entry" have the same size
        total_count = sum(counts[set_type])
        mean_weighted = np.sum([m * c for m, c in zip(means[set_type], counts[set_type])]) / total_count
        std_weighted = np.sqrt(np.sum([((s**2) * c) for s, c in zip(stds[set_type], counts[set_type])]) / total_count)
        if (set_type, 'min') not in rows:
            rows[(set_type, 'min')] = [min(mins[set_type])]
            rows[(set_type, 'max')] = [max(maxs[set_type])]
            rows[(set_type, 'mean')] = [mean_weighted]
            rows[(set_type, 'std')] = [std_weighted]
        else:
            rows[(set_type, 'min')].append(min(mins[set_type]))
            rows[(set_type, 'max')].append(max(maxs[set_type]))
            rows[(set_type, 'mean')].append(mean_weighted)
            rows[(set_type, 'std')].append(std_weighted)
    return rows

def collectStats(valid_vals, set_type, mask, mins, maxs, means, stds, counts):
    if set_type not in mins:
        mins    [set_type] = [np.min    (valid_vals)]
        maxs    [set_type] = [np.max    (valid_vals)]
        means   [set_type] = [np.mean   (valid_vals)]
        stds    [set_type] = [np.std    (valid_vals)]
        counts  [set_type] = [np.sum(mask)]
    else:
        mins    [set_type].append(np.min    (valid_vals))
        maxs    [set_type].append(np.max    (valid_vals))
        means   [set_type].append(np.mean   (valid_vals))
        stds    [set_type].append(np.std    (valid_vals))
        counts  [set_type].append(np.sum(mask))
    # already modified the object but for the readibility
    return mins, maxs, means, stds, counts

def transformData(values, scaler, mask, fake_candle, cols_idx):
    """Apply the respective scaler to each slice and concatenate"""
    values = np.array(values)  # Ensure conversion
    if len(values) > 0:
        values[:, :, cols_idx] = scaler.transform(values[:, :, cols_idx])
        values[~mask] = fake_candle
    return values


def scale(split_dict, cols_to_group_for_scale, columns_idx, path):
    path = Path(path)
    inputs_train, inputs_val, inputs_test = [], [], []
    outputs_train, outputs_val, outputs_test = [], [], []
    inputs_mask_train, inputs_mask_val, inputs_mask_test = [], [], []
    outputs_mask_train, outputs_mask_val, outputs_mask_test = [], [], []
    index = []
    rows = {}
    fake_candle = np.zeros(next(iter(split_dict.values()))['inputs']['train'][0].shape[1])  
    
    for key, values in tqdm(split_dict.items(), total=len(split_dict), mininterval=1):
        market = key.split('_')[0].lower()
        index.append(key)
        
        values, rows = computeGroups(values, rows, path, cols_to_group_for_scale, market, columns_idx, fake_candle)

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
    
    columns = pd.MultiIndex.from_tuples(rows.keys(), names=["Set", "Stat"])
    df = pd.DataFrame(rows, index=index)
    df.columns = columns
    print(df)

    return inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test, inputs_mask_train, inputs_mask_val, inputs_mask_test, outputs_mask_train, outputs_mask_val, outputs_mask_test


def scaleAndShuffle(datas_path, id, split_str, split_dict, cols_to_group_for_scale, columns_idx, seed=42):
    print('\nScaler and Shuffle datas set')
    scaler_path = datas_path / f'scaler/{id}/{split_str}'
    scaler_path.mkdir(parents=True, exist_ok=True)
    
    inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test, inputs_mask_train, inputs_mask_val, inputs_mask_test, outputs_mask_train, outputs_mask_val, outputs_mask_test = scale(split_dict, cols_to_group_for_scale, columns_idx, path=scaler_path)
    
    inputs_train, outputs_train, inputs_mask_train, outputs_mask_train = sklearn.utils.shuffle(inputs_train, outputs_train, inputs_mask_train, outputs_mask_train, random_state=seed)
    inputs_val, outputs_val, inputs_mask_val, outputs_mask_val = sklearn.utils.shuffle(inputs_val, outputs_val, inputs_mask_val, outputs_mask_val, random_state=seed)
    inputs_test, outputs_test, inputs_mask_test, outputs_mask_test = sklearn.utils.shuffle(inputs_test, outputs_test, inputs_mask_test, outputs_mask_test, random_state=seed)
    datas_dict = {'inputs_train':inputs_train, 'inputs_val':inputs_val, 'inputs_test':inputs_test, 'outputs_train':outputs_train, 'outputs_val':outputs_val, 'outputs_test':outputs_test, 'inputs_mask_train':inputs_mask_train, 'inputs_mask_val':inputs_mask_val, 'inputs_mask_test':inputs_mask_test, 'outputs_mask_train':outputs_mask_train, 'outputs_mask_val':outputs_mask_val, 'outputs_mask_test':outputs_mask_test}
    return datas_dict