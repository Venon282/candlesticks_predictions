import time
import pandas as pd
from pathlib import Path
from itertools import combinations
import numpy as np
import joblib

from tqdm import tqdm

# internal
from data_preparation.prepareDataSet import prepareDataSet
from data_preparation.handleColumns import handleColumns
from data_preparation.dfsDtToDatas import dfsDtToDatas
from data_preparation.splitDataSet import splitDataSet
from data_preparation.scaleAndShuffle import scaleAndShuffle
from data_preparation.saveDatas import saveDatas



    
def main(n_candle_input_min = 10, n_candle_input_max = 60, 
         n_candle_output_min = 1, n_candle_output_max = 6, 
         step=10, size_coherence=True,  split_rates=[0.7, 0.15, 0.15],
         indicators_to_add=['rsi', 'macd', 'bollinger'], 
         cols_to_drop=['date', 'time'],
         sep_file = '_', datas_path='./datas', seed=42):
    """
        size_coherence if true, n candle output can't be greater than n candle input in the random
    """
    datas_path = Path(datas_path)
    
    # Verify split rates
    if not np.isclose(sum(split_rates), 1.0):
        raise ValueError(f'split_rates sum must be 1')
    
    # prepare split name folder
    split_str = sep_file.join(map(str, split_rates))
    split_str = split_str.replace('0.', '').replace('.', '')
    
    # prepare id name folder
    id = f'({n_candle_input_min}-{n_candle_input_max}){sep_file}({n_candle_output_min}-{n_candle_output_max}){sep_file}step={step}{sep_file}sc={size_coherence}'
    if len(indicators_to_add) > 0:
        id +=  sep_file + sep_file.join(indicators_to_add)
    if len(cols_to_drop) > 0:
        id += sep_file + 'drop(' + sep_file.join(cols_to_drop) + ')'
        
    print(f'{id=}')
    print(f'{split_str=}')
        
    #todo add the parameters for the technical indicators
    dfs_dict = prepareDataSet(datas_path, cols_to_drop, indicators_to_add)
    
    columns, columns_idx, cols_to_group_for_scale = handleColumns(dfs_dict)
    
    datas_dict = dfsDtToDatas(dfs_dict, n_candle_input_min, n_candle_input_max, 
                                                       n_candle_output_min, n_candle_output_max,
                                                       size_coherence=size_coherence, step=step,
                                                       max_workers=4, seed=seed)
    
    split_dict = splitDataSet(datas_dict, split_rates, seed=seed)
    
    datas_dict = scaleAndShuffle(datas_path, id, split_str, split_dict, cols_to_group_for_scale, columns_idx, seed=seed)
    saveDatas(datas_path, id, split_str, datas_dict)

if __name__ == '__main__':
    start_time = time.time()
    indicators = ['rsi', 'macd', 'bollinger']
    combined_indicators = list(combinations(indicators, 2)) + list(combinations(indicators, 3))
    for indicators_to_add in combined_indicators:
        main(
                n_candle_input_min = 40, 
                n_candle_input_max = 40, 
                n_candle_output_min = 1, 
                n_candle_output_max = 1, 
                step=1, 
                size_coherence=True,  
                split_rates=[0.935, 0.05, 0.015],
                indicators_to_add=indicators_to_add,
                cols_to_drop=['date', 'time', 'tickvol', 'vol', 'spread'],
                sep_file = '_', 
                datas_path=r'E:\csp',
                seed=42
            )
    print(f'Time: {time.time() - start_time}s')
