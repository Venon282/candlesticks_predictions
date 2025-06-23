import time
from pathlib import Path
from itertools import combinations, product
import numpy as np

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
         cols_group={},
         sep_file = '_', datas_path='./datas', seed=42, id=''):
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
    folder_name = f'({n_candle_input_min}-{n_candle_input_max}){sep_file}({n_candle_output_min}-{n_candle_output_max}){sep_file}step={step}{sep_file}sc={size_coherence}'
    if len(indicators_to_add) > 0:
        folder_name +=  sep_file + sep_file.join(indicators_to_add)
    if len(cols_to_drop) > 0:
        folder_name += sep_file + 'drop(' + sep_file.join(cols_to_drop) + ')'
    folder_name += sep_file + id
        
    print(f'{folder_name=}')
    print(f'{split_str=}')
        
    #todo add the parameters for the technical indicators
    dfs_dict = prepareDataSet(datas_path, cols_to_drop, indicators_to_add)
    
    columns, columns_idx, cols_to_group_for_scale = handleColumns(dfs_dict, cols_group)
    
    datas_dict = dfsDtToDatas(dfs_dict, n_candle_input_min, n_candle_input_max, 
                                                       n_candle_output_min, n_candle_output_max,
                                                       size_coherence=size_coherence, step=step,
                                                       max_workers=4, seed=seed)
    
    split_dict = splitDataSet(datas_dict, split_rates, seed=seed)
    
    datas_dict = scaleAndShuffle(datas_path, folder_name, split_str, split_dict, cols_to_group_for_scale, columns_idx, seed=seed)
    saveDatas(datas_path, folder_name, split_str, datas_dict)

if __name__ == '__main__':
    start_time = time.time()
    indicators = ['rsi', 'macd', 'bollinger']
    combined_indicators = [[i] for i in indicators] + list(combinations(indicators, 2)) + list(combinations(indicators, 3))
    columns_groups_1 = {
        'StandardGlobalScaler':[['open', 'high', 'low', 'close'], 
                                ['bollinger_sma', 'bollinger_upper', 'bollinger_lower'],
                                ['macd_hist'],
                                ['macd_line', 'macd_signal']],
        'MinMaxGlobalScaler':[['rsi']]
    }
    columns_groups_2 = {
        'StandardGlobalScaler':[['macd_hist'],
                                ['macd_line', 'macd_signal']],
        'MinMaxGlobalScaler':[['open', 'high', 'low', 'close'],
                              ['bollinger_sma', 'bollinger_upper', 'bollinger_lower'],
                              ['rsi']]
    }
    cols_groups = [['sgs', columns_groups_1], ['mmgs', columns_groups_2]]
    c = ['tickvol', 'vol', 'spread']
    cols_to_drops = list(combinations(c, 2)) + list(combinations(c, 3))
    
    for  cols_to_drop , [id, cols_group] in product(cols_to_drops, cols_groups): # indicators_to_add,product(combined_indicators, cols_groups):
        main(
                n_candle_input_min = 40, 
                n_candle_input_max = 40, 
                n_candle_output_min = 1, 
                n_candle_output_max = 1, 
                step=1, 
                size_coherence=True,  
                split_rates=[0.935, 0.05, 0.015],
                indicators_to_add=[], #indicators_to_add,
                cols_to_drop=['date', 'time'] + list(cols_to_drop), #['date', 'time', 'tickvol', 'vol', 'spread'],
                cols_group=cols_group,
                sep_file = '_', 
                datas_path=r'E:\csp',
                seed=42,
                id = id
            )
    print(f'Time: {time.time() - start_time}s')
