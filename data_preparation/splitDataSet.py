from py_libraries.ml.preprocessing import trainTestSplit
from tqdm import tqdm
import pandas as pd
import numpy as np

def displaySplit(split_dict):
    column_names = pd.DataFrame([['inputs', '', 'train'],
                             ['inputs', '', 'val'],
                             ['inputs', '', 'test'],
                             ['inputs', 'mask', 'train'],
                             ['inputs', 'mask', 'val'],
                             ['inputs', 'mask', 'test'],
                             ['outputs', '', 'train'],
                             ['outputs', '', 'val'],
                             ['outputs', '', 'test'],
                             ['outputs', 'mask', 'train'],
                             ['outputs', 'mask', 'val'],
                             ['outputs', 'mask', 'test']],
                            columns=['Type', '', 'Set'])
    
    rows = [[np.array(val['inputs']['train']).shape,
             np.array(val['inputs']['val']).shape,
             np.array(val['inputs']['test']).shape,
             np.array(val['inputs_mask']['train']).shape,
             np.array(val['inputs_mask']['val']).shape,
             np.array(val['inputs_mask']['test']).shape,
             np.array(val['outputs']['train']).shape,
             np.array(val['outputs']['val']).shape,
             np.array(val['outputs']['test']).shape,
             np.array(val['outputs_mask']['train']).shape,
             np.array(val['outputs_mask']['val']).shape,
             np.array(val['outputs_mask']['test']).shape,
             ] for val in split_dict.values()]
    
    columns = pd.MultiIndex.from_frame(column_names)
    iddex = list(split_dict.keys())
    print(pd.DataFrame(rows, columns=columns, index=iddex))
    
def splitDict(datas_dict, train=0.7, val=0.15, test=0.15, seed=42):
    split_dict = {}
    
    # For each market split the datas into 3 sets
    for key, values in tqdm(datas_dict.items(), total=len(datas_dict), mininterval=1):
        inputs_train, inputs_temp, outputs_train, outputs_temp, inputs_mask_train, inputs_mask_temp, outputs_mask_train, outputs_mask_temp = trainTestSplit(values['inputs'], values['outputs'], values['inputs_mask'], values['outputs_mask'], test_size=val+test, random_state=seed)
        inputs_val, inputs_test, outputs_val, outputs_test, inputs_mask_val, inputs_mask_test, outputs_mask_val, outputs_mask_test = trainTestSplit(inputs_temp, outputs_temp, inputs_mask_temp, outputs_mask_temp, test_size=test / (val+test), random_state=seed)
        split_dict[key] = {'inputs':{'train':inputs_train, 'val':inputs_val, 'test':inputs_test},
                           'outputs':{'train':outputs_train, 'val':outputs_val, 'test':outputs_test},
                           'inputs_mask':{'train':inputs_mask_train, 'val':inputs_mask_val, 'test':inputs_mask_test},
                           'outputs_mask':{'train':outputs_mask_train, 'val':outputs_mask_val, 'test':outputs_mask_test}}
    return split_dict

def splitDataSet(datas_dict, split_rates, seed=42):
    print('\nSplit datas set')
    split_dict = splitDict(datas_dict, train=split_rates[0], val=split_rates[1], test=split_rates[2], seed=seed)
    displaySplit(split_dict)
    
    return split_dict