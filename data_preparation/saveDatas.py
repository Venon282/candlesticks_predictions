import joblib
from tqdm import tqdm
import numpy as np
import pandas as pd

def saveAndDisplayDatas(datas_dict, save_path='./datas'):
    columns = ['Shape', 'Min', 'Max', 'Mean', 'Std']
    index_names = []
    rows = []
    for key, values in tqdm(datas_dict.items(), total=len(datas_dict), mininterval=1):
        if len(values) > 0:
            rows.append([np.array(values).shape, np.min(values), np.max(values), np.mean(values), np.std(values)])
            key_split = key.split('_')
            index_names.append([key_split[0], 'mask' if len(key_split) == 3 else '', key_split[-1]])
            file_path = save_path  / f'{key}.pkl'
            if (file_path).exists():
                print(f'Warning: {key}.pkl already exists and will be overwritten.')
            if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                print(f'Warning: {key} contains NaNs or infs!')
            joblib.dump(np.array(values), file_path)
    index_names = pd.MultiIndex.from_frame(pd.DataFrame(index_names, columns=['Type', '', 'Set']))
    print(pd.DataFrame(rows, columns=columns, index=index_names))

def saveDatas(datas_path, id, split_str, datas_dict):
    print('\nSave datas')
    
    save_path = datas_path / f'split/{id}/{split_str}'
    save_path.mkdir(parents=True, exist_ok=True)
    saveAndDisplayDatas(datas_dict, save_path=save_path)