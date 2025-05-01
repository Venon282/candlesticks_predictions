import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm

# internal
from TechnicalIndicators import TechnicalIndicators

def dfFirstValidIndex(df):
    first_valid_index = 0
    # Get the max first valid index across all columns
    for col_name, data in df.items():
        first_valid_index = max(first_valid_index, data.first_valid_index())
    return first_valid_index



def addIndicatorsToDataSets(dfs_dict, indicators_to_add=None, params=None):
    tech_ind = TechnicalIndicators(indicators=indicators_to_add, params=params) # Create the TechnicalIndicators object
    
    # For each dataframe, add the indicators
    for key, df in dfs_dict.items():
        tech_ind.add_indicators(df, inplace=True)
        
        # remove firsts indexs with nan. Insure not nan drop in the middle of the data set.
        first_idx = dfFirstValidIndex(df)
        dfs_dict[key] = df.loc[first_idx:].reset_index(drop=True)
    return dfs_dict
        
def improveDf(df, dtype, cols_to_drop=[]):
    df.rename(columns=lambda x: re.sub(r'[<>]','',x.lower()), inplace=True) # Simplify column names
    df.drop(cols_to_drop, axis=1, inplace=True)                             # Drop undesired cols
    df = df.map(lambda x: x.replace(',', '.') if isinstance(x, str) else x) # Correct number format
    
    # Convert all object columns to float
    obj_cols = df.select_dtypes(include=['object']).columns
    df[obj_cols] = df[obj_cols].astype(float)
    
    # Convert cols to the good type
    if dtype is not None:
        df = df.astype({key: value for key, value in dtype.items() if key in df} )
   
    return df

def dfsDict(path, sep=',', encoding='utf-8-sig', n_cols=None, cols_to_drop=None, dtype=None):
    dfs_dict = {}                               # dataframes dictionary
    files_path = list(Path(path).glob('*.csv')) # list all file paths
    
    for file_path in tqdm(files_path, total=len(files_path), mininterval=1):

        df = pd.read_csv(file_path, sep=sep, encoding=encoding, dtype=str)
        
        # define the number of cols in the file
        expected_cols = n_cols or len(df)
        
        # If a problem with a csv, try to correct it with the separator
        if len(df.columns) != expected_cols:
            for separator in (r'\t', '  ', ',', ';'):
                df = pd.read_csv(file_path, encoding=encoding, sep=separator, on_bad_lines='skip')
                if len(df.columns) == expected_cols:
                    df.to_csv(file_path, sep=sep, index=False)
                    break
            else:
                raise ValueError(f'{file_path.stem} have {len(df.columns)} cols instead of {expected_cols}')
        
        # improve and add it to the dictionary
        dfs_dict[file_path.stem] = improveDf(df, dtype, cols_to_drop=cols_to_drop)
    return dfs_dict

def prepareDataSet(datas_path, cols_to_drop, indicators_to_add):
    print('\nPrepare datas set')
    
    # Read datas and store them into a dictionary
    dfs_dict = dfsDict(datas_path / 'raw', sep=';', n_cols=9, cols_to_drop=cols_to_drop,
                       dtype={'<DATE>':str, '<TIME>':str, '<OPEN>':'float', '<HIGH>':'float', '<LOW>':'float', '<CLOSE>':'float', '<TICKVOL>':'int', '<VOL>':'int', '<SPREAD>':'int'})
    
    # Add the desire indicators
    dfs_dict = addIndicatorsToDataSets(dfs_dict, indicators_to_add = indicators_to_add) 
    print(dfs_dict[next(iter(dfs_dict))].head(5), '\n')
    return dfs_dict