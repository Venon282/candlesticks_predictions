from py_libraries.object import flatten

def colsToGroupForScale(columns, base={}):
    similar_cols = {}
    base_flatten = flatten(base)

    for col in columns:    
        # Skip cols already handled                   
        if col in base_flatten:     
            continue
        
        head = col.split('_')[0] # Get group name  
        
        # Create the groups
        if head in similar_cols:
            similar_cols[head].append(col)
        else:
            similar_cols[head] = [col]
    
    # Extend the base with the new groups
    cols = list(similar_cols.values())
    if len(cols) > 0:
        if 'MinMaxGlobalScaler' not in base:
            base['MinMaxGlobalScaler'] = cols
        else:
            base['MinMaxGlobalScaler'].extend(cols)
    
    return base

def handleColumns(dfs_dict, cols_group={}):
    # Prepare similar columns to have the same/different scaler
    columns = next(iter(dfs_dict.values())).columns.to_list()   # Get columns
    
    columns_idx = {col: i for i, col in enumerate(columns)}     # Get columns index
    cols_to_group_for_scale = colsToGroupForScale(columns, base=cols_group)
    
    print('cols to group: ', cols_to_group_for_scale)
    
    return columns, columns_idx, cols_to_group_for_scale