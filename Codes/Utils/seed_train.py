def get_seed_train_test_data(data_set, seed_data_type):
    '''
    return train and test data based with/without seed data.
    
    Params
    ------
        data_set: (train data, test data)
            Tuple of original train and test data set.
        seed_data_type: int
            The seed data type No.
            0: no seed data.
            1: all seed data (80, 400, N-1).
            2: N-1 seed data.
            3: N-2 and N-1 seed data.
    Returns
    -------
        (train data, test data)
    '''
    
    # columns to drop
    cols_to_drop = ['input_step_N-1', 'input_step_N-1_N-2', 'input_step_N-1_N-2_N-3', 'input_step_N-2']

    # get train and test data
    train_data, test_data = data_set
    
    if seed_data_type == 1:
        input_step_col = 'input_step_N-1_N-2_N-3'
        train_df = train_data.copy()
        test_df = test_data.copy()
        scale_df = train_data[train_data['Stage']=='N'].copy()
    
    elif seed_data_type == 0:
        input_step_col = 'input_step'
        train_df = train_data[train_data['Stage']=='N'].copy()
        train_df = train_df.reset_index(drop=True)
        test_df = test_data[test_data['Stage']=='N'].copy()
        test_df = test_df.reset_index(drop=True)
        scale_df = train_data[train_data['Stage']=='N'].copy()

        
    elif seed_data_type == 2:
        input_step_col = 'input_step_N-1'
        train_df = train_data[(train_data['Stage']=='N-1')|(train_data['Stage']=='N')].copy()
        train_df = train_df.reset_index(drop=True)
        test_df = test_data[(test_data['Stage']=='N-1')|(test_data['Stage']=='N')].copy()
        test_df = test_df.reset_index(drop=True)
        scale_df = train_data[train_data['Stage']=='N'].copy()

    elif seed_data_type == 3:
        input_step_col = 'input_step_N-1_N-2'
        train_df = train_data[(train_data['Stage']=='N-2')|(train_data['Stage']=='N-1')|(train_data['Stage']=='N')].copy()
        train_df = train_df.reset_index(drop=True)
        test_df = test_data[(test_data['Stage']=='N-2')|(test_data['Stage']=='N-1')|(test_data['Stage']=='N')].copy()
        test_df = test_df.reset_index(drop=True)
        scale_df = train_data[train_data['Stage']=='N'].copy()

    elif seed_data_type == 4:
        input_step_col = 'input_step_N-1_N-2_N_3'
        train_df = train_data[(train_data['Stage']=='N-3')|(train_data['Stage']=='N')].copy()
        train_df = train_df.reset_index(drop=True)
        test_df = test_data[(test_data['Stage']=='N-3')|(test_data['Stage']=='N')].copy()
        test_df = test_df.reset_index(drop=True)
        scale_df = train_data[train_data['Stage']=='N'].copy()

    elif seed_data_type == 5:
        input_step_col = 'input_step_N-1_N-2_N_3'
        train_df = train_data[(train_data['Stage']=='N-3')|(train_data['Stage']=='N-1')|(train_data['Stage']=='N')].copy()
        train_df = train_df.reset_index(drop=True)
        test_df = test_data[(test_data['Stage']=='N-3')|(test_data['Stage']=='N-1')|(test_data['Stage']=='N')].copy()
        test_df = test_df.reset_index(drop=True)
        scale_df = train_data[train_data['Stage']=='N'].copy()

    else:
        print("seed_data_type must be 0, 1, 2, or 3.")
        return None, None
    
    # get and updata input step data
    train_df['input_step'] = train_df[input_step_col]
    test_df['input_step'] = test_df[input_step_col]
    scale_df['input_step'] = scale_df[input_step_col]

    # Drop the columns
    train_df.drop(columns=cols_to_drop, inplace=True)
    test_df.drop(columns=cols_to_drop, inplace=True)
    scale_df.drop(columns=cols_to_drop, inplace=True)

    return (train_df, test_df, scale_df)