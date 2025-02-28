# Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence, pad_sequence
# count the number of the same value in the run_id_length
from collections import Counter
def get_scaler(scaler):
    # Select Prefered Scaler
    if scaler=='min_max':
        return MinMaxScaler(feature_range=(-1, 1))
    elif scaler=='standard':
        return  StandardScaler()
    elif scaler=='max_abs':
        return MaxAbsScaler()
    elif scaler=='robust':
        return RobustScaler(quantile_range=(5.0, 95.0))#quantile_range=(5.0, 95.0)
    else:
        return 'min_max'

def custom_transform(values, scaler1, scaler2):
    # use the same scaler for glc and glc_after
    values_glc = values[:,0].reshape(-1,1)
    values_glc_after = values[:,1].reshape(-1,1)
    values_others = values[:,2:]
    values = np.concatenate((scaler1.transform(values_glc),scaler1.transform(values_glc_after),scaler2.transform(values_others)), axis=1)
    return values

class MyTensorDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    # tensors: Tuple[Tensor, ...]

    def __init__(self, *tensors) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors), index

    def __len__(self):
        return self.tensors[0].size(0)


class DatasetGenerator:
    """
    Dataset Generator.
    This will generate datasets for training, validation, and testing.
    """
    def __init__(
            self, 
            input_length: int, 
            output_length: int, 
            train_df: pd.DataFrame, 
            scale_df: pd.DataFrame, 
            valid_df: pd.DataFrame=None, 
            test_df: pd.DataFrame=None,
            scaling: str='min_max',
            feature_columns: List=None, 
            label_columns: List=None,
            batch_size: int=2, 
            shuffle: bool=True, 
            drop_last: bool=True,
            time_col: str='Time',
            id_col: str='ID',
    ) -> None:
        self.id_col = id_col

        # get Scaler
        scaler1_1 = get_scaler(scaler=scaling)
        scaler1_2 = get_scaler(scaler=scaling)
        scaler2_1 = get_scaler(scaler=scaling)
        scaler2_2 = get_scaler(scaler=scaling)

        # work with train data
        # fill nan values in the glc_after column with the value in the glc column
        train_df['glc_after'] = train_df['glc_after'].fillna(train_df['glc'])
        self.train_data = train_df.copy()    # used for predictions for train data

        # initialization
        self.train_size = train_df[id_col].unique().size
        self.train_df = train_df[feature_columns].copy()
        
        # work with scale data
        # fill nan values in the glc_after column with the value in the glc column
        scale_df['glc_after'] = scale_df['glc_after'].fillna(scale_df['glc'])
        self.scale_data = scale_df.copy()    # used for predictions for train data

        # initialization
        self.scale_size = scale_df[id_col].unique().size
        self.scale_df = scale_df[feature_columns].copy()

        
        # extract data for scaling
        time_col = train_df[train_df['Stage']!='N']['input_step'].unique()

        # split the data into two parts, production and seed train data
        self.scale_df1 = self.train_data[~self.train_data['input_step'].isin(time_col)].copy()
        self.scale_df2 = self.train_data[self.train_data['input_step'].isin(time_col)].copy()
        
        # get scaler and scaling for each part
        # Production train data
        x1 = self.scale_df1[feature_columns].values
        scaler1_1.fit(x1[:,1].reshape(-1,1))
        scaler1_2.fit(x1[:,2:])
        self.scale_values1 = custom_transform(self.scale_df1[feature_columns].values, scaler1_1, scaler1_2)

        self.scale_df1[feature_columns] = self.scale_values1

        # Seed train data
        if not self.scale_df2[feature_columns].empty:
            x2 = self.scale_df2[feature_columns].values
            scaler2_1.fit(x2[:,1].reshape(-1,1))
            scaler2_2.fit(x2[:,2:])
            self.scale_values2 = custom_transform(self.scale_df2[feature_columns].values, scaler2_1, scaler2_2)
            self.scale_df2[feature_columns] = self.scale_values2

        # merge the scaled values
        self.scale_data = pd.concat([self.scale_df1, self.scale_df2], axis=0).sort_index()

        self.train_data[feature_columns] = self.scale_data[feature_columns]
        self.train_data['glc_after'] = self.train_data['glc_after'] #- self.train_data['glc']
        self.train_values = self.train_data[feature_columns].values
        
        # get the run ID list of train data
        train_run_id_list = train_df['ID'].unique().tolist()

        # count the length of each run ID
        train_df_length = []
        for run_id in train_run_id_list:
            train_df_length.append(len(train_df[(train_df['ID'] == run_id)]))

        # convert the train_df_length to a torch tensor
        self.train_df_length = torch.tensor(train_df_length)

        # work with validation data
        if valid_df is not None:
            # fill nan values in the glc_after column with the value in the glc column
            valid_df['glc_after'] = valid_df['glc_after'].fillna(valid_df['glc'])
            valid_df1 = valid_df[~valid_df['input_step'].isin(time_col)].copy()
            valid_df2 = valid_df[valid_df['input_step'].isin(time_col)].copy()
            valid_values1 = custom_transform(valid_df1[feature_columns].values, scaler1_1, scaler1_2)
            valid_df1[feature_columns] = valid_values1

            if not self.scale_df2[feature_columns].empty:
                valid_values2  = custom_transform(valid_df2[feature_columns].values, scaler2_1, scaler2_2)

                valid_df2[feature_columns] = valid_values2

            self.valid_df = pd.concat([valid_df1, valid_df2], axis=0).sort_index()
            self.valid_size = self.valid_df[id_col].unique().size
            self.valid_df['glc_after'] = self.valid_df['glc_after'] #- self.valid_df['glc']
            self.valid_values = valid_df[feature_columns].values

            # get the run ID list of valid data
            valid_run_id_list = valid_df['ID'].unique().tolist()

            # count the length of each run ID
            valid_df_length = []
            for run_id in valid_run_id_list:
                valid_df_length.append(len(valid_df[(valid_df['ID'] == run_id)]))

            # convert the valid_df_length to a torch tensor
            self.valid_df_length = torch.tensor(valid_df_length)


        # work with test data
        if test_df is not None:
            # fill nan values in the glc_after column with the value in the glc column
            test_df['glc_after'] = test_df['glc_after'].fillna(test_df['glc'])
            test_df1 = test_df[~test_df['input_step'].isin(time_col)].copy()
            test_df2 = test_df[test_df['input_step'].isin(time_col)].copy()
            test_values1 = custom_transform(test_df1[feature_columns].values, scaler1_1, scaler1_2)

            test_df1[feature_columns] = test_values1

            if not self.scale_df2[feature_columns].empty:
                test_values2  = custom_transform(test_df2[feature_columns].values, scaler2_1, scaler2_2)
                test_df2[feature_columns] = test_values2
            test_df = pd.concat([test_df1, test_df2], axis=0).sort_index()
            self.test_data = test_df.copy()    # used for predictions
            self.test_size = test_df[id_col].unique().size
            self.test_data['glc_after'] = self.test_data['glc_after'] #- self.test_data['glc']
            self.test_values = self.test_data[feature_columns].values

            # get the run ID list of test data
            test_run_id_list = test_df['ID'].unique().tolist()

            # count the length of each run ID
            test_df_length = []
            for run_id in test_run_id_list:
                test_df_length.append(len(test_df[(test_df['ID'] == run_id)]))
            
            # convert the test_df_length to a torch tensor
            self.test_df_length = torch.tensor(test_df_length)

        # Store other parameters
        self.feature_columns = feature_columns
        if feature_columns is not None:
            self.feature_columns_indices = {name: i for i, name in enumerate(self.feature_columns)}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Create the label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns = [name for name in feature_columns if name in label_columns]
            self.label_columns_indices = {name: i for i, name in enumerate(self.label_columns)}
        self.column_indices = {name: i for i, name in enumerate(feature_columns)}

        # Create the dataset parameters
        self.input_length = input_length
        self.output_length = output_length
        self.total_length = input_length + output_length

        self.input_slice = slice(0, input_length)
        self.input_indices = np.arange(self.total_length)[self.input_slice]

        self.label_start = self.total_length - self.output_length
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_length)[self.labels_slice]

    def split_dataset(
            self, 
            features: List
    ) -> Tuple[List, List]:
        """
        Split data into datasets.

        Args:
            features: list of features or parameters.
        
        Returns:
            List: list of features or input parameters.
            List: list of labels or output parameters.
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]

        if self.label_columns is not None:
            stack_lst = [labels[:, :, self.column_indices[name]] for name in self.label_columns]
            labels = torch.stack(stack_lst, dim=-1)

        self.inputs = inputs
        self.labels = labels

        return inputs, labels

    def make_dataset_loader(
            self, 
            data: pd.DataFrame.values,
            sequence_length: torch.tensor, 
            data_size: int, 
            batch_size: int, 
            shuffle: bool, 
            drop_last: bool
    ) -> DataLoader:
        """
        Create pytorch data loader.
        
        Returns:
            DataLoader:
        """
        # Split dataframe based on the list of sequence length
        split_array = np.split(data, np.cumsum(sequence_length)[:-1])


        # Split each data into time sequences
        input_list, label_list = list(), list()
        for data in split_array:
            stack_df = []
            stack_df.append(torch.tensor(data, dtype=torch.float32))
            data_dataset = torch.stack(stack_df)
            inputs, labels = self.split_dataset(data_dataset)
            input_list.append(inputs)
            label_list.append(labels.squeeze())

        # create TensorDataset with label output sequences that have variable length
        # get the lengths of the label sequences
        lengths = torch.tensor([label.size()[0] for label in label_list])

        # Pad multivariate sequences to the same length
        label_list = pad_sequence(label_list, batch_first=True)

        # # sort the input and label sequences by length
        input_list = torch.cat(input_list, dim=0)

        ds = MyTensorDataset(input_list, label_list)
        dl = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        drop_last=drop_last)
        return dl, lengths

    @property
    def train(self):
        ''''''
        return self.make_dataset_loader(self.train_values, self.train_df_length, self.train_size, self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)
    
    @property
    def train_full(self):
        ''''''
        return self.make_dataset_loader(self.train_values, self.train_df_length, self.train_size, 1, shuffle=False, drop_last=False)

    @property
    def valid(self):
        ''''''
        return self.make_dataset_loader(self.valid_values, self.valid_df_length, self.valid_size, self.valid_size, shuffle=self.shuffle, drop_last=self.drop_last)

    @property
    def test(self):
        '''
        shape of (RunID, Sequences, # features)
        '''
        return self.make_dataset_loader(self.test_values, self.test_df_length, self.test_size, batch_size=1, shuffle=False, drop_last=False)

    def __repr__(self):
        return '\n'.join([f'Total dataset size: {self.total_length}',
                          f'Input Indices: {self.input_indices}',
                          f'Label indices: {self.label_indices}',
                          f'Feature column name(s): {self.feature_columns}',
                          f'Label column name(s): {self.label_columns}',
                          # f'Batch size: {self.batch_size}',
                          # f'Data loader shuffle: {self.shuffle}',
                          # f'Data loader Drop Last: {self.drop_last}',
                          ])