import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.model_selection import train_test_split, KFold
from ax.service.ax_client import AxClient, ObjectiveProperties
from .KFold import KFoldsCrossValidator

def load_data(
        file_path: str,
        id_col: str,
        test_ratio: float=0.8,
        valid_ratio: float=None,
        shuffle: bool=False,
        random_state: int=None,

) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataset and split data into training and test sets.

    Args:
        file_path: path to a csv file.
        train_size: proportion of the train data to the test data.
        shuffle: wheather shuffle data when spliting data into train and test sets.
        random_state: control the shuffling before spliting data into train and test sets.
    
    Returns:
        pd.DataFrame: train data
        pd.DataFrame: test data
    """
    df = pd.read_csv(file_path)

    # Split Run IDs into a pair of IDs for train and test
    run_ids = df[id_col].unique()
    train_ids, test_ids = train_test_split(run_ids,
                                           test_size=test_ratio,
                                           shuffle=shuffle, 
                                           random_state=random_state)
    
    # Split Run IDs into test and valid pairs
    if valid_ratio:
        valid_ids, test_ids = train_test_split(test_ids,
                                                test_size=valid_ratio,
                                                shuffle=shuffle, 
                                                random_state=random_state)


        return (
            df[df[id_col].isin(train_ids)], 
            df[df[id_col].isin(valid_ids)],
            df[df[id_col].isin(test_ids)]
        )
    
    else:
        return (
            df[df[id_col].isin(train_ids)], 
            df[df[id_col].isin(test_ids)]
        )

def kfold_data_split(
        df: pd.DataFrame,
        id_col: str,
        n_splits: int=5,
        shuffle: bool=True,
        random_state: int=None,

) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load dataset and split data into k-fold cross validation datasets.

    Args:
        file_path: path to a csv file.
        train_size: proportion of the train data to the test data.
        shuffle: wheather shuffle data when spliting data into train and test sets.
        random_state: control the shuffling before spliting data into train and test sets.
    
    Returns:
        pd.DataFrame: train data
        pd.DataFrame: test data
    """

    # Split Run IDs into a pair of IDs for train and test
    run_ids = df[id_col].unique()

    # Split Run IDs into 5-fold sets for cross validation
    kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    # store the train and test IDs in a list
    train_ids_list = []
    test_ids_list = []

    for train_index, test_index in kfold.split(run_ids):
        train_ids = run_ids[train_index]
        test_ids = run_ids[test_index]
        # store the train and test IDs in a list
        train_ids_list.append(train_ids)
        test_ids_list.append(test_ids)

    # create tuple to store the train and test data
    data_list = []
    for train_ids, test_ids in zip(train_ids_list, test_ids_list):
        data_list.append((
            df[df[id_col].isin(train_ids)], 
            df[df[id_col].isin(test_ids)]
        ))
    
    return data_list


def load_data_top_middle_btm(file_path: str,
        id_col: str,):
    df = pd.read_csv(file_path)
    return (df[df['Titer_group']!='Middle 60%'], df[df['Titer_group']=='Middle 60%'])

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models


# utilities for BO hyperparameter tuning
def create_ax_client(experiment_name, parameters, seed = 8):
    """ 
    Create an experiment with required arguments: name, parameters, and objective_name.
    """
    
    # Define the initial Sobol step with the desired number of trials
    initial_sobol_samples = 5  # Set this to your desired number

    sobol_step = GenerationStep(
        model=Models.SOBOL,
        num_trials=initial_sobol_samples,  # Number of Sobol trials
        model_kwargs={"seed": seed},  # Set the seed for Sobol sampling
    )

    # Define the Bayesian Optimization step
    bo_step = GenerationStep(
        model=Models.GPEI,
        num_trials=-1,  # -1 means that Bayesian Optimization will continue indefinitely
    )

    # Create the GenerationStrategy
    generation_strategy = GenerationStrategy(
        steps=[sobol_step, bo_step]
    )
    # Initialize AxClient with the custom generation strategy
    ax_client = AxClient(generation_strategy=generation_strategy)
    ax_client.create_experiment(
        name=experiment_name,  # The name of the experiment.
        parameters=parameters,
        objectives={"validation_loss": ObjectiveProperties(minimize=True)},  # The objective name and minimization setting.
        # parameter_constraints: Optional, a list of strings of form "p1 >= p2" or "p1 + p2 <= some_bound".
        # outcome_constraints: Optional, a list of strings of form "constrained_metric <= some_bound".
    )    
    return ax_client
