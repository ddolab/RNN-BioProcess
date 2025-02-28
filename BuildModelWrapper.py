import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

import torch

# Import Classes and Functions
from Codes.DatasetGenerator import DatasetGenerator
from Codes.Utils.utils import load_data_top_middle_btm, load_data, create_ax_client, kfold_data_split
from Codes.Utils.seed_train import get_seed_train_test_data


# Import Model 
from Codes.DeepLearningModels.Vec2Seq_RNN import Vec2Seq
from Codes.DeepLearningModels.Vec2Seq_MLP import Vec2Seq_MLP
from Codes.DeepLearningModels.Seq2seq import Seq2Seq

# Import Directory Constants
from Codes.DIRs import RAW_DATA_DIR

import argparse


def train_evaluate(num_epochs, features, labels, data_sets, input_step, seed_data_type, model_type, PARAMS, SEED, folder_path):
    
    # define the hyperparameters
    if model_type == 'Vec2Seq' or model_type == 'Vec2Seq_MLP':
        # hyperparameters
        alpha = PARAMS['alpha']
        hidden_size = PARAMS['hidden_size']
        l_rate = PARAMS['l_rate']
        batch_size = PARAMS['batch_size']
        wd = PARAMS['wd']
    
    elif model_type == 'Seq2Seq' or model_type == 'Seq2Seq_attn' or model_type == 'Seq2Seq_multimodal':
        # hyperparameters
        alpha = PARAMS['alpha']
        encoder_hidden_size = PARAMS['hidden_size']
        decoder_hidden_size = PARAMS['hidden_size']
        l_rate = PARAMS['l_rate']
        batch_size = PARAMS['batch_size']
        wd = PARAMS['wd']


    # fixed parameters
    num_epochs = num_epochs
    model_structure = 'gru'
    if model_type == 'Vec2Seq_MLP':
        non_linearity = 'ReLU'
        dropout = 0.0
        num_layers = 3
    else:
        non_linearity = 'tanh'
        dropout = 0.0
        num_layers = 2
    
    # create empty list to store the loss
    train_losses = []
    valid_losses = []

    # k-fold cross validation
    for k_fold, data_set in enumerate(data_sets):
        print(f'======================================== fold #{k_fold}==============================================')
        folder_path_new = f'{folder_path}/fold_{k_fold}'
        # create train, test, scale data
        train_df, test_df, scale_df = get_seed_train_test_data(data_set=data_set, seed_data_type=seed_data_type)
        train_df['Stage'].unique(), test_df['Stage'].unique()
        num_external_inputs = len(features) - len(labels)


        # count the control inputs
        if num_external_inputs > 0:
            external_input_ind = True
        else:
            external_input_ind = False


        input_step_col = 'input_step'

        seed_input_step = train_df[train_df['Stage']!='N'][input_step_col].unique().size
        input_length = seed_input_step + input_step
        output_length = train_df[input_step_col].unique().size - input_length



        dataset = DatasetGenerator(id_col='ID', 
                                            time_col=input_step_col,
                                            input_length=input_length, 
                                            output_length=output_length, 
                                            train_df=train_df,
                                            scale_df=scale_df,
                                            test_df=test_df, 
                                            scaling='min_max', #standard, min_max, mean 
                                            feature_columns=features, 
                                            label_columns=features, 
                                            batch_size=batch_size)
        # Data loader
        (train_dl, train_lengths), (test_dl, test_lengths) = dataset.train, dataset.test
        # indices of parameters whoes "glc feed" true values are used in training
        indices = dataset.label_columns_indices

        num_input = len(features)- 1
        num_output = len(labels) - 1

        if model_type == 'Seq2Seq':
            test_model = Seq2Seq(model_struct=model_structure,
                            num_features=num_input,
                            num_external_inputs=num_external_inputs,
                            num_labels=num_output,
                            encoder_hidden_size=encoder_hidden_size,
                            decoder_hidden_size=decoder_hidden_size,
                            num_layers=num_layers,
                            non_linearity=non_linearity,
                            dropout=dropout,
                            bi_direct=False,
                            export_path=folder_path_new,external_input_ind=external_input_ind, alpha=alpha)
        elif model_type == 'Vec2Seq':
            test_model = Vec2Seq(model_struct=model_structure,
                            num_features=num_input,
                            num_external_inputs=num_external_inputs,
                            num_labels=num_output,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            non_linearity=non_linearity,
                            dropout=dropout,
                            bi_direct=False,
                            export_path=folder_path_new,external_input_ind=external_input_ind, alpha=alpha)
            
        elif model_type == 'Vec2Seq_MLP':
            test_model = Vec2Seq_MLP(model_struct=model_structure,
                            num_features=num_input,
                            num_external_inputs=num_external_inputs,
                            num_labels=num_output,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            non_linearity=non_linearity,
                            dropout=dropout,
                            bi_direct=False,
                            export_path=folder_path_new,external_input_ind=external_input_ind, alpha=alpha)

        train_loss, valid_loss, best_model_state = test_model.train_BO(dataset, label_indices=indices,
                        train_data_loader=train_dl,
                        train_lengths=train_lengths,
                        valid_data_loader=test_dl,
                        valid_lengths=test_lengths,
                        epochs=num_epochs,
                        lr=l_rate,
                        wd=wd,
                        log_epochs=10,
                        save_fig=False
                        )
        # store the loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
    
    # return the average loss
    train_loss = np.mean(train_losses)
    valid_loss = np.mean(valid_losses)

    return train_loss, valid_loss

def test_evaluate(num_epochs, features, labels, data_set, input_step, seed_data_type, model_type, PARAMS, SEED, folder_path):
    # define the hyperparameters
    if model_type == 'Vec2Seq' or model_type == 'Vec2Seq_MLP':

        # hyperparameters
        alpha = PARAMS['alpha']
        hidden_size = PARAMS['hidden_size']
        l_rate = PARAMS['l_rate']
        batch_size = PARAMS['batch_size']
        wd = PARAMS['wd']
    
    elif model_type == 'Seq2Seq' or model_type == 'Seq2Seq_attn' or model_type == 'Seq2Seq_multimodal':
        # hyperparameters
        alpha = PARAMS['alpha']
        encoder_hidden_size = PARAMS['hidden_size']
        decoder_hidden_size = PARAMS['hidden_size']
        l_rate = PARAMS['l_rate']
        batch_size = PARAMS['batch_size']
        wd = PARAMS['wd']

    

    # fixed parameters
    num_epochs = num_epochs
    model_structure = 'gru'
    if model_type == 'Vec2Seq_MLP':
        non_linearity = 'ReLU'
        dropout = 0.0
        num_layers = 3
    else:
        non_linearity = 'tanh'
        dropout = 0.0
        num_layers = 2

    print('===================================== train on the full dataset and report test errors =======================================')
    
    print(f'best parameters: {PARAMS}')
    print(f'input_step={input_step}, seed_data_type={seed_data_type}, model_type={model_type}')
    # create train, test, scale data
    train_df, test_df, scale_df = get_seed_train_test_data(data_set=data_set, seed_data_type=seed_data_type)
    train_df['Stage'].unique(), test_df['Stage'].unique()
    num_external_inputs = len(features) - len(labels)


    # count the control inputs
    if num_external_inputs > 0:
        external_input_ind = True
    else:
        external_input_ind = False


    input_step_col = 'input_step'

    seed_input_step = train_df[train_df['Stage']!='N'][input_step_col].unique().size
    input_length = seed_input_step + input_step
    output_length = train_df[input_step_col].unique().size - input_length #+ 1
    shift_width = output_length #- 1



    dataset = DatasetGenerator(id_col='ID', 
                                        time_col=input_step_col,
                                        input_length=input_length, 
                                        output_length=output_length, 
                                        train_df=train_df,
                                        scale_df=scale_df,
                                        test_df=test_df, 
                                        scaling='min_max', #standard, min_max, mean 
                                        feature_columns=features, 
                                        label_columns=features, 
                                        batch_size=batch_size)
    # Data loader
    (train_dl, train_lengths), (test_dl, test_lengths) = dataset.train, dataset.test
    # indices of parameters whoes "glc feed" true values are used in training
    indices = dataset.label_columns_indices

    num_input = len(features)- 1
    num_output = len(labels) - 1

    if model_type == 'Seq2Seq':
        test_model = Seq2Seq(model_struct=model_structure,
                        num_features=num_input,
                        num_external_inputs=num_external_inputs,
                        num_labels=num_output,
                        encoder_hidden_size=encoder_hidden_size,
                        decoder_hidden_size=decoder_hidden_size,
                        num_layers=num_layers,
                        non_linearity=non_linearity,
                        dropout=dropout,
                        bi_direct=False,
                        export_path=folder_path,external_input_ind=external_input_ind, alpha=alpha)
    elif model_type == 'Vec2Seq':
        test_model = Vec2Seq(model_struct=model_structure,
                        num_features=num_input,
                        num_external_inputs=num_external_inputs,
                        num_labels=num_output,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        non_linearity=non_linearity,
                        dropout=dropout,
                        bi_direct=False,
                        export_path=folder_path,external_input_ind=external_input_ind, alpha=alpha)
        
    elif model_type == 'Vec2Seq_MLP':
        num_layers = 3
        test_model = Vec2Seq_MLP(model_struct=model_structure,
                        num_features=num_input,
                        num_external_inputs=num_external_inputs,
                        num_labels=num_output,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        non_linearity=non_linearity,
                        dropout=dropout,
                        bi_direct=False,
                        export_path=folder_path,external_input_ind=external_input_ind, alpha=alpha)

    train_loss, valid_loss, best_model_state = test_model.train_BO(dataset, label_indices=indices,
                    train_data_loader=train_dl,
                    train_lengths=train_lengths,
                    valid_data_loader=test_dl,
                    valid_lengths=test_lengths,
                    epochs=num_epochs,
                    lr=l_rate,
                    wd=wd,
                    log_epochs=10,
                    save_fig=True
                    )
    

    return train_loss, valid_loss


def main():
    """
    To executre the code, input the following lines in the command window.
    python BuildModelWrapper.py -seed_id 0 -epoch 250 -seed_type 1 -model_type 1 # for Seq2Seq model, seed data type N-1+N-2+N-3
    python BuildModelWrapper.py -seed_id 0 -epoch 250 -seed_type 2 -model_type 1 # for Seq2Seq model, seed data type N-1
    python BuildModelWrapper.py -seed_id 0 -epoch 250 -seed_type 3 -model_type 1 # for Seq2Seq model, seed data type N-1+N-2
    python BuildModelWrapper.py -seed_id 0 -epoch 250 -seed_type 0 -model_type 0 # for RNN (N-only) model
    python BuildModelWrapper.py -seed_id 0 -epoch 250 -seed_type 0 -model_type 2 # for FNN (MLP) model
    """
    # Collect input for model parameter assignment.
    parser = argparse.ArgumentParser(description='BO4IO algorithm for standard pooling problems')
    optional = parser._action_groups.pop()  # creates group of optional arguments
    required = parser.add_argument_group('required arguments')  # creates group of required arguments
    # required input
    # optional input
    optional.add_argument('-input_step', '--input_step', help='input step length', type=int, default = 1)
    optional.add_argument('-seed_type', '--seed_type', help='seed data type. 0: no seed, 1: all, 2: 2000L, 3: 2000L + 400L ', type=int, default = 3)
    optional.add_argument('-epoch', '--epoch', help='number of epochs', type=int, default = 100)
    optional.add_argument('-seed_id', '--seed_id', help='random seed', type=int, default = 8)
    optional.add_argument('-model_type', '--model_type', help='model_type: 0: Vec2Seq 1: Seq2Seq, 2: Seq2Seq_attn, 3: Vec2Seq_MLP, 4: Seq2Seq_multimodal', type=int, default = 1)
    optional.add_argument('-base', '--base', help='0: no base input, 1: with base input', type=int, default = 1)
    optional.add_argument('-train_group', '--train_group', help='0: mix, 1: top/bottom 20%, 2: middle 60%', type=int, default = 0)



    parser._action_groups.append(optional)  # add optional values to the parser
    args = parser.parse_args()  # get the arguments from the program input, set them to args


    # intialization
    input_step = args.input_step
    num_epochs = args.epoch
    BO_TRIALS = 30
    seed_data_type = args.seed_type
    if args.model_type == 0:
        model_type = 'Vec2Seq'
    elif args.model_type == 1:
        model_type = 'Seq2Seq'
    elif args.model_type == 2:
        model_type = 'Vec2Seq_MLP'

    sns.set_theme(style="darkgrid")

    pd.set_option('display.max_columns', 200)

    # print the configuration
    print("=====================================================================================================================================")
    print('Building model with the following configuration')
    print(f'num_epochs={num_epochs}, BO_TRIALS={BO_TRIALS}')
    print(f'input_step={input_step}, seed_data_type={seed_data_type}, model_type={model_type}')

    use_cuda = torch.cuda.is_available()
    print('Use CUDA:', use_cuda)

    # Set the seed for reproducibility
    SEED = args.seed_id
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Loding data and splitting it into train and test sets
    id_col = 'ID'
    # Import CSV file to dataframe
    data_file = 'Data_input_template.csv'
    data_file_path = os.path.join(RAW_DATA_DIR, data_file)

    # load and split data
    if args.train_group == 0: # train with mix titer group
        (train_data_full, test_data_full) = load_data(data_file_path, id_col, test_ratio=0.2, valid_ratio=None, shuffle=True, random_state=SEED)
    else:
        (top_btm_data_full, middle_data_full) = load_data_top_middle_btm(data_file_path, id_col)
        if args.train_group == 1: # train with top/bottom 20% titer group
            train_data_full = top_btm_data_full
            test_data_full = middle_data_full
        elif args.train_group == 2: # train with middle 60% titer group
            train_data_full = middle_data_full
            test_data_full = top_btm_data_full

    data_sets = kfold_data_split(train_data_full, id_col, n_splits=5, shuffle=True, random_state=SEED+5)
    data_sets = [*data_sets]


    # define input and output features
    if args.base == 0:
        features = ['glc', 'glc_after','lac', 'vcd', 'viab']
        labels = ['glc', 'glc_after','lac', 'vcd', 'viab']
    elif args.base == 1:
        features = ['glc', 'glc_after','lac', 'vcd', 'viab','base']
        labels = ['glc', 'glc_after','lac', 'vcd', 'viab']

    # Define the list of hyperparameters and their search ranges for the Bayesian Optimization
    # ax_client parameters
    client_params = [
        # Range parameters
        {"name": "batch_size", "type": "range", "bounds": [10, 40], "value_type": "int",},
        {"name": "l_rate", "type": "range", "bounds": [1e-4, 1e-2], "value_type": "float","log_scale": True},
        {"name": "hidden_size", "type": "range", "bounds": [16,128], "value_type": "int"},
        {"name": "alpha", "type": "range", "bounds": [1e-8, 1e-4], "value_type": "float","log_scale": True},
        {"name": "wd", "type": "range", "bounds": [1e-4, 1e-1], "value_type": "float","log_scale": True},
    ]

    # Parameters
    PARAMS = {
        "alpha": 0.0000001,
        "hidden_size": 64,
        "l_rate": 0.001,
        "batch_size": 10,
        "wd": 0.1
    }
            

    # Hyperparameters
    hyper_parameters = PARAMS.copy()
    print("=====================================================================================================================================")
    print('Initiating Bayesian Optimization')
    # creating ax_client
    ax_client = create_ax_client(experiment_name=f'{model_type}_BO_tuning', parameters=client_params, seed=SEED)
    print(ax_client.get_max_parallelism())
    # Attach the trial
    print('Attach the first trial to the AX client')
    ax_client.attach_trial(parameters=hyper_parameters)

    # Get the parameters and run the trial for the intial baseline
    print("=====================================================================================================================================")
    print('Baseline Trial')
    baseline_parameters = ax_client.get_trial_parameters(trial_index=0)

    trial_index = 0
    # folder path to save data
    if args.base == 0:
        folder_path = f'output_models/{model_type}/step={input_step}_seed_data_type={seed_data_type}_seed={SEED}_no_base'
    else:
        folder_path = f'output_models/{model_type}/step={input_step}_seed_data_type={seed_data_type}_seed={SEED}_base'
        

    train_loss, valid_loss = train_evaluate(num_epochs, features, labels, data_sets, input_step, seed_data_type, model_type, baseline_parameters, SEED,f'{folder_path}/BOtrial={trial_index}')


    ax_client.complete_trial(trial_index=0, 
                             raw_data=valid_loss
                            )
    
    # Train a model with the base parametrs and evaluate the test data
    initial_best_parameters, values = ax_client.get_best_parameters()

    # print current best parameters
    print(f"============== Parameters at trial {trial_index} ======================")
    print(initial_best_parameters)
    trials = [0]
    test_losses = [np.nan]
    train_losses_full = [np.nan]
    train_losses = [train_loss]

    best_train_loss = train_loss
    best_valid_loss = valid_loss

    best_train_losses = [best_train_loss]
    best_valid_losses = [best_valid_loss]

    # test evaluate indicator
    test_evaluate_ind = True

    # Optimization loop
    print("=====================================================================================================================================")
    print("Start optimization loop")
    for i in range(1, BO_TRIALS+1):
        print(f'Optimization Loop: [{i}/{BO_TRIALS}]')
        tuning_parameters, trial_index = ax_client.get_next_trial()

        # evaluate the valid loss with the new parameters
        
        train_loss, valid_loss = train_evaluate(num_epochs, features, labels, data_sets, input_step, seed_data_type, model_type, tuning_parameters, SEED,f'{folder_path}/BOtrial={i}')

        # store the loss
        train_losses.append(train_loss)
        trials.append(i)

        # complete the trial and update the ax_client
        ax_client.complete_trial(trial_index=i, 
                                raw_data=valid_loss
                                )
        
        

        # Get the best parameters
        current_best_parameters, values = ax_client.get_best_parameters()


        
        

        # update the best loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_train_loss = train_loss
            best_train_losses.append(best_train_loss)
            best_valid_losses.append(best_valid_loss)
            # indicator to evaluate the test data
            test_evaluate_ind = True

            # print the current best parameters
            print(f"============== Current best parameters at {trial_index} trial ======================")
            print(initial_best_parameters)
            # store the best parameters
            file_path = f"{folder_path}/params.txt"
            with open(file_path, 'w') as file:
                for key, value in current_best_parameters.items():
                    file.write(f"{key}: {value}\n")
                print('Parameters file saved successfully.')

        else:
            best_train_losses.append(best_train_loss)
            best_valid_losses.append(best_valid_loss)


        # Plotting BO performance
        BO_config = ax_client.get_optimization_trace()[0]
        x = [i-1 for i in BO_config['data'][0]['x']]
        y = BO_config['data'][0]['y']
        # plt.plot(x, y, label='Validation')    
        plt.clf()
        plt.plot(trials, best_train_losses, label='Training')
        plt.plot(trials, best_valid_losses, label='Validation')
        plt.yscale('log')
        plt.title("Model performance vs. # Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.xlim([0, BO_TRIALS])
        plt.legend()
        plt.savefig(f'{folder_path}/BO_trace_plot.png')
        plt.close()
        
        # if found better hyperparameters, train on the full dataset and evaluate the test data
        if i%5==0 and i>=10 and test_evaluate_ind:
            # Train the model with the best parameters on the full datasets
            best_parameters, values = ax_client.get_best_parameters()
            train_loss_full, test_loss_full = test_evaluate(num_epochs, features, labels, (train_data_full, test_data_full), input_step, seed_data_type, model_type, best_parameters, SEED, f'{folder_path}/test_BOtrial={i}')
            test_losses.append(test_loss_full)
            train_losses_full.append(train_loss_full)

            plt.clf()
            # plot the full data training and testing losses, ignore the nan values and connect the points
            np_tmp = np.array(train_losses_full)
            plt.plot(np.array(trials)[~np.isnan(np_tmp)], np.array(train_losses_full)[~np.isnan(np_tmp)], label='Training')
            plt.plot(np.array(trials)[~np.isnan(np_tmp)], np.array(test_losses)[~np.isnan(np_tmp)], label='Testing')
            plt.yscale('log')
            plt.title("Model performance vs. # Iterations")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.xlim([0, BO_TRIALS])
            plt.legend()
            plt.savefig(f'{folder_path}/BO_trace_plot_full_data.png')
            plt.close()
            test_evaluate_ind = False
            print("========================== test evaluate ends ==================================")
        else:
            train_losses_full.append(np.nan)
            test_losses.append(np.nan)

        # save loss
        df_stats = ax_client.get_trials_data_frame()
        df_stats['train_loss'] = train_losses
        df_stats['test_loss'] = test_losses
        df_stats['train_loss_full'] = train_losses_full
        df_stats.to_csv(f"{folder_path}/BO_stats.csv")
        print(df_stats)
    
    # Saving ax_client data
    ax_client.save_to_json_file(filepath=f'{folder_path}/ax_client.json')
    print("End optimization loop")
    print("=====================================================================================================================================")

    

if __name__ ==  '__main__':
    main()