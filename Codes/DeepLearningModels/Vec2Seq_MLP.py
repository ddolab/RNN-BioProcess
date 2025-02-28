from typing import Tuple
from datetime import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from  torch.utils.data.dataloader import DataLoader

# from .Decoder_only import Decoder_only
from .MLP import MLPModel

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import PredictionErrorDisplay
import copy


def create_glc_afeter_feed(s: torch.tensor)->torch.tensor:
    '''
    create tensor ['glc_after_feed', ..., 'other'] from ['glc', 'feed', ..., 'other'].
    '''
    # split x into s1=['glc', 'feed'] and s2=['others']
    s1, s2 = torch.split(s, [2, s.size(2)-2], dim=2)
    # 'glc' + 'feed' = 'glc_after'
    s1 = s1.sum(dim=2, keepdim=True)
    # combine x1 and x2 for the input=['glc_after', 'others']
    return torch.concat([s1, s2], dim=2)

class Vec2Seq_MLP(nn.Module):
    '''
    A Seq2seq with an encoder-decoder model for time-series squences.
    The RNN structure in the encoder and decoder can be chosen from "RNN", "LSTM", and "GRU".
    '''
    def __init__(self,
                 model_struct: str,
                 num_features: int, 
                 num_external_inputs: int, 
                 num_labels: int, 
                 hidden_size: int,
                 num_layers: int=1,
                 non_linearity: str="tanh",
                 dropout: float=0.0,
                 bi_direct: bool=False,
                 export_path: str=None,
                 external_input_ind: bool=False,
                 alpha: float=0.000001,
                 alpha_elu: float=1.0,
                 ) -> None:
        
        '''
        Args:
            model_struct: str
                The structure of the RNN model. 
                Can be either "rnn", "lstm", or "gru".
            num_features: int
                The number of features in the input.
            num_labels: int
                The number of labels in the output.
            num_external_inputs: int
                The number of external inputs = num_features - num_labels.
            hidden_size: int
                The number of features in the hidden layer.
            num_layers: int, optinal, default=1
                The number of stacked reccurent layers.
            non_linearity: str, optional, default="tanh"
                Non-linear function to use, if model_struct=="rnn".
                Can be either "relu" or "tanh".
            dropout: float, optional, default=0.0
                Probability of dropout in the dropout layer.
            bi_direct: bool, optional, default=False
                If True, the model becomes the bidirectional RNN.
            external_input_ind: bool, optional, default=False
                If true, there are extra external inputs (not predicted at output)
        '''
        super(Vec2Seq_MLP, self).__init__()
        self.model_struct = model_struct
        self.num_features = num_features
        self.num_external_inputs = num_external_inputs
        self.num_labels = num_labels
        self.H_out = hidden_size
        self.num_layers = num_layers
        self.dropout=dropout
        self.D=bi_direct
        self.external_input_ind=external_input_ind
        self.alpha = alpha
        # # Encoder
        # self.encoder = Encoder(model_struct=model_struct,
        #                        num_features=num_features,
        #                        hidden_size=hidden_size,
        #                        num_layers=num_layers,
        #                        non_linearity=non_linearity,
        #                        dropout=dropout,
        #                        bi_direct=bi_direct)

        # Decoder
        self.decoder = MLPModel(
                                n_features=num_features,
                               n_labels=num_labels,
                               hidden_dim=hidden_size,
                               layer_dim=num_layers,
                               non_linearity=non_linearity,
                               dropout=dropout, 
                               alpha_elu=alpha_elu)
        
        # Loss function
        self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()


        # Folder name to save data
        self.folder_name = None

        # Create a folder to save data
        if export_path is not None:
            # get current time
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y_%m_%d_%H_%M")
            folder_name = os.path.join(export_path, formatted_time)
            folder_name = export_path

            # Check if the folder already exists
            if not os.path.exists(folder_name):
                # Create the folder
                os.makedirs(folder_name)
                print(f"Folder '{folder_name}' created successfully.")
                self.folder_name = folder_name
            else:
                print(f"Folder '{folder_name}' already exists.")
                self.folder_name = folder_name
                # return None
            
    def compute_loss(self, y, y_hat, batch_lengths):
        '''
        Args:
            y: target matrix.
            y_hat: output matrix.
            bath_lengths: lengths of output sequences.
        Return:
            loss
        '''
        # create a mask for the padded outputs based on the batch_lengths
        mask = torch.zeros_like(y_hat)
        for i, length in enumerate(batch_lengths):
            mask[i, length:, :] = 1

        # compute the loss, ignoring the padded outputs
        y_hat = y_hat.masked_fill(mask.bool(), 0)
        y = y.masked_fill(mask.bool(), 0)
        
        return self.criterion(y, y_hat)

    def elastic_net_regularization(self, l1_lambda, l2_lambda):
        # calculate penalty term of the L1 and L2 regularization
        l1_reg = 0
        l2_reg = 0
        
        for param in self.parameters():
            l1_reg += torch.norm(param, 1)
            l2_reg += torch.norm(param, 2)
        
        return l1_lambda * l1_reg + l2_lambda * l2_reg
    
    def optimize(self, y, y_hat, batch_lengths):
        '''
        Args:
            y: target matrix.
            y_hat: output matrix.
            bath_lengths: lengths of output sequences.
        Return:
            loss
        '''
        # Computes loss and the gradients
        loss = self.compute_loss(y=y, y_hat=y_hat, batch_lengths=batch_lengths)

        original_loss = loss.item()
        
        # Hyperparameters for Elastic Net
        l1_lambda = self.alpha
        l2_lambda = self.alpha
        # Add Elastic Net regularization
        loss += self.elastic_net_regularization(l1_lambda, l2_lambda)
        
        # Computes gradients
        loss.backward() # Computes gradients
        
        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), original_loss
    
    def one_step_predict(self, x, y, teacher_forcing = True, teacher_forcing_ratio=0.0):
        '''
        compute prediction for one batch.

        Args:
            x: input batch.
            y: target batch.
            teacher_forcing: bool, optional, default=False
                If True, the model uses teacher forcing.
            teacher_forcing_ratio: float, optional, default=0.5
                The probability of using teacher forcing.
        Return:
            (prediction, true value)
        '''
        # # get the encoder input
        decoder_input = create_glc_afeter_feed(s=x)

        # split y_target into to ['glc', 'others'] and ['feed']
        y_feed = y[:, :, 1:2]
        y_targets = y[:, :, self.target_indices]

        # init output tensor
        output_batch = torch.zeros_like(y_targets)

        # predict output recursively for the decoder
        for t in range(y_targets.size(1)):
            # get one step prediction from decoder
            decoder_out = self.decoder(decoder_input)

            # change the output shape
            if len(decoder_out.size()) < 3:
                decoder_out = decoder_out.view([1, 1, -1])
            
            # store output
            output_batch[:, t:t+1, :] = decoder_out

            # concatenate ['glc'], ['feed'], and ['others'] with the true value of ['feed']
            if self.external_input_ind:
                # teacher forcing
                if teacher_forcing and torch.rand(1).item() < teacher_forcing_ratio:
                    input_tensor = y[:,t:t+1,:]
                else:
                    input_tensor = torch.concat([decoder_out[:, :, 0:1], y_feed[:, t:t+1, :], decoder_out[:, :, 1:], y[:,t:t+1,y.size(2)-self.num_external_inputs:]], dim=2)
                    
            else:
                # teacher forcing
                if teacher_forcing and torch.rand(1).item() < teacher_forcing_ratio:
                    input_tensor = y[:,t:t+1,:]
                else:
                    input_tensor = torch.concat([decoder_out[:, :, 0:1], y_feed[:, t:t+1, :], decoder_out[:, :, 1:]], dim=2)
            
            # get input tensor
            decoder_input = create_glc_afeter_feed(input_tensor)

        return (output_batch, y_targets)

    def train_BO(self, dataset, label_indices: list, 
              epochs: int, 
              train_data_loader: DataLoader, 
              train_lengths,
              valid_data_loader: DataLoader=None,    
              valid_lengths=None,          
              lr: float=0.001, 
              betas=(0.9, 0.999), 
              wd=0.01,
              log_epochs: int=False,
              save_fig: bool=False,
              kfold_ind = False):
        '''
        Args:
            label_indices: dict
                The label indices used in dataseting data.
            epochs: int
                The number of epochs.
            train_data_loader: dataloader.
                The dataloader of training data.
            valid_data_loader: dataloader, optional, default=None
                The dataloader of validation data.
            lr: float, optional, default=0.001.
                Laerning rate. 
            betas: Tuple[float, float], optional, default=(0.9, 0.999).
                Coefficients for computing running averages of gradient and its square.
            wd: float, optional, default=0.01.
                Weight decay coefficient.
        '''
        self.train_loss = []
        self.valid_loss = []
        self.valid_loss_nonnan = []
        self.best_valid_loss = np.inf
        self.best_train_loss = np.inf
        # Optimizer
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, betas=betas, weight_decay=wd)
        
        # get label indices without 'feed'
        self.target_indices = [i for s, i in label_indices.items() if s != 'glc_after']
        # self.glc_feed_indices = [i for s, i in label_indices.items() if s == 'glc_feed']
        self.target_indices = self.target_indices[: len(self.target_indices)-self.num_external_inputs]
        teacher_forcing_ratio = 0.0
        
        for epoch in range(1, epochs+1):
            batch_losses = []
            batch_original_losses = []
            # Sets encoder and decoder models to train mode
            self.decoder.train()

            train_loss_val = np.nan

            for (x_batch, y_batch), batch_indices in train_data_loader:
                batch_lengths = train_lengths[batch_indices]
                # perform prediction for one batch
                prediction_batch, target_batch = self.one_step_predict(x=x_batch, y=y_batch, teacher_forcing_ratio=teacher_forcing_ratio)

                # compute loss and get gradients
                loss, original_loss = self.optimize(y=target_batch, y_hat=prediction_batch, batch_lengths=batch_lengths)
                batch_losses.append(loss)
                batch_original_losses.append(original_loss)

            if epoch%10==0 and not kfold_ind:
                train_loss_val = self.predict_train(epoch, dataset)
                # evaluation
                if valid_data_loader is not None:
                    valid_loss = self.evaluate(valid_data_loader, valid_lengths)
                    self.valid_loss.append(valid_loss)
                    self.valid_loss_nonnan.append(valid_loss)
                    # save the current best model
                    if valid_loss < self.best_valid_loss:
                        self.best_valid_loss = valid_loss
                        self.best_train_loss = train_loss_val
                        self.best_model_state = copy.deepcopy(self.state_dict())                       
                

            else:
                self.valid_loss.append(np.nan)

            # work with losses
            training_loss = np.mean(batch_losses)
            self.train_loss.append(training_loss)
            
            # print out training log
            if log_epochs and (epoch<=10 or epoch%log_epochs==0):
                loss_log = f'[{epoch}/{epochs}] Train loss: {training_loss:.6f}'
                if valid_data_loader is not None:
                    loss_log += f' Valid loss: {self.valid_loss[-1]:.6f}'
                loss_log += f' Train loss reeval: {train_loss_val:.6f}'
                print(loss_log)

        if save_fig:
            if not os.path.exists(f"{self.folder_name}/{epoch}"):
                # Create the folder
                os.makedirs(f"{self.folder_name}/{epoch}")

            # save model
            torch.save(self.best_model_state, os.path.join(f"{self.folder_name}/{epoch}", 'torch_model.pth'))
            # load the model with the best state
            self.load_state_dict(self.best_model_state)
            
            # create dataframe of the loss in each epoch
            self.loss_data = pd.DataFrame(data={"Epochs": list(range(1, len(self.train_loss)+1)),
                                                'Train Loss': self.train_loss})
            if valid_data_loader is not None:
                self.loss_data['Valid/Test Loss'] = self.valid_loss

            # save loss data
            self.loss_data.to_csv(os.path.join(f"{self.folder_name}/{epoch}", 'loss_data.csv'))
            self.predict(epoch, dataset)
            self.predict_train(epoch, dataset, save_fig=save_fig)
            # save loss curve
            self.plot(log_scale=True, valid_loss_label='Testing loss', fig_size=(15, 6), title='Loss Curve')
        return self.best_train_loss,self.best_valid_loss, self.best_model_state
    
    def evaluate(self, data_loader, lengths):
        '''evaluate the model performance.'''
        batch_losses = []
        predictions = []

        with torch.no_grad():
            # Sets encoder and decoder models to evaluation mode
            self.decoder.eval()

            for (x_batch, y_batch), batch_indices in data_loader:
                batch_lengths = lengths[batch_indices]
                # perform prediction for one batch
                prediction_batch, target_batch = self.one_step_predict(x=x_batch, y=y_batch, teacher_forcing=False)

                # compute loss and get gradients
                loss = self.compute_loss(y=target_batch, y_hat=prediction_batch, batch_lengths=batch_lengths)
                batch_losses.append(loss)

                predictions.append(prediction_batch.detach().squeeze().numpy())

        # self.predictions = predictions
        return np.mean(batch_losses)
    
    def plot(self, title=None, xlabel="Epochs", ylabel="Loss", 
             train_loss_label="Training loss", valid_loss_label="Validation loss",
             log_scale=False, fig_size=(6, 4), display=False):
        '''Plot training and validation/testing loss.'''

        epochs = list(range(1, len(self.train_loss)+1))
        plt.figure(figsize=fig_size)
        plt.plot(epochs, self.train_loss, label=train_loss_label)
        plt.plot(epochs, self.valid_loss, label=valid_loss_label)
        plt.xlabel(xlabel)
        plt.xlim([1, len(self.train_loss)])
        plt.legend()
        if log_scale:
            plt.yscale('log')
            plt.ylabel(ylabel + ' (log)')
        else:
            plt.ylabel(ylabel)
        if title is not None:
            plt.title(title)
        
        # Check if the folder already exists
        if not os.path.exists(f"{self.folder_name}/{epochs[-1]}"):
            # Create the folder
            os.makedirs(f"{self.folder_name}/{epochs[-1]}")
        
        # save data
        if self.folder_name is not None:
            # save prediction data
            plt.savefig(os.path.join(f"{self.folder_name}/{epochs[-1]}", 'train_plot.png'))
        if display:
            plt.show()
    
        plt.close("all") 

    
    def predict(self, epoch, dataset, id_col='ID'):
        ''''''
        data_type_col = 'Data Type'
        predictions = []
        batch_losses = []
        with torch.no_grad():
            # Sets encoder and decoder models to evaluation mode
            self.decoder.eval()
            (test_dl, test_lengths) = dataset.test
            for (x, y), batch_indices in test_dl:
                batch_lengths = test_lengths[batch_indices]

                # perform prediction for one batch
                prediction_batch, target_batch = self.one_step_predict(x=x, y=y, teacher_forcing=True, teacher_forcing_ratio=0.0)
                predictions.append(prediction_batch.detach().squeeze().numpy())
                # compute loss and get gradients
                loss = self.compute_loss(y=target_batch, y_hat=prediction_batch, batch_lengths=batch_lengths)
                batch_losses.append(loss)

        # get original test data
        df = dataset.test_data.copy()
        
        # Get datasets parameters
        in_width = dataset.input_width
        out_width = in_width + dataset.label_width - 1
        label_columns = dataset.label_columns.copy()
        label_columns.remove('glc_after')
        label_columns=label_columns[:len(label_columns)-self.num_external_inputs]

        df_list = []
        for i, id in enumerate(df[id_col].unique()):
            id_mask = df[id_col]==id
            temp = df[id_mask].copy().reset_index(drop=True)

            # Input data
            input = temp.loc[: in_width-1, :].copy()
            input[data_type_col] = 'Input'
            # Actual data
            actual = temp.loc[in_width: out_width, :].copy()
            actual[data_type_col] = 'Actual'
            # Pred data
            pred = actual.copy()
            pred[data_type_col] = 'Predicted'
            pred.loc[: , label_columns] = predictions[i][:test_lengths[i],:]
            df_list.append(pd.concat([input, actual, pred], axis=0))
        
        data = pd.concat(df_list, axis=0)

        data['Input Seq.'] = in_width
        self.predictin_data = data

        df_scores, fig = R_squared_cal(data, label_columns)
        
        # Check if the folder already exists
        if not os.path.exists(f"{self.folder_name}/{epoch}"):
            # Create the folder
            os.makedirs(f"{self.folder_name}/{epoch}")
        
        
        # save data
        if self.folder_name is not None:
            # save prediction data
            self.predictin_data.to_csv(os.path.join(f"{self.folder_name}/{epoch}", 'data.csv'))
            df_scores.to_csv(os.path.join(f"{self.folder_name}/{epoch}", 'scores.csv'), index=False)
            fig.savefig(os.path.join(f"{self.folder_name}/{epoch}", 'R2_plots.png'), dpi=300)
        return data

    def predict_train(self, epoch, dataset, id_col='ID', save_fig=False):
        ''''''
        data_type_col = 'Data Type'
        predictions = []
        batch_losses = []
        with torch.no_grad():
            # Sets encoder and decoder models to evaluation mode
            # self.encoder.eval()
            self.decoder.eval()
            (train_dl, train_lengths) = dataset.train_full
            for (x, y), batch_indices in train_dl:
                batch_lengths = train_lengths[batch_indices]
                
                # perform prediction for one batch
                prediction_batch, target_batch = self.one_step_predict(x=x, y=y, teacher_forcing=False)
                predictions.append(prediction_batch.detach().squeeze().numpy())
                # compute loss and get gradients
                loss = self.compute_loss(y=target_batch, y_hat=prediction_batch, batch_lengths=batch_lengths)
                batch_losses.append(loss)
        
        if save_fig:

            # get original train data
            df = dataset.train_data.copy()

            # Get datasets parameters
            in_width = dataset.input_width
            out_width = in_width + dataset.label_width - 1
            label_columns = dataset.label_columns.copy()
            label_columns.remove('glc_after')
            label_columns=label_columns[:len(label_columns)-self.num_external_inputs]

            df_list = []
            for i, id in enumerate(df[id_col].unique()):
                id_mask = df[id_col]==id
                temp = df[id_mask].copy().reset_index(drop=True)

                # Input data
                input = temp.loc[: in_width-1, :].copy()
                input[data_type_col] = 'Input'
                # Actual data
                actual = temp.loc[in_width: out_width, :].copy()
                actual[data_type_col] = 'Actual'
                # Pred data
                pred = actual.copy()
                pred[data_type_col] = 'Predicted'
                pred.loc[: , label_columns] = predictions[i][:train_lengths[i],:]

                df_list.append(pd.concat([input, actual, pred], axis=0))
            
            
            data = pd.concat(df_list, axis=0)

            data['Input Seq.'] = in_width
            self.predictin_data = data

            df_scores, fig = R_squared_cal(data, label_columns)
            
            # Check if the folder already exists
            if not os.path.exists(f"{self.folder_name}/{epoch}"):
                # Create the folder
                os.makedirs(f"{self.folder_name}/{epoch}")
            
            
            # save data
            if self.folder_name is not None:
                # save prediction data
                self.predictin_data.to_csv(os.path.join(f"{self.folder_name}/{epoch}", 'data_train.csv'))
                df_scores.to_csv(os.path.join(f"{self.folder_name}/{epoch}", 'scores_train.csv'), index=False)
                fig.savefig(os.path.join(f"{self.folder_name}/{epoch}", 'R2_plots_train.png'), dpi=300)
            return data, np.mean(batch_losses)
        else:
            return np.mean(batch_losses)

def R_squared_cal(df, labels, n_cols = 3):
    f, axs = plt.subplots(int(np.ceil(len(labels)/n_cols)), n_cols, figsize = (n_cols*5, np.ceil(len(labels)/n_cols)*4))#, sharey=True, )
    axs_lst = []
    if int(np.ceil(len(labels)/n_cols)) != 1:
        for i in range(int(np.ceil(len(labels)/n_cols))):
            for j in range(n_cols): 
                axs_lst.append((i,j))
    else:
        for j in range(n_cols): 
                axs_lst.append(j)
    
    df_scores = pd.DataFrame()
    r2_lst = []
    MSE_lst = []
    MAE_lst = []
    MAPE_lst = []
    r2_lst_first50_lst = []
    r2_lst_sec50_lst = []

    for idx, label in enumerate(labels):
        y = df[df['Data Type']=='Actual'][label].tolist()
        y_hat = df[df['Data Type']=='Predicted'][label].tolist()
        input_seq_lst = df[df['Data Type']=='Actual']['input_step'].unique().tolist()
        n_half = round(len(input_seq_lst)/2)
        first_half = input_seq_lst[:n_half]
        sec_half = input_seq_lst[n_half:]
        y_hat_first50 = df[(df['Data Type']=='Predicted')&(df['input_step'].isin(first_half))][label].tolist()
        y_hat_sec50 = df[(df['Data Type']=='Predicted')&(df['input_step'].isin(sec_half))][label].tolist()
        y_first50 = df[(df['Data Type']=='Actual')&(df['input_step'].isin(first_half))][label].tolist()
        y_sec50 = df[(df['Data Type']=='Actual')&(df['input_step'].isin(sec_half))][label].tolist()

        # score calculation
        r2 = r2_score(y, y_hat)
        r2_first50 = r2_score(y_first50, y_hat_first50)
        r2_sec50 = r2_score(y_sec50, y_hat_sec50)
        mse = mean_squared_error(y, y_hat)
        mae = mean_absolute_error(y, y_hat)
        mape = mean_absolute_percentage_error(y, y_hat)

        # store results
        r2_lst.append(r2)
        r2_lst_first50_lst.append(r2_first50)
        r2_lst_sec50_lst.append(r2_sec50)
        MSE_lst.append(mse)
        MAE_lst.append(mae)
        MAPE_lst.append(mape)
        try:
            PredictionErrorDisplay.from_predictions(
                y,
                y_hat,
                kind="actual_vs_predicted",
                ax=axs[axs_lst[idx]],
                scatter_kwargs={"alpha": 0.5},
            )
            axs[axs_lst[idx]].plot([], [], " ", label="$R^2$=%.3f"%r2)
            axs[axs_lst[idx]].legend(loc="upper left", frameon=False, fontsize = 12)
            axs[axs_lst[idx]].set_title(f"{label}")
            # recompute the ax.dataLim
            axs[axs_lst[idx]].relim()
            # update ax.viewLim using the new dataLim
            axs[axs_lst[idx]].autoscale_view(True)
        except:
            PredictionErrorDisplay.from_predictions(
                y,
                y_hat,
                kind="actual_vs_predicted",
                ax=axs[axs_lst[idx][1]],
                scatter_kwargs={"alpha": 0.5},
            )
            axs[axs_lst[idx][1]].plot([], [], " ", label="$R^2$=%.3f"%r2)
            axs[axs_lst[idx][1]].legend(loc="upper left", frameon=False, fontsize = 12)
            axs[axs_lst[idx][1]].set_title(f"{label}")
            # recompute the ax.dataLim
            axs[axs_lst[idx][1]].relim()
            # update ax.viewLim using the new dataLim
            axs[axs_lst[idx][1]].autoscale_view(True)

    # store results
    labels.append('Average')
    r2_lst.append(np.mean(r2_lst))
    r2_lst_first50_lst.append(np.mean(r2_lst_first50_lst))
    r2_lst_sec50_lst.append(np.mean(r2_lst_sec50_lst))
    MSE_lst.append(np.mean(MSE_lst))
    MAE_lst.append(np.mean(MAE_lst))
    MAPE_lst.append(np.mean(MAPE_lst))
    

    df_scores['label'] = labels
    df_scores['R2'] = r2_lst
    df_scores['R2 first half'] = r2_lst_first50_lst
    df_scores['R2 sec half'] = r2_lst_sec50_lst
    df_scores['MSE'] = MSE_lst
    df_scores['MAE'] = MAE_lst
    df_scores['MAPE'] = MAPE_lst
    
    f.tight_layout()

    return df_scores, f
