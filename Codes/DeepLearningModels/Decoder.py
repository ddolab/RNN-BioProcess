from typing import Tuple
import torch
import torch.nn as nn

class Decoder(nn.Module):
    '''
    A decoder model for time-series sequences. 
    The RNN structure can be chosen from "RNN", "LSTM", and "GRU".
    '''

    def __init__(self, 
                 model_struct: str,
                 num_features: int, 
                 num_labels: int, 
                 hidden_size: int, 
                 num_layers: int=1,
                 non_linearity: str="tanh",
                 dropout: float=0.0,
                 bi_direct: bool=False) -> None:
        '''
        Args:
            model_struct: str
                The structure of the RNN model. 
                Can be either "rnn", "lstm", or "gru".
            num_features: int
                The number of features in the input.
            num_labels: int
                The number of labels in the output.
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
        '''
        super(Decoder, self).__init__()
        self.model_struct = model_struct
        self.num_labels = num_labels
        self.H_out = hidden_size
        self.num_layers = num_layers
        self.non_linearity = non_linearity
        self.dropout = dropout
        self.D = 2 if bi_direct else 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # choose structure of RNN.
        '''
        if batch first is True, 
        then the shapes of the input and output tensors are (N, L, H_in) and (N, L, H_out), respectively.
            N: batch size
            L: sequence length
            H_in: input size/num_features
            H_out: hidden size
        '''
        if model_struct=='rnn': # RNN model
            self.rnn = nn.RNN(input_size=num_features, 
                              hidden_size=hidden_size, 
                              num_layers=num_layers, 
                              nonlinearity=non_linearity, 
                              bias=True, 
                              batch_first=True, 
                              dropout=dropout, 
                              bidirectional=bi_direct)
        
        elif model_struct=='lstm': # LSTM model
            self.rnn = nn.LSTM(input_size=num_features,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bias=True,
                                batch_first=True,
                                dropout=dropout,
                                bidirectional=bi_direct)
            
        elif model_struct=='gru': # GRU model
            self.rnn = nn.GRU(input_size=num_features,
                               hidden_size=hidden_size, 
                               num_layers=num_layers, 
                               bias=True, 
                               batch_first=True, 
                               dropout=dropout, 
                               bidirectional=bi_direct)
        else:
            return print('model_struct can be either "rnn", "lstm", or "gru".')

        # Fully connected linear layer
        self.linear = nn.Linear(hidden_size, num_labels)
        
        
        # self.linear = nn.Linear(hidden_size, 16)

        # # Add a activation function
        # self.activation = nn.Tanh()
        # # Add anoter linear layer
        # self.linear2 = nn.Linear(16, num_labels)


        # self.linear3 = nn.Linear(num_features, 16)

        # # Add a activation function
        # self.activation2 = nn.Tanh()
        # # Add anoter linear layer
        # self.linear4 = nn.Linear(16, num_features)

    def forward(self, input, context_vector):
        '''
        Args:
            input: torhc.tensor
                The input tensor with the shape of (N, L, H_in).
            context_vector: torch.tensor if model_struct=="rnn" or "gru", (torch.tensor, torch.tensor) if model_struct=="lstm".
                h_n, the final hidden state of the shape of (N,L,D*H_out) if model_struct=="rnn" or "gru".
                (h_n, c_n), the final hidden state and final cell state of the shape of (N,L,D*H_out) if model_struct=="lstm".   

        Returns:
            output: torch.tensor
                The output tensor with the shape of (N,L,num_labels)
            hidden: torch.tensor if model_struct=="rnn" or "gru", (torch.tensor, torch.tensor) if model_struct=="lstm".
                h_n, the final hidden state of the shape of (N,L,D*H_out) if model_struct=="rnn" or "gru".
                (h_n, c_n), the final hidden state and final cell state of the shape of (N,L,D*H_out) if model_struct=="lstm".
        '''
        
        
        # input = self.linear3(input)
        # input = self.activation2(input)
        # input = self.linear4(input)
        
        output, hidden = self.rnn(input, context_vector)
        self.lstm_hidden = hidden
        # project the tensor of the shape (N,L,H_out) on (N,L,num_labels)
        output = self.linear(output.squeeze(0))
        
        # output = self.activation(output)
        # output = self.linear2(output.squeeze(0))
        return output, hidden