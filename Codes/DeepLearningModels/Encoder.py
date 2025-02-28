from typing import Tuple
import torch
import torch.nn as nn

class Encoder(nn.Module):
    '''
    An encoder model for time-series sequences. 
    The RNN structure can be chosen from "RNN", "LSTM", and "GRU".
    '''
    def __init__(self,
                 model_struct: str,
                 num_features: int, 
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
        super(Encoder, self).__init__()
        self.model_struct = model_struct
        self.num_features = num_features
        self.H_out = hidden_size
        self.num_layers = num_layers
        self.non_linearity = non_linearity
        self.dropout = dropout
        self.D = 2 if bi_direct else 1

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

    def forward(self, input) -> Tuple[torch.tensor, torch.tensor]:
        '''
        Args:
            input: torhc.tensor
                The input tensor with the shape of (N, L, H_in).
        Returns:
            output: torch.tensor
                The output tensor with the shape of (N,L,D*H_out)
            hidden: torch.tensor if model_struct=="rnn" or "gru", (torch.tensor, torch.tensor) if model_struct=="lstm".
                h_n, the final hidden state of the shape of (N,L,D*H_out) if model_struct=="rnn" or "gru".
                (h_n, c_n), the final hidden state and final cell state of the shape of (N,L,D*H_out) if model_struct=="lstm".   
        '''
        batch_size = input.size(0)
        self.init_hidden(N=batch_size)
        output, hidden = self.rnn(input, self.hidden)
        self.hidden = hidden

        return output, hidden
    
    def init_hidden(self, N):
        '''
        Initialize h_0 if model_struct=="rnn" or "gru", 
        h_0 and c_0 if model_struct=="lstm", 
        with the shape of (D*num_layers, N, H_out).

        Args:
            N: int
                The size of batch. x_input.size(0).
        '''
        # model_struct=='rnn' or 'gru'
        h_0 = torch.zeros(self.D*self.num_layers, N, self.H_out).requires_grad_()
        hidden = h_0
        if self.model_struct=='lstm':
            c_0 = torch.zeros(self.D*self.num_layers, N, self.H_out).requires_grad_()
            hidden = (h_0, c_0)        
        self.hidden = hidden