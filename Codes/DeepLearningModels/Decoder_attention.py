from typing import Tuple
import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
    
    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) 
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(encoder_outputs.shape[0], 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        return torch.softmax(attention, dim=1)


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
                 attention,
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
        self.attention = attention

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
            self.rnn = nn.LSTM(input_size=num_features+hidden_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bias=True,
                                batch_first=True,
                                dropout=dropout,
                                bidirectional=bi_direct)
            
        elif model_struct=='gru': # GRU model
            self.rnn = nn.GRU(input_size=num_features+hidden_size,
                               hidden_size=hidden_size, 
                               num_layers=num_layers, 
                               bias=True, 
                               batch_first=True, 
                               dropout=dropout, 
                               bidirectional=bi_direct)
        else:
            return print('model_struct can be either "rnn", "lstm", or "gru".')

        # Fully connected linear layer
        self.linear = nn.Linear(hidden_size*2+num_features, num_labels)
        
        
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

    def forward(self, input, context_vector, encoder_outputs):
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

        # attention mechanism
        # input = input.squeeze(1)
        # print(input.shape)
        a = self.attention(context_vector[-1], encoder_outputs)
        a = a.unsqueeze(1)
        # print("a", a.shape)
        weighted = torch.bmm(a, encoder_outputs)
        # print("weighted", weighted.shape)
        rnn_input = torch.cat((input, weighted), dim=2)
        # print("rnn_input", rnn_input.shape)
        output, hidden = self.rnn(rnn_input, context_vector)
        self.lstm_hidden = hidden
        output = output.squeeze(1)
        input = input.squeeze(1)
        weighted = weighted.squeeze(1)
        output = self.linear(torch.cat((output, weighted, input), dim=1))

        # project the tensor of the shape (N,L,H_out) on (N,L,num_labels)
        # output = self.linear(output.squeeze(0))
        
        # output = self.activation(output)
        # output = self.linear2(output.squeeze(0))
        return output.unsqueeze(1), hidden