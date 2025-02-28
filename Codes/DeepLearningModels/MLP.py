import torch.nn as nn

class MLPModel(nn.Module):
    '''Multi Layer Perceptron Model.'''
    def __init__(self, n_features, n_labels,
                 hidden_dim, layer_dim, non_linearity = None, dropout=0, alpha_elu=1.0):
        super(MLPModel, self).__init__()
        # Input Layer
        input_layer = nn.Linear(n_features, hidden_dim)
        # Activation layer
        if non_linearity == "elu":
            print("Using ELU as activation function")  
            act_layer = nn.ELU(alpha=alpha_elu)
        elif non_linearity == "tanh":
            print("Using tanh as activation function")  
            act_layer = nn.Tanh()
        else:
            print("Using ReLU as activation function")  
            act_layer = nn.ReLU()

        # Dropout layer
        drop_layer = nn.Dropout(dropout)

        layers_list = [input_layer, act_layer, drop_layer]
        # Hidden layers and activation functions
        for i in range(layer_dim):
            if i < layer_dim-1:
                layer = nn.Linear(hidden_dim, hidden_dim)
                layers_list.append(layer)
                layers_list.append(act_layer)
                layers_list.append(drop_layer)
            else:
                # Output layer
                layer = nn.Linear(hidden_dim, n_labels)
                layers_list.append(layer)
        self.module_list = nn.ModuleList(layers_list)

    def forward(self, x):
        batch_size, _, _  = x.shape
        x = x.contiguous().view(batch_size, -1)

        for f in self.module_list:
            x = f(x)
        out = x.contiguous().view(batch_size, -1).unsqueeze(1)

        return out