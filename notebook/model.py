import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import swanlab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_units = 128

def tanh_activation(X):
    return np.tanh(X)
def softmax(X):
    exp_X = np.exp(X)
    exp_X_sum = np.sum(exp_X, axis=1).reshape(-1, 1)
    exp_X = exp_X / exp_X_sum
    return exp_X
def tanh_derivative(X):
    return 1 - (X**2)


def init_lstm_state(batch_size, hidden_units, device):
    return (torch.zeros((batch_size, hidden_units), device=device), 
            torch.zeros((batch_size, hidden_units), device=device))

def initialize_parameters(vocab_size, hidden_units, device):
    std = 0.01
    output_units = vocab_size
    input_units = 7  # 7 天，mat 形状应该为 135*128
    def normal(shape):
        return torch.randn(size=shape, device=device) * std
    forget_gate_weights = normal((input_units + hidden_units, hidden_units))
    input_gate_weights = normal((input_units + hidden_units, hidden_units))
    output_gate_weights = normal((input_units + hidden_units, hidden_units))
    c_tilda_gate_weights = normal((input_units + hidden_units, hidden_units))
    forget_gate_bias = torch.zeros((1, hidden_units), device=device)
    input_gate_bias = torch.zeros((1, hidden_units), device=device)
    output_gate_bias = torch.zeros((1, hidden_units), device=device)
    c_tilda_gate_bias = torch.zeros((1, hidden_units), device=device)

    hidden_output_weights = normal((hidden_units, output_units))
    output_bias = torch.zeros((1, output_units), device=device)

    parameters = {
        'fgw': forget_gate_weights,
        'igw': input_gate_weights,
        'ogw': output_gate_weights,
        'cgw': c_tilda_gate_weights,
        'fgb': forget_gate_bias,
        'igb': input_gate_bias,
        'ogb': output_gate_bias,
        'cgb': c_tilda_gate_bias,
        'how': hidden_output_weights,
        'ob': output_bias
    }

    for param in parameters.values():
        param.requires_grad_(True)

    return parameters

def lstm_cell(batch_dataset, prev_hidden_state, prev_cell_state, parameters):
    # get parameters
    fgw = parameters['fgw']
    igw = parameters['igw']
    ogw = parameters['ogw']
    cgw = parameters['cgw']

    fgb = parameters['fgb']
    igb = parameters['igb']
    ogb = parameters['ogb']
    cgb = parameters['cgb']

    # 检查输入数据和隐藏状态的批量大小是否一致
    # print("batch_dataset size:", batch_dataset.shape) # torch.Size([32, 7, 1]）
    batch_dataset = batch_dataset.squeeze(-1)
    # print("batch_dataset shape after squeeze:", batch_dataset.shape)
    # print("prev_hidden_state size:", prev_hidden_state.shape) # ([32, 128])

    # 串联 data 和 prev_hidden_state
    concat_dataset = torch.cat((batch_dataset, prev_hidden_state), axis=1)

    # forget gate activations
    F = torch.sigmoid(torch.matmul(concat_dataset, fgw) + fgb)

    # input gate activations
    I = torch.sigmoid(torch.matmul(concat_dataset, igw) + igb)

    # output gate activations
    O = torch.sigmoid(torch.matmul(concat_dataset, ogw) + ogb)

    # cell_tilda gate activations
    C_tilda = torch.tanh(torch.matmul(concat_dataset, cgw) + cgb)

    # 更新 cell state, hidden_state
    cell_state = F * prev_cell_state + I * C_tilda
    hidden_state =  O * torch.tanh(cell_state)

    # store four gate weights to be used in back propagation
    lstm_activations = {
        'F': F,
        'I': I,
        'O': O,
        'C_tilda': C_tilda
    }
    
    return lstm_activations, hidden_state, cell_state

def output_cell(hidden_state, parameters):
    how = parameters['how']
    ob = parameters['ob']
    output = torch.matmul(hidden_state, how)+ ob
    return output

def lstm(inputs, initail_state, parameters):
    hidden_state, cell_state = initail_state
    outputs = []
    _, hidden_state, cell_state = lstm_cell(inputs, hidden_state, cell_state, parameters)
    outputs.append(output_cell(hidden_state, parameters))    
    return outputs, (hidden_state, cell_state)

# 定义一个RNN 类来训练LSTM
import torch.nn.functional as F

class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params(vocab_size, hidden_units, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        # print(type(X))   torch.Tensor
        # X = F.one_hot(X.T.long(), self.vocab_size).type(torch.float32)
        X = X.type(torch.float32)
        return self.forward_fn(X, state, self.params)
    
    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
    
model = RNNModelScratch(1, hidden_units, device, initialize_parameters, init_lstm_state, lstm)
print(model)