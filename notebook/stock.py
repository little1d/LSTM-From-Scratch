import swanlab
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from copy import deepcopy as dc
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from model import *

class TimeSeriesDataset(Dataset):
    """
    定义数据集类
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def prepare_dataframe_for_lstm(df, n_steps):
    """
    处理数据集，使其适用于LSTM模型
    """
    df = dc(df)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    for i in range(1, n_steps+1):
        df[f'close(t-{i})'] = df['close'].shift(i)
        
    df.dropna(inplace=True)
    return df


def get_dataset(file_path, lookback, split_ratio=0.9):
    """
    归一化数据、划分训练集和测试集
    """
    data = pd.read_csv(file_path)
    data = data[['date','close']]
    
    shifted_df_as_np = prepare_dataframe_for_lstm(data, lookback)

    scaler = MinMaxScaler(feature_range=(-1,1))
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]

    X = dc(np.flip(X,axis=1))

    # 划分训练集和测试集
    split_index = int(len(X) * split_ratio)
    
    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    X_train = X_train.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    # 转换为Tensor
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()
    
    return scaler, X_train, X_test, y_train, y_test


def train(model, train_loader, optimizer, criterion, num_epochs, device):
    for epoch in range(num_epochs):
        total_loss = 0
        for X, y in train_loader:
            # print(X.shape)  torch.Size([32, 7, 1])  [batch_size, seq_len, feature_dim]
            # print(X.shape[0])
            state = model.begin_state(batch_size=X.shape[0], device=device)
            X, y = X.to(device), y.to(device)

            y_pred, _ = model(X, state)
            # print(type(y_pred), type(y))  <class 'list'> <class 'torch.Tensor'>
            loss = criterion(y_pred[0], y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")
        swanlab.log({"train/loss": total_loss}, step=epoch)

if __name__ == '__main__':
    # ------------------- 初始化一个SwanLab实验 -------------------
    swanlab.init(
        project='Google-Stock-Prediction',
        experiment_name="LSTM",
        description="根据前7天的数据预测下一日股价",
        config={ 
            "learning_rate": 1e-3,
            "vocab_size": 1,
            "hidden_units": 128,
            "epochs": 50,
            "batch_size": 32,
            "lookback": 7,
            "spilt_ratio": 0.9, 
            "save_path": "./checkpoint",
            "optimizer": "Adam",
            "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        },
    )
    
    config = swanlab.config
    device = torch.device(config.device)

    
    # ------------------- 定义数据集 -------------------
    scaler, X_train, X_test, y_train, y_test = get_dataset(file_path='./GOOG.csv',
                                                           lookback=config.lookback,
                                                           split_ratio=config.spilt_ratio,)
    
    train_dataset = TimeSeriesDataset(X_train, y_train)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # ------------------- 定义模型、超参数 -------------------
    # model = LSTMModel(input_size=1, output_size=1)
    model = RNNModelScratch(1, hidden_units, device, initialize_parameters, init_lstm_state, lstm)

    vocab_size = config.vocab_size
    hidden_units = config.hidden_units

    # 将参数字典的值提取为 list，符合 optim 优化器的输入格式
    optimizer = optim.Adam(list(model.params.values()), lr=config.learning_rate)
    criterion = nn.MSELoss()

    # ------------------- 训练与验证 -------------------
    # for epoch in range(1, config.epochs):
    train(model, train_loader, optimizer, criterion, config.epochs, device)