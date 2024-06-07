import torch
import torch.nn as nn
from utils.usersvectors import UsersVectors
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class LSTM_Transformer(nn.Module):
    def __init__(self, n_layers, tr_layers, n_heads_first, n_heads_second, input_dim, hidden_dim, output_dim, dropout, logsoftmax=True, input_twice=False):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.input_twice = input_twice
        self.tr_layers = tr_layers
        self.n_heads_first = n_heads_first
        self.n_heads_second = n_heads_second

        self.input_fc = nn.Sequential(nn.Linear(input_dim, input_dim * 2),
                                      nn.Dropout(dropout),
                                      nn.ReLU(),
                                      nn.Linear(input_dim * 2, self.hidden_dim),
                                      nn.Dropout(dropout),
                                      nn.ReLU())

        self.input_fc1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                      nn.Dropout(dropout),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim // 2, self.hidden_dim // 2),
                                      nn.Dropout(dropout),
                                      nn.ReLU())

        self.main_task = nn.LSTM(input_size=self.hidden_dim,
                                 hidden_size=self.hidden_dim,
                                 batch_first=True,
                                 num_layers=self.n_layers,
                                 dropout=dropout)

        self.last_task = nn.LSTM(input_size=self.hidden_dim // 2,
                                 hidden_size=self.hidden_dim,
                                 batch_first=True,
                                 num_layers=self.n_layers,
                                 dropout=dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim , nhead=self.n_heads_first,
                                                        dim_feedforward=self.hidden_dim*4 , #82.063, 82.238, 82.429
                                                        dropout=dropout,
                                                        activation='relu',
                                                        batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=self.tr_layers).double()

        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=self.n_heads_second,
                                                        dim_feedforward=self.hidden_dim * 4,
                                                        dropout=dropout,
                                                        activation='relu',
                                                        batch_first=True)
        self.transformer2 = nn.TransformerEncoder(self.encoder_layer, num_layers=self.tr_layers).double()

        seq = [nn.Linear(self.hidden_dim + (input_dim if self.input_twice else 0), self.hidden_dim // 2),
               nn.ReLU(),
               nn.Linear(self.hidden_dim // 2, self.output_dim)]
        if logsoftmax:
            seq += [nn.LogSoftmax(dim=-1)]

        self.output_fc = nn.Sequential(*seq)

        self.user_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=self.n_layers)
        self.game_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=self.n_layers)

        self.weight_lstm = nn.Parameter(torch.rand(1), requires_grad=True)
        self.weight_transform = nn.Parameter(torch.rand(1), requires_grad=True)
        self.weight_transform2 = nn.Parameter(torch.rand(1), requires_grad=True)
        self.weight_attn = nn.Parameter(torch.rand(1), requires_grad=True)

    def init_game(self, batch_size=1):
        return torch.stack([self.game_vectors.init_user] * batch_size, dim=0)

    def init_user(self, batch_size=1):
        return torch.stack([self.user_vectors.init_user] * batch_size, dim=0)
    def attention(self, tensor):
        mask = torch.triu(torch.ones(tensor.shape[1], tensor.shape[1]) * float('-inf'), diagonal=1)
        l = []
        for i in range(tensor.shape[0]):
            l.append((tensor[i] @ tensor[i].transpose(0, 1)) + mask)

        x = []
        for i in l:
            softed = F.softmax(i, dim=1)
            x.append(softed)

        output = []
        for i in range(tensor.shape[0]):
            output.append(x[i] @ tensor[i])
        return torch.stack(output)

    def forward(self, input_vec, game_vector, user_vector):

        lstm_input = self.input_fc(input_vec)
        input_vec1 = self.attention(lstm_input)

        mask = torch.triu(torch.ones(lstm_input.shape[1], lstm_input.shape[1]) * float('-inf'), diagonal=1)
        transformer_output1 = self.transformer(lstm_input, mask=mask)
        transformer_output2 = self.transformer2(lstm_input, mask=mask)

        lstm_shape = lstm_input.shape
        shape = user_vector.shape
        assert game_vector.shape == shape
        if len(lstm_shape) != len(shape):
            lstm_input = lstm_input.reshape((1,) * (len(shape) - 1) + lstm_input.shape)
        user_vector1 = user_vector.reshape(shape[:-1][::-1] + (shape[-1],))
        game_vector1 = game_vector.reshape(shape[:-1][::-1] + (shape[-1],))

        lstm_output, (game_vector, user_vector) = self.main_task(lstm_input.contiguous(),
                                                                (game_vector1.contiguous(),
                                                                  user_vector1.contiguous()))

        if hasattr(self, "input_twice") and self.input_twice:
            lstm_output = torch.cat([lstm_output, input_vec], dim=-1)

        lstm_output = self.weight_transform*transformer_output1 + self.weight_lstm*lstm_output + \
                      self.weight_attn*input_vec1 + self.weight_transform2*transformer_output2
        lstm_output /= (self.weight_lstm + self.weight_attn + self.weight_transform + self.weight_transform2)

        lstm_input = self.input_fc1(lstm_output)

        final_output, (game_vector, user_vector) = self.last_task(lstm_input.contiguous(),
                                                                (game_vector1.contiguous(),
                                                                user_vector1.contiguous()))
        user_vector = user_vector.reshape(shape)
        game_vector = game_vector.reshape(shape)

        output = self.output_fc(final_output)

        if len(output.shape) != len(lstm_shape):
            output.reshape(-1, output.shape[-1])
        if self.training:
            return {"output": output, "game_vector": game_vector, "user_vector": user_vector}
        else:
            return {"output": output, "game_vector": game_vector.detach(), "user_vector": user_vector.detach()}