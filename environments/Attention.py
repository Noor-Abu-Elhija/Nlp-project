import torch
import torch.nn as nn
from utils.usersvectors import UsersVectors
import torch.nn.functional as F
import math


class FCAtenttion(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, dropout, logsoftmax=True, input_twice=False):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.input_twice = input_twice

        self.input_fc = nn.Sequential(nn.Linear(input_dim, input_dim * 2),
                                      nn.Dropout(dropout),
                                      nn.ReLU(),
                                      nn.Linear(input_dim * 2, self.hidden_dim),
                                      nn.Dropout(dropout),
                                      nn.ReLU())
        seq = [nn.Linear(self.hidden_dim + (input_dim if self.input_twice else 0), self.hidden_dim // 2),
               nn.ReLU(),
               nn.Linear(self.hidden_dim // 2, self.output_dim)]
        if logsoftmax:
            seq += [nn.LogSoftmax(dim=-1)]

        self.output_fc = nn.Sequential(*seq)
        self.user_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=self.n_layers)
        self.game_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=self.n_layers)
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


        lstm_shape = lstm_input.shape
        shape = user_vector.shape
        assert game_vector.shape == shape
        user_vector1 = user_vector.reshape(shape[:-1][::-1] + (shape[-1],))
        game_vector1 = game_vector.reshape(shape[:-1][::-1] + (shape[-1],))
        user_vector = user_vector.reshape(shape)
        game_vector = game_vector.reshape(shape)

        output = self.output_fc(input_vec1)

        if len(output.shape) != len(lstm_shape):
            output.reshape(-1, output.shape[-1])
        if self.training:
            return {"output": output, "game_vector": game_vector, "user_vector": user_vector}
        else:
            return {"output": output, "game_vector": game_vector.detach(), "user_vector": user_vector.detach()}