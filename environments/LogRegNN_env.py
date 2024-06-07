from environments import environment
import torch
import torch.nn as nn
from consts import *


class LogisticRegressionNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionNN, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, vectors):
        x = vectors["x"]

        batch_size, seq_len, dim = x.shape
        x = x.view(batch_size * seq_len, dim)

        x = self.fc(x)
        x = self.logsoftmax(x)

        x = x.view(batch_size, seq_len, -1)

        return {"output": x}


class LogReg_env(environment.Environment):
    def init_model_arc(self, config):
        input_dim = config["REVIEW_DIM"] + STRATEGY_DIM
        output_dim = 2
        self.model = LogisticRegressionNN(input_dim, output_dim)
        self.model.eval()

    def predict_proba(self, x):
        return {"proba": torch.exp(self.model(x))}
