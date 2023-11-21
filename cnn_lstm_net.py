import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


class CNNLSTMNetwork(nn.Module):
    def __init__(self, cnn_network, lstm_hidden_units, lstm_layers, lstm_dropout):
        super(CNNLSTMNetwork, self).__init__()
        
        self.cnn = cnn_network
        self.lstm_hidden_units = lstm_hidden_units
        self.output_size = 2
        self.input_size = 128 # input_size derived from size of cnn `outputs`
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=lstm_hidden_units, num_layers=lstm_layers, dropout=lstm_dropout)
        self.fc = nn.Linear(self.lstm_hidden_units, self.output_size)

    def forward(self, x, hx_cx=None):
        seq_len = x.shape[1]
        batch_size = x.shape[0]

        if hx_cx is None:
            hx_cx = Variable(torch.zeros(1, self.lstm_hidden_units)).cuda()
            hx_cx = (hx_cx, hx_cx)

        hx, cx = hx_cx
        
        # input should be in shape: (batches, breaths in seq, chans, 224)
        batches = batch_size
        outputs = self.cnn(x[0]).squeeze()
        outputs = outputs.unsqueeze(dim=0)

        
        for i in range(1, batches):
            block_out = self.cnn(x[i]).squeeze()
            block_out = block_out.unsqueeze(dim=0)
            outputs = torch.cat([outputs, block_out], dim=0)

        for i in reversed(range(1, seq_len)):

            output, (hx, cx) = self.lstm(outputs[:, -i, :], (hx, cx))

        output = self.fc(output)

        # output = torch.sum(output, 1) / output.size(1)

        return output