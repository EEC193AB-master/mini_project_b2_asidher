import torch
from torch.autograd import Variable
import torch.nn as nn


class CNNLSTMNetwork(nn.Module):
    def __init__(self, cnn_network, lstm_hidden_units, lstm_layers, lstm_dropout):
        super(CNNLSTMNetwork, self).__init__()

        self.cnn = cnn_network
        self.lstm_hidden_units = lstm_hidden_units
        # XXX Here is where you need to add an LSTM to this network.
        # Use the PyTorch standard LSTM layer

    def forward(self, x, hx_cx):
        # input should be in shape: (batches, breaths in seq, chans, 224)
        batches = x.shape[0]
        outputs = self.cnn(x[0]).squeeze()
        outputs = outputs.unsqueeze(dim=0)

        for i in range(1, batches):
            block_out = self.cnn(x[i]).squeeze()
            block_out = block_out.unsqueeze(dim=0)
            outputs = torch.cat([outputs, block_out], dim=0)

        # XXX add LSTM and linear layer here
