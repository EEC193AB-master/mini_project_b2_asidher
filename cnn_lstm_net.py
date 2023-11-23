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
        batch_size = x.shape[0]

        if hx_cx is None:
            hx_cx = Variable(torch.zeros(1, self.lstm_hidden_units)).cuda()
            hx_cx = (hx_cx, hx_cx)

        hx, cx = hx_cx
        
        # input should be in shape: (batches, breaths in seq, chans, 224)
        batches = batch_size
        outputs = self.cnn(x[0]).squeeze()
        outputs = outputs.unsqueeze(dim=0)

        # Initialize the lstm_output for the first batch output from the CNN
        lstm_output, (hx, cx) = self.lstm(outputs[0], (hx, cx))
        lstm_output = lstm_output.unsqueeze(dim=0)
        
        # For each batch, extract features for the 20 breaths via the CNN and then feed into LSTM
        for i in range(1, batches):
            block_out = self.cnn(x[i]).squeeze()
            block_out = block_out.unsqueeze(dim=0)
            outputs = torch.cat([outputs, block_out], dim=0)
 
            # For each block out of CNN, run LSTM
            lstm_block_out, (hx, cx) = self.lstm(block_out[0], (hx, cx))

            # Store LSTM results for fully connected layer
            lstm_block_out = lstm_block_out.unsqueeze(dim=0)
            lstm_output = torch.cat([lstm_output, lstm_block_out], dim=0)

        output = self.fc(lstm_output[:, -1, :])

        return output