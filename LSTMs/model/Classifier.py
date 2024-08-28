import torch
from torch import nn

"""
Code inspired by https://github.com/Junhua-Liao/Light-ASD
"""

class BGRU(nn.Module):
    def __init__(self, channel):
        super(BGRU, self).__init__()

        self.gru_forward = nn.GRU(input_size = channel, hidden_size = channel, num_layers = 1, bidirectional = False, bias = True, batch_first = True)
        self.gru_backward = nn.GRU(input_size = channel, hidden_size = channel, num_layers = 1, bidirectional = False, bias = True, batch_first = True)
        
        self.gelu = nn.GELU()
        self.__init_weight()

    def forward(self, x):
        x, _ = self.gru_forward(x)
        x = self.gelu(x)
        x = torch.flip(x, dims=[1])
        x, _ = self.gru_backward(x)
        x = torch.flip(x, dims=[1])
        x = self.gelu(x)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                torch.nn.init.kaiming_normal_(m.weight_ih_l0)
                torch.nn.init.kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()

class BLSTM(nn.Module):
    def __init__(self, channel):
        super(BLSTM, self).__init__()
        self.lstm_forward = nn.LSTM(input_size=channel, hidden_size=channel, num_layers=1, bidirectional=False, bias=True, batch_first=True)
        self.lstm_backward = nn.LSTM(input_size=channel, hidden_size=channel, num_layers=1, bidirectional=False, bias=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer
        self.gelu = nn.GELU()
        self.__init_weight()

    def forward(self, x):
        x, _ = self.lstm_forward(x)
        x = self.dropout(x)  # Apply dropout
        x = self.gelu(x)
        x = torch.flip(x, dims=[1])
        x, _ = self.lstm_backward(x)
        # x = self.dropout(x)  # Apply dropout
        x = torch.flip(x, dims=[1])
        x = self.gelu(x)
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                torch.nn.init.kaiming_normal_(m.weight_ih_l0)
                torch.nn.init.kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()

class GPTBLSTM(nn.Module):
    def __init__(self, channel):
        super(GPTBLSTM, self).__init__()
        # self.lstm = nn.LSTM(input_size=channel, hidden_size=channel // 2, num_layers=1, bidirectional=True, bias=True, batch_first=True)
        self.lstm_forward = nn.LSTM(input_size=channel, hidden_size=channel, num_layers=1, bidirectional=False, bias=True, batch_first=True)
        self.lstm_backward = nn.LSTM(input_size=channel, hidden_size=channel, num_layers=1, bidirectional=False, bias=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)  # Adjusted dropout rate
        self.gelu = nn.GELU()
        self.__init_weights()

    def forward(self, x):
        x, _ = self.lstm_forward(x)
        x = self.gelu(x)
        x = torch.flip(x, dims=[1])
        x, _ = self.lstm_backward(x)
        x = torch.flip(x, dims=[1])
        # x, _ = self.lstm(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x
    
    def __init_weights(self):
        # Initialize weights and biases for lstm_forward
        for name, param in self.lstm_forward.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Initialize weights and biases for lstm_backward
        for name, param in self.lstm_backward.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    # def __init_weights(self):
    #     for name, param in self.lstm.named_parameters():
    #         if 'weight_ih' in name:
    #             nn.init.xavier_uniform_(param.data)
    #         elif 'weight_hh' in name:
    #             nn.init.xavier_uniform_(param.data)
    #         elif 'bias' in name:
    #             param.data.fill_(0)


class GPTGRU(nn.Module):
    def __init__(self, channel):
        super(GPTGRU, self).__init__()
        # self.lstm = nn.LSTM(input_size=channel, hidden_size=channel // 2, num_layers=1, bidirectional=True, bias=True, batch_first=True)
        self.gru_forward = nn.GRU(input_size=channel, hidden_size=channel, num_layers=1, bidirectional=False, bias=True, batch_first=True)
        self.gru_backward = nn.GRU(input_size=channel, hidden_size=channel, num_layers=1, bidirectional=False, bias=True, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)  # Adjusted dropout rate
        self.gelu = nn.GELU()
        self.__init_weights()

    def forward(self, x):
        x, _ = self.gru_forward(x)
        x = self.gelu(x)
        x = torch.flip(x, dims=[1])
        x, _ = self.gru_backward(x)
        x = torch.flip(x, dims=[1])
        # x, _ = self.lstm(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x
    
    def __init_weights(self):
        # Initialize weights and biases for gru_forward
        for name, param in self.gru_forward.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Initialize weights and biases for gru_backward
        for name, param in self.gru_backward.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    # def __init_weights(self):
    #     for name, param in self.lstm.named_parameters():
    #         if 'weight_ih' in name:
    #             nn.init.xavier_uniform_(param.data)
    #         elif 'weight_hh' in name:
    #             nn.init.xavier_uniform_(param.data)
    #         elif 'bias' in name:
    #             param.data.fill_(0)

