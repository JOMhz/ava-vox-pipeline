import torch
import torch.nn as nn

from model.Classifier import BGRU, BLSTM, GPTBLSTM, GPTGRU
# from model.Encoder import visual_encoder, audio_encoder

class ASD_Model(nn.Module):
    def __init__(self, channels):
        super(ASD_Model, self).__init__()
        self.channels = channels

        self.LSTM = GPTGRU(channels)


    
    def forward_audio_visual_backend(self, x):
        x = self.LSTM(x)
        x = torch.reshape(x, (-1, self.channels))
        return x   

    def forward(self, feature):
        outsAV = self.forward_audio_visual_backend(feature)
        return outsAV