import torch
import torch.nn as nn
from model.vox.audio_encoder      import audioEncoder
from model.vox.visual_encoder     import visualFrontend, visualTCN, visualConv1D, NonTemporalVisualFrontend, NonTemporalVisualTCN, NonTemporalVisualConv1D
from model.vox.attention_layer    import attentionLayer
# VoxSenseModel code inspired by source [3]
class VoxSenseModel(nn.Module):
    def __init__(self, temporal=True):
        super(VoxSenseModel, self).__init__()

        if temporal:
            # Visual Temporal Encoder
            self.visualFrontend = visualFrontend() # Visual Frontend     
            self.visualTCN = visualTCN() # Visual Temporal Network TCN
            self.visualConv1D = visualConv1D() # Visual Temporal Network Conv1d

            # Audio Temporal Encoder 
            self.audioEncoder = audioEncoder(layers = [3, 4, 6, 3],  num_filters = [16, 32, 64, 128])
            
        else:
            # Visual Non-Temporal Encoder

            self.visualFrontend = NonTemporalVisualFrontend()  # Visual Frontend
            # Replace placeholders with actual non-temporal dimensionality reducers
            self.nonTemporalVisualTCN = NonTemporalVisualTCN()  # Non-temporal version of visualTCN
            self.nonTemporalVisualConv1D = NonTemporalVisualConv1D()  # Non-temporal version of visualConv1D

            # Audio Non-Temporal Encoder
            self.audioEncoder = audioEncoder(layers = [3, 4, 6, 3],  num_filters = [16, 32, 64, 128], temporal=False)
        
        # Audio-visual Cross Attention
        self.crossA2V = attentionLayer(d_model = 128, nhead = 8)
        self.crossV2A = attentionLayer(d_model = 128, nhead = 8)

        # Audio-visual Self Attention
        self.selfAV = attentionLayer(d_model = 256, nhead = 8)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape  
        x = x.view(B*T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)        
        x = x.transpose(1,2)     
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1,2)
        return x
    
    def forward_non_temporal_visual(self, x):
        B, T, W, H = x.shape
        x = x.view(B * T, 1, W, H)  # Adapted for 2D convolutions
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)  # Adjust dimension according to visualFrontend's output

        # Use the non-temporal visual processing components
        x = x.transpose(1, 2)  # Transpose to prepare for convolution over what was the T dimension
        x = self.nonTemporalVisualTCN(x)  # Non-temporal visual TCN
        x = self.nonTemporalVisualConv1D(x)  # Non-temporal visual Conv2D
        x = x.transpose(1, 2)  # Transpose back
        return x

    def forward_audio_frontend(self, x):    
        x = x.unsqueeze(1).transpose(2, 3)        
        x = self.audioEncoder(x)
        return x

    def forward_non_temporal_audio(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        x = self.audioEncoder(x)
        return x

    def forward_cross_attention(self, x1, x2):
        x1_c = self.crossA2V(src = x1, tar = x2)
        x2_c = self.crossV2A(src = x2, tar = x1)        
        return x1_c, x2_c

    def forward_audio_visual_backend(self, x1, x2): 
        x = torch.cat((x1,x2), 2)    
        x = self.selfAV(src = x, tar = x)       
        x = torch.reshape(x, (-1, 256))
        return x    

    def forward_audio_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_visual_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

# End of source [3]