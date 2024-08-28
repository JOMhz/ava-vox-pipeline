import torch
from torch.nn import Module, ModuleList, Conv1d, Sequential, ReLU, Dropout, Parameter
from torch_geometric.nn import Linear, EdgeConv, GATv2Conv, SAGEConv, BatchNorm, GCNConv
import random
import torch.nn.init as init

class SPELLA(Module):
    def __init__(self, cfg):
        super(SPELLA, self).__init__()
        self.use_spf = cfg['use_spf'] # whether to use the spatial features
        self.use_ref = cfg['use_ref']
        self.num_modality = cfg['num_modality']
        self.pre_computed_features = cfg['pre_computed_features']
        channels = [cfg['channel1'], cfg['channel2']]
        final_dim = cfg['final_dim']
        num_att_heads = cfg['num_att_heads']
        dropout = cfg['dropout']
        

        if self.use_spf:
            self.layer_spf1 = Linear(-1, cfg['proj_dim']) # projection layer for spatial features
            self.layer_spf2 = Linear(-1, cfg['lstm_proj_dim']) # projection layer for spatial features

        self.layer011 = Linear(-1, channels[0])
        if self.num_modality == 2:
            self.layer012 = Linear(-1, channels[0])
        self.layer013 = Linear(-1, channels[0])

        self.batch01 = BatchNorm(channels[0])
        self.relu = ReLU()
        self.dropout = Dropout(dropout)

        self.layer11 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch11 = BatchNorm(channels[0])
        self.layer12 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch12 = BatchNorm(channels[0])
        self.layer13 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch13 = BatchNorm(channels[0])

        if num_att_heads > 0:
            self.layer21 = GATv2Conv(channels[0], channels[1], heads=num_att_heads)
        else:
            self.layer21 = SAGEConv(channels[0], channels[1])
            num_att_heads = 1
        self.batch21 = BatchNorm(channels[1]*num_att_heads)

        self.layer31 = SAGEConv(channels[1]*num_att_heads, final_dim)
        self.layer32 = SAGEConv(channels[1]*num_att_heads, final_dim)
        self.layer33 = SAGEConv(channels[1]*num_att_heads, final_dim)

        if self.use_ref:
            self.layer_ref1 = Refinement(final_dim)
            self.layer_ref2 = Refinement(final_dim)
            self.layer_ref3 = Refinement(final_dim)


    def forward(self, x, edge_index, edge_attr, c=None):
        feature_dim = x.shape[1] - self.pre_computed_features
        # print(x[:, :feature_dim//self.num_modality].shape)
        # print(x[:, feature_dim:].shape)
        # print(x[:, feature_dim//self.num_modality:-self.pre_computed_features].shape)
        if self.use_spf:
            x_visual = self.layer011(torch.cat((x[:, :feature_dim//self.num_modality], self.layer_spf1(c)), dim=1))
            x_computed = self.layer013(torch.cat((x[:, feature_dim:], self.layer_spf2(c)), dim=1))
        else:
            x_visual = self.layer011(x[:, :feature_dim//self.num_modality])
            x_computed = self.layer013(x[:, feature_dim:])

        if self.num_modality == 1:
            x = x_visual
        elif self.num_modality == 2:
            x_audio = self.layer012(x[:, feature_dim//self.num_modality:-self.pre_computed_features])
            x = x_visual + x_audio

        x = x + x_computed

        x = self.batch01(x)
        x = self.relu(x)

        edge_index_f = edge_index[:, edge_attr<=0]
        edge_index_b = edge_index[:, edge_attr>=0]

        # Forward-graph stream
        x1 = self.layer11(x, edge_index_f)
        x1 = self.batch11(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.layer21(x1, edge_index_f)
        x1 = self.batch21(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)

        # Backward-graph stream
        x2 = self.layer12(x, edge_index_b)
        x2 = self.batch12(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.layer21(x2, edge_index_b)
        x2 = self.batch21(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)

        # Undirected-graph stream
        x3 = self.layer13(x, edge_index)
        x3 = self.batch13(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)
        x3 = self.layer21(x3, edge_index)
        x3 = self.batch21(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)

        x1 = self.layer31(x1, edge_index_f)
        x2 = self.layer32(x2, edge_index_b)
        x3 = self.layer33(x3, edge_index)

        out = x1+x2+x3

        if self.use_ref:
            xr0 = torch.permute(out, (1, 0)).unsqueeze(0)
            xr1 = self.layer_ref1(torch.softmax(xr0, dim=1))
            xr2 = self.layer_ref2(torch.softmax(xr1, dim=1))
            xr3 = self.layer_ref3(torch.softmax(xr2, dim=1))
            out = torch.stack((xr0, xr1, xr2, xr2), dim=0).squeeze(1).transpose(2, 1).contiguous()

        return out
    
class VOXGNN(Module):
    def __init__(self, cfg):
        super(VOXGNN, self).__init__()
        self.use_spf = cfg['use_spf'] # whether to use the spatial features
        self.use_ref = cfg['use_ref']
        self.num_modality = cfg['num_modality']
        self.pre_computed_features = cfg['pre_computed_features']
        channels = [cfg['channel1'], cfg['channel2']]
        final_dim = cfg['final_dim']
        num_att_heads = cfg['num_att_heads']
        dropout = cfg['dropout']
        

        if self.use_spf:
            self.layer_spf1 = Linear(-1, cfg['proj_dim']) # projection layer for spatial features
            self.layer_spf2 = Linear(-1, cfg['lstm_proj_dim']) # projection layer for spatial features

        self.layer011 = Linear(-1, channels[0])
        if self.num_modality == 2:
            self.layer012 = Linear(-1, channels[0])
        self.layer013 = Linear(-1, channels[0])

        self.batch01 = BatchNorm(channels[0])
        self.relu = ReLU()
        self.dropout = Dropout(dropout)

        self.layer11 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch11 = BatchNorm(channels[0])
        self.layer12 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch12 = BatchNorm(channels[0])
        self.layer13 = EdgeConv(Sequential(Linear(2*channels[0], channels[0]), ReLU(), Linear(channels[0], channels[0])))
        self.batch13 = BatchNorm(channels[0])

        if num_att_heads > 0:
            self.layer21 = GATv2Conv(channels[0], channels[1], heads=num_att_heads)
        else:
            self.layer21 = SAGEConv(channels[0], channels[1])
            num_att_heads = 1
        self.batch21 = BatchNorm(channels[1]*num_att_heads)

        self.layer31 = SAGEConv(channels[1]*num_att_heads, final_dim)
        self.layer32 = SAGEConv(channels[1]*num_att_heads, final_dim)
        self.layer33 = SAGEConv(channels[1]*num_att_heads, final_dim)

        if self.use_ref:
            self.layer_ref1 = Refinement(final_dim)
            self.layer_ref2 = Refinement(final_dim)
            self.layer_ref3 = Refinement(final_dim)


    def forward(self, x, edge_index, edge_attr, c=None):
        feature_dim = x.shape[1] - self.pre_computed_features
        if self.use_spf:
            x_visual = self.layer011(torch.cat((x[:, :feature_dim//self.num_modality], self.layer_spf1(c)), dim=1))
            x_computed = self.layer013(torch.cat((x[:, feature_dim:], self.layer_spf2(c)), dim=1))
        else:
            x_visual = self.layer011(x[:, :feature_dim//self.num_modality])
            x_computed = self.layer013(x[:, feature_dim:])

        if self.num_modality == 1:
            x = x_visual
        elif self.num_modality == 2:
            x_audio = self.layer012(x[:, feature_dim//self.num_modality:-self.pre_computed_features])
            x = x_visual + x_audio

        x = x + x_computed

        x = self.batch01(x)
        x = self.relu(x)

        edge_index_f = edge_index[:, edge_attr<=0]
        edge_index_b = edge_index[:, edge_attr>=0]

        # Forward-graph stream
        x1 = self.layer11(x, edge_index_f)
        x1 = self.batch11(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.layer21(x1, edge_index_f)
        x1 = self.batch21(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)

        # Backward-graph stream
        x2 = self.layer12(x, edge_index_b)
        x2 = self.batch12(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.layer21(x2, edge_index_b)
        x2 = self.batch21(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)

        # Undirected-graph stream
        x3 = self.layer13(x, edge_index)
        x3 = self.batch13(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)
        x3 = self.layer21(x3, edge_index)
        x3 = self.batch21(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)

        x1 = self.layer31(x1, edge_index_f)
        x2 = self.layer32(x2, edge_index_b)
        x3 = self.layer33(x3, edge_index)

        out = x1+x2+x3

        if self.use_ref:
            xr0 = torch.permute(out, (1, 0)).unsqueeze(0)
            xr1 = self.layer_ref1(torch.softmax(xr0, dim=1))
            xr2 = self.layer_ref2(torch.softmax(xr1, dim=1))
            xr3 = self.layer_ref3(torch.softmax(xr2, dim=1))
            out = torch.stack((xr0, xr1, xr2, xr2), dim=0).squeeze(1).transpose(2, 1).contiguous()

        return out

class SPELLB(Module):
    def __init__(self, cfg):
        super(SPELLB, self).__init__()
        channels = [cfg['channel1'], cfg['channel2']]
        in_channels = cfg['channel1']
        out_channels = cfg['channel2']
        final_dim = cfg['final_dim']
        num_att_heads = cfg['num_att_heads']
        dropout = cfg['dropout']

        if self.use_spf:
            self.layer_spf = Linear(-1, cfg['proj_dim']) # projection layer for spatial features

        self.layer011 = Linear(-1, channels[0])
        if self.num_modality == 2:
            self.layer012 = Linear(-1, channels[0])

        self.batch01 = BatchNorm(in_channels)

        # Define the layers for each stream using different convolutional approaches
        # Forward-graph stream
        self.layer11 = EdgeConv(Sequential(
            Linear(2 * in_channels, in_channels), 
            ReLU(), 
            Linear(in_channels, out_channels)
        ))
        self.batch11 = BatchNorm(out_channels)
        self.layer21 = SAGEConv(out_channels, out_channels)  # GATv2Conv layer
        self.batch21 = BatchNorm(out_channels)
        self.layer31 = SAGEConv(out_channels, final_dim)  # SAGEConv layer
        self.batch31 = BatchNorm(final_dim)

        # Backward-graph stream (layers are repeated for each stream)
        self.layer12 = EdgeConv(Sequential(
            Linear(2 * in_channels, in_channels), 
            ReLU(), 
            Linear(in_channels, out_channels)
        ))
        self.batch12 = BatchNorm(out_channels)
        self.layer22 = SAGEConv(out_channels, out_channels)
        self.batch22 = BatchNorm(out_channels)
        self.layer32 = SAGEConv(out_channels, final_dim)
        self.batch32 = BatchNorm(final_dim)

        # Undirected-graph stream
        self.layer13 = EdgeConv(Sequential(
            Linear(2 * in_channels, in_channels), 
            ReLU(), 
            Linear(in_channels, out_channels)
        ))
        self.batch13 = BatchNorm(out_channels)
        self.layer23 = SAGEConv(out_channels, out_channels)
        self.batch23 = BatchNorm(out_channels)
        self.layer33 = SAGEConv(out_channels, final_dim)
        self.batch33 = BatchNorm(final_dim)
        

        self.relu = ReLU()
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, c=None):
        if x.size(1) < 5:
            raise ValueError("Input tensor must have at least five dimensions")

        # print(x.shape)
        # print(x[1])
        # print(x[2])
        # print(x[3])
        # print(x[4])
        # import sys
        # sys.exit()
        

        # # Apply weights and biases to compute the modified values
        # self.custom_weights = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]).cuda()  # Example weights

        # weighted_sum = torch.sum(x[:, 1024:1029] * self.custom_weights, dim=1)
        # # weighted_sum = torch.sum(x[:, 0:5] * self.custom_weights, dim=1)
        
        # # Clone x to prevent in-place modificationss
        # new_x = x.clone()
        
        # # Update the first column of new_x with the modified weighted_sum
        # new_x[:, 0] = weighted_sum
        
        # # Extract the modified first column and maintain the dimensions
        # new_x = new_x[:, 0].unsqueeze(1)

        # Edge splits based on attributess
        edge_index_f = edge_index[:, edge_attr <= 0]
        edge_index_b = edge_index[:, edge_attr >= 0]

        x = self.batch01(x)
        x = self.relu(x)

        x1 = self.layer11(x, edge_index_f)
        x1 = self.batch11(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.layer21(x1, edge_index_f)
        x1 = self.batch21(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.layer31(x1, edge_index_f)
        x1 = self.batch31(x1)

        # Backward-graph stream
        x2 = self.layer12(x, edge_index_b)
        x2 = self.batch12(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.layer22(x2, edge_index_b)
        x2 = self.batch22(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.layer32(x2, edge_index_b)
        x2 = self.batch32(x2)

        # Undirected-graph stream
        x3 = self.layer13(x, edge_index)
        x3 = self.batch13(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)
        x3 = self.layer23(x3, edge_index)
        x3 = self.batch23(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)
        x3 = self.layer33(x3, edge_index)
        x3 = self.batch33(x3)

        # print(x1)
        # Combine outputs from all streams
        out = x1 + x2 + x3
        # print(new_x[0])
        # print(x1[0])
        # print(x2[0])
        # print(x3[0])
        # print(out[0])
        # print("------------------")
        # out = new_x
        return out
    
# class SPELLB(Module):
#     def __init__(self, cfg):
#         super(SPELLB, self).__init__()
#         self.weights = Parameter(torch.full((5,), 1.2))

#     def forward(self, x, edge_index, edge_attr, c=None):
#         # print(x[1])
#             # Check if the input tensor has enough dimensions and width
#         if x.size(1) < 5:
#             raise ValueError("Input tensor must have at least five dimensions")

#         # Compute the mean of each row across the specified columns (0 through 4)
#         mean_values = torch.mean(x[:, 0:5], dim=1)  # Mean across five values of each row

#         # Replace the entire second column with the corresponding mean values for each row
#         x[:, 1] = mean_values

#         # Extract the modified second column, keep the dimensions
#         x = x[:, 1].unsqueeze(1)

#         # Printing the specific element to show its modified value, which is the mean of the respective row's first five numbers
#         return x