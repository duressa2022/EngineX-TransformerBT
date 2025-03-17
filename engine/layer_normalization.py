import torch
import torch.nn as nn
class LayerNormConnection(nn.Module):
    def __init__(self,d_model):
        super(LayerNormConnection,self).__init__()
        #define normalized layer/for normalization
        self.norm_layer=nn.LayerNorm(d_model)

    def forward(self,x,sub_layer):
        #x:(batch_size,seq_length,d_model)
        #sub_layer:(batch_size,seq_length,d_model)
        #add skip connection for performance
        add_layer=x+sub_layer
        #normalize skipped connection layer
        return self.norm_layer(add_layer)