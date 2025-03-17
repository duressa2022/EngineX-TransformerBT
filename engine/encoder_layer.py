import torch
import torch.nn as nn 
from engine.multi_head_attention import MultiHeadAttention
from engine.feed_forward_layer import FeedForwardLayer
from engine.layer_normalization import LayerNormConnection

class EncoderLayer(nn.Module):
    def __init__(self, h,dk,dv,d_model,d_ff,rate):
        super(EncoderLayer,self).__init__()

        #define multi head attention for encoder 
        self.multi_head_attention=MultiHeadAttention(h,dk,dv,d_model)
        #define first drop out layer/for layer1
        self.dropout_layer1=nn.Dropout(rate)
        #define first layer normalization/layer1
        self.layer_norm1=LayerNormConnection(d_model)
        #define feedforward layer/ for layer1
        self.feed_forward=FeedForwardLayer(d_model,d_ff)
        #define second drop out layer/for layer2
        self.dropout_layer2=nn.Dropout(rate)
        #define second layer normalization/layer2
        self.layer_norm2=LayerNormConnection(d_model)

    def forward(self,x,padding_mask,training=False):
        #pass input through multihead attention
        attention_output=self.multi_head_attention(x,x,x,padding_mask)
        #apply dropout for first output from heads
        attention_output=self.dropout_layer1(attention_output) if training else attention_output
        #apply layer normalization on the first layer 
        layer_norm_output=self.layer_norm1(x,attention_output)
        #pass through feed forward layer 
        feed_output=self.feed_forward(layer_norm_output)
        #apply second dropout on the feed out 
        feed_output=self.dropout_layer2(feed_output) if training else feed_output
        #apply layer normalization on feed out 
        output=self.layer_norm2(layer_norm_output,feed_output)
        #return the final result 
        return output

