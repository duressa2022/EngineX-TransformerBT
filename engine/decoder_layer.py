import torch.nn as nn
from engine.multi_head_attention import MultiHeadAttention
from engine.layer_normalization import LayerNormConnection
from engine.feed_forward_layer import FeedForwardLayer
class DecoderLayer(nn.Module):
    def __init__(self,h,dk,dv,d_model,d_ff,rate):
        super(DecoderLayer,self).__init__()
        #define masked attention layer 
        self.masked_attention_layer=MultiHeadAttention(h,dk,dv,d_model)
        #define first drop out layer
        self.drop_out=nn.Dropout(rate)
        #define first normalization layer 
        self.layer_norm1=LayerNormConnection(d_model)
        #define second attention layer
        self.attention_layer=MultiHeadAttention(h,dk,dv,d_model)
        #define second drop out layer
        self.drop_out2=nn.Dropout(rate)
        #define second normalization layer
        self.layer_norm2=LayerNormConnection(d_model)
        #define feed forward layerf
        self.feed_forward=FeedForwardLayer(d_model,d_ff)
        #define third drop out layer
        self.drop_out3=nn.Dropout(rate)
        #define third normalization layer
        self.layer_norm3=LayerNormConnection(d_model)
    def forward(self,x,encoder_out,look_ahead_mask,padding_mask,training=False):
        #x:(batch_size,seq_length,d_model)
        #encoder_out:(batch_size,seq_length,d_model)


        #pass through first attention layer 
        mask_output=self.masked_attention_layer(x,x,x,look_ahead_mask)
        #pass through first dropout layer 
        mask_output=self.drop_out(mask_output) if training else mask_output
        #pass through first normalization layer 
        mask_output=self.layer_norm1(x,mask_output)
        #pass through second attention layer 
        attention_output=self.attention_layer(mask_output,encoder_out,encoder_out,padding_mask)
        #pass through second dropout layer 
        attention_output=self.drop_out2(attention_output) if training else attention_output
        #pass through second normalization layer 
        attention_output=self.layer_norm2(mask_output,attention_output)
        #pass through feed forward layer 
        feed_forward_output=self.feed_forward(attention_output)
        #pass through third dropout layer 
        feed_forward_output=self.drop_out3(feed_forward_output) if training else feed_forward_output
        #pass through third normalization layer 
        feed_forward_output=self.layer_norm3(attention_output,feed_forward_output)
        return feed_forward_output





