import torch.nn as nn
import torch
from engine.decoder_layer import DecoderLayer
from engine.position_embedding import PositionEmbedding
class Decoder(nn.Module):
    def __init__(self,dk,dv,h,d_model,d_ff,layers,rate,seq_length,voc_nums,n):
        super(Decoder,self).__init__()
        #define positional encoding layer 
        self.pos_encoding=PositionEmbedding(seq_length,n,voc_nums,d_model)
        #define dropout layer:layer pos 
        self.drop_out=nn.Dropout(rate)
        #define decoder stack:stack of decoder 
        self.decoder_layers=nn.ModuleList([DecoderLayer(h,dk,dv,d_model,d_ff,rate) for _ in range(layers)])

    def forward(self,target_sequence,encoder_output,look_ahead_mask,padding_mask,training=False):
        #pass through positional encoding 
        encoded=self.pos_encoding(target_sequence)
        #pass through drop out layer 
        output=self.drop_out(encoded) if training else encoded
        #pass through stack of layers 
        for layer in self.decoder_layers:
            output=layer(output,encoder_output,look_ahead_mask,padding_mask,training)
        return output
#test code  
h = 8
dk = 64
dv = 64
dff = 2048
d_model = 512
layers = 6
batch_size = 64
dropout_rate = 0.1
dec_vocab_size = 20
input_seq_length = 5
rate = 0.1

output_target = torch.randint(0, dec_vocab_size, (batch_size, input_seq_length)) 
enc_output = torch.randn(batch_size, input_seq_length, d_model) 
decoder = Decoder(dk, dv, h, d_model, dff, layers, rate, input_seq_length, dec_vocab_size, 10000)
output = decoder(output_target, enc_output, None, None, True)
print("Output shape:", output.shape)
print(output)