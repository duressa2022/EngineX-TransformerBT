import torch.nn as nn
import torch
from engine.transformer_encoder import Encoder
from engine.transformer_decoder import Decoder
class Transformer(nn.Module):
    def __init__(self,dk,dv,h,d_model,d_ff,layers,rate,en_seq_length,en_voc_nums,dec_seq_length,dec_voc_nums,n):
        super(Transformer,self).__init__()
        #define encoder layer for transformer
        self.encoder=Encoder(dk,dv,h,d_model,d_ff,layers,rate,en_seq_length,en_voc_nums,n)
        #define decoder layer for transformer 

        self.decoder=Decoder(dk,dv,h,d_model,d_ff,layers,rate,dec_seq_length,dec_voc_nums,n)
        #define final layer:model output
        self.connected=nn.Linear(in_features=d_model,out_features=dec_voc_nums)
        nn.init.xavier_uniform_(self.connected.weight, gain=1.0)
    
    def forward(self,encoder_input,decoder_input,training=False):
        #encoder_input:(batch_size,seq_length,d_model)
        #decoder_input:(batch_size,seq_length,d_model)
        #create encoder padding mask 
        encoder_padding_mask=self.padding_mask(encoder_input)
        #create decoder padding mask 
        decoder_padding_mask=self.padding_mask(decoder_input)
        #create look ahead padding
        look_ahead_mask=self.lookahead_mask(decoder_input.size(1))
        #create combined padding 
        combined_mask=torch.max(decoder_padding_mask,look_ahead_mask.unsqueeze(0).unsqueeze(0))

        #pass through encoder layer
        encoder_ouput=self.encoder(encoder_input,encoder_padding_mask,training)
        #pass through decoder layer
        decoder_output=self.decoder(decoder_input,encoder_ouput,combined_mask,encoder_padding_mask,training)
        #pass through final layer
        model_output=self.connected(decoder_output)
        #return decoder output
        return model_output
    
    def padding_mask(self,input):
        #input:(batch_size,seq_length)
        
        #create look ahead mask 
        mask=(input==0).float()
        #reshape into:(batch_size,1,1,seq_length)
        #mask has to be:(batch_size,h,seq_length,seq_length)
        mask=mask.unsqueeze(1).unsqueeze(2)
        return mask
    

    def lookahead_mask(self,length):
        #input:length

        #create look ahead mask 
        mask=1-torch.tril(torch.ones((length,length),dtype=torch.float32))
        return mask
#test 
h, d_k, d_v, d_ff, d_model, layers,n = 8, 64, 64, 2048, 512, 6,10000
batch_size, dropout_rate = 64, 0.1
enc_vocab_size, dec_vocab_size = 20, 20
enc_seq_length, dec_seq_length = 5, 5

encoder_input = torch.randint(0, enc_vocab_size, (batch_size, enc_seq_length))
decoder_input = torch.randint(0, dec_vocab_size, (batch_size, dec_seq_length))

model=Transformer(d_k,d_v,h,d_model,d_ff,layers,dropout_rate,enc_seq_length,enc_vocab_size,dec_seq_length,dec_vocab_size,n)

output=model(encoder_input,decoder_input,True)
print(output)
print(output.shape)
    
