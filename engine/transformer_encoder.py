import torch.nn as nn
from engine.encoder_layer import EncoderLayer
from engine.position_embedding import PositionEmbedding
from engine.text_processing_layer import TextProcessingLayer

class Encoder(nn.Module):
    def __init__(self,dk,dv,h,d_model,d_ff,layers,rate,seq_length,voc_nums,n):
        super(Encoder,self).__init__()
        #define positional encoding layer:(batch_size,seq_length,d_model)
        self.pos_encoding=PositionEmbedding(seq_length,n,voc_nums,d_model)
        #define dropout layer for first one
        self.drop_out=nn.Dropout(rate)
        #define stack of encoder layer:(batch_size,seq_length,d_model)
        self.encoder_layers=nn.ModuleList([EncoderLayer(h,dk,dv,d_model,d_ff,rate) for _ in range(layers)])
    
    def forward(self,input_sentences,padding_mask,training=False):
        #input_sentences:(batch_size,seq_length)
        #apply positional encoding on sent embedding 
        encoded=self.pos_encoding(input_sentences)
        #apply dropout on the next layer
        output=self.drop_out(encoded) if training else encoded
        #apply an encoder stack through the data 
        for layer in self.encoder_layers:
            output=layer(output,padding_mask,training)
        return output
#test code
h = 8          
d_k = 64      
d_v = 64       
d_ff = 2048   
d_model = 512
layers = 6
rate = 0.1
batch_size = 64
number_vocs = 20  
seq_length = 50
n = 10000

processor = TextProcessingLayer(number_vocs=number_vocs)
sentences = ["Duressa is your token playing with playing"]
input_sentences = ["here is an input one"]

# Train tokenizer and get actual vocab size
vocab = processor.train_tokenizer(sentences)
actual_vocab_size = len(vocab)

# Encode input
encoded = processor.encode(input_sentences)
vectorized = processor.vectorize(encoded, vocab, max_length=seq_length)

# Initialize encoder with ACTUAL vocab size
encoder = Encoder(d_k, d_v, h, d_model, d_ff, layers, rate, seq_length, actual_vocab_size, n)
output = encoder(vectorized, None, True)
print(output.size())
print(output)


