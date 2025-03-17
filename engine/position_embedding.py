import torch
import torch.nn as nn
import numpy as np
from engine.text_processing_layer import TextProcessingLayer
import logging
logger=logging.getLogger(__name__)

class PositionEmbedding(nn.Module):
    def __init__(self,seq_length,n,voc_nums,dim):
        super(PositionEmbedding,self).__init__()
        #define word embedding 
        self.word_embedding=nn.Embedding(voc_nums,dim)
        #generate pos encoding
        pos_encoding=self.get_position_encoding(seq_length,dim,n)
        #register pos encoding as buffer
        self.register_buffer("pos_encoding",torch.tensor(pos_encoding,dtype=torch.float32))
        #set pos encoding non trainable
        self.pos_encoding.requires_grad=False

    def forward(self,input):
        #input: (batch_size,seq_length)
        #pass through word embedding 
        embedding=self.word_embedding(input)

        #correct dim of pos encoding 
        size=input.size(1)
        encoding=self.pos_encoding[:size,:].unsqueeze(0).expand(input.size(0),-1,-1)
        #add positional encoding to embedding

        return embedding+encoding


    def get_position_encoding(self,seq_length,dim,n=10000):
        #define array with:(seq_length,dim)
        encoding=np.zeros((seq_length,dim))
        #fill the array by using sin and cos
        for k in range(seq_length):
            for i in range(dim//2):
                #fill even index by using sine and 
                encoding[k,2*i]=np.sin(k/(n**((2*i)/dim)))
                #fill odd index by using cosine 
                encoding[k,2*i+1]=np.cos(k/(n**((2*i)/dim)))

        #finally return the encoding 
        return encoding

#test code here 
number_voc=100
seq_length=40
ss=["This is my first sentences",
    "This is my second sentences",
    "This is my third sentences",
    "This is my fourth sentences",
    "This is my fifth sentences"]
processer=TextProcessingLayer(number_vocs=number_voc)
embedding=PositionEmbedding(seq_length,10000,number_voc,5)
vocab=processer.train_tokenizer(ss)
#print(vocab)
test=["other sentences",
      "let me add one",
      "let me add one more"]
encoded=processer.encode(test)
#print(encoded)

vectors=processer.vectorize(encoded,vocab,seq_length)
em=embedding(vectors)
print(em.size())








    