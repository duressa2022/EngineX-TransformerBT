import torch
import torch.nn as nn
from engine.dot_product_attention import ScaledDotProductAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, h,dk,dv,d_model):
        super(MultiHeadAttention,self).__init__()
        #define number heads
        self.h=h
        #define dim for key value query 
        self.dk=dk
        #define dim for value matrix 
        self.dv=dv
        #define dim for layer output
        self.d_model=d_model
        #define attention heads
        self.attention=ScaledDotProductAttention()
        #define learnable q projection
        self.wq=nn.Linear(in_features=self.d_model,out_features=self.h*self.dk)
        #define learnable k projection 
        self.wk=nn.Linear(in_features=self.d_model,out_features=self.h*self.dk)
        #define learnable v projection
        self.wv=nn.Linear(in_features=self.d_model,out_features=self.h*self.dv)
        #define final ouput projection 
        self.wo=nn.Linear(in_features=self.dv*self.h,out_features=self.d_model)
        nn.init.xavier_uniform_(self.wq.weight, gain=1.0)
        nn.init.xavier_uniform_(self.wk.weight, gain=1.0)
        nn.init.xavier_uniform_(self.wv.weight, gain=1.0)
        nn.init.xavier_uniform_(self.wo.weight, gain=1.0)

    def forward(self,queries,keys,values,mask=None):
        #queries: (batch_size,seq_length,dk)
        #keys:    (batch_size,seq_length,dk)
        #values:  (batch_size,seq_length,dv)
        #mask:    (batch_size,seq_length,seq_length)

        #projection of queries:->(batch_size,seq_length,h*dk)
        q=self.wq(queries) 
        #projection of keys:   ->(batch_size,seq_length,h*dk)
        k=self.wk(keys)
        #projection of values: ->(batch_size,seq_length,h*dv)
        v=self.wv(values)

        #reshape queries:->(batch_size ,h,seq_length,dk)
        q=self.reshape_tensor(x=q,flag=True)
        #reshape keys: -->(batch_size,h,seq_length,dk)
        k=self.reshape_tensor(x=k,flag=True)
        #reshape values:-->(batch_size,h,seq_length,dv)
        v=self.reshape_tensor(x=v,flag=True)


        #calulate attention scores:(batch_size,h,seq_length,dv)
        attention_values=self.attention(q,k,v,self.dk,mask)
    
        #concatnet attention values:(batch_size,seq_length,h*dv)
        attention_values=self.reshape_tensor(x=attention_values,flag=False)

        #finally output layer for:(batch_size,seq_length,d_model)
        return self.wo(attention_values) 


    def reshape_tensor(self,x,flag):
        #get batch size from  input
        batch_size=x.size(0)
        #get seq length from input 
        seq_length=x.size(1)
        #get number of head from input
        num_heads=x.size(2)

        #reshape :(batch_size,seq_length,h*dk or h*dv)->(batch_size,h,seq_length,dk or dv)
        #or 
        #reshape:(batch_size,h,seq_length,dk or dv)->(batch_size,seq_length,h*dk or h*dv)
        if flag:
            #for parallel multihead opreation
            x=x.view(batch_size,seq_length,self.h,-1)
            x=x.transpose(1,2)
        else:
            #for concatination opreations
            x=x.transpose(1,2).contiguous()
            x=x.view(batch_size,num_heads,-1)
        
        #return reshaped input as output
        return x
    
# Test code
h = 8
d_k = 64
d_v = 64
d_model = 512
batch_size = 64
input_seq_length = 5

queries = torch.rand(batch_size, input_seq_length, d_model)
keys = torch.rand(batch_size, input_seq_length, d_model)
values = torch.rand(batch_size, input_seq_length, d_model)

mask = torch.triu(torch.ones(input_seq_length, input_seq_length), diagonal=1)  # Look-ahead mask
mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_length, seq_length)

mha = MultiHeadAttention(h, d_k, d_v, d_model)
output = mha(queries, keys, values, mask)
print("Output shape (with mask):", output.shape)
print(output)


        




    