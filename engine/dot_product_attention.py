import torch
import torch.nn as nn
import torch.nn.functional as f

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention,self).__init__()

    def forward(self,queries,keys,values,dk,mask=None):
        #queries: (batch_size,seq_length,dk)
        #keys:    (batch_size,seq_length,dk)
        #values:  (batch_size,seq_length,dv)
        #mask:    (batch_size,seq_length,seq_length)

        #calculate attention scores: (batch_size,seq_length,seq_length)
        attention_scores=torch.matmul(queries,keys.transpose(-1,-2))
        #normalize attention scores: (batch_size,seq_length,seq_length)

        normalized_score=attention_scores/torch.sqrt(torch.tensor(dk,dtype=torch.float32))

        #apply mask future attendance:(batch_size,seq_length,seq_length)
        if mask is not None:
            if mask.dim() == 3:           # (batch_size, seq_length, seq_length)
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_length, seq_length)
            normalized_score=normalized_score.masked_fill(mask == 1, -1e9)

        #calculate attention weight: (batch_size,seq_length,seq_length)
        attention_weights=f.softmax(normalized_score,dim=-1)

        #calculte attention values: (batch_size,seq_length,dv)
        attention_values=torch.matmul(attention_weights,values)

        #finally return attention values 
        return attention_values




