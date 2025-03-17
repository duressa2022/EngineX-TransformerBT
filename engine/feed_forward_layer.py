import torch.nn as nn
class FeedForwardLayer(nn.Module):
    def __init__(self, d_model:int,d_ff:int):
        super(FeedForwardLayer,self).__init__()
        #define first linear layer
        self.feed_layer1=nn.Linear(in_features=d_model,out_features=d_ff)
        #define second linear layer 
        self.feed_layer2=nn.Linear(in_features=d_ff,out_features=d_model)
        #define activation layer 
        self.activation=nn.ReLU()
    
    def forward(self,x):
        #x:(batch_size,seq_length,d_model)
        #pass through first linear layer 
        x=self.feed_layer1(x)
        #apply activation function/layer 
        x=self.activation(x)
        #pass through second linear layer 
        x=self.feed_layer2(x)
        #return the out of the last layer 
        return x


    