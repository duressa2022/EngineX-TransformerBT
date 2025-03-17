import torch.optim as optim
import torch.nn as nn
import torch
from engine.transformer_dataset_json import TransformerDatasetJson
from engine.transformer_network import Transformer
from torch.utils.data import DataLoader,Dataset
import numpy as np
import os 
import time
import pickle
class LRSchedular:
    def __init__(self,optimizer,d_model,warm_steps):
        #create an optimizer for the model
        self.optimizer=optimizer
        #create the size of input embedding
        self.d_model=d_model
        #create warmup steps for the training
        self.warm_steps=warm_steps
        #create number of steps that is taken 
        self.step_nums=0
    def step(self):
        #update  number steps 
        self.step_nums+=1
        #create the learning rate
        lr=(self.d_model**0.5)*min(self.step_nums**(-0.5),self.step_nums*self.warm_steps**(-1.5))
        #update the params group for optimizer 
        for param_group in self.optimizer.param_groups:
            #update the learning rate 
            param_group["lr"]=lr
#create utility function 
def create_padding_mask(seq):
    return (seq!=0).float()
def loss_fn(target,prediction):
    #target: ground_truth:(batch_size,seq_length)
    #prediction: from model: (batch_size,seq_length,num_vocab)

    #create cross entropy loss function
    criterion=nn.CrossEntropyLoss(reduction="none")
    #creating padding mask:(batch_size,seq_length)
    mask=create_padding_mask(target)
    #compute a loss :(batch_size,seq_length)
    loss=criterion(prediction.transpose(1,2),target)
    #apply mask to loss: (batch_size,seq_length)
    masked_loss=loss*mask
    #compute mean over non padded element
    return  masked_loss.sum() / (mask.sum() + 1e-9)
def accuracy_fn(target,prediction):
    #target: ground_truth:(batch_size,seq_length)
    #prediction: from model: (batch_size,seq_length,num_vocab)

    #compute ids for max preds: (batch_size,seq_length)
    pd_ids=torch.argmax(prediction,dim=-1) 
    #create amask 1: for real 0: for padding 
    mask=create_padding_mask(target)
    #compute matching data from target
    correct=(pd_ids==target).float()*mask 
    #return the average values 
    return correct.sum() / (mask.sum() + 1e-9)


#build the transformer here 
dataset=TransformerDatasetJson(fileName="english_amharic_pairs.json",
                               enc_path="future-translation-engine\\engine\\__apps_model\\enc_tokenizer.pkl",
                               dec_path="future-translation-engine\\engine\\__apps_model\\dec_tokenizer.pkl")
#define model params here
h, d_k, d_v, d_ff, d_model, layers,n = 8, 64, 64, 2048, 256, 3,10000
batch_size, dropout_rate,warm_steps,epochs = 20, 0.1,75,20
#define model data here
enc_vocab_size, dec_vocab_size = len(dataset.en_vocab), len(dataset.dec_vocab)
enc_seq_length, dec_seq_length = dataset.max_length, dataset.max_length
#create a dataloader here
print("LOADING DATA..............")
train_loader=DataLoader(dataset,batch_size,shuffle=True)
var_loader=DataLoader([(dataset.val_X[i],dataset.val_Y[i]) for i in range(len(dataset.val_X))],batch_size,shuffle=False)
#create a model here 
print("LOADING MODEL........")
model=Transformer(d_k,d_v,h,d_model,d_ff,layers,dropout_rate,enc_seq_length,enc_vocab_size,dec_seq_length,dec_vocab_size,n)
#create an optimizer here 
optimizer=optim.Adam(model.parameters(),lr=0, betas=(0.9, 0.98), eps=1e-9)
#create lr schedular here
lr_schedular=LRSchedular(optimizer,d_model,warm_steps)
#define a device here 
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#put the model to device 
model.to(device=device)
#traning_loss tracking 
#validation loss tracking 
train_loss_track={}
val_loss_track={}
#start training the model 
print("STARTING TRAINING.......")
start=time.time()
for epoch in range(epochs):
    #put the model on train mode
    model.train()
    #define the total loss var 
    total_loss=0
    #define the total acc var 
    total_acc=0
    #interate through train_loader 
    for index,(src,tgt) in enumerate(train_loader):
        #put the data on the device
        src,tgt=src.to(device),tgt.to(device)
        #encoder input: the src 
        encoder_input=src
        #decoder input: the tgt but not the last 
        decoder_input=tgt[:,:-1]
        #decoder target: The tgt but the last one
        decoder_target=tgt[:,1:]

        #set the grad to zero 
        optimizer.zero_grad()
        #predict by using model 
        prediction=model(encoder_input,decoder_input,True)
        #compute loss
        loss=loss_fn(decoder_target,prediction)
        #compute accuracy 
        acc=accuracy_fn(decoder_target,prediction)
        
        #backpropgate error 
        loss.backward()
        #clip the params
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #optimizer weights 
        optimizer.step()
        #move the schedular 
        lr_schedular.step()

        #update total loss
        total_loss+=loss.item()
        #update total acc
        total_acc+=acc.item()

        #log training data 
        print(f"epochs: {epoch+1}, batch_id: {index}, loss: {loss.item()}:.4f, acc: {acc.item()}:.4f")
    print("------MODEL--VALIDATION------")
    #put the model on eval mode
    model.eval()
    #define the total loss var 
    val_loss=0

    with torch.no_grad():
        for index,(src,tgt) in enumerate(var_loader):
            src,tgt=src.to(device),tgt.to(device)
            encoder_input=src
            decoder_input=tgt[:,:-1]
            decoder_target=tgt[:,1:]
            prediction=model(encoder_input,decoder_input,False)
            loss=loss_fn(decoder_target,prediction)
            acc=accuracy_fn(decoder_target,prediction)
            val_loss+=loss.item()
        
    average_loss_val=val_loss/(len(var_loader)+1e-9)
    average_loss=total_loss/len(train_loader)
    average_acc=total_acc/len(train_loader)
    val_loss_track[epoch+1]=average_loss_val
    train_loss_track[epoch+1]=average_loss
    #save the model weights 
    model_path=os.path.join(os.path.dirname(__file__),"__apps_model",f"weights{epoch+1}.pt")
    torch.save(model.state_dict(),model_path)
    #log training data 
    print(f"epochs: {epoch+1}, average_loss: {average_loss}:.4f, average_acc: {average_acc}:.4f")
    print("=============MODEL=====================")

#save loss for validation/training 
with open(os.path.join(os.path.dirname(__file__),"__apps_model","train_loss.pkl"),"wb") as file:
    pickle.dump(train_loss_track,file)
with open(os.path.join(os.path.dirname(__file__),"__apps_model","val_loss.pkl"),"wb") as file:
    pickle.dump(val_loss_track,file)
print("time took to train: ",time.time()-start,"seconds")
















    
