import pickle
from engine.text_processing_layer import TextProcessingLayer
from torch.utils.data import DataLoader,Dataset
import numpy as np
class TransformerDatasetPickle(Dataset):
    def __init__(self,fileName,n_sentences=10000,train_split=0.9,max_length=50):
        super(TransformerDatasetPickle,self).__init__()
        #load data from the filename 
        with open(fileName,"rb") as data:
            cleaned_data=pickle.load(data)
        #reduce the size| create a dataset
        self.Dataset=cleaned_data[:n_sentences]
        #add <START> and <EOS> on each data
        for index in range(len(self.Dataset)):
            self.Dataset[index,0]=f"<START> {self.Dataset[index,0]} <EOS>"
            self.Dataset[index,1]=f"<START> {self.Dataset[index,1]} <EOS>"
        #shuffle randomly the dataset 
        np.random.shuffle(self.Dataset)
        #create/make the traning size
        train_size=int(n_sentences*train_split)
        #make the traning dataset 
        self.train_data=self.Dataset[:train_size]
        #create a tokenizer for encoder
        self.enc_processer=TextProcessingLayer(number_vocs=1000000)
        #create a tokenizer for decoder 
        self.dec_processer=TextProcessingLayer(number_vocs=1000000)
        #make encoder traning sentences 
        en_senetences=self.train_data[:0].tolist()
        #make decoder training senetences 
        dec_senetencs=self.train_data[:1].tolist()
        #train encoder tokenizer 
        self.en_vocab=self.enc_processer.train_tokenizer(en_senetences)
        #train decoder tokenizer 
        self.dec_vocab=self.dec_processer.train_tokenizer(dec_senetencs)
        #get max enc seq length 
        self.enc_seq_length = max_length
        #get max dec seq length 
        self.dec_seq_length = max_length
        #set the max seq length

        self.max_length=max_length
        #enode the enc_sequence 
        encode_x=self.enc_processer.encode(en_senetences)
        #encode the dec sequence
        encode_y=self.dec_processer.encode(dec_senetencs)
        #vectorize the encode_x
        self.train_X=self.enc_processer.vectorize(encode_x,self.en_vocab,self.max_length)
        #vectorize the encode_y
        self.train_Y=self.dec_processer.vectorize(encode_y,self.dec_vocab,max_length)
    def __len__(self):
        #get length of dataset
        return len(self.train_X)
    def __getitem__(self, index):
        #get x and y train from the dataset 
        return self.train_X[index],self.train_Y[index]

#test code here 
dataset=TransformerDatasetPickle(fileName="")
dataloader=DataLoader(dataset,batch_size=64,shuffle=True)
for x,y in dataloader:
    print(x,y)

    






        



            
    
