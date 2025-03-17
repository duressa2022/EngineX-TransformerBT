import json
from engine.text_processing_layer import TextProcessingLayer
from torch.utils.data import DataLoader,Dataset
import numpy as np
import pickle
class TransformerDatasetJson(Dataset):
    def __init__(self,fileName,n_sentences=1000,train_split=0.9,val_splits=0.1,max_length=50,enc_path=None,dec_path=None,test_path=None):
        super(TransformerDatasetJson,self).__init__()
        #load data from the filename 
        with open(fileName,"r",encoding='utf-8') as data:
            cleaned_data=json.load(data)
        #reduce the size| create a dataset
        self.Dataset=cleaned_data[:n_sentences]
        #add <START> and <EOS> on each data
        for index in range(len(self.Dataset)):
            self.Dataset[index]["english"]=f"<START> {self.Dataset[index]["english"].lower()} <EOS>"
            self.Dataset[index]["amharic"]=f"<START> {self.Dataset[index]["amharic"].lower()} <EOS>"
        #shuffle randomly the dataset 
        np.random.shuffle(self.Dataset)
        #create/make the traning size
        train_size=int(n_sentences*train_split)
        #create/make the validation size
        val_size=int(n_sentences*val_splits)
        #create/make the testing size
        test_size=n_sentences-train_size-val_size
        #make the traning dataset 
        self.train_data=self.Dataset[:train_size]
        #make the validation dataset
        self.val_data=self.Dataset[train_size:train_size+val_size]
        #make the testing dataset
        self.testing_data=self.Dataset[train_size+val_size:]
        #create a tokenizer for encoder
        self.enc_processer=TextProcessingLayer(number_vocs=30000)
        #create a tokenizer for decoder 
        self.dec_processer=TextProcessingLayer(number_vocs=30000)

        #create sentences for tokenization 
        en_senetences=np.array([item["english"] for item in self.train_data])
        dec_senetencs=np.array([item["amharic"] for item in self.train_data])
        print(en_senetences,dec_senetencs)

        #train encoder and decoder tokenizer 
        self.en_vocab=self.enc_processer.train_tokenizer(en_senetences,enc_path)
        self.dec_vocab=self.dec_processer.train_tokenizer(dec_senetencs,dec_path)

        #get max and seq length
        # self.enc_seq_length=max(len(s.split()) for s in en_senetences)
        # self.dec_seq_length=max(len(s.split()) for s in dec_senetencs)
        self.max_length=max_length

        #encode and vectorize the traning data
        en_senetences_train=np.array([item["english"] for item in self.train_data])
        dec_senetencs_train=np.array([item["amharic"] for item in self.train_data])
        encode_x=self.enc_processer.encode(en_senetences_train)
        encode_y=self.dec_processer.encode(dec_senetencs_train)
        self.train_X=self.enc_processer.vectorize(encode_x,self.en_vocab,self.max_length)
        self.train_Y=self.dec_processer.vectorize(encode_y,self.dec_vocab,self.max_length)

        #make the same with validation data
        en_senetences_val=np.array([item["english"] for item in self.val_data])
        dec_senetencs_val=np.array([item["amharic"] for item in self.val_data])
        encode_x_val=self.enc_processer.encode(en_senetences_val)
        encode_y_val=self.dec_processer.encode(dec_senetencs_val)
        self.val_X=self.enc_processer.vectorize(encode_x_val,self.en_vocab,self.max_length)
        self.val_Y=self.dec_processer.vectorize(encode_y_val,self.dec_vocab,self.max_length)

        #make the same with testing data
        en_senetences_test=np.array([item["english"] for item in self.testing_data])
        dec_senetencs_test=np.array([item["amharic"] for item in self.testing_data])
        encode_x_test=self.enc_processer.encode(en_senetences_test)
        encode_y_test=self.dec_processer.encode(dec_senetencs_test)
        self.test_X=self.enc_processer.vectorize(encode_x_test,self.en_vocab,self.max_length)
        self.test_Y=self.dec_processer.vectorize(encode_y_test,self.dec_vocab,self.max_length)

        #save encoder and decoder tokenizer
        # self.save_tokenizer(self.enc_processer.tokenizer,enc_path)
        # self.save_tokenizer(self.dec_processer.tokenizer,dec_path)

        #compute seq length of enc or dec
        self.enc_seq_length=self.train_X.shape[1]
        self.dec_seq_length=self.train_Y.shape[1]
    


    def save_tokenizer(self,tokenizer,path):
        with open(path,"wb") as file:
            pickle.dump(tokenizer,file,protocol=4)

    def load_tokenizer(self,path):
        with open(path,"rb") as file:
            tokenizer=pickle.load(file)
        return tokenizer
    

    def __len__(self):
        #get length of dataset
        return len(self.train_X)
    def __getitem__(self, index):
        #get x and y train from the dataset 
        return self.train_X[index],self.train_Y[index]

#test code here 
dataset=TransformerDatasetJson(fileName="english_amharic_pairs.json",
                               enc_path="future-translation-engine\\engine\\__apps_model\\enc_tokenizer.pkl",
                               dec_path="future-translation-engine\\engine\\__apps_model\\dec_tokenizer.pkl")
dataloader=DataLoader(dataset,batch_size=64,shuffle=True)
# for x,y in dataloader:
#     print(x)