from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch
import json
import pickle

class TextProcessingLayer:
    def __init__(self, number_vocs):
        # Initialize tokenizer with WordPiece model
        self.tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        #here pre tokenize your sentences 
        self.tokenizer.pre_tokenizer = Whitespace()
        # Configure trainer with special tokens
        self.trainer = WordPieceTrainer(
            vocab_size=number_vocs,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"] 
        )
    
    def train_tokenizer(self,input,save_path=None):
        #train tokenizer on input data
        self.tokenizer.train_from_iterator(input, trainer=self.trainer)
        #save tokenizer for path
        if save_path:
            with open(save_path,"wb") as handler:
                pickle.dump(self.tokenizer,handler)
        return self.tokenizer.get_vocab()

    def load_tokenizer(self,save_path):
        #load tokenizer for working 
        with open(save_path,"rb") as handler:
            self.tokenizer=pickle.load(handler)
        #return saved tokenizer 
        return self.tokenizer.get_vocab()

    def encode(self,sentences):
        #return enocoded data: ids,tokens
        return [self.tokenizer.encode(sentence) for sentence in sentences]
    def decode(self,ids):
        #return decoded token: token 
        return self.tokenizer.decode(ids)

    def vectorize(self, encoded_sequences, vocab, max_length):
        """Convert encoded sequences to padded tensors"""
        pad_idx = vocab["[PAD]"]  
        
        vectorized = []
        for enc in encoded_sequences:
            # Extract token IDs
            indices = enc.ids[:max_length]  
            
            # Calculate padding needed
            padding = [pad_idx] * (max_length - len(indices))
            
            # Combine actual tokens and padding
            vectorized.append(indices + padding)

        # Convert to tensor and return
        return torch.tensor(vectorized, dtype=torch.long)
    def build_string(self, ids, vocabs):
        #map index or idx with the token from vocabs
        idx_to_token = {idx: token for token, idx in vocabs.items()}
        #get token by using idx
        sentence_tokens = [idx_to_token[idx.item()] for idx in ids]
        #return token from used 
        return sentence_tokens
    def join_subwords(self, tokens):
        # Remove special tokens
        cleaned_tokens = [token for token in tokens if token not in ["[PAD]", "[CLS]", "[SEP]"]]
        # Join subwords
        sentence = []
        for i, token in enumerate(cleaned_tokens):
            if token.startswith("##"):
                if sentence:  
                    sentence[-1] += token.replace("##", "")
            else:
                sentence.append(token)
        return " ".join(sentence).strip()

    

# Example usage
if __name__ == "__main__":
    # Initialize processing layer
    processor = TextProcessingLayer(
        number_vocs=30  
    )
    
    # Sample sentences
    english_sentences = []
    oromo_senetences=[]
    with open("C:\\Users\\HP\\Downloads\\english_afaan_oromo_pairs.json","r") as file:
        data=json.load(file)
        for item in data:
            english_sentences.append(f"<START> {item["english"].lower()} <EOS>")
            oromo_senetences.append(f"<START> {item["afaan_oromo"].lower()} <EOS>")

    
    # train your tokenizer here
    #vocab_for_english=processor.train_tokenizer(english_sentences,save_path="C:\\Users\\HP\\Documents\\learningpath\\pytorch\\future-translation-engine\\engine\\__apps_model\\enc_tokenizer.pkl")
    #vocab_for_oromo=processor.train_tokenizer(english_sentences,save_path="C:\\Users\\HP\\Documents\\learningpath\\pytorch\\future-translation-engine\\engine\\__apps_model\\dec_tokenizer.pkl")
    vocab=processor.load_tokenizer(save_path="C:\\Users\\HP\\Documents\\learningpath\\pytorch\\future-translation-engine\\engine\\__apps_model\\dec_tokenizer.pkl")
    #encode your input to the tome 
    encoded=processor.encode(oromo_senetences)
    
    # Vectorize sequences with padding
    vectorized = processor.vectorize(encoded, vocab, max_length=30)
    for vector in vectorized:
       tokens=processor.build_string(vector,vocab)
       words=processor.join_subwords(tokens)
       print("vector: ",vector)
       print("word: ",words)

    #decode encoded one into token 
    # decoded=processor.decode(encoded[0].ids)
    # print("decoded",decoded)

    #bulid string from vectorized 
    # ans=processor.build_string(vectorized[1],vocab)
    # print("ans",ans)

    #build words from tokens 
    # ans=processor.join_subwords(ans)
    # print("ans",ans)
    
    # print("Vectorized output:")
    # print(vectorized)
    # print("\nVocabulary tokens:")
    # print(list(vocab.keys()))