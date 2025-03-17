from engine.inference_engine import Translation
from engine.transformer_network import Transformer
import torch
import pickle
import os
#define model params 
h = 8
d_k = 64
d_v = 64
d_model = 512
d_ff = 2048
layers = 6
n = 10000
drop_out = 0

# Define dataset params
enc_seq_length = 20
dec_seq_length = 20
enc_vocab_size = None
dec_vocab_size = None
encoder_path ="future-translation-engine\\engine\\__apps_model\\enc_tokenizer.pkl"
decoder_path = "future-translation-engine\\engine\\__apps_model\\dec_tokenizer.pkl"
load_epoch = 3
model_path = f"future-translation-engine\engine\\__apps_model\\weights{load_epoch}.pt"

# Method for loading tokenizer
def load_tokenizer(path):
    with open(path, "rb") as handler:
        tokenizer = pickle.load(handler)
    return tokenizer

enc_tokenizer = load_tokenizer(encoder_path)
dec_tokenizer = load_tokenizer(decoder_path)
enc_vocab_size = len(enc_tokenizer.get_vocab())
dec_vocab_size = len(dec_tokenizer.get_vocab())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create inference model
model = Transformer(
    dk=d_k,
    dv=d_v,
    h=h,
    d_model=d_model,
    d_ff=d_ff,
    layers=layers,
    rate=drop_out,
    en_seq_length=enc_seq_length,
    en_voc_nums=enc_vocab_size,
    dec_seq_length=dec_seq_length,
    dec_voc_nums=dec_vocab_size,
    n=n
).to(device=device)

model.load_state_dict(torch.load(model_path))
model.eval()
translate = Translation(
    model=model,
    enc_tokenizer=enc_tokenizer,
    dec_tokenizer=dec_tokenizer,
    device=device,
    enc_seq_length=enc_seq_length,
    dec_seq_length=dec_seq_length
)

#define a service method here 
def translate_fn(sentence):
    return translate(sentence=sentence)