import torch
from engine.transformer_network import Transformer
from engine.text_processing_layer import TextProcessingLayer
import pickle
import os

class Translation:
    def __init__(self, model: Transformer, enc_tokenizer, dec_tokenizer, device,enc_seq_length,dec_seq_length):
        self.model = model
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.device = device
        self.enc_seq_length = enc_seq_length
        self.dec_seq_length = dec_seq_length


    def __call__(self, sentence):
        # Process the sentence
        input_sen = f"<START> {sentence.lower()} <EOS>"
        enc_input = self.enc_tokenizer.encode(input_sen).ids
        enc_input = enc_input[:self.enc_seq_length] + [0] * (self.enc_seq_length - len(enc_input))
        enc_input = torch.tensor([enc_input], dtype=torch.long).to(self.device)

        start_input = self.dec_tokenizer.encode("<START>").ids[0]
        end_input = self.dec_tokenizer.encode("<EOS>").ids[0]
        dec_input_list = [start_input]
        dec_input = torch.tensor([dec_input_list], dtype=torch.long).to(self.device)

        # Inference loop
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.dec_seq_length):
                output = self.model(enc_input, dec_input, False)
                next_token = torch.argmax(output[:, -1, :], dim=-1)
                print("token are: ",output[:,-1,:].topk(5))
                next_ids = next_token.item()

                dec_input_list.append(next_ids)
                dec_input = torch.tensor([dec_input_list], dtype=torch.long).to(self.device)

                if next_ids == end_input:
                    break

        # Decode output
        vocabs = self.dec_tokenizer.get_vocab()
        processer = TextProcessingLayer(number_vocs=len(vocabs))
        # Flatten dec_input to a 1D tensor before passing
        print("other: ",dec_input)
        output = processer.build_string(dec_input[0], vocabs=vocabs)  # Pass the sequence, not batch
        output = processer.join_subwords(output)
        return output

# Define model params
h = 8
d_k = 64
d_v = 64
d_model = 256
d_ff = 2048
layers = 3
n = 10000
drop_out = 0

# Define dataset params
enc_seq_length = 50
dec_seq_length = 50
enc_vocab_size = None
dec_vocab_size = None
encoder_path = os.path.join(os.path.dirname(__file__), "__apps_model", "enc_tokenizer.pkl")
decoder_path = os.path.join(os.path.dirname(__file__), "__apps_model", "dec_tokenizer.pkl")
load_epoch = 20
model_path = os.path.join(os.path.dirname(__file__), "__apps_model", f"weights{load_epoch}.pt")

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

print(translate(sentence="i love programming"))