# engineX-transformerBT

![Python](https://img.shields.io/badge/python-3.8+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg)

**engineX-transformerBT** is a Transformer-based machine translation project focused on translating English to Afaan Oromo, a widely spoken language in Ethiopia. This project implements a custom Transformer architecture from scratch, including encoder and decoder layers, attention mechanisms, and utilities for text processing, dataset creation, and training. It also includes a learning rate scheduler, loss function, and accuracy metric tailored for sequence-to-sequence tasks with padding handling.

The project is designed for researchers and developers interested in NLP, machine translation, or building Transformer models for low-resource languages like Afaan Oromo.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Dataset Generation](#dataset-generation)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Custom Transformer Implementation**: A from-scratch implementation of the Transformer architecture, including encoder, decoder, multi-head attention, and feed-forward layers.
- **English-to-Afaan Oromo Translation**: Supports translation for the low-resource language pair, with a generated dataset of 1000 sentence pairs.
- **Text Processing**: Includes a `TextProcessingLayer` for tokenization, encoding, and vectorization using the Hugging Face `tokenizers` library.
- **Padding-Aware Training**: Handles variable-length sequences with padding masks, masked loss, and accuracy functions.
- **Learning Rate Scheduler**: Implements the Transformer learning rate schedule with warmup and decay, as per the "Attention Is All You Need" paper.
- **Modular Design**: Organized into reusable modules for easy experimentation and extension.

---

## Project Structure

The project is organized into two main directories: `engine` (core implementation) and apis. Below is the folder structure:

```
engineX-transformerBT/
├── engine/                          # Core implementation modules
│   ├── __apps__.py                  # Application utilities (if any)
│   ├── __model__.py                 # Model-related utilities (if any)
│   ├── __pycache__/                 # Python cache files
│   ├── build_transformer.py         # Main script to build the Transformer model
│   ├── decoder_layer.py             # Decoder layer implementation
│   ├── dot_product_attention.py     # Scaled dot-product attention mechanism
│   ├── encoder_layer.py             # Encoder layer implementation
│   ├── feed_forward_layer.py        # Feed-forward neural network layer
│   ├── inference_engine.py          # Inference logic for translation
│   ├── layer_normalization.py       # Layer normalization implementation
│   ├── model_evaluation.py          # Evaluation metrics (e.g., accuracy, loss)
│   ├── multi_head_attention.py      # Multi-head attention mechanism
│   ├── position_embedding.py        # Positional encoding for input embeddings
│   ├── text_processing_layer.py     # Tokenization and text processing utilities
│   ├── transformer_dataset_json.py  # Dataset generation for English-to-Afaan Oromo pairs
│   ├── transformer_decoder.py       # Full Transformer decoder implementation
│   ├── transformer_encoder.py       # Full Transformer encoder implementation
│   └── transformer_network.py       # Complete Transformer network (encoder + decoder)
├── models/                          # Directory for saving/loading models
├── routes/                          # Directory for API routes (if applicable)
├── service/                         # Service-related scripts (if applicable)
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
└── train.py                         # Training script (to be created)
```

- **`engine/`**: Contains all the core components of the Transformer model, including layers, attention mechanisms, and utilities for text processing and dataset creation.
- **`models/`**: for request and reponse model definition 
- **`routes/` and `service/`**: for route definition and service provision/translate.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/engineX-transformerBT.git
   cd engineX-transformerBT
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install the required packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` doesn’t exist yet, create it with the following content:
   ```
   torch>=2.0.0
   tokenizers>=0.13.0
   numpy>=1.21.0
   ```
   Then run the above command.

4. **Verify Installation**:
   Run a quick test to ensure PyTorch is installed:
   ```python
   import torch
   print(torch.__version__)
   ```

---

## Usage

### Dataset Generation

The project includes a script to generate a synthetic dataset of 1000 English-to-Afaan Oromo sentence pairs, located in `engine/transformer_dataset_json.py`. Run the following to generate the dataset:

```bash
python -m engine.transformer_dataset_json
```

This will create `english_oromo_1000_pairs.json` with sentence pairs like `["The house is big", "Manni kun guddaa dha"]`.

### Training the Model

You can train the Transformer model using a script like `train.py`. Below is an example script to train the model on the generated dataset:

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import json
from engine.transformer import Transformer
from engine.text_processing_layer import TextProcessingLayer

# Load dataset
with open("english_oromo_1000_pairs.json", "r") as f:
    pairs = json.load(f)
english_sentences, oromo_sentences = zip(*pairs)

# Initialize text processor
processor = TextProcessingLayer(number_vocs=1000)
processor.train_tokenizer(english_sentences + oromo_sentences)
vocab = processor.get_vocab()

# Vectorize inputs and targets
enc_encoded = processor.encode(english_sentences)
dec_encoded = processor.encode(oromo_sentences)
enc_vectors = processor.vectorize(enc_encoded, vocab, max_length=5)
dec_vectors = processor.vectorize(dec_encoded, vocab, max_length=5)

# Model hyperparameters
h, d_k, d_v, d_ff, d_model, layers, n = 8, 64, 64, 2048, 512, 6, 10000
batch_size, dropout_rate = 64, 0.1
enc_vocab_size, dec_vocab_size = len(vocab), len(vocab)
enc_seq_length, dec_seq_length = 5, 5

# Initialize model, optimizer, and scheduler
model = Transformer(d_k, d_v, h, d_model, d_ff, layers, dropout_rate, enc_seq_length, enc_vocab_size, dec_seq_length, dec_vocab_size, n)
optimizer = optim.Adam(model.parameters(), lr=1.0)

# Learning rate scheduler
class LRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

scheduler = LRScheduler(optimizer, d_model, warmup_steps=4000)

# Loss and accuracy functions
def create_padding_mask(seq):
    return (seq != 0).float()

def loss_fcn(target, prediction):
    criterion = nn.CrossEntropyLoss(reduction='none')
    mask = create_padding_mask(target)
    loss = criterion(prediction.transpose(1, 2), target)
    masked_loss = loss * mask
    return masked_loss.sum() / mask.sum()

def accuracy_fcn(target, prediction):
    pred_ids = torch.argmax(prediction, dim=-1)
    mask = create_padding_mask(target)
    correct = (pred_ids == target).float() * mask
    return correct.sum() / mask.sum()

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    # Process batch (assuming data fits in one batch for simplicity)
    prediction = model(enc_vectors[:batch_size], dec_vectors[:batch_size], training=True)
    loss = loss_fcn(dec_vectors[:batch_size], prediction)
    accuracy = accuracy_fcn(dec_vectors[:batch_size], prediction)
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.7f}")

# Save the model
torch.save(model.state_dict(), "models/transformer_model.pth")
```

Run the training script:

```bash
python train.py
```

### Evaluating the Model

To evaluate the model on a test set, you can modify the training script to include an evaluation phase or use `engine/model_evaluation.py` to compute metrics like accuracy:

```python
# Evaluate
model.eval()
with torch.no_grad():
    prediction = model(enc_vectors[:batch_size], dec_vectors[:batch_size], training=False)
    accuracy = accuracy_fcn(dec_vectors[:batch_size], prediction)
    print(f"Test Accuracy: {accuracy.item():.4f}")
```

---

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- `tokenizers` (Hugging Face) for text processing
- NumPy for numerical operations

Install them via `requirements.txt` as shown in the [Installation](#installation) section.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Inspired by the "Attention Is All You Need" paper by Vaswani et al. (2017).
- Thanks to the PyTorch and Hugging Face communities for their excellent libraries.
- Special appreciation to contributors working on low-resource language translation.

---
