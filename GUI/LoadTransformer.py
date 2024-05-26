import torch

import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_dim, dropout=0.1, max_len=64):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, dropout, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x, mask = None):

        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        if mask == None:
            x = self.transformer_encoder(x)
        else:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)

        x = self.fc_out(x)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=64):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LoadTransformerModel:



    def __init__(self, local_path_to_model):
        self.load_model(local_path_to_model)


    def load_model(self, file_path):
        checkpoint = torch.load(file_path, map_location=torch.device('cpu'))

        # Reconstruct the text processor
        self._w2i = checkpoint['w2i']
        self._i2w = checkpoint['i2w']

        # Retrieve model parameters
        embed_size = checkpoint['embedding_dim']
        hidden_dim = checkpoint['hidden_size']
        num_layer = checkpoint['num_layers']
        vocab_size = checkpoint['vocab_size']
        drop_out = checkpoint['drop_out']
        num_heads = checkpoint['num_heads']
        self.seq_length = checkpoint['sequence_length']

        # Reconstruct the model with parameters loaded from the checkpoint
        self.model = TransformerModel(vocab_size, embed_size, num_heads, num_layer, hidden_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def get_next_k_word(self, input_text, k):
        tokens = input_text
        if len(tokens) > self.seq_length:
            tokens = tokens[-self.seq_length:]
        input_ids = torch.tensor([self._w2i.get(word, self._w2i['<UNK>']) for word in tokens]).unsqueeze(0)
        outputs = self.model(input_ids)
        logits = outputs[0, -1]

        top_k_probs, top_k_indices = torch.topk(logits, k)

        top_k_words = [self._i2w[index] for index in top_k_indices.tolist()]
        top_k_probs = top_k_probs.tolist()

        top_k_words_probs = sorted(zip(top_k_words, top_k_probs), key=lambda x: x[1], reverse=True)
        sorted_top_k_words = [word for word, prob in top_k_words_probs if word != "<UNK>" ]

        return sorted_top_k_words