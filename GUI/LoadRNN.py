import torch
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_size=256, num_layers=2, drop_out=0.0):
        super(DeepNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=drop_out)
        self.final = nn.Linear(hidden_size, vocab_size)

    def forward(self, t, hidden):
        embeddings = self.embedding(t)
        out, hidden = self.rnn(embeddings, hidden)
        out = self.final(out[:, -1, :])
        return out, hidden


class RNN_GUI:
    def __init__(self, file_path, device='cpu'):
        self.device = device
        self.load_model(file_path)

    def load_model(self, file_path):
        checkpoint = torch.load(file_path, map_location=self.device)

        # Reconstruct the text processor
        self._w2i = checkpoint['w2i']
        self._i2w = checkpoint['i2w']

        # Retrieve model parameters
        embedding_dim = checkpoint['embedding_dim']
        hidden_size = checkpoint['hidden_size']
        num_layers = checkpoint['num_layers']
        vocab_size = checkpoint['vocab_size']
        drop_out = checkpoint['drop_out']

        # Reconstruct the model with parameters loaded from the checkpoint
        self.model = DeepNetwork(vocab_size, embedding_dim, hidden_size, num_layers=num_layers).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Load losses and accuracies
        self.losses = checkpoint['losses']
        self.accuracies = checkpoint['accuracies']
        self.accuracies_words = checkpoint['accuracies_words']
        self.epoch_acc = checkpoint['epoch_acc']

        self.sequence_length = checkpoint['sequence_length']
        self.num_layers = checkpoint['num_layers']
        self.hidden_size = checkpoint['hidden_size']
        self.eval_text = checkpoint['eval_text']

        print(f"Model loaded from {file_path}")

        self.hidden = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)

    def _generate_words(self, input_indices, neighbours, hidden):
        input_tensor = torch.LongTensor([input_indices]).to(self.device)
        with torch.no_grad():
            output, hidden = self.model(input_tensor, hidden)

        output = output.squeeze(0)

        probabilities = F.softmax(output, dim=0).cpu().numpy()

        top_indices = np.argsort(probabilities)[-neighbours:][::-1]
        top_words = [self._i2w[idx] for idx in top_indices]
        top_probs = probabilities[top_indices]

        return top_indices, top_words, hidden
    

    def gui_generate(self, input_words, hidden=None, neighbours=100, nr_of_suggestions=5):
          input_indices = deque(maxlen=self.sequence_length-1)

          for word in input_words[-self.sequence_length:]:
              idx = self._w2i.get(word, self._w2i['<UNK>'])
              input_indices.append(idx)
              top_indices, top_words, self.hidden = self._generate_words(input_indices, neighbours, self.hidden)
            

          return top_words