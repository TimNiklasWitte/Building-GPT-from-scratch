import torch
from torch.utils.data import Dataset


class NextTokenPredictionDataset(Dataset):

    def __init__(self, corpus_tokenized, seq_len, vocab):
     
        self.corpus_tokenized = corpus_tokenized
        self.seq_len = seq_len

        self.map_token_to_id = {token:id for id, token in enumerate(vocab)}

        self.num_data = len(self.corpus_tokenized) - self.seq_len

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):

        window = self.corpus_tokenized[index:index+self.seq_len]
        window = [self.map_token_to_id[token] for token in window]
        window = torch.tensor(window, dtype=torch.long)

        next_token = self.corpus_tokenized[index+self.seq_len]
        next_token_id = self.map_token_to_id[next_token]
        next_token_id = torch.tensor(next_token_id, dtype=torch.long)

        return window, next_token_id




