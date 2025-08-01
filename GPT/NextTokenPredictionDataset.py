import torch
from torch.utils.data import Dataset


class NextTokenPredictionDataset(Dataset):

    def __init__(self, corpus_tokenized, block_size, vocab):
        
        #
        # Convert token into token IDs
        #
        self.corpus_tokenized = corpus_tokenized
        self.map_token_to_id = {token:id for id, token in enumerate(vocab)}
        self.corpus_token_ids = [self.map_token_to_id[token] for token in self.corpus_tokenized]

        self.block_size = block_size

        self.num_data = len(self.corpus_tokenized) - self.block_size

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):

        x = self.corpus_token_ids[index:index+self.block_size]
        x = torch.tensor(x, dtype=torch.long)

        y = self.corpus_token_ids[index+1:index+self.block_size+1]
        y = torch.tensor(y, dtype=torch.long)

        return x, y




