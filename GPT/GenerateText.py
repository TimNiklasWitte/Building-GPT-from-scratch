import sys
sys.path.append("./../milestone_1")

import os
import torch 
import torch.nn.functional as F

import numpy as np

from BytePairEncoding import *
from GPT import *

@torch.no_grad
def generate_text_sampling(model, bpe, context, device):

    context_tokenized = bpe.segment(context)

    map_token_to_id = {
        token:id for id, token in enumerate(bpe.vocab)
    }

    token_ids = [map_token_to_id[id] for id in context_tokenized]
    
    stop_token_id = map_token_to_id[". "]

    window_size = model.block_size
    
    cnt_tokens = len(token_ids)

    vocab_size = len(bpe.vocab)

    while cnt_tokens != 100:

        token_ids_window = token_ids[-window_size:]

        token_ids_window = torch.tensor(token_ids_window, dtype=torch.long, device=device)

        # token_ids_window: (n_tokens, )

        # add batch dim
        token_ids_window = token_ids_window.unsqueeze(0)

        # token_ids_window: (1, n_tokens)

        preds = model(token_ids_window)

        logits = preds[0, -1, :]
        distri = F.softmax(logits, dim=0).cpu().numpy()

        next_token_id = np.random.choice(vocab_size, p=distri)

        token_ids.append(next_token_id)

        cnt_tokens += 1

        if next_token_id == stop_token_id:
            break


    # Token ids -> tokens
    
    tokens = [bpe.vocab[token_id] for token_id in token_ids]

    # tokens -> text
    text = "".join(tokens)

    return text

@torch.no_grad
def generate_text_argmax(model, bpe, context, device):

    context_tokenized = bpe.segment(context)

    map_token_to_id = {
        token:id for id, token in enumerate(bpe.vocab)
    }

    token_ids = [map_token_to_id[id] for id in context_tokenized]
    
    stop_token_id = map_token_to_id[". "]

    window_size = model.block_size
    
    cnt_tokens = len(token_ids)


    while cnt_tokens != 100:
        
        token_ids_window = token_ids[-window_size:]

        token_ids_window = torch.tensor(token_ids_window, dtype=torch.long, device=device)

        # token_ids_window: (n_tokens, )

        # add batch dim
        token_ids_window = token_ids_window.unsqueeze(0)

        # token_ids_window: (1, n_tokens)

        preds = model(token_ids_window)

        logits = preds[0, -1, :].cpu().numpy()
     
        next_token_id = np.argmax(logits)

        token_ids.append(next_token_id)

        cnt_tokens += 1

        if next_token_id == stop_token_id:
            break


    # Token ids -> tokens
    
    tokens = [bpe.vocab[token_id] for token_id in token_ids]

    # tokens -> text
    text = "".join(tokens)

    return text


@torch.no_grad
def generate_text_topK(model, bpe, context, k, device):

    context_tokenized = bpe.segment(context)

    map_token_to_id = {
        token:id for id, token in enumerate(bpe.vocab)
    }

    token_ids = [map_token_to_id[id] for id in context_tokenized]
    
    stop_token_id = map_token_to_id[". "]

    window_size = model.block_size
    
    cnt_tokens = len(token_ids)

    vocab_size = len(bpe.vocab)

    while cnt_tokens != 100:
        
        token_ids_window = token_ids[-window_size:]

        token_ids_window = torch.tensor(token_ids_window, dtype=torch.long, device=device)

        # token_ids_window: (n_tokens, )

        # add batch dim
        token_ids_window = token_ids_window.unsqueeze(0)

        # token_ids_window: (1, n_tokens)

        preds = model(token_ids_window)

        logits = preds[0, -1, :]
        distri = F.softmax(logits, dim=0).cpu().numpy()

        #
        # sample only from top k tokens
        #

        # get idxs of top k tokens

        idxs = np.argsort(distri)[::-1] # reverse
        idxs = idxs[:k] # take top k
        
        distri_topK = np.zeros_like(distri)
        distri_topK[idxs] = distri[idxs]

        # sum of probs must be 1

        distri_topK = distri_topK / np.sum(distri_topK)
        
        next_token_id = np.random.choice(vocab_size, p=distri_topK)

        token_ids.append(next_token_id)

        cnt_tokens += 1

        if next_token_id == stop_token_id:
            break


    # Token ids -> tokens
    
    tokens = [bpe.vocab[token_id] for token_id in token_ids]

    # tokens -> text
    text = "".join(tokens)

    return text


@torch.no_grad
def generate_text_topP(model, bpe, context, p, device):

    context_tokenized = bpe.segment(context)

    map_token_to_id = {
        token:id for id, token in enumerate(bpe.vocab)
    }

    token_ids = [map_token_to_id[id] for id in context_tokenized]
    
    stop_token_id = map_token_to_id[". "]

    window_size = model.block_size
    
    cnt_tokens = len(token_ids)

    vocab_size = len(bpe.vocab)

    while cnt_tokens != 100:
        
        token_ids_window = token_ids[-window_size:]

        token_ids_window = torch.tensor(token_ids_window, dtype=torch.long, device=device)

        # token_ids_window: (n_tokens, )

        # add batch dim
        token_ids_window = token_ids_window.unsqueeze(0)

        # token_ids_window: (1, n_tokens)

        preds = model(token_ids_window)

        logits = preds[0, -1, :]
        distri = F.softmax(logits, dim=0).cpu().numpy()

        #
        # sample only from the X best tokens (assuming their combined prob mass is over p)
        #

        # 

        distri_sorted_idxs = np.argsort(distri)[::-1] # reverse order

        distri_sorted = np.sort(distri)[::-1] # reverse order

        distri_sorted_cumsum = np.cumsum(distri_sorted)

        cutoff = np.where(p < distri_sorted_cumsum)[0][0] + 1
        idxs = distri_sorted_idxs[:cutoff]

        distri_topP = np.zeros_like(distri)
        
        distri_topP[idxs] = distri[idxs]

        # renormalize: sum of probs must be 1

        distri_topP = distri_topP / np.sum(distri_topP)
        
        next_token_id = np.random.choice(vocab_size, p=distri_topP)

        token_ids.append(next_token_id)

        cnt_tokens += 1

        if next_token_id == stop_token_id:
            break


    # Token ids -> tokens
    
    tokens = [bpe.vocab[token_id] for token_id in token_ids]

    # tokens -> text
    text = "".join(tokens)

    return text




def main():
    
    #
    # Device
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #
    # Load data
    #

    # Train
    file_path = "./../data/Shakespeare_clean_train.txt"
    with open(file_path) as file:
        corpus_train = file.read()


    context = "hello julia"

    embedd_dim_list = [16, 32, 64]
    k_list = [50, 150, 250, 1000]

    for k in k_list:

        #
        # Train BytePairEncoding
        #

        bpe = BytePairEncoding(num_types=k)

        bpe.train(corpus_train)


        for emb_dim in embedd_dim_list:

            #
            # Init GPT
            #

            # learning rate does not matter for evaluation!
            model = GPT(block_size=64, vocab_size=len(bpe.vocab), emb_dim=emb_dim)

            #
            # Load weights
            #

            root = f"./saved_models/{emb_dim}_{k}/"
            file = os.listdir(root)[-1]
                
            path = root + file 
                    
            model.load_state_dict(torch.load(path, weights_only=True))

            model.to(device)

            model.eval()

            #
            # Generate text
            #

            root = f"./generated_texts/{emb_dim}_{k}"
            os.makedirs(root, exist_ok=True)

            # Sampling
            text = generate_text_sampling(model, bpe, context, device)

            path = f"{root}/Sampling.txt"
            with open(path, "w") as file:
                file.write(text)

            # Argmax
            text = generate_text_argmax(model, bpe, context, device)
            
            path = f"{root}/Argmax.txt"
            with open(path, "w") as file:
                file.write(text)

            # TopK
            text = generate_text_topK(model, bpe, context, k=10, device=device)

            path = f"{root}/TopK.txt"
            with open(path, "w") as file:
                file.write(text)

            # TopP
            text = generate_text_topP(model, bpe, context, p=0.5, device=device)

            path = f"{root}/TopP.txt"
            with open(path, "w") as file:
                file.write(text)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")