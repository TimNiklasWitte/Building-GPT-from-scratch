from collections import defaultdict
from util import save_vocabulary, save_document, open_text_file, extract_test_set

def learner(corpus, merge_count=10):
    corpus = corpus.lower()
    words = [list(word) + ['_'] for word in corpus.split()]

    merges = []

    for m in range(merge_count):
        vocab = defaultdict(int)
        for word in words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                vocab[pair] += 1

        most_frequent = max(vocab, key=vocab.get)
        merges.append(most_frequent)

        new_token = ''.join(most_frequent)
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word):
                # Merge durchführen
                if i < len(word) - 1 and (word[i], word[i+1]) == most_frequent:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        words = new_words  # Corpus aktualisieren

        print(f"Iteration {m+1}: merged {most_frequent}")

    token_set = set()
    for word in words:
        for token in word:
            token_set.add(token)
    token_list = sorted(token_set)

    return words, merges, token_list

def segmenter(corpus, merges):
    words = [list(word) + ['_'] for word in corpus.lower().split()]
    for merge in merges:
        new_token = ''.join(merge)
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == merge:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        words = new_words
        text = [''.join(word).strip('_') for word in words]
    return text

text_path = 'data/shakespeare.txt'
corpus = open_text_file(text_path)

#train_corpus, test_corpus = extract_test_set(corpus, test_size=0.1)
#save_document(train_corpus, 'data/train_corpus.txt')
#save_document(test_corpus, 'data/test_corpus.txt')

train_corpus = open_text_file('data/train_corpus.txt')

new_corpus, merges, tokens = learner(train_corpus, merge_count=200)
save_vocabulary(tokens, 'data/vocabulary.txt')



