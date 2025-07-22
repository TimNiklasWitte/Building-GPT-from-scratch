from BPE import *

def main():

    bpe = BPE(num_types=50)

    bpe.vocab.insert(0, "un")
    bpe.vocab.insert(0, "fort")
    bpe.vocab.insert(0, "ly")

    #bpe.vocab.append("un")

    print(bpe.vocab)
    tokenization = bpe.tokenize_word("unfortunately ")
    print(tokenization)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")