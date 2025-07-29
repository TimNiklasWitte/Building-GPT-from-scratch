from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

import numpy as np

def main():

    ppl_list = []
    for seq_len in range(1, 13):
        df = load_dataframe(f"./../logs/{seq_len}/")

        ppl = np.min(df.loc[:, "train perplexity"])
        
        ppl_list.append(ppl)

    n_values = list(range(2, 14))

    plt.plot(n_values, ppl_list)

    plt.xlabel("n")
 
    plt.ylabel("Perplexity")
    plt.title("Perplexity of Neural N-Grams")

    plt.grid()
    plt.tight_layout()

    plt.savefig("./plots/Perplexity.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
