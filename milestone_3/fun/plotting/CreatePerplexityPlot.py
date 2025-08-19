from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

import numpy as np

def main():

    

    for k in [50, 150, 250, 1000]:

        ppl_list = []

        for seq_len in [1, 8, 16, 64]:
            df = load_dataframe(f"./../logs/{k}/{seq_len}/")

            ppl = np.min(df.loc[:, "test perplexity"])
            
            ppl_list.append(ppl)

        n_values = [1, 8, 16, 64]

        plt.plot(n_values, ppl_list, label=str(k))

    plt.xlabel("N")
 
    plt.ylabel("Perplexity")
    plt.title("Perplexity of Neural N-Grams")

    plt.grid()
    plt.legend(title="Vocabulary size")
    plt.tight_layout()

    plt.savefig("./plots/Perplexity.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
