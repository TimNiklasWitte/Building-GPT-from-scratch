from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

import numpy as np

def main():

    learning_rate_list = [0.0001, 0.0005, 0.001]
    k_list = [100, 200, 300, 500]

    for learning_rate in learning_rate_list:
        for k in k_list:
            df = load_dataframe(log_dir=f"./../logs/{learning_rate}_{k}")

            fig, axs = plt.subplots(nrows=1, ncols=2)
    
            axs[0].plot(df.loc[:, "train loss"], label="train")
            axs[0].plot(df.loc[:, "validation loss"], label="valid")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("Loss")
            axs[0].grid()
            axs[0].legend()
            axs[0].set_title("Loss")

            axs[1].plot(df.loc[:, "train perplexity"], label="train")
            axs[1].plot(df.loc[:, "validation perplexity"], label="valid")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("Perplexity")
            axs[1].grid()
            axs[1].legend()
            axs[1].set_title("Perplexity")
            
            plt.suptitle(f"Learning rate: {learning_rate}, k: {k}")
            plt.tight_layout()
          
            plt.savefig(f"./plots/{learning_rate}_{k}.png", dpi=200)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
