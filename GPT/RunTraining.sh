#!/bin/bash
set -x #echo on

emb_dims=(
    "16"
    "32"
    "64"
)

ks=(
   "50"
   "150"
   "200"
   "250"
   "1000"
)




for emb_dim in "${emb_dims[@]}"; do
    for k in "${ks[@]}"; do
        python3 Training.py "$emb_dim" "$k"
    done
done