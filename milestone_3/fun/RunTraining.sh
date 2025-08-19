#!/bin/bash
#set -x #echo on

ks=(
   "50"
   "150"
   "250"
   "1000"
)

seq_lens=(
   "1"
   "8"
   "16"
   "32"
   "64"
)

for k in "${ks[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        python3 Training.py "$k" "$seq_len"
    done
done