#!/bin/bash
set -x #echo on

learning_rates=(
    "0.0001"
    "0.0005"
    "0.001"
)

ks=(
    "100"
    "200"
    "300"
    "500"
)


for learning_rates in "${learning_rates[@]}"; do
    for k in "${ks[@]}"; do
        python3 Training.py "$learning_rates" "$k"
    done
done