#!/bin/bash

set -u

# set method and results folder
method="amin"
folder="./results/"
embed="sentence-transformers/all-mpnet-base-v2"
write=100

# setting common options
options="--method=$method --write_every=$write --folder=$folder --embed_model=$embed"
NO_MIMIC="false"

# check for the --nomimic flag in the arguments
for arg in "$@"; do
    shift
    case "$arg" in
        --nomimic) NO_MIMIC="true" ;;
        *) set -- "$@" "$arg";;
    esac
done


if [[ "$NO_MIMIC" == "true" ]]; then
    echo "Skipping MIMIC dataset generation"
else

    # All MIMIC generations using TinyLLaMA 1B
    dataset="mimic"
    model="tinyllama1B"
    num=1000
    toks=500
    ptoks=100
    minibatch=16

    # for k = |V| (full vocabulary generations)
    for temp in 0.8 1.0 1.2
    do
    for eps in 10 5 3 1
    do
    for batch in 32 64 128
    do
    for seed in 0 1 2
    do
        time python3 -u generate_amin.py --model=$model --dataset=$dataset --eps=$eps --temp=$temp --batch=$batch --minibatch=$minibatch --num=$num --seed=$seed --tokens=$toks --ptokens=$ptoks $options
    done 
    done
    done
    done
fi

# All YELP generations using TinyLLaMA 1B
dataset="yelp"
model="tinyllama1B"
num=500
toks=200
ptoks=100
seed=42
minibatch=16

# for k = |V| (full vocabulary generations)
for temp in 0.8 1.0 1.2
do
for eps in 10 5 3 1
do
for batch in 32 64 128
do
    time python3 -u generate_amin.py --model=$model --dataset=$dataset --eps=$eps --temp=$temp --batch=$batch --minibatch=$minibatch --num=$num --seed=$seed --tokens=$toks --ptokens=$ptoks $options
done
done
done

# All TAB generations using TinyLLaMA 1B
dataset="tab"
model="tinyllama1B"
num=100
toks=500
ptoks=100
temp=1.2
batch=8
minibatch=8
seed=42

# for k = |V| (full vocabulary generations)
for eps in 10 5 3 1
do
    time python3 -u generate_amin.py --model=$model --dataset=$dataset --eps=$eps --temp=$temp --batch=$batch --minibatch=$minibatch --num=$num --seed=$seed --tokens=$toks --ptokens=$ptoks $options
done