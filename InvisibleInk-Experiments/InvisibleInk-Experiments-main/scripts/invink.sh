#!/bin/bash

set -u

# set method and results folder
method="invink"
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
    minibatch=16

    # for k = |V| (full vocabulary generations)
    for temp in 0.8 0.9 1.0 1.1 1.2
    do
    for eps in 10 5 3 1
    do
    for batch in 4 8 16 32
    do
    for seed in 0 1 2
    do
        time python3 -u generate.py --model=$model --dataset=$dataset --eps=$eps --temp=$temp --batch=$batch --minibatch=$minibatch --num=$num --seed=$seed --tokens=$toks $options
    done 
    done
    done
    done

    # for k < |V|
    temp=1.1

    for eps in 10 5 3 1
    do
    for batch in 8 16 32
    do
    for topk in 10 50 100 500
    do
    for seed in 0 1 2
    do
        time python3 -u generate.py --model=$model --dataset=$dataset --eps=$eps --temp=$temp --topk=$topk --batch=$batch --minibatch=$minibatch --num=$num --seed=$seed --tokens=$toks $options
    done 
    done
    done
    done

    # LLaMA3.2 1B model generations
    dataset="mimic"
    model="llama3.2-1B"
    num=1000
    toks=500
    temp=1.1
    minibatch=8

    # for k < |V|
    for eps in 10 5 3 1
    do
    for batch in 8
    do
    for topk in 10 50 100 500
    do
    for seed in 0 1 2
    do
        time python3 -u generate.py --model=$model --dataset=$dataset --eps=$eps --temp=$temp --topk=$topk --batch=$batch --minibatch=$minibatch --num=$num --seed=$seed --tokens=$toks $options
    done 
    done
    done
    done

    # for k = |V| (full vocabulary generations)
    for eps in 10 5 3 1
    do
    for batch in 8
    do
    for seed in 0 1 2
    do
        time python3 -u generate.py --model=$model --dataset=$dataset --eps=$eps --temp=$temp --batch=$batch --minibatch=$minibatch --num=$num --seed=$seed --tokens=$toks $options
    done 
    done
    done


    # LLaMA3 8B model generations
    dataset="mimic"
    model="llama3-8B"
    num=1000
    toks=500
    temp=1.2
    minibatch=4

    # for k < |V|
    for eps in 10 5 3 1
    do
    for batch in 8
    do
    for topk in 10 50 100 500 1000 5000
    do
    for seed in 0 1 2
    do
        time python3 -u generate.py --model=$model --dataset=$dataset --eps=$eps --temp=$temp --topk=$topk --batch=$batch --minibatch=$minibatch --num=$num --seed=$seed --tokens=$toks $options
    done 
    done
    done
    done
fi

# All YELP generations
dataset="yelp"
model="tinyllama1B"
num=500
toks=200
temp=1.2
seed=42
minibatch=16

# for k < |V|
for eps in 10 5 3 1
do
for batch in 8 16 32
do
for topk in 10 50 100
do
    time python3 -u generate.py --model=$model --dataset=$dataset --eps=$eps --temp=$temp --topk=$topk --batch=$batch --minibatch=$minibatch --num=$num --seed=$seed --tokens=$toks $options
done 
done
done

# for k = |V| (full vocabulary generations)
for eps in 10 5 3 1
do
for batch in 4 8 16 32
do
    time python3 -u generate.py --model=$model --dataset=$dataset --eps=$eps --temp=$temp --batch=$batch --minibatch=$minibatch --num=$num --seed=$seed --tokens=$toks $options
done 
done


# All TAB Generations
dataset="tab"
model="tinyllama1B"
num=100
toks=500
temp=1.2
seed=42
batch=8

# for k < |V|
for eps in 10 5 3 1
do
for topk in 10 50 100
do
    time python3 -u generate.py --model=$model --dataset=$dataset --eps=$eps --temp=$temp --topk=$topk --batch=$batch --minibatch=$minibatch --num=$num --seed=$seed --tokens=$toks $options
done 
done

# for k = |V|
for eps in 10 5 3 1
do
    time python3 -u generate.py --model=$model --dataset=$dataset --eps=$eps --temp=$temp --batch=$batch --minibatch=$minibatch --num=$num --seed=$seed --tokens=$toks $options
done