#!/bin/bash

set -u

# set eval params
name='roberta_yelp50class'

# setting options
options="--name=$name"


# yelp expts
folder="./results/"
model='tinyllama1B'
for method in 'invink' 'adapmixed' 'amin' 
do
    time python3 -u classify_yelp.py --folder=$folder --dataset="yelp" --model=$model --method=$method $options
done