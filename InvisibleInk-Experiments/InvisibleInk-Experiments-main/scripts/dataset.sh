#!/bin/bash

set -u
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
    echo "Skipping the MIMIC dataset."
else
    python3 dataset.py --dataset="mimic" --folder="./data/mimic" --embed_model="sentence-transformers/all-mpnet-base-v2" --embed
fi

python3 dataset.py --dataset="yelp" --folder="./data/yelp" --embed_model="sentence-transformers/all-mpnet-base-v2" --embed
python3 dataset.py --dataset="tab" --folder="./data/tab" --embed_model="sentence-transformers/all-mpnet-base-v2" --embed