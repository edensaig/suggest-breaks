#! /bin/bash
in_file=$1
dataset=$2
seed=$3
out_file=./output/${dataset}_${seed}.ipynb
echo Reading from ${in_file}, writing to ${out_file}
RANDOM_STATE=${seed} INPUT_DATASET=${dataset} conda run -n lvml --no-capture-output jupyter nbconvert --to notebook --execute ${in_file} --output ${out_file}
# Example: for i in {1..5}; do echo dispatching run with seed $i; ./run_nb.sh run_experiments.ipynb movielens $i &; done
